import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import httpx
import redis
from fastapi import FastAPI, HTTPException, Query
from load_examples import custom_openapi
from loguru import logger
from pydantic_models import FeatureRequest
from utils import debug_logging_decorator

app = FastAPI()

SEQRP_MODEL_SERVER_URL = os.getenv("SEQRP_MODEL_SERVER_URL", "http://localhost:3000")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
FEAST_ONLINE_SERVER_HOST = os.getenv("FEAST_ONLINE_SERVER_HOST", "localhost")
FEAST_ONLINE_SERVER_PORT = os.getenv("FEAST_ONLINE_SERVER_PORT", 6566)

seqrp_url = f"{SEQRP_MODEL_SERVER_URL}/predict"
redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
)
redis_output_i2i_key_prefix = "output:i2i:"
redis_feature_recent_items_key_prefix = "feature:user:recent_items:"
redis_output_popular_key = "output:popular"

# Set the custom OpenAPI schema with examples
app.openapi = lambda: custom_openapi(
    app,
    redis_client,
    redis_output_i2i_key_prefix,
    redis_feature_recent_items_key_prefix,
)


def get_recommendations_from_redis(
    redis_key: str, count: Optional[int]
) -> Dict[str, Any]:
    rec_data = redis_client.get(redis_key)
    if not rec_data:
        error_message = f"[DEBUG] No recommendations found for key: {redis_key}"
        logger.error(error_message)
        raise HTTPException(status_code=404, detail=error_message)
    rec_data_json = json.loads(rec_data)
    rec_item_ids = rec_data_json.get("rec_item_ids", [])
    rec_scores = rec_data_json.get("rec_scores", [])
    if count is not None:
        rec_item_ids = rec_item_ids[:count]
        rec_scores = rec_scores[:count]
    return {"rec_item_ids": rec_item_ids, "rec_scores": rec_scores}


@app.get("/recs/i2i")
@debug_logging_decorator
async def get_recommendations_i2i(
    item_id: str = Query(..., description="ID of the item to get recommendations for"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    redis_key = f"{redis_output_i2i_key_prefix}{item_id}"
    recommendations = get_recommendations_from_redis(redis_key, count)
    return {
        "item_id": item_id,
        "recommendations": recommendations,
    }


@app.get(
    "/recs/u2i/last_item_i2i",
    summary="Get recommendations for users based on their most recent items",
)
@debug_logging_decorator
async def get_recommendations_u2i_last_item_i2i(
    user_id: str = Query(..., description="ID of the user"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    logger.debug(f"Getting recent items for user_id: {user_id}")

    # Step 1: Get the recent items for the user
    item_sequences = await feature_store_fetch_item_sequence(user_id)
    last_item_id = item_sequences["item_sequence"][-1]

    logger.debug(f"Most recently interacted item: {last_item_id}")

    # Step 2: Call the i2i endpoint internally to get recommendations for that item
    recommendations = await get_recommendations_i2i(last_item_id, count, debug)

    # Step 3: Format and return the output
    result = {
        "user_id": user_id,
        "last_item_id": last_item_id,
        "recommendations": recommendations["recommendations"],
    }

    return result


@app.get("/recs/u2i/rerank", summary="Get recommendations for users")
@debug_logging_decorator
async def get_recommendations_u2i_rerank(
    user_id: str = Query(
        ..., description="ID of the user to provide recommendations for"
    ),
    top_k_retrieval: Optional[int] = Query(
        100, description="Number of retrieval results to use"
    ),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    # Step 1: Get popular and i2i recommendations concurrently
    popular_recs, last_item_i2i_recs = await asyncio.gather(
        get_recommendations_popular(count=top_k_retrieval, debug=False),
        get_recommendations_u2i_last_item_i2i(
            user_id=user_id, count=top_k_retrieval, debug=False
        ),
    )

    # Step 2: Merge popular and i2i recommendations
    all_items = set(popular_recs["recommendations"]["rec_item_ids"]).union(
        set(last_item_i2i_recs["recommendations"]["rec_item_ids"])
    )
    all_items = list(all_items)

    logger.debug("Retrieved items: {}", all_items)

    # Step 3: Get item_sequence features
    item_sequences = await feature_store_fetch_item_sequence(user_id)
    item_sequences = item_sequences["item_sequence"]

    # Step 4: Remove rated items
    set_item_sequences = set(item_sequences)
    set_all_items = set(all_items)
    already_rated_items = list(set_item_sequences.intersection(set_all_items))
    logger.debug(
        f"Removing {len(already_rated_items)} items already rated by this user: {already_rated_items}..."
    )
    all_items = list(set_all_items - set_item_sequences)

    # Step 5: Rerank
    reranked_recs = await score_seq_rating_prediction(
        user_ids=[user_id] * len(all_items),
        item_sequences=[item_sequences] * len(all_items),
        item_ids=all_items,
    )

    # Step 6: Extract scores from the result
    scores = reranked_recs.get("scores", [])
    returned_items = reranked_recs.get("item_ids", [])
    if not scores or len(scores) != len(all_items):
        error_message = "[DEBUG] Mismatch sizes between returned scores and all items"
        logger.debugr("{}", error_message)
        raise HTTPException(status_code=500, detail=error_message)

    # Create a list of tuples (item_id, score)
    item_scores = list(zip(returned_items, scores))

    # Sort the items based on the scores in descending order
    item_scores.sort(key=lambda x: x[1], reverse=True)

    # Unzip the sorted items and scores
    sorted_item_ids, sorted_scores = zip(*item_scores)

    # Step 7: Return the reranked recommendations
    result = {
        "user_id": user_id,
        "features": {"item_sequence": item_sequences},
        "recommendations": {
            "rec_item_ids": list(sorted_item_ids)[:count],
            "rec_scores": list(sorted_scores)[:count],
        },
    }

    return result


@app.get("/recs/popular")
@debug_logging_decorator
async def get_recommendations_popular(
    count: Optional[int] = Query(10, description="Number of popular items to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    recommendations = get_recommendations_from_redis(redis_output_popular_key, count)
    return {"recommendations": recommendations}


# New endpoint to connect to external service
@app.post("/score/seq_rating_prediction")
@debug_logging_decorator
async def score_seq_rating_prediction(
    user_ids: List[str],
    item_sequences: List[List[str]],
    item_ids: List[str],
    debug: bool = Query(False, description="Enable debug logging"),
):
    logger.debug(
        f"Calling seq_rating_predicting with user_ids: {user_ids}, item_sequences: {item_sequences} and item_ids: {item_ids}"
    )

    # Step 1: Prepare the payload for the external service
    payload = {
        "input_data": {
            "user_ids": user_ids,
            "item_sequences": item_sequences,
            "item_ids": item_ids,
        }
    }

    logger.debug(f"Payload prepared: {payload}")

    # Step 2: Make the POST request to the external service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                seqrp_url,
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

        # Step 3: Handle response
        if response.status_code == 200:
            logger.debug(f"Response from external service: {response.json()}")
            result = response.json()

            return result
        else:
            error_message = (
                f"[DEBUG] External service returned an error: {response.text}"
            )
            logger.error(error_message)
            raise HTTPException(
                status_code=response.status_code,
                detail=error_message,
            )

    except httpx.HTTPError as e:
        error_message = f"[DEBUG] Error connecting to external service: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/feature_store/fetch")
async def fetch_features(request: FeatureRequest):
    # Define the URL for the feature store's endpoint
    feature_store_url = f"http://{FEAST_ONLINE_SERVER_HOST}:{FEAST_ONLINE_SERVER_PORT}/get-online-features"
    logger.info(f"Sending request to {feature_store_url}...")

    # Create the payload to send to the feature store
    payload_fresh = {
        "entities": request.entities,
        "features": request.features,
        "full_feature_names": True,
    }

    # Make the POST request to the feature store
    async with httpx.AsyncClient() as client:
        response = await client.post(feature_store_url, json=payload_fresh)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error fetching features: {response.text}",
        )


@app.get("/feature_store/fetch/item_sequence")
@debug_logging_decorator
async def feature_store_fetch_item_sequence(user_id: str):
    """
    Quick work around to get feature sequences from both streaming sources and common online sources
    """
    fr = FeatureRequest(
        entities={"user_id": [user_id]},
        features=[
            "user_rating_stats_fresh:user_rating_list_10_recent_asin",
            "user_rating_stats:user_rating_list_10_recent_asin",
        ],
    )
    response = await fetch_features(fr)

    # Since the values returned from Fease Server will contain an array containing [user_id] + features
    # So the feature idx is offset by 1
    fresh_idx = 1
    common_idx = 2

    feature_results = response["results"]
    get_feature_values = lambda idx: feature_results[idx]["values"][0]
    fresh_item_sequence_str = get_feature_values(fresh_idx)
    if fresh_item_sequence_str is not None:
        item_sequence = fresh_item_sequence_str.split(",")
    else:
        common_item_sequence_str = get_feature_values(common_idx)
        item_sequence = common_item_sequence_str.split(",")

    return {"user_id": user_id, "item_sequence": item_sequence}
