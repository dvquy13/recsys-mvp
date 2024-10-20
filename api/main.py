import asyncio
import json
import os
from typing import List, Optional

import httpx
import redis
from fastapi import FastAPI, HTTPException, Query
from load_examples import custom_openapi
from loguru import logger
from pydantic_models import FeatureRequest

app = FastAPI()

I2V_MODEL_SERVER_URL = os.getenv("I2V_MODEL_SERVER_URL", "http://localhost:3000")
SEQRP_MODEL_SERVER_URL = os.getenv("SEQRP_MODEL_SERVER_URL", "http://localhost:3001")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
FEAST_ONLINE_SERVER_HOST = os.getenv("FEAST_ONLINE_SERVER_HOST", "localhost")
FEAST_ONLINE_SERVER_PORT = os.getenv("FEAST_ONLINE_SERVER_PORT", 6566)

i2v_url = f"{I2V_MODEL_SERVER_URL}/predict"
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


@app.get("/recs/i2i")
async def get_recommendations_i2i(
    item_id: str = Query(
        ..., description="ID of the item to get recommendations for"
    ),  # ... denotes required param in FastAPI
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    debug_info = []
    if debug:
        logger.info(f"Getting recommendations for item_id: {item_id}")
        debug_info.append(f"Getting recommendations for item_id: {item_id}")

    # Step 1: Get recommendations for the item ID from Redis
    recommendations_key = redis_output_i2i_key_prefix + item_id
    rec_data = redis_client.get(recommendations_key)

    if not rec_data:
        error_message = f"No recommendations found for item_id: {item_id}"
        if debug:
            logger.error(error_message)
            debug_info.append(error_message)
        raise HTTPException(status_code=404, detail=error_message)

    if debug:
        logger.info(f"Retrieved recommendations data: {rec_data}")
        debug_info.append(f"Retrieved recommendations data: {rec_data}")

    # Parse the stored recommendation data
    rec_data_json = json.loads(rec_data)
    rec_item_ids = rec_data_json.get("rec_item_ids", [])
    rec_scores = rec_data_json.get("rec_scores", [])

    # Step 2: Limit the output by count if count is provided
    if count is not None:
        rec_item_ids = rec_item_ids[:count]
        rec_scores = rec_scores[:count]

    if debug:
        logger.info(f"Recommendations after limiting: {rec_item_ids}, {rec_scores}")
        debug_info.append(
            f"Recommendations after limiting: {rec_item_ids}, {rec_scores}"
        )

    # Step 3: Format and return the output
    result = {
        "item_id": item_id,
        "recommendations": {"rec_item_ids": rec_item_ids, "rec_scores": rec_scores},
    }

    if debug:
        result["debug_info"] = debug_info

    return result


@app.get(
    "/recs/u2i/last_item_i2i",
    summary="Get recommendations for users based on their most recent items",
)
async def get_recommendations_u2i_last_item_i2i(
    user_id: str = Query(..., description="ID of the user"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    debug_info = []
    if debug:
        logger.info(f"Getting recent items for user_id: {user_id}")
        debug_info.append(f"Getting recent items for user_id: {user_id}")

    # Step 1: Get the recent items for the user
    fr = FeatureRequest(
        features=["user_rating_stats:user_rating_list_10_recent_asin"],
        entities={"user_id": [user_id]},
    )

    features = await fetch_features(fr)
    item_sequences = features["results"][1]["values"][0].split(",")
    last_item_id = item_sequences[-1]

    if debug:
        logger.info(f"Most recently interacted item: {last_item_id}")
        debug_info.append(f"Most recently interacted item: {last_item_id}")

    # Step 2: Call the i2i endpoint internally to get recommendations for that item
    recommendations = await get_recommendations_i2i(last_item_id, count, debug)

    # Step 3: Format and return the output
    result = {
        "user_id": user_id,
        "last_item_id": last_item_id,
        "recommendations": recommendations["recommendations"],
    }

    if debug:
        result["debug_info"] = debug_info

    return result


@app.get("/recs/u2i/rerank", summary="Get recommendations for users")
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
    debug_info = []
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

    if debug:
        _msg = f"Retrieved items: {all_items}"
        logger.info(_msg)
        debug_info.append(_msg)

    # Step 3: Get item_sequence features
    fr = FeatureRequest(
        features=["user_rating_stats:user_rating_list_10_recent_asin"],
        entities={"user_id": [user_id]},
    )

    features = await fetch_features(fr)
    item_sequences = features["results"][1]["values"][0].split(",")
    if debug:
        _msg = f"Retrieved features: {features}, item_sequences: {item_sequences}"
        logger.info(_msg)
        debug_info.append(_msg)

    # Step 4: Rerank
    reranked_recs = await score_seq_rating_prediction(
        user_ids=[user_id] * len(all_items),
        item_sequences=[item_sequences] * len(all_items),
        item_ids=all_items,
    )

    # Step 5: Extract scores from the result
    scores = reranked_recs.get("scores", [])
    returned_items = reranked_recs.get("item_ids", [])
    if not scores or len(scores) != len(all_items):
        error_message = "Mismatch sizes between returned scores and all items"
        if debug:
            logger.error(error_message)
            debug_info.append(error_message)
        raise HTTPException(status_code=500, detail=error_message)

    # Create a list of tuples (item_id, score)
    item_scores = list(zip(returned_items, scores))

    # Sort the items based on the scores in descending order
    item_scores.sort(key=lambda x: x[1], reverse=True)

    # Unzip the sorted items and scores
    sorted_item_ids, sorted_scores = zip(*item_scores)

    # Step 4: Return the reranked recommendations
    result = {
        "user_id": user_id,
        "features": {"item_sequence": item_sequences},
        "recommendations": {
            "rec_item_ids": list(sorted_item_ids)[:count],
            "rec_scores": list(sorted_scores)[:count],
        },
    }

    if debug:
        result["debug_info"] = debug_info

    return result


@app.get("/recs/popular")
async def get_recommendations_popular(
    count: Optional[int] = Query(10, description="Number of popular items to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    debug_info = []
    if debug:
        logger.info(f"Getting popular recommendations")
        debug_info.append(f"Getting popular recommendations")

    # Step 1: Get popular recommendations from Redis
    rec_data = redis_client.get(redis_output_popular_key)

    if not rec_data:
        error_message = "No popular recommendations found"
        if debug:
            logger.error(error_message)
            debug_info.append(error_message)
        raise HTTPException(status_code=404, detail=error_message)

    if debug:
        logger.info(f"Retrieved popular recommendations data: {rec_data}")
        debug_info.append(f"Retrieved popular recommendations data: {rec_data}")

    # Parse the stored recommendation data
    rec_data_json = json.loads(rec_data)
    rec_item_ids = rec_data_json.get("rec_item_ids", [])
    rec_scores = rec_data_json.get("rec_scores", [])

    # Step 2: Limit the output by count if count is provided
    if count is not None:
        rec_item_ids = rec_item_ids[:count]
        rec_scores = rec_scores[:count]

    if debug:
        logger.info(
            f"Popular recommendations after limiting: {rec_item_ids}, {rec_scores}"
        )
        debug_info.append(
            f"Popular recommendations after limiting: {rec_item_ids}, {rec_scores}"
        )

    # Step 3: Format and return the output
    result = {
        "recommendations": {"rec_item_ids": rec_item_ids, "rec_scores": rec_scores},
    }

    if debug:
        result["debug_info"] = debug_info

    return result


# New endpoint to connect to external service
@app.post("/score/seq_rating_prediction")
async def score_seq_rating_prediction(
    user_ids: List[str],
    item_sequences: List[List[str]],
    item_ids: List[str],
    debug: bool = Query(False, description="Enable debug logging"),
):
    debug_info = []
    if debug:
        _msg = f"Calling item2vec_predict with user_ids: {user_ids}, item_sequences: {item_sequences} and item_ids: {item_ids}"
        logger.info(_msg)
        debug_info.append(_msg)

    # Step 1: Prepare the payload for the external service
    payload = {
        "input_data": {
            "user_ids": user_ids,
            "item_sequences": item_sequences,
            "item_ids": item_ids,
        }
    }

    if debug:
        _msg = f"Payload prepared: {payload}"
        logger.info(_msg)
        debug_info.append(_msg)

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
            if debug:
                _msg = f"Response from external service: {response.json()}"
                logger.info(_msg)
                debug_info.append(_msg)
            result = response.json()
            if debug:
                result["debug_info"] = debug_info

            return result
        else:
            error_message = f"External service returned an error: {response.text}"
            if debug:
                logger.error(error_message)
                debug_info.append(error_message)
            raise HTTPException(
                status_code=response.status_code,
                detail=error_message,
            )

    except httpx.HTTPError as e:
        error_message = f"Error connecting to external service: {str(e)}"
        if debug:
            logger.error(error_message)
            debug_info.append(error_message)
        raise HTTPException(status_code=500, detail=error_message)


@app.post("/feature_store/fetch")
async def fetch_features(request: FeatureRequest):
    # Define the URL for the feature store's endpoint
    feature_store_url = f"http://{FEAST_ONLINE_SERVER_HOST}:{FEAST_ONLINE_SERVER_PORT}/get-online-features"
    logger.info(f"Sending request to {feature_store_url}...")

    # Create the payload to send to the feature store
    payload = {"features": request.features, "entities": request.entities}

    # Make the POST request to the feature store
    async with httpx.AsyncClient() as client:
        response = await client.post(feature_store_url, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error fetching features: {response.text}",
        )
