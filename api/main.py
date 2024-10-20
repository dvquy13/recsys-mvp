import json
import os
from typing import List, Optional

import httpx
import redis
from fastapi import FastAPI, HTTPException, Query
from load_examples import custom_openapi
from loguru import logger

app = FastAPI()

I2V_MODEL_SERVER_URL = os.getenv("I2V_MODEL_SERVER_URL", "http://localhost:3000")
SEQRP_MODEL_SERVER_URL = os.getenv("SEQRP_MODEL_SERVER_URL", "http://localhost:3001")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

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
def get_recommendations_i2i(
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


@app.get("/recs/u2i")
def get_recommendations_u2i(
    user_id: str = Query(..., description="ID of the user"),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    debug_info = []
    if debug:
        logger.info(f"Getting recent items for user_id: {user_id}")
        debug_info.append(f"Getting recent items for user_id: {user_id}")

    # Step 1: Get the recent items for the user
    recent_items_key = redis_feature_recent_items_key_prefix + user_id
    item_ids_str = redis_client.get(recent_items_key)

    if not item_ids_str:
        error_message = f"No recent items found for user_id: {user_id}"
        if debug:
            logger.error(error_message)
            debug_info.append(error_message)
        raise HTTPException(status_code=404, detail=error_message)

    if debug:
        logger.info(f"Retrieved recent items: {item_ids_str}")
        debug_info.append(f"Retrieved recent items: {item_ids_str}")

    # Step 2: Split the item IDs string by "__"
    item_ids = item_ids_str.split("__")

    # Step 3: Get the most recently interacted item ID
    last_item_id = item_ids[-1]

    if debug:
        logger.info(f"Most recently interacted item: {last_item_id}")
        debug_info.append(f"Most recently interacted item: {last_item_id}")

    # Step 4: Call the i2i endpoint internally to get recommendations for that item
    recommendations = get_recommendations_i2i(last_item_id, count, debug)

    # Step 5: Format and return the output
    result = {
        "user_id": user_id,
        "last_item_id": last_item_id,
        "recommendations": recommendations["recommendations"],
    }

    if debug:
        result["debug_info"] = debug_info

    return result


@app.get("/recs/popular")
def get_recommendations_popular(
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
@app.post("/score/i2i")
async def score_i2i(
    item_1_ids: List[str],
    item_2_ids: List[str],
    debug: bool = Query(False, description="Enable debug logging"),
):
    debug_info = []
    if debug:
        logger.info(
            f"Calling item2vec_predict with item_1_ids: {item_1_ids} and item_2_ids: {item_2_ids}"
        )
        debug_info.append(
            f"Calling item2vec_predict with item_1_ids: {item_1_ids} and item_2_ids: {item_2_ids}"
        )

    # Step 1: Prepare the payload for the external service
    payload = {
        "input_data": {
            "item_1_ids": item_1_ids,
            "item_2_ids": item_2_ids,
        }
    }

    if debug:
        logger.info(f"Payload prepared: {payload}")
        debug_info.append(f"Payload prepared: {payload}")

    # Step 2: Make the POST request to the external service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                i2v_url,
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

        # Step 3: Handle response
        if response.status_code == 200:
            if debug:
                logger.info(f"Response from external service: {response.json()}")
                debug_info.append(f"Response from external service: {response.json()}")
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


@app.get("/recs/i2i/rerank")
async def get_recommendations_i2i_rerank(
    item_id: str = Query(
        ..., description="ID of the item to rerank recommendations for"
    ),
    top_k_retrieval: Optional[int] = Query(
        100, description="Number of retrieval results to use"
    ),
    count: Optional[int] = Query(10, description="Number of recommendations to return"),
    debug: bool = Query(False, description="Enable debug logging"),
):
    debug_info = []
    if debug:
        logger.info(f"Getting reranked recommendations for item_id: {item_id}")
        debug_info.append(f"Getting reranked recommendations for item_id: {item_id}")

    # Step 1: Get popular items
    rec_data = redis_client.get(redis_output_popular_key)

    if not rec_data:
        error_message = "No popular recommendations found"
        if debug:
            logger.error(error_message)
            debug_info.append(error_message)
        raise HTTPException(status_code=404, detail=error_message)

    if debug:
        logger.info(f"Retrieved popular recommendations: {rec_data}")
        debug_info.append(f"Retrieved popular recommendations: {rec_data}")

    # Parse the stored popular recommendation data
    rec_data_json = json.loads(rec_data)
    popular_item_ids = rec_data_json.get("rec_item_ids", [])

    # Limit the popular items by top_k_retrieval
    if top_k_retrieval is not None:
        popular_item_ids = popular_item_ids[:top_k_retrieval]

    if debug:
        logger.info(f"Popular items to rerank: {popular_item_ids}")
        debug_info.append(f"Popular items to rerank: {popular_item_ids}")

    # Step 2: Call the existing item2vec_predict function to get similarity scores
    result = await score_i2i(
        item_1_ids=[item_id] * len(popular_item_ids),
        item_2_ids=popular_item_ids,
        debug=debug,
    )

    # Step 3: Extract scores from the result
    scores = result.get("scores", [])
    if not scores or len(scores) != len(popular_item_ids):
        error_message = "Mismatch between returned scores and popular items"
        if debug:
            logger.error(error_message)
            debug_info.append(error_message)
        raise HTTPException(status_code=500, detail=error_message)

    # Create a list of tuples (item_id, score)
    item_scores = list(zip(popular_item_ids, scores))

    # Sort the items based on the scores in descending order
    item_scores.sort(key=lambda x: x[1], reverse=True)

    # Unzip the sorted items and scores
    sorted_item_ids, sorted_scores = zip(*item_scores)

    # Step 4: Return the reranked recommendations
    result = {
        "item_id": item_id,
        "recommendations": {
            "rec_item_ids": list(sorted_item_ids)[:count],
            "rec_scores": list(sorted_scores)[:count],
        },
    }

    if debug:
        result["debug_info"] = debug_info

    return result
