import json
import os
from datetime import datetime
from typing import List

import requests
from loguru import logger

FEAST_ONLINE_SERVER_HOST = os.getenv("FEAST_ONLINE_SERVER_HOST")
FEAST_ONLINE_SERVER_PORT = os.getenv("FEAST_ONLINE_SERVER_PORT")


def get_recommendations(user_id, top_k_retrieval=100, count=10, debug=False):
    """
    Fetch recommendations for a user by making a GET request to the specified endpoint.

    Args:
        user_id (str): The ID of the user.
        top_k_retrieval (int, optional): Number of items to retrieve before reranking. Defaults to 100.
        count (int, optional): Number of recommendations to return. Defaults to 10.
        debug (bool, optional): Whether to include debug information. Defaults to False.

    Returns:
        dict: The server's response parsed as JSON.

    Raises:
        requests.exceptions.HTTPError: If the HTTP request returned an unsuccessful status code.
    """
    # Define the URL
    url = "http://localhost:8000/recs/u2i/rerank"

    # Set the headers
    headers = {
        "accept": "application/json",
    }

    # Set the query parameters
    params = {
        "user_id": user_id,
        "top_k_retrieval": str(top_k_retrieval),
        "count": str(count),
        "debug": str(debug).lower(),
    }

    try:
        # Make the GET request
        response = requests.get(url, headers=headers, params=params)
        # Raise an exception for HTTP errors
        response.raise_for_status()
        # Return the JSON response
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle exceptions (e.g., network errors, HTTP errors)
        print(f"An error occurred: {e}")
        return None


def get_user_features(user_id):
    # Define the URL
    url = "http://localhost:8000/feature_store/fetch"

    headers = {
        "accept": "application/json",
    }
    payload = {
        "entities": {"user_id": [user_id]},
    }

    try:
        # Make the GET request
        response = requests.post(url, headers=headers, json=payload)
        # Raise an exception for HTTP errors
        response.raise_for_status()
        # Return the JSON response
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle exceptions (e.g., network errors, HTTP errors)
        print(f"An error occurred: {e}")
        return None


def push_new_item_sequence(
    user_id: str, new_items: List[str], sequence_length: int = 10
):
    features = get_user_features(user_id)
    _idx = features["metadata"]["feature_names"].index(
        "user_rating_list_10_recent_asin"
    )
    item_sequence_str = features["results"][_idx]["values"][0]
    item_sequences = item_sequence_str.split(",")
    new_item_sequences = item_sequences + new_items
    new_item_sequences = new_item_sequences[-sequence_length:]
    new_item_sequences_str = ",".join(new_item_sequences)

    event_dict = {
        "user_id": [user_id],
        "timestamp": [str(datetime.now())],
        "dedup_rn": [
            1
        ],  # Mock to conform with current offline schema TODO: Remove this column in the future
        "user_rating_cnt_90d": [1],  # Mock
        "user_rating_avg_prev_rating_90d": [4.5],  # Mock
        "user_rating_list_10_recent_asin": [new_item_sequences_str],
    }
    push_data = {
        "push_source_name": "user_rating_stats_push_source",
        "df": event_dict,
        "to": "online",
    }
    logger.info(f"{event_dict=}")
    r = requests.post(
        f"http://{FEAST_ONLINE_SERVER_HOST}:{FEAST_ONLINE_SERVER_PORT}/push",
        data=json.dumps(push_data),
    )

    if r.status_code != 200:
        logger.error(f"Error: {r.status_code} {r.text}")
