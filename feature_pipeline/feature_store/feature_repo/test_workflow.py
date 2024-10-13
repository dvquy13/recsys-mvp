import subprocess
from datetime import datetime

import pandas as pd
from feast import FeatureStore
from feast.data_source import PushMode

end_date_string = "2020-08-27"
end_date_object = datetime.strptime(end_date_string, "%Y-%m-%d")


def run_demo():
    store = FeatureStore(repo_path=".")
    print("\n--- Run feast apply to setup feature store on Postgres ---")
    subprocess.run(["feast", "apply"])

    print("\n--- Historical features for training ---")
    fetch_historical_features_entity_df(store, for_batch_scoring=False)

    print("\n--- Historical features for batch scoring ---")
    fetch_historical_features_entity_df(store, for_batch_scoring=True)

    print("\n--- Load features into online store ---")
    # Use the fix end date to simulate testing with historical data
    # store.materialize_incremental(end_date=datetime.now())
    store.materialize_incremental(end_date=end_date_object)

    # Reinitiate to fix error not recognizing feature service
    store = FeatureStore(repo_path=".")

    print("\n--- Online features ---")
    fetch_online_features(store)

    print("\n--- Online features retrieved (instead) through a feature service---")
    fetch_online_features(store, source="feature_service")

    print("\n--- Simulate a stream event ingestion of the user rating stats df ---")
    event_df = pd.DataFrame.from_dict(
        {
            "user_id": ["AE2DMKKQV7GGCYWZ7HTGMECN5UWQ"],
            "timestamp": [
                datetime.now(),
            ],
            "user_rating_cnt_90d": [1],
            "user_rating_avg_prev_rating_90d": [4.5],
            "user_rating_list_10_recent_asin": ["AAAAAAAAA,BBBBBBBBB"],
        }
    )
    print(event_df)
    store.push("user_rating_stats_push_source", event_df, to=PushMode.ONLINE)

    print("\n--- Online features again with updated values from a stream push---")
    fetch_online_features(store, source="push")

    print("\n--- Run feast teardown ---")
    # Looks like there is currently a bug that prevent the date clean up on Postgre
    store.teardown()


def fetch_historical_features_entity_df(store: FeatureStore, for_batch_scoring: bool):
    entity_df = pd.DataFrame.from_dict(
        {
            "user_id": [
                "AE254X5CLDBVBPAQKQ5YK2WXRX6A",
                "AE2DMKKQV7GGCYWZ7HTGMECN5UWQ",
                "AEBORGHNCXOCGYEJ4WCULQCQWLAA",
            ],
            "event_timestamp": [
                datetime(2020, 8, 25, 20, 0, 0),
                datetime(2020, 8, 25, 1, 7, 0),
                datetime(2020, 8, 27, 9, 0, 0),
            ],
        }
    )
    if for_batch_scoring:
        entity_df["event_timestamp"] = pd.to_datetime("now", utc=True)

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "user_rating_stats:user_rating_cnt_90d",
            "user_rating_stats:user_rating_avg_prev_rating_90d",
            "user_rating_stats:user_rating_list_10_recent_asin",
        ],
    ).to_df()
    print(training_df.head())


def fetch_online_features(store, source: str = ""):
    entity_rows = [
        {"user_id": "AE254X5CLDBVBPAQKQ5YK2WXRX6A"},
        {"user_id": "AE2DMKKQV7GGCYWZ7HTGMECN5UWQ"},
        {"user_id": "AEBORGHNCXOCGYEJ4WCULQCQWLAA"},
    ]
    if source == "feature_service":
        features_to_fetch = store.get_feature_service("user_rating_v1")
    elif source == "push":
        features_to_fetch = store.get_feature_service("user_rating_v1_fresh")
    else:
        features_to_fetch = [
            "user_rating_stats:user_rating_cnt_90d",
            "user_rating_stats:user_rating_avg_prev_rating_90d",
            "user_rating_stats:user_rating_list_10_recent_asin",
        ]
    returned_features = store.get_online_features(
        features=features_to_fetch,
        entity_rows=entity_rows,
    ).to_dict()
    for key, value in sorted(returned_features.items()):
        print(key, " : ", value)


if __name__ == "__main__":
    run_demo()
