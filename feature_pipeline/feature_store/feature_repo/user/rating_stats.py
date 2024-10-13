from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, PushSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float32, Int64, String

# Define an entity for the user. You can think of an entity as a primary key used to
# fetch features.
user = Entity(name="user", join_keys=["user_id"])

# Define the PostgreSQL source for the new data
user_rating_stats_source = PostgreSQLSource(
    name="user_rating_stats_source",
    query="SELECT * FROM feature_store_offline.user_rating_stats",
    timestamp_field="timestamp",
)

# Define the new Feature View for user rating stats
user_rating_stats_fv = FeatureView(
    name="user_rating_stats",
    entities=[user],
    ttl=timedelta(
        days=10000
    ),  # Define this to be very long for demo purpose otherwise null data
    schema=[
        Field(name="user_rating_cnt_90d", dtype=Int64),
        Field(name="user_rating_avg_prev_rating_90d", dtype=Float32),
        Field(name="user_rating_list_10_recent_asin", dtype=String),
    ],
    online=True,
    source=user_rating_stats_source,
    tags={"domain": "user_rating"},
)

# Example FeatureService with the new Feature View
user_activity_v1 = FeatureService(
    name="user_rating_v1",
    features=[
        user_rating_stats_fv,
    ],
)

# Defines a way to push data (to be available offline, online or both) into Feast.
user_rating_stats_push_source = PushSource(
    name="user_rating_stats_push_source",
    batch_source=user_rating_stats_source,
)

# Defines a slightly modified version of the feature view from above, where the source
# has been changed to the push source. This allows fresh features to be directly pushed
# to the online store for this feature view.
user_rating_stats_fresh_fv = FeatureView(
    name="user_rating_stats_fresh",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="user_rating_cnt_90d", dtype=Int64),
        Field(name="user_rating_avg_prev_rating_90d", dtype=Float32),
        Field(name="user_rating_list_10_recent_asin", dtype=String),
    ],
    online=True,
    source=user_rating_stats_push_source,  # Changed from above
    tags={"domain": "user_rating"},
)

# Fresh source
user_activity_v1_fresh = FeatureService(
    name="user_rating_v1_fresh",
    features=[
        user_rating_stats_fresh_fv,
    ],
)
