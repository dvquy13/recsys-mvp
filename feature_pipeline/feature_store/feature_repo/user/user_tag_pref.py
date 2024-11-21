from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import String

# Define an entity for the user. You can think of an entity as a primary key used to
# fetch features.
user = Entity(name="user", join_keys=["user_id"])

# Define the PostgreSQL source for the new data
user_tag_pref_source = PostgreSQLSource(
    name="user_tag_pref_source",
    query="SELECT * FROM feature_store_offline.user_tag_pref_v1_features",
    timestamp_field="timestamp",
)

schema = [
    Field(name="user_tag_pref_score_full_list", dtype=String),
]

# Define the new Feature View for user rating stats
user_tag_pref_fv = FeatureView(
    name="user_tag_pref",
    entities=[user],
    ttl=timedelta(
        days=10000
    ),  # Define this to be very long for demo purpose otherwise null data
    schema=schema,
    online=True,
    source=user_tag_pref_source,
    tags={"domain": "user_rating"},
)

# Example FeatureService with the new Feature View
user_tag_pref_v1 = FeatureService(
    name="user_tag_pref_v1",
    features=[
        user_tag_pref_fv,
    ],
)
