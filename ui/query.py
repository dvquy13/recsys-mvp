import os
from typing import List

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

table_name: str = "user_rating_stats"

# PostgreSQL connection details
username = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("POSTGRES_HOST")
port = os.getenv("POSTGRES_PORT")
database = os.getenv("POSTGRES_DB")
schema = os.getenv("POSTGRES_FEATURE_STORE_OFFLINE_SCHEMA")

# Create a connection string and engine outside the function
connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
engine = create_engine(connection_string)


def get_users():
    query = f"select distinct user_id from {schema}.{table_name}"
    return pd.read_sql(query, engine)["user_id"].values.tolist()


def get_items_df(user_id):
    query = f"""
select
	parent_asin as item_id,
	timestamp,
	title,
    categories
from 
	oltp.amz_review_rating_raw arrr
where
	user_id = '{user_id}'
order by
    timestamp desc
"""
    return pd.read_sql(query, engine)


def get_items_metadata(item_ids: List[str]):
    item_ids_str = ",".join([f"'{item_id}'" for item_id in item_ids])
    query = f"""
select distinct
    parent_asin as item_id,
    title,
    categories
from 
    oltp.amz_review_rating_raw arrr
where
    parent_asin in ({item_ids_str})
"""
    return pd.read_sql(query, engine)
