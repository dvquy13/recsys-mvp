import os

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
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


def get_user_id():
    query = f"select user_id from {schema}.{table_name} limit 1;"
    return pd.read_sql(query, engine)["user_id"].iloc[0]


logger.info(f"Random user_id: <user_id>{get_user_id()}</user_id>")
