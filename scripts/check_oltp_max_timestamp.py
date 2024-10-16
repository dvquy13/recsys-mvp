import os

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine

load_dotenv()

table_name: str = "amz_review_rating_raw"

# PostgreSQL connection details
username = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("POSTGRES_HOST")
port = os.getenv("POSTGRES_PORT")
database = os.getenv("POSTGRES_DB")
schema = os.getenv("POSTGRES_OLTP_SCHEMA")

# Create a connection string and engine outside the function
connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
engine = create_engine(connection_string)


def get_curr_oltp_max_timestamp():
    query = f"select max(timestamp) as max_timestamp from {schema}.{table_name};"
    return pd.read_sql(query, engine)["max_timestamp"].iloc[0]


logger.info(f"Max timestamp in OLTP: {get_curr_oltp_max_timestamp()}")
