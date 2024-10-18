import pandas as pd
from loguru import logger

holdout_fp = "./data/holdout.parquet"

holdout_df = pd.read_parquet(holdout_fp)

user_id = holdout_df["user_id"].iloc[0]

logger.info(f"Random holdout user_id: <user_id>{user_id}</user_id>")
