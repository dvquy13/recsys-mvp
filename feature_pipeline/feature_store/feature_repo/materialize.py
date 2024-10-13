import os
from datetime import datetime

from dotenv import load_dotenv
from feast import FeatureStore
from feast.repo_config import load_repo_config
from loguru import logger

load_dotenv("../../../.env")

logger.info(f"{os.environ.get('POSTGRES_DB')=}")

# Use the fix end date to simulate testing with historical data
end_date_string = "2020-08-30"
end_date_object = datetime.strptime(end_date_string, "%Y-%m-%d")


def run_test():
    repo_config = load_repo_config(
        repo_path="feature_store.yaml", fs_yaml_file=f"./feature_store.yaml"
    )
    store = FeatureStore(repo_path=".", config=repo_config)
    store.materialize_incremental(end_date=end_date_object)


if __name__ == "__main__":
    run_test()
