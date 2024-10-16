import os
import sys
import time

import papermill as pm
from loguru import logger

sys.path.insert(0, "../../")

from src.io_utils import init_s3_client

run_timestamp = int(time.time())
output_dir = f"output/{run_timestamp}"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"{run_timestamp=}")
logger.info(f"Notebook outputs will be saved to {output_dir}")

pm.execute_notebook(
    "002-append-holdout-to-oltp.ipynb", f"{output_dir}/002-append-holdout-to-oltp.ipynb"
)

if os.getenv("S3_ENDPOINT_URL") is not None:
    s3 = init_s3_client()

    bucket_name = "runs"
    run_fp = f"{output_dir}/002-append-holdout-to-oltp.ipynb"

    s3.upload_file(run_fp, bucket_name, run_fp)

    logger.info(
        f"Notebook run {os.path.abspath(run_fp)} uploaded successfully to S3 at {bucket_name}/{run_fp}!"
    )
