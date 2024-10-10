import os
import time

import papermill as pm
from loguru import logger

run_timestamp = int(time.time())
output_dir = f"output/{run_timestamp}"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"{run_timestamp=}")
logger.info(f"Notebook outputs will be saved to {output_dir}")

pm.execute_notebook(
    "012-batch-precompute.ipynb", f"{output_dir}/012-batch-precompute.ipynb"
)
pm.execute_notebook(
    "013-store-batch-recs.ipynb", f"{output_dir}/013-store-batch-recs.ipynb"
)
pm.execute_notebook(
    "014-store-user-item-sequence.ipynb",
    f"{output_dir}/014-store-user-item-sequence.ipynb",
)
