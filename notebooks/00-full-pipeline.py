import os
import time

import papermill as pm
from loguru import logger

run_timestamp = int(time.time())
output_dir = f"output/{run_timestamp}"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"{run_timestamp=}")
logger.info(f"Notebook outputs will be saved to {output_dir}")

pm.execute_notebook("000-prep-data.ipynb", f"{output_dir}/000-prep-data.ipynb")
pm.execute_notebook("001-features.ipynb", f"{output_dir}/001-features.ipynb")
pm.execute_notebook("010-prep-item2vec.ipynb", f"{output_dir}/010-prep-item2vec.ipynb")
pm.execute_notebook(
    "011-item2vec.ipynb",
    f"{output_dir}/011-item2vec.ipynb",
    parameters={"max_epochs": 2},
)
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
