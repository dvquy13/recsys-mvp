import os
import time

import papermill as pm
from loguru import logger

run_timestamp = int(time.time())
output_dir = f"output/{run_timestamp}"
os.makedirs(output_dir, exist_ok=True)
logger.info(f"{run_timestamp=}")
logger.info(f"Notebook outputs will be saved to {output_dir}")

pm.execute_notebook("001-features.ipynb", f"{output_dir}/001-features.ipynb")
pm.execute_notebook(
    "020-negative-sample.ipynb", f"{output_dir}/020-negative-sample.ipynb"
)
pm.execute_notebook("010-prep-item2vec.ipynb", f"{output_dir}/010-prep-item2vec.ipynb")
pm.execute_notebook(
    "011-item2vec.ipynb",
    f"{output_dir}/011-item2vec.ipynb",
    parameters={"max_epochs": 100},
)
pm.execute_notebook(
    "021-sequence-modeling.ipynb",
    f"{output_dir}/021-sequence-modeling.ipynb",
    parameters={"max_epochs": 100},
)
