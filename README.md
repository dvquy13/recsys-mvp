# Implement an MVP RecSys

# Prerequisite
- Poetry 1.8.3
- Miniconda or alternatives that can create new Python environment with a specified Python version
- Docker

> [!IMPORTANT] Increase Docker memory to 16GB
> By default after installing Docker it might get only 8 GB of RAM from the host machine.
> Due to this project's poor optimization at the moment, it's required to increase the Docker allocatable memory to at least 12 GB.

# Set up
- Create a new `.env` file based on `.env.example` and populate the variables there
- Create a new Python 3.11.9 environment: `conda create --prefix .venv python=3.11.9`
- Make sure Poetry use the new Python 3.11.9 environment: `poetry env use .venv/bin/python`
- Install Python dependencies with Poetry: `poetry install`

# Start services
- Run `docker compose -f compose.yml up -d mlflow_server redis qdrant` to start the supporting services

# Dev
- Run `make notebook-up` to start Jupyter Lab. The `.env` file would define connection credentials necessary to connect with the supporting services.
- To prep data and train the model, run: `poetry run python 00-training-pipeline.py`

# Docker Run
- Run `docker compose -f compose.pipeline.yml run --rm --build training_pipeline` to train the model
- Run `docker compose -f compose.pipeline.yml run --rm --build batch_reco_pipeline` to run batch recommendations

# API
```shell
make requirements-txt
docker compose -f compose.api.yml up
```

# Troubleshooting
- If you run into Kernel Died error while runninng build training_pipeline, it might possibly due to Docker is not granted enough memory. You can try increasing the Docker memory allocation.