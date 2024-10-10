# Implement an MVP RecSys

# Prerequisite
- Poetry 1.8.3
- Miniconda or alternatives that can create new Python environment with a specified Python version
- Docker

# Set up
- Create a new `.env` file based on `.env.example` and populate the variables there
- Create a new Python 3.11.9 environment: `conda create --prefix .venv python=3.11.9`
- Make sure Poetry use the new Python 3.11.9 environment: `poetry env use .venv/bin/python`
- Install Python dependencies with Poetry: `poetry install`

# Run
- Run `docker compose -f compose.yml up -d mlflow_server redis` to start MLflow Server and Redis
- Run `docker compose -f compose.training.yml run training_pipeline` to train the model
