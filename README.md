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
- Run `docker compose -f compose.yml up -d mlflow_server kv_store qdrant` to start the supporting services

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

# Airflow
> [!WARNING] Local Docker Airflow requires some serious resources
> You might need to monitor the Airflow service logs at startup to check if they complain anything about your available resources

```shell
# Create a new shell starts at root dir recsys-mvp/feature_pipeline
export ROOT_DIR=$(pwd)
cd airflow
mkdir -p ./dags ./logs ./plugins ./config
cd $ROOT_DIR
export AIRFLOW_UID=$(id -u)
sed -i '' "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$AIRFLOW_UID/" .env
export $(cat .env | grep -v "^#")
docker compose -f compose.airflow.yml up -d
```

# Feature pipeline
Refer to the [Feature Pipeline README](feature_pipeline/README.md)

## Simulate transaction data
- On a new shell, navigate the feature_pipeline dir
- Run `export ROOT_DIR=$(pwd)`
- Run `cd .. && make ml-platform-up` to start PostgreSQL and MinIO services
- Run `cd $ROOT_DIR/notebooks`
- Execute the notebook to populate the raw data into PostgreSQL: `poetry run papermill 001-simulate-oltp.ipynb papermill-output/001-simulate-oltp.ipynb`

## Build feature table with dbt
```shell
# Create a new shell starts at root dir recsys-mvp/feature_pipeline
export ROOT_DIR=$(pwd)
export $(cat .env | grep -v "^#")
cd dbt/feature_store
# Specify credential for dbt to connect to PostgreSQL
cat <<EOF > profiles.yml
feature_store:
  outputs:
    dev:
      dbname: $POSTGRES_DB
      host: $POSTGRES_HOST
      pass: $POSTGRES_PASSWORD
      port: $POSTGRES_PORT
      schema: $POSTGRES_FEATURE_STORE_OFFLINE_SCHEMA
      threads: 1
      type: postgres
      user: $POSTGRES_USER
  target: dev
EOF

# Specify the source data for dbt transformation
cat <<EOF > models/marts/amz_review_rating/sources.yml
version: 2

sources:
  - name: amz_review_rating
    database: $POSTGRES_DB
    schema: $POSTGRES_OLTP_SCHEMA
    tables:
      - name: amz_review_rating_raw
EOF

# Run dbt tranformation
poetry run dbt build --models marts.amz_review_rating
```

## Feature Store

```shell
# Create a new shell starts at root dir recsys-mvp/feature_pipeline
export ROOT_DIR=$(pwd)
export $(cat .env | grep -v "^#")
cd feature_store/feature_repo
poetry run feast apply
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
poetry run feast materialize-incremental $CURRENT_TIME
```

## Append the holdout data to the OLTP source
```shell
cd $ROOT_DIR/notebooks
poetry run papermill 002-append-holdout-to-oltp.ipynb papermill-output/002-append-holdout-to-oltp.ipynb
# To undo, unfollow and run the following
# poetry run papermill 003-undo-append.ipynb papermill-output/003-undo-append.ipynb
```

## Run the feature pipeline using Airflow
```shell
# On a new shell at recsys-mvp root
export ROOT_DIR=$(pwd)
make build-pipeline
# Check the OLTP table to see the latest timestamp
poetry run python scripts/check_oltp_max_timestamp.py
# Expect to see something like 2022-06-15. Later after we run the Airflow pipeline to trigger 002-append-hold-to-oltp notebook we should see new max timestamp denoting new data added
```

- Now go to Airflow UI localhost:8080, username=airflow password=airflow
- Trigger the DAG named `append_oltp`. Check the DAG run logs to see if there are any errors.
- If not, running `poetry run python scripts/check_oltp_max_timestamp.py` again should yield a later date like 2022-07-16.

## Clean up
```shell
poetry run feast teardown
cd $ROOT_DIR
make down
rm -rf db
```

# Troubleshooting
- If you run into Kernel Died error while runninng build training_pipeline, it might possibly due to Docker is not granted enough memory. You can try increasing the Docker memory allocation.