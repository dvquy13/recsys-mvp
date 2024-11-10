# Implement an MVP RecSys

# Prerequisite
- Poetry >= 1.8.3
- Miniconda or alternatives that can create new Python environment with a specified Python version
- Docker
- PostgreSQL
  - For Mac, run: `brew install postgresql`
  - For Ubuntu, run: `sudo apt-get update && sudo apt-get install -y gcc libpq-dev`

> [!IMPORTANT]
> **Increase Docker memory to 16GB**
> On MacOS, By default after installing Docker Desktop it might get only 8 GB of RAM from the host machine.
> Due to this project's poor optimization at the moment, it's required to increase the Docker allocatable memory to at least 14 GB.

# Set up
- Create a new `.env` file based on `.env.example` and populate the variables there
- Set up env var $ROOT_DIR: `export ROOT_DIR=$(pwd) && sed "s|^ROOT_DIR=.*|ROOT_DIR=$ROOT_DIR|" .env > .tmp && mv .tmp .env`
- Run `export $(grep -v '^#' .env | xargs)` to load the variables
- Create a new Python 3.11.9 environment: `conda create --prefix .venv python=3.11.9`
- Make sure Poetry use the new Python 3.11.9 environment: `poetry env use .venv/bin/python`
- Install Python dependencies with Poetry: `poetry install`

> [!TIP]
> **VSCode auto load `.env`**
> For convenience, you should enable your IDE to automatically load the `.env` as environment vars.
> If using VSCode then this is done automatically as long as you have VSCode Python Extension installed.

# Start services
## Common services
- Run `make ml-platform-up && make ml-platform-logs` to start the supporting services
- Wait until you see "Booting worker with pid..." then you can Ctrl + C to exit the logs following process

## Airflow
```shell
cd $ROOT_DIR
make airflow-up && make airflow-logs
# To check airflow logs: `make airflow-logs`

# Below 4 lines are there just in case Airflow does not start correctly due to permission issue
# export AIRFLOW_UID=$(id -u)
# sed "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$AIRFLOW_UID/" .env > .tmp && mv .tmp .env
# export $(cat .env | grep -v "^#")
# docker compose -f compose.airflow.yml up -d
```

- Wait until you see "airflow-webserver: Booting worker with pid..." then you can Ctrl + C to exit the logs following process

> [!WARNING]
> **Local Docker Airflow requires some serious resources**
> You might need to monitor the Airflow service logs at startup to check if they complain anything about your available resources

# Prepare data
## Sample data
```shell
echo "To start, we need to sample our main dataset from the bigger upstream dataset"
cd $ROOT_DIR/notebooks && poetry run python 00-prep-data.py
```

## Simulate transaction data
```shell
echo "Execute the notebook to populate the raw data into PostgreSQL"
mkdir -p $ROOT_DIR/feature_pipeline/notebooks/papermill-output
cd $ROOT_DIR/feature_pipeline/notebooks && poetry run papermill 001-simulate-oltp.ipynb papermill-output/001-simulate-oltp.ipynb
```

# Feature pipeline
The goal of feature pipeline is to keep the feature in feature store updated via daily batch jobs.

> [!IMPORTANT]
> This section assumes you have `make ml-platform-up` and `make airflow-up` running

## Build feature table with dbt
```shell
cd $ROOT_DIR/feature_pipeline/dbt/feature_store
echo "Specify credential for dbt to connect to PostgreSQL"
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

echo "Specify the source data for dbt transformation"
cat <<EOF > models/marts/amz_review_rating/sources.yml
version: 2

sources:
  - name: amz_review_rating
    database: $POSTGRES_DB
    schema: $POSTGRES_OLTP_SCHEMA
    tables:
      - name: amz_review_rating_raw
EOF

echo "Run dbt tranformation"
poetry run dbt deps
poetry run dbt build --models marts.amz_review_rating
```

## Feature Store

### Initial materialization
```shell
# CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
# We can not use CURRENT_TIME here since it would mark the latest ingestion at CURRENT_TIME which is way pass the last timestamp for our data
# So later we can not demo the flow to update feature store
cd $ROOT_DIR && MATERIALIZE_CHECKPOINT_TIME=$(poetry run python scripts/check_oltp_max_timestamp.py 2>&1 | awk -F'<ts>|</ts>' '{print $2}')
echo "MATERIALIZE_CHECKPOINT_TIME=$MATERIALIZE_CHECKPOINT_TIME"
cd $ROOT_DIR/feature_pipeline/feature_store/feature_repo
poetry run feast apply
poetry run feast materialize-incremental $MATERIALIZE_CHECKPOINT_TIME
```

### Feature Server
Set up Feature Server to serve both online and offline features

```shell
cd $ROOT_DIR
make feature-server-up
sleep 5 && echo "Visit Feate Store Web UI at: http://localhost:${FEAST_UI_PORT:-8887}"
```

Make feature request to Feature Server:
```shell
# Create a new shell
USER_ID=$(poetry run python scripts/get_holdout_user_id.py 2>&1 | awk -F'<user_id>|</user_id>' '{print $2}') && echo $USER_ID
# Use double quotes in curl -d to enable env var $USER_ID substitution
curl -X POST \
  "http://localhost:6566/get-online-features" \
  -d "{
    \"features\": [
        \"user_rating_stats:user_rating_cnt_90d\",
        \"user_rating_stats:user_rating_avg_prev_rating_90d\",
        \"user_rating_stats:user_rating_list_10_recent_asin\"
    ],
    \"entities\": {
      \"user_id\": [
        \"$USER_ID\"
      ]
    }
  }" | jq
```

- Note down the request value, later we would compare that after update our online feature store via Airflow batch jobs

## Run the feature pipeline using Airflow

### Append holdout data to OLTP
Here we manually update our source OLTP data with new data, simulating new data generated by users.

```shell
cd $ROOT_DIR
make build-pipeline
echo "Check the OLTP table to see the latest timestamp"
poetry run python scripts/check_oltp_max_timestamp.py
echo "Expect to see something like 2022-06-15. Later after we run the Airflow pipeline to trigger 002-append-hold-to-oltp notebook we should see new max timestamp denoting new data added"
```

- Now go to Airflow UI http://localhost:8081, username=airflow password=airflow
- Trigger the DAG named `append_oltp`. Check the DAG run logs to see if there are any errors.
  - In case the error log says: "Failed to establish connection to Docker host unix://var/run/docker.sock: Error while fetching server API version: ('Connection aborted.', PermissionError(13, 'Permission denied'))", it's likely you need to grant 'rw' permission to all users for the docker.sock file. Do so by running `sudo chmod 666 /var/run/docker.sock`. Read [this SO](https://stackoverflow.com/questions/62499661/airflow-dockeroperator-fails-with-permission-denied-error) for more details.
- If no error, running `poetry run python scripts/check_oltp_max_timestamp.py` again should yield a later date like 2022-07-16, which means just now we have a new round of OLTP data in our system.

> [!NOTE]
> **Undo the append**
> In case you want to undo the append, run: `cd $ROOT_DIR/feature_pipeline/notebooks && poetry run papermill 003-undo-append.ipynb papermill-output/003-undo-append.ipynb`

### Update features
- Now after we have new data in our OLTP source, we should be able to update our Feature Store
- Let's try to use the scheduling functionality from Airflow this time
- Go to [feature_pipeline/dags/update_features.py](feature_pipeline/dags/update_features.py), update `schedule_interval` to some minutes later in the future
- On Airflow Web UI, turn on the `update_features` DAG, then wait and see Airflow trigger run for DAG `update_features`

Now we run the request to online feature store again to see if the $USER_ID has updated features:
```shell
curl -X POST \
  "http://localhost:6566/get-online-features" \
  -d "{
    \"features\": [
        \"user_rating_stats:user_rating_cnt_90d\",
        \"user_rating_stats:user_rating_avg_prev_rating_90d\",
        \"user_rating_stats:user_rating_list_10_recent_asin\"
    ],
    \"entities\": {
      \"user_id\": [
        \"$USER_ID\"
      ]
    }
  }" | jq
echo "We should expect to see new feature values corresponding to new timestamp"
```

# Training
You can choose to run either Non-docker version or Docker version below. The non-docker runs faster while the docker version is used to test packaged version of the run, which can be deployed on remote containerized computing infras.
## Non-docker version
```shell
cd $ROOT_DIR/notebooks && poetry run python 00-training-pipeline.py
cd $ROOT_DIR/notebooks && poetry run python 00-batch-reco-pipeline.py
```

## Docker version
```shell
# Train the Item2Vec and Sequence Rating Prediction models
docker compose -f compose.pipeline.yml run --rm --build training_pipeline
# Run batch pre-recommendations for Item2Vec models and persist to Redis
docker compose -f compose.pipeline.yml run --rm --build batch_reco_pipeline
```

# API
```shell
cd $ROOT_DIR
make requirements-txt
make api-up
echo "Visit http://localhost:8000/docs to interact with the APIs"
```

# Demo interaction and streaming feature update
This section assumes we have run `make ml-platform-up`, `make feature-server-up` and `make api-up`
```shell
cd $ROOT_DIR/ui
poetry run gradio app.py
```

Then you can try to rate some items and then see if the recommendations are updated accordingly.

# Improve
## Add new features
- Double check `cfg/run_cfg.py` to have both `use_sbert_features` and `use_item_tags_from_llm` set to False
- Run [notebook 022-ranker](./notebooks/022-ranker.ipynb) to refit the ranker with more features where we aim to reach ROC-AUC ~ 0.87
- In between check out the [notebook 030-error-analysis](./notebooks/030-error-analysis.ipynb) to dive into the model resutls

## Use LLM features
### Use SBERT to get embeddings of long textual features
- Run [notebook 016-sbert](./notebooks/016-sentence-transformers.ipynb) to build SBERT features for text item descriptions
### Extract item tags from LLM
- Update [RunCfg](./cfg/run_cfg.py) with `use_item_tags_from_llm = True`
- Run notebook [040](./notebooks/040-retrieve-item-tags-from-llm.ipynb) to extract the item tags using LLM
- Run notebook [002](./notebooks/002-features-v2.ipynb) to create new item metadata transformation pipeline
- Re-run notebook [020](./notebooks/020-negative-sample.ipynb) to update datasets
- Re-run notebook [022](./notebooks/022-ranker.ipynb) to re-fit model

---

## Clean up
```shell
make clean
```

---

# Troubleshooting
- If you run into Kernel Died error while runninng build training_pipeline, it might possibly due to Docker is not granted enough memory. You can try increasing the Docker memory allocation.
- If your Redis (kv_store) can not start due to "# Can't handle RDB format version 12", just remove the data and try again: `make clean`
