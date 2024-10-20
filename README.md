# Implement an MVP RecSys

# Prerequisite
- Poetry 1.8.3
- Miniconda or alternatives that can create new Python environment with a specified Python version
- Docker

> [!TIP]
> **VSCode auto load `.env`**
> For convenience, you should enable your IDE to automatically load the `.env` as environment vars.
> If using VSCode then this is done automatically as long as you have VSCode Python Extension installed.

> [!IMPORTANT]
> **Increase Docker memory to 16GB**
> By default after installing Docker it might get only 8 GB of RAM from the host machine.
> Due to this project's poor optimization at the moment, it's required to increase the Docker allocatable memory to at least 12 GB.

# Set up
- Create a new `.env` file based on `.env.example` and populate the variables there
- Set up env var $ROOT_DIR: `sed -i '' "s|^ROOT_DIR=.*|ROOT_DIR=$(pwd)|" .env`
- Create a new Python 3.11.9 environment: `conda create --prefix .venv python=3.11.9`
- Make sure Poetry use the new Python 3.11.9 environment: `poetry env use .venv/bin/python`
- Install Python dependencies with Poetry: `poetry install`

# Start services
## Common services
- Run `make ml-platform-up && make ml-platform-logs` to start the supporting services
- Wait until you see "Booting worker with pid..." then you can Ctrl + C to exit the logs following process

## Airflow
> [!WARNING]
> **Local Docker Airflow requires some serious resources**
> You might need to monitor the Airflow service logs at startup to check if they complain anything about your available resources

```shell
# Create a new shell starts at root dir recsys-mvp/feature_pipeline
cd airflow
mkdir -p ./dags ./logs ./plugins ./config
cd $ROOT_DIR
make airflow-up && make airflow-logs
# To check airflow logs: `make airflow-logs`

# Below 4 lines are there just in case Airflow does not start correctly due to permission issue
# export AIRFLOW_UID=$(id -u)
# sed -i '' "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$AIRFLOW_UID/" .env
# export $(cat .env | grep -v "^#")
# docker compose -f compose.airflow.yml up -d
```

- Wait until you see "airflow-webserver: Booting worker with pid..." then you can Ctrl + C to exit the logs following process


# Feature pipeline
The goal of feature pipeline is to keep the feature in feature store updated via daily batch jobs.

> [!IMPORTANT]
> This section assumes you have `make ml-platform-up` and `make airflow-up` running

## Simulate transaction data
```shell
echo "Execute the notebook to populate the raw data into PostgreSQL"
cd $ROOT_DIR/feature_pipeline/notebooks && poetry run papermill 001-simulate-oltp.ipynb papermill-output/001-simulate-oltp.ipynb
```

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
Set up Feature Server to serve online features

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

- Now go to Airflow UI http://localhost:8080, username=airflow password=airflow
- Trigger the DAG named `append_oltp`. Check the DAG run logs to see if there are any errors.
- If no error, running `poetry run python scripts/check_oltp_max_timestamp.py` again should yield a later date like 2022-07-16.

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
```shell
# Train the Item2Vec and Sequence Rating Prediction models
docker compose -f compose.pipeline.yml run --rm --build training_pipeline
# Run batch pre-recommendations for Item2Vec models and persist to Redis
docker compose -f compose.pipeline.yml run --rm --build batch_reco_pipeline
```

# API
```shell
make requirements-txt
make api-up
echo "Visit http://localhost:8000/docs to interact with the APIs"
```

# Demo interaction and streaming feature update
```shell
cd $ROOT_DIR/ui
poetry run gradio app.py
```

Then you can try to rate some items and then see if the recommendations are updated accordingly.

## Clean up
```shell
make clean
```

# Troubleshooting
- If you run into Kernel Died error while runninng build training_pipeline, it might possibly due to Docker is not granted enough memory. You can try increasing the Docker memory allocation.
