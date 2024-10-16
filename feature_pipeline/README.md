# Feature pipeline with DBT and Feature Store

In this module we will simulate how data is ETL from source systems into ML-specific data storeage.

We first push the interaction data into PostgreSQL to simulate transaction data.
Then we would create model using the dbt framework to extract and transform the source data into features used in ML model.
Finally we materialize the features from offline store into online store where it's stored in a Key-value manner which is optimized for inference use case.

# Prerequisite
- Poetry 1.8.3
- Miniconda or alternatives that can create new Python environment with a specified Python version
- Docker
- PostgreSQL
  - For Mac, run: `brew install postgresql`

# Set up
- Run `export ROOT_DIR=$(pwd)` for easier nagivation
- Create a new `.env` file based on `.env.example` and populate the variables there
- Create a new Python 3.11.9 environment: `conda create --prefix .venv python=3.11.9`
- Make sure Poetry use the new Python 3.11.9 environment: `poetry env use .venv/bin/python`
- Install Python dependencies with Poetry: `poetry install`

# Simulate transaction data
- Run `make up` to start PostgreSQL service
- Execute the notebook [001-simulate-oltp.ipynb](notebooks/001-simulate-oltp.ipynb) to populate the raw data into PostgreSQL

# Build feature table
```shell
# Create a new shell starts at root dir recsys-mvp/feature_pipeline
export ROOT_DIR=$(pwd)
export $(cat .env | grep -v "^#")
cd dbt/feast
# Specify credential for dbt to connect to PostgreSQL
cat <<EOF > profiles.yml
feast:
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

# Feature Store

```shell
# Create a new shell starts at root dir recsys-mvp/feature_pipeline
export ROOT_DIR=$(pwd)
export $(cat .env | grep -v "^#")
cd feature_store/feature_repo
poetry run feast apply
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
poetry run feast materialize-incremental $CURRENT_TIME
```

# Clean up
```shell
poetry run feast teardown
cd $ROOT_DIR
make down
```
