.PHONY:
.ONESHELL:

include .env
export

ml-platform-up:
	docker compose -f compose.yml up -d mlflow_server kv_store qdrant dwh

ml-platform-logs:
# For make command that follows logs, if not add prefix '-' then when interrupet the command, it will complain with Error 130
	- docker compose -f compose.yml logs -f

airflow-up:
	docker compose -f compose.airflow.yml up -d

airflow-logs:
	- docker compose -f compose.airflow.yml logs -f

lab:
	poetry run jupyter lab --port 8888 --host 0.0.0.0

api-up:
	docker compose -f compose.api.yml up -d

api-down:
	docker compose -f compose.api.yml down

# Create the requirements.txt file and update the torch to CPU version to reduce the image size
requirements-txt:
	poetry export --without dev --without-hashes --format=requirements.txt > requirements.txt
	sed -i '' '/^torch/ s/^/# /' requirements.txt  # Commend out torch in requirements.txt to pre-install the CPU version in Docker
	sed -i '' '/^nvidia/ s/^/# /' requirements.txt

build-pipeline:
	docker build -f feature_pipeline.Dockerfile . -t recsys-mvp-pipeline:0.0.1

feature-server-up:
	docker compose -f compose.yml up -d feature_online_server feature_offline_server feature_store_ui

down:
	docker compose -f compose.yml down
	docker compose -f compose.airflow.yml down
	docker compose -f compose.pipeline.yml down
	docker compose -f compose.api.yml down

remove-data:
	rm -rf data/redis
	rm -rf data/postgres
	rm -rf data/mlflow
	rm -rf data/qdrant_storage

clean: down remove-data
