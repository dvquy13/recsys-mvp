.PHONY:
.ONESHELL:

include .env
export

ml-platform-up:
	docker compose -f compose.yml up -d mlflow_server redis qdrant

down:
	docker compose -f compose.yml down

notebook-up:
	poetry run jupyter lab --port 8888 --host 0.0.0.0

api-up:
	docker compose -f compose.api.yml up

# Create the requirements.txt file and update the torch to CPU version to reduce the image size
requirements-txt:
	poetry export --without dev --without-hashes --format=requirements.txt > requirements.txt
	sed -i '' '/^torch/ s/^/# /' requirements.txt  # Commend out torch in requirements.txt to pre-install the CPU version in Docker
	sed -i '' '/^nvidia/ s/^/# /' requirements.txt
