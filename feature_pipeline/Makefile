.PHONY:
.ONESHELL:


up:
	docker compose -f compose.yml up -d 

down:
	docker compose down -v

build-pipeline:
	docker build -f Dockerfile . -t recsys-mvp-pipeline:0.0.1
