import os
from datetime import timedelta

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

from airflow import DAG

envs = {
    "POSTGRES_USER": os.getenv("POSTGRES_USER"),
    "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
    "POSTGRES_HOST": os.getenv("POSTGRES_HOST"),
    "POSTGRES_PORT": os.getenv("POSTGRES_PORT"),
    "POSTGRES_DB": os.getenv("POSTGRES_DB"),
    "POSTGRES_OLTP_SCHEMA": os.getenv("POSTGRES_OLTP_SCHEMA"),
    "S3_ENDPOINT_URL": os.getenv("S3_ENDPOINT_URL"),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
}

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "append_oltp",
    default_args=default_args,
    description="Append OLTP",
    schedule_interval=timedelta(days=1),
)

t1 = DockerOperator(
    task_id="002-append-holdout-to-oltp",
    image="recsys-mvp-pipeline:0.0.1",
    api_version="auto",
    auto_remove=True,
    command="sh -c 'poetry run papermill 002-append-holdout-to-oltp.ipynb papermill-output/002-append-holdout-to-oltp.ipynb'",
    docker_url="unix://var/run/docker.sock",
    network_mode="host",  # Need to use host most since we need to access to services in compose.yml file when running the pipeline
    dag=dag,
    environment=envs,
)

t1
