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
    "example_docker_operator",
    default_args=default_args,
    description="A simple DockerOperator example",
    schedule_interval=timedelta(days=1),
)

t1 = DockerOperator(
    task_id="docker_command_task",
    image="ubuntu:latest",
    api_version="auto",
    auto_remove=True,
    command="sh -c '/bin/sleep 10 && echo \"$POSTGRES_DB\"'",
    docker_url="unix://var/run/docker.sock",
    network_mode="bridge",
    dag=dag,
    environment=envs,
)

t1
