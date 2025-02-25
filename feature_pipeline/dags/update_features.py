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
    "POSTGRES_FEATURE_STORE_OFFLINE_SCHEMA": os.getenv(
        "POSTGRES_FEATURE_STORE_OFFLINE_SCHEMA"
    ),
    "POSTGRES_FEATURE_STORE_ONLINE_SCHEMA": os.getenv(
        "POSTGRES_FEATURE_STORE_ONLINE_SCHEMA"
    ),
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
    "update_features",
    default_args=default_args,
    description="Update features",
    catchup=False,
    schedule_interval="5 18 * * *",  # 2:05 AM GMT+8
)

t1 = DockerOperator(
    task_id="update-features",
    image="recsys-mvp-pipeline:0.0.1",
    api_version="auto",
    auto_remove=True,
    command="sh -c 'cd ../dbt/feature_store && uv run dbt deps && uv run dbt build --models marts.amz_review_rating'",
    docker_url="unix://var/run/docker.sock",
    network_mode="host",
    dag=dag,
)

t2 = DockerOperator(
    task_id="apply-and-materialize-features",
    image="recsys-mvp-pipeline:0.0.1",
    api_version="auto",
    auto_remove=True,
    command="""
        sh -c 'cd ../feature_store/feature_repo && 
        uv run feast apply &&
        CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S") &&
        uv run feast materialize-incremental $CURRENT_TIME'
    """,
    docker_url="unix://var/run/docker.sock",
    network_mode="host",
    dag=dag,
    environment=envs,
)

# Set task dependencies
t1 >> t2
