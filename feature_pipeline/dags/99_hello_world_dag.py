import os
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG


def print_hello():
    print("Hello, world!")
    print(f"{os.getenv('POSTGRES_DB')=}")


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["your_email@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2023, 1, 1),
}

dag = DAG(
    "hello_world_dag",
    default_args=default_args,
    description="A simple hello world DAG",
    schedule_interval=timedelta(days=1),
)

hello_task = PythonOperator(
    task_id="hello_task",
    python_callable=print_hello,
    dag=dag,
)

hello_task
