from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_DIR = Path(os.getenv("PROJECT_DIR", "/opt/airflow/project"))
PYTHON_EXECUTABLE = os.getenv("PROJECT_PYTHON", sys.executable)
DEPLOY_COMPOSE_FILE = PROJECT_DIR / "code" / "deployment" / "docker-compose.yml"


def run_python_script(script_path: str) -> None:
    full_path = PROJECT_DIR / script_path
    subprocess.run(
        [PYTHON_EXECUTABLE, str(full_path)],
        cwd=PROJECT_DIR,
        check=True,
    )


def deploy_api_and_app() -> None:
    subprocess.run(
        ["docker", "compose", "-f", str(DEPLOY_COMPOSE_FILE), "build"],
        cwd=PROJECT_DIR,
        check=True,
    )
    subprocess.run(
        ["docker", "compose", "-f", str(DEPLOY_COMPOSE_FILE), "up", "-d"],
        cwd=PROJECT_DIR,
        check=True,
    )


default_args = {
    "owner": "student",
    "retries": 0,
}


with DAG(
    dag_id="ai_layoff_pipeline",
    description="Download data, prepare data, train model, and deploy API/app.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="*/5 * * * *",
    catchup=False,
    tags=["pmldl", "mlops", "ai-layoffs"],
) as dag:
    download_data = PythonOperator(
        task_id="download_data",
        python_callable=run_python_script,
        op_kwargs={"script_path": "code/datasets/download_data.py"},
    )

    prepare_data = PythonOperator(
        task_id="prepare_data",
        python_callable=run_python_script,
        op_kwargs={"script_path": "code/datasets/prepare_data.py"},
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_python_script,
        op_kwargs={"script_path": "code/models/train_model.py"},
    )

    deploy = PythonOperator(
        task_id="deploy_api_and_app",
        python_callable=deploy_api_and_app,
    )

    download_data >> prepare_data >> train_model >> deploy

