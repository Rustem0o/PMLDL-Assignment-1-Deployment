# PMLDL Assignment 1: AI Layoffs Deployment Pipeline

This repository is a simple MLOps assignment project. It uses a real open dataset about US WARN layoff notices and AI attribution, then builds an automated pipeline with data engineering, model engineering, and deployment.

The goal is to predict whether a WARN layoff event belongs to the **high AI-attribution layoff group**. The target is created from the dataset's `ai_score_imputed` column, but this score is not used as a model input.

## Dataset

Dataset: **WARN Layoff Atlas (US)** by Meo Advisors.

Source links:

- Hugging Face dataset: https://huggingface.co/datasets/Meo-Advisors/warn-layoff-atlas
- AI Workforce Data GitHub page: https://github.com/meoadvisors/ai-workforce-data
- Dataset explorer/methodology page: https://meoadvisors.com/research/warn-layoff-atlas/

The dataset contains 12,623 WARN layoff notices. Real columns used in this project include:

- `company_name`
- `state`
- `naics_2`
- `affected_workers`
- `event_type`
- `event_bucket`
- `filing_received_date`
- `notice_date`
- `layoff_start_date`
- `layoff_end_date`
- `ai_score_imputed`

License note: the dataset is published as CC BY-NC 4.0 by Meo Advisors. This project is intended for education/research use with attribution.

## Problem

The model predicts:

```text
1 = high AI-attribution layoff group
0 = lower AI-attribution layoff group
```

The label is created during data preparation:

```text
ai_score_imputed >= 75th percentile of ai_score_imputed
```

Important: `ai_score_imputed` is used only to create the target label. It is removed from the model input features to avoid direct target leakage.

## Project Structure

```text
.
├── code
│   ├── common
│   │   └── features.py
│   ├── datasets
│   │   ├── download_data.py
│   │   └── prepare_data.py
│   ├── deployment
│   │   ├── api
│   │   │   ├── Dockerfile
│   │   │   └── main.py
│   │   ├── app
│   │   │   ├── Dockerfile
│   │   │   └── app.py
│   │   └── docker-compose.yml
│   └── models
│       └── train_model.py
├── data
│   ├── raw
│   └── processed
├── models
├── notebooks
├── services
│   └── airflow
│       ├── dags
│       │   └── ai_layoff_pipeline_dag.py
│       ├── Dockerfile
│       ├── docker-compose.yml
│       └── logs
├── requirements.txt
├── requirements-airflow.txt
└── run_pipeline.py
```

## Quick Start

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the complete local data and training pipeline:

```bash
python3 run_pipeline.py
```

This downloads the raw CSV, prepares train/test files, trains the model, saves metrics, and packages the trained model.

Generated files:

```text
data/raw/warn_layoffs.csv
data/processed/train.csv
data/processed/test.csv
data/processed/data_info.json
models/ai_layoff_model.joblib
models/metrics.json
mlflow.db
mlruns/
```

## Run API and Streamlit App

After the model is trained, start the API and app in separate Docker containers:

```bash
docker compose -f code/deployment/docker-compose.yml up --build
```

Open:

- FastAPI docs: http://localhost:8000/docs
- Streamlit app: http://localhost:8501

The Streamlit app sends user input to the FastAPI service and displays the prediction returned by the API.

Stop the containers:

```bash
docker compose -f code/deployment/docker-compose.yml down
```

## MLflow

Training metrics and the model are logged to MLflow.

Run the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///$(pwd)/mlflow.db --port 5000
```

Open:

```text
http://localhost:5000
```

Logged metrics include accuracy, precision, recall, F1 score, and ROC AUC.

MLflow uses:

```text
mlflow.db     # tracking database
mlruns/       # local artifact folder
```

## Airflow Automation

The Airflow DAG is located at:

```text
services/airflow/dags/ai_layoff_pipeline_dag.py
```

Schedule:

```text
*/5 * * * *
```

This means the pipeline is scheduled to run every 5 minutes.

The DAG tasks are:

```text
download_data
    ↓
prepare_data
    ↓
train_model
    ↓
deploy_api_and_app
```

To run Airflow with Docker:

```bash
docker compose -f services/airflow/docker-compose.yml up --build
```

Open:

```text
http://localhost:8080
```

Default login:

```text
username: admin
password: admin
```

Then enable the DAG named:

```text
ai_layoff_pipeline
```

The final deployment task uses Docker Compose, so Docker must be running on the machine.

## Manual Step-by-Step Commands

If you want to run each stage separately:

```bash
python3 code/datasets/download_data.py
python3 code/datasets/prepare_data.py
python3 code/models/train_model.py
docker compose -f code/deployment/docker-compose.yml up --build
```

## API Example

Request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "CA",
    "naics_2": "51",
    "affected_workers": 120,
    "event_bucket": "layoff",
    "filing_received_date": "2026-03-11",
    "notice_date": "2026-03-11",
    "layoff_start_date": "2026-12-11"
  }'
```

Response:

```json
{
  "prediction": 0,
  "prediction_label": "Lower AI-attribution layoff group",
  "high_ai_probability": 0.31
}
```

The exact probability depends on the trained model.

## Assignment Criteria Mapping

| Criterion | Where it is implemented |
|---|---|
| Data engineering | `code/datasets/download_data.py`, `code/datasets/prepare_data.py` |
| Missing values and outliers | `clean_data()` and `remove_worker_outliers()` in `prepare_data.py` |
| Train/test split | `split_data()` in `prepare_data.py` |
| Feature engineering | `code/common/features.py` |
| Model training/evaluation/package | `code/models/train_model.py` |
| MLflow logging | `train_model.py`, local `mlruns/` folder |
| FastAPI model API | `code/deployment/api/main.py` |
| Streamlit app | `code/deployment/app/app.py` |
| Separate Docker containers | `code/deployment/docker-compose.yml` |
| Airflow automation every 5 minutes | `services/airflow/dags/ai_layoff_pipeline_dag.py` |
| Repository structure | This repository layout |

## Before GitHub Submission

1. Run `python3 run_pipeline.py`.
2. Run `docker compose -f code/deployment/docker-compose.yml up --build`.
3. Open the Streamlit app and make one prediction.
4. Run or show the Airflow DAG scheduled every 5 minutes.
5. Push the project to a public GitHub repository.
6. Include the public GitHub link in the submission.
