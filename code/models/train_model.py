from __future__ import annotations

from pathlib import Path
import json
import os
import sys

import joblib

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from common.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    add_features,
)


TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_FILE = PROJECT_ROOT / "data" / "processed" / "test.csv"
DATA_INFO_FILE = PROJECT_ROOT / "data" / "processed" / "data_info.json"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "ai_layoff_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlruns"

TARGET_COLUMN = "target_high_ai_layoff"


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "Processed train/test files were not found."
        )

    return pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)


def create_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": round(
            float(accuracy_score(y_test, predictions)), 4
        ),
        "precision": round(
            float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "roc_auc": round(
            float(
                roc_auc_score(
                    y_test,
                    probabilities,
                )
            ),
            4,
        ),
    }


def save_model(model: Pipeline, metrics: dict) -> None:
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_info = {}

    if DATA_INFO_FILE.exists():
        data_info = json.loads(
            DATA_INFO_FILE.read_text(
                encoding="utf-8"
            )
        )

    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "metrics": metrics,
        "data_info": data_info,
    }

    joblib.dump(
        model_bundle,
        MODEL_FILE,
    )

    METRICS_FILE.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved model to {MODEL_FILE}"
    )
    print(
        f"Saved metrics to {METRICS_FILE}"
    )


def log_to_mlflow(
    model: Pipeline,
    metrics: dict,
) -> None:
    MLFLOW_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracking_uri = f"sqlite:///{MLFLOW_DB}"
    mlflow.set_tracking_uri(tracking_uri)

    artifact_location = os.getenv(
        "MLFLOW_ARTIFACT_LOCATION",
        f"file://{MLFLOW_ARTIFACT_DIR.resolve()}",
    )

    experiment_name = "ai_layoff_assignment"

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )

    mlflow.set_experiment(
        experiment_name
    )

    with mlflow.start_run(
        run_name="logistic_regression_ai_layoffs"
    ):
        mlflow.log_param(
            "model_type",
            "LogisticRegression",
        )

        mlflow.log_param(
            "target",
            TARGET_COLUMN,
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURE_COLUMNS),
        )

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            serialization_format=(
                mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

        mlflow.log_artifact(
            str(MODEL_FILE)
        )

        mlflow.log_artifact(
            str(METRICS_FILE)
        )

    print(
        f"Logged run to MLflow database at {MLFLOW_DB}"
    )


def main() -> None:
    train_data, test_data = load_processed_data()

    x_train = add_features(train_data)
    y_train = train_data[TARGET_COLUMN]

    x_test = add_features(test_data)
    y_test = test_data[TARGET_COLUMN]

    model = create_model()

    model.fit(
        x_train,
        y_train,
    )

    metrics = evaluate_model(
        model,
        x_test,
        y_test,
    )

    save_model(
        model,
        metrics,
    )

    log_to_mlflow(
        model,
        metrics,
    )

    print("Evaluation metrics:")

    for name, value in metrics.items():
        print(
            f"{name}: {value}"
        )


if __name__ == "__main__":
    main()