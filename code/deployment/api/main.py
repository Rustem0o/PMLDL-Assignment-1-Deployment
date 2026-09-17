from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional
import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


CURRENT_FILE = Path(__file__).resolve()
sys.path.insert(0, "/app")

from common.features import add_features  # noqa: E402


DEFAULT_MODEL_PATH = Path("/app/models/ai_layoff_model.joblib")
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

app = FastAPI(title="AI Layoffs Prediction API")


class LayoffInput(BaseModel):
    state: str = Field("CA", description="US state abbreviation")
    naics_2: str = Field("51", description="Two-digit NAICS industry code")
    affected_workers: float = Field(120, ge=1, description="Number of affected workers")
    event_bucket: str = Field("layoff", description="Event group, for example layoff")
    filing_received_date: Optional[date] = Field(None)
    notice_date: Optional[date] = Field(None)
    layoff_start_date: Optional[date] = Field(None)


def input_to_dict(input_data: LayoffInput) -> dict:
    if hasattr(input_data, "model_dump"):
        return input_data.model_dump()
    return input_data.dict()


@lru_cache(maxsize=1)
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at {MODEL_PATH}. Train the model first."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def root() -> dict:
    return {"message": "AI Layoffs Prediction API is running."}


@app.get("/health")
def health() -> dict:
    bundle = load_model_bundle()
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "features": bundle["feature_columns"],
    }


@app.post("/predict")
def predict(input_data: LayoffInput) -> dict:
    bundle = load_model_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    row = input_to_dict(input_data)
    input_frame = pd.DataFrame([row])
    features = add_features(input_frame)[feature_columns]

    probability = float(model.predict_proba(features)[0, 1])
    prediction = int(probability >= 0.5)

    label = (
        "High AI-attribution layoff group"
        if prediction == 1
        else "Lower AI-attribution layoff group"
    )

    return {
        "prediction": prediction,
        "prediction_label": label,
        "high_ai_probability": round(probability, 4),
    }
