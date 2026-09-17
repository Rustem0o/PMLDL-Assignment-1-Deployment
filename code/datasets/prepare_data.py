from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "warn_layoffs.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
DATA_INFO_FILE = PROCESSED_DIR / "data_info.json"

TARGET_COLUMN = "target_high_ai_layoff"
RANDOM_STATE = 42

KEEP_COLUMNS = [
    "company_name",
    "state",
    "naics_2",
    "affected_workers",
    "event_type",
    "event_bucket",
    "filing_received_date",
    "notice_date",
    "layoff_start_date",
    "layoff_end_date",
    "main_event_date",
    "ai_score_imputed",
    TARGET_COLUMN,
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data was not found at {path}. Run code/datasets/download_data.py first."
        )

    return pd.read_csv(path)


def first_available_date(data: pd.DataFrame) -> pd.Series:
    filing_date = pd.to_datetime(data["filing_received_date"], errors="coerce")
    notice_date = pd.to_datetime(data["notice_date"], errors="coerce")
    layoff_date = pd.to_datetime(data["layoff_start_date"], errors="coerce")

    return layoff_date.fillna(notice_date).fillna(filing_date)


def remove_worker_outliers(data: pd.DataFrame) -> pd.DataFrame:
    q1 = data["affected_workers"].quantile(0.25)
    q3 = data["affected_workers"].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr

    return data[data["affected_workers"] <= upper_limit].copy()


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned = cleaned.drop_duplicates()

    cleaned["ai_score_imputed"] = pd.to_numeric(
        cleaned["ai_score_imputed"], errors="coerce"
    )
    cleaned["affected_workers"] = pd.to_numeric(
        cleaned["affected_workers"], errors="coerce"
    )

    cleaned = cleaned.dropna(subset=["ai_score_imputed"])

    positive_workers = cleaned.loc[cleaned["affected_workers"] > 0, "affected_workers"]
    median_workers = positive_workers.median()
    cleaned["affected_workers"] = cleaned["affected_workers"].fillna(median_workers)
    cleaned = cleaned[cleaned["affected_workers"] > 0].copy()
    cleaned = remove_worker_outliers(cleaned)

    for column in ["company_name", "state", "event_type", "event_bucket"]:
        cleaned[column] = cleaned[column].fillna("unknown").astype(str)

    cleaned["naics_2"] = pd.to_numeric(cleaned["naics_2"], errors="coerce")
    cleaned["main_event_date"] = first_available_date(cleaned)
    cleaned = cleaned.dropna(subset=["main_event_date"])

    threshold = cleaned["ai_score_imputed"].quantile(0.75)
    cleaned[TARGET_COLUMN] = (cleaned["ai_score_imputed"] >= threshold).astype(int)

    return cleaned[KEEP_COLUMNS].copy(), float(threshold)


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COLUMN],
    )

    return train_data, test_data


def save_outputs(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    raw_rows: int,
    cleaned_rows: int,
    target_threshold: float,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(TRAIN_FILE, index=False)
    test_data.to_csv(TEST_FILE, index=False)

    data_info = {
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "train_rows": len(train_data),
        "test_rows": len(test_data),
        "target_column": TARGET_COLUMN,
        "target_rule": "ai_score_imputed >= 75th percentile",
        "target_ai_score_threshold": round(target_threshold, 4),
        "positive_rate_train": round(float(train_data[TARGET_COLUMN].mean()), 4),
        "positive_rate_test": round(float(test_data[TARGET_COLUMN].mean()), 4),
    }

    DATA_INFO_FILE.write_text(json.dumps(data_info, indent=2), encoding="utf-8")
    print(f"Saved train data to {TRAIN_FILE}")
    print(f"Saved test data to {TEST_FILE}")
    print(f"Saved data info to {DATA_INFO_FILE}")


def main() -> None:
    raw_data = load_data(RAW_FILE)
    cleaned_data, target_threshold = clean_data(raw_data)
    train_data, test_data = split_data(cleaned_data)
    save_outputs(
        train_data,
        test_data,
        raw_rows=len(raw_data),
        cleaned_rows=len(cleaned_data),
        target_threshold=target_threshold,
    )


if __name__ == "__main__":
    main()

