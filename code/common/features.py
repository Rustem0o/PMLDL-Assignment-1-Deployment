from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "affected_workers",
    "workers_log",
    "event_year",
    "event_month",
    "days_notice_to_layoff",
]

CATEGORICAL_FEATURES = [
    "state",
    "naics_2",
    "event_bucket",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def parse_date_column(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(pd.NaT, index=data.index)
    return pd.to_datetime(data[column], errors="coerce")


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create simple model features from cleaned layoff data."""
    features = data.copy()

    features["affected_workers"] = pd.to_numeric(
        features.get("affected_workers"), errors="coerce"
    )
    features["workers_log"] = np.log1p(features["affected_workers"].clip(lower=0))

    filing_date = parse_date_column(features, "filing_received_date")
    notice_date = parse_date_column(features, "notice_date").fillna(filing_date)
    layoff_start_date = parse_date_column(features, "layoff_start_date")
    main_event_date = parse_date_column(features, "main_event_date")

    event_date = layoff_start_date.fillna(main_event_date).fillna(notice_date)

    features["event_year"] = event_date.dt.year
    features["event_month"] = event_date.dt.month
    features["days_notice_to_layoff"] = (layoff_start_date - notice_date).dt.days

    features["state"] = features.get("state", "unknown").fillna("unknown").astype(str)
    features["event_bucket"] = (
        features.get("event_bucket", "unknown").fillna("unknown").astype(str)
    )

    naics_2 = pd.to_numeric(features.get("naics_2"), errors="coerce")
    features["naics_2"] = naics_2.astype("Int64").astype(str).replace("<NA>", "unknown")

    return features[FEATURE_COLUMNS]

