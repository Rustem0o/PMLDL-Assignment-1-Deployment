from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "warn_layoffs.csv"

DATA_URL = (
    "https://huggingface.co/datasets/Meo-Advisors/warn-layoff-atlas/"
    "resolve/main/warn-layoffs.csv"
)

EXPECTED_COLUMNS = {
    "company_name",
    "city",
    "county",
    "state",
    "naics_2",
    "affected_workers",
    "event_type",
    "event_bucket",
    "filing_received_date",
    "notice_date",
    "layoff_start_date",
    "layoff_end_date",
    "ai_score_imputed",
}


def download_csv(url: str, output_path: Path, force: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"Raw data already exists: {output_path}")
        return output_path

    print(f"Downloading data from {url}")
    response = requests.get(url, timeout=90)
    response.raise_for_status()

    temp_path = output_path.with_suffix(".tmp")
    temp_path.write_bytes(response.content)
    temp_path.replace(output_path)

    print(f"Saved raw data to {output_path}")
    return output_path


def check_columns(csv_path: Path) -> None:
    sample = pd.read_csv(csv_path, nrows=5)
    missing_columns = sorted(EXPECTED_COLUMNS - set(sample.columns))

    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    print("Dataset columns look correct.")
    print(f"Columns: {list(sample.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Download the file again.")
    args = parser.parse_args()

    csv_path = download_csv(DATA_URL, RAW_FILE, force=args.force)
    check_columns(csv_path)


if __name__ == "__main__":
    main()

