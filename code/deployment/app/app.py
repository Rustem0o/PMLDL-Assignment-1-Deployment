from __future__ import annotations

from datetime import date
import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000/predict")


st.set_page_config(page_title="AI Layoff Predictor", page_icon="📉")

st.title("AI Layoff Predictor")
st.write("Predict whether a WARN layoff event is in the high AI-attribution group.")

with st.form("prediction_form"):
    state = st.selectbox(
        "State",
        ["CA", "NY", "TX", "FL", "WA", "IL", "MA", "OH", "GA", "NC", "Other"],
    )
    if state == "Other":
        state = st.text_input("Enter state abbreviation", value="CA")

    naics_2 = st.selectbox(
        "Industry NAICS-2 code",
        ["11", "21", "22", "23", "31", "32", "33", "42", "44", "45", "48", "49", "51", "52", "54", "56", "61", "62", "72", "81"],
        index=12,
    )

    affected_workers = st.number_input(
        "Affected workers",
        min_value=1,
        max_value=50000,
        value=120,
        step=1,
    )

    event_bucket = st.selectbox(
        "Event type",
        ["layoff", "closure", "mass_layoff", "other", "unknown"],
    )

    filing_received_date = st.date_input("Filing received date", value=date(2026, 3, 11))
    notice_date = st.date_input("Notice date", value=date(2026, 3, 11))
    layoff_start_date = st.date_input("Layoff start date", value=date(2026, 12, 11))

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "state": state,
        "naics_2": naics_2,
        "affected_workers": affected_workers,
        "event_bucket": event_bucket,
        "filing_received_date": filing_received_date.isoformat(),
        "notice_date": notice_date.isoformat(),
        "layoff_start_date": layoff_start_date.isoformat(),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction")
        st.metric("Result", result["prediction_label"])
        st.write(f"High AI-attribution probability: {result['high_ai_probability']}")
    except requests.RequestException as error:
        st.error(f"Could not get prediction from the API: {error}")

