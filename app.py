"""Streamlit analyst dashboard for batch monitoring and threshold tuning.

Run:  streamlit run app.py
"""
from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import GeneratorConfig, ScoringConfig
from src.model import DEFAULT_ARTIFACT_PATH, generate_transactions, reason_codes

st.set_page_config(page_title="Fraud Transaction Monitoring", layout="wide")


@st.cache_resource
def load_artifact():
    # Read-only: the dashboard never writes to the artifact file.
    return joblib.load(DEFAULT_ARTIFACT_PATH)


@st.cache_data(show_spinner=False)
def score_batch(n: int, seed: int) -> pd.DataFrame:
    artifact = load_artifact()
    frame = generate_transactions(GeneratorConfig.from_env(n_transactions=n, seed=seed))
    frame["risk_score"] = artifact["model"].predict_proba(frame[artifact["features"]])[:, 1]
    return frame


st.title("Fraud Transaction Monitoring Dashboard")
st.info(
    "All data in this dashboard is **synthetic**. No real customers, cards or accounts are involved, "
    "and every metric below is computed on synthetic data only."
)
st.warning(
    "Responsible use: the model supports human review and does not prove that a transaction is fraudulent."
)

artifact = load_artifact()
defaults = ScoringConfig(
    review_threshold=float(artifact["threshold"]),
    high_risk_threshold=float(artifact["high_risk_threshold"]),
)
gen_defaults = GeneratorConfig.from_env()

with st.sidebar:
    st.header("Controls")
    n_transactions = st.number_input("Number of transactions", min_value=100, max_value=100000, value=min(gen_defaults.n_transactions, 100000), step=500)
    # Offset so the dashboard batch differs from the training data by default.
    seed = st.number_input("Generator seed", min_value=0, max_value=2**31 - 1, value=gen_defaults.seed + 1, step=1)
    review_threshold = st.slider("Review threshold", 0.0, 0.99, float(min(defaults.review_threshold, 0.99)), 0.01)
    high_min = float(review_threshold)
    high_default = float(min(max(defaults.high_risk_threshold, high_min), 1.0))
    high_risk_threshold = st.slider("High-risk threshold", high_min, 1.0, high_default, 0.01)

scoring = ScoringConfig(review_threshold=float(review_threshold), high_risk_threshold=float(high_risk_threshold))

data = score_batch(int(n_transactions), int(seed)).copy()
data["risk_band"] = data["risk_score"].map(scoring.band)
data["decision"] = data["risk_score"].map(scoring.decision)
flagged_mask = data["decision"] == "review"

y_true = data["is_fraud"].to_numpy()
y_pred = flagged_mask.astype(int).to_numpy()
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

c1, c2, c3 = st.columns(3)
c1.metric("Total transactions", f"{len(data):,}")
c2.metric("Flagged for review", f"{int(flagged_mask.sum()):,}")
c3.metric("Synthetic fraud count", f"{int(data['is_fraud'].sum()):,}")
c4, c5, c6 = st.columns(3)
c4.metric("Precision", f"{precision:.2%}")
c5.metric("Recall", f"{recall:.2%}")
c6.metric("F1", f"{f1:.2%}")
st.caption("Precision, recall and F1 are calculated on synthetic data and do not reflect real-world performance.")

st.subheader("Top 25 flagged transactions")
alerts = data[flagged_mask].sort_values("risk_score", ascending=False).head(25).copy()
if alerts.empty:
    st.write("No transactions are flagged at the current review threshold.")
else:
    alerts["reason_codes"] = alerts.apply(lambda r: "; ".join(reason_codes(r)), axis=1)
    alerts["risk_score"] = alerts["risk_score"].round(4)
    columns = [
        "transaction_id", "customer_ref", "risk_score", "risk_band", "decision",
        "amount", "hour", "velocity_1h", "distance_from_home_km", "merchant_risk",
        "device_age_days", "is_foreign", "card_present", "reason_codes",
    ]
    st.dataframe(alerts[columns], width="stretch", hide_index=True)
