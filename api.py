"""Flask REST API for scoring individual synthetic transactions.

Run:  python api.py
"""
from __future__ import annotations

import math
import threading

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.config import ScoringConfig
from src.model import DEFAULT_ARTIFACT_PATH, FEATURES, reason_codes

app = Flask(__name__)

# The artifact is loaded once, at startup.
_artifact = joblib.load(DEFAULT_ARTIFACT_PATH)
MODEL = _artifact["model"]
MODEL_FEATURES = list(_artifact["features"])

_config_lock = threading.Lock()
_scoring = ScoringConfig(
    review_threshold=float(_artifact["threshold"]),
    high_risk_threshold=float(_artifact["high_risk_threshold"]),
)


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/config")
def get_config():
    with _config_lock:
        return jsonify(
            {
                "review_threshold": _scoring.review_threshold,
                "high_risk_threshold": _scoring.high_risk_threshold,
            }
        )


@app.post("/config")
def set_config():
    global _scoring
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.")
    unknown = sorted(set(payload) - {"review_threshold", "high_risk_threshold"})
    if unknown:
        return _error(f"Unknown field(s): {unknown}")
    if not payload:
        return _error("Provide review_threshold and/or high_risk_threshold.")
    with _config_lock:
        try:
            new_config = ScoringConfig(
                review_threshold=payload.get("review_threshold", _scoring.review_threshold),
                high_risk_threshold=payload.get("high_risk_threshold", _scoring.high_risk_threshold),
            )
        except ValueError as exc:
            return _error(str(exc))
        _scoring = new_config
        return jsonify(
            {
                "review_threshold": _scoring.review_threshold,
                "high_risk_threshold": _scoring.high_risk_threshold,
            }
        )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.")

    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return _error(f"Missing required field(s): {missing}")
    non_numeric = [f for f in FEATURES if not _is_number(payload[f])]
    if non_numeric:
        return _error(f"Field(s) must be numeric: {non_numeric}")

    with _config_lock:
        scoring = _scoring

    override = request.args.get("threshold")
    if override is not None:
        try:
            value = float(override)
            # One-request-only override: the global configuration is not modified.
            scoring = ScoringConfig(
                review_threshold=value,
                high_risk_threshold=max(scoring.high_risk_threshold, value),
            )
        except ValueError as exc:
            return _error(f"Invalid threshold override: {exc}")

    row = {f: payload[f] for f in FEATURES}
    frame = pd.DataFrame([row], columns=MODEL_FEATURES)
    score = float(MODEL.predict_proba(frame)[0, 1])

    return jsonify(
        {
            "risk_score": round(score, 4),
            "risk_band": scoring.band(score),
            "decision": scoring.decision(score),
            "reason_codes": reason_codes(row),
            "threshold_used": scoring.review_threshold,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
