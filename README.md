# Fraud Transaction Monitoring (Synthetic Data, Educational Project)

## 1. Project overview

A small, modular fraud-monitoring demo built on **synthetic data only**. It trains a
`RandomForestClassifier` to produce a fraud risk score, converts the score into a
Low / Medium / High band and a `clear` / `review` decision, explains each alert with
human-readable reason codes, serves scores through a Flask REST API, and offers a
Streamlit dashboard for batch monitoring and threshold tuning.

> **This is not a validated real-world fraud detection system.** See
> [Responsible-use limitations](#18-responsible-use-limitations).

## 2. Architecture

```
            +-------------------------+
            |  src/config.py          |  GeneratorConfig, FraudLogitWeights,
            |  (dataclasses + env)    |  ScoringConfig (band / decision)
            +-----------+-------------+
                        |
            +-----------v-------------+        +---------------------------+
            |  src/model.py           |        | data/                     |
            |  generate_transactions  |<-------| fraud_transactions_       |
            |  validate_privacy_safe  |  CSV   | synthetic.csv             |
            |  reason_codes           |        +---------------------------+
            +-----+-------------+-----+
                  |             |
        +---------v----+   +----v-----------------------+
        | src/train.py |   |  api.py (Flask)            |
        | split 60/20/ |   |  /health /config /predict  |
        | 20, RF,      |   +----------------------------+
        | thresholds   |   +----------------------------+
        +------+-------+   |  app.py (Streamlit)        |
               |           |  batch scoring + sliders   |
               v           +-------------^--------------+
   artifacts/fraud_model.joblib ---------+  (read-only at runtime)
```

## 3. Features

- Synthetic generator (log-normal, Poisson, exponential, Beta distributions, logistic fraud probability)
- Privacy allow-list and PII-fragment checks on every generated frame
- RandomForest with stratified 60/20/20 train/validation/test split
- Threshold strategies: fixed, F1-maximising, precision-targeted (chosen on validation only)
- Reason codes (up to three, deterministic, strongest signal first)
- Flask API with runtime config and per-request threshold override
- Streamlit analyst dashboard with threshold sliders and top-25 alert table
- Environment-variable configuration, reproducible via seed
- Nine pytest tests

## 4. Project structure

```
fraud-transaction-monitoring/
├── src/
│   ├── __init__.py
│   ├── config.py        # dataclasses, env loading, band()/decision()
│   ├── model.py         # generator, privacy checks, reason codes, loaders
│   └── train.py         # training CLI, threshold selection, artifact export
├── tests/
│   └── test_model.py
├── artifacts/
│   └── fraud_model.joblib
├── data/
│   └── fraud_transactions_synthetic.csv
├── app.py               # Streamlit dashboard
├── api.py               # Flask API
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

`data_demo.csv` (1,500 rows) is written to the project root on each training run.

## 5. Installation

Requires **Python 3.10 to 3.13** (developed and verified on 3.13). Python 3.14 is not supported by the pinned package versions: pip would try to compile numpy from source and fail on Windows. Check your version with `python --version`; if needed, install 3.13 and create the environment with `py -3.13 -m venv .venv`.

## 6. Virtual environment setup (Windows, VS Code terminal)

```
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

(`python -m venv .venv` also works if your default `python` is 3.10-3.13.)

If PowerShell blocks activation, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 7. Dataset

`data/fraud_transactions_synthetic.csv` is synthetic. Columns are exactly:

| Column | Meaning |
| --- | --- |
| `transaction_id` | Sequential synthetic id (`txn_0000000`) |
| `customer_ref` | Opaque synthetic reference `cust_<12 hex>` |
| `amount` | Transaction amount |
| `hour` | Hour of day (0-23) |
| `velocity_1h` | Transactions in the past hour |
| `distance_from_home_km` | Distance from home location |
| `merchant_risk` | Merchant risk score (0-1) |
| `device_age_days` | Age of the device in days |
| `is_foreign` | 1 if foreign transaction |
| `card_present` | 1 if card was present |
| `is_fraud` | Synthetic label |

`load_transactions()` in `src/model.py` loads a CSV and runs the same privacy checks. Note: `src/train.py` currently trains on data produced by `generate_transactions()` (seeded, reproducible); the CSV in `data/` is a sample of that generator's output and is not read by the training script.

## 8. How synthetic data generation works

`generate_transactions(config)` uses `numpy.random.default_rng(seed)`:

- `amount` ~ log-normal, `velocity_1h` ~ Poisson, `distance_from_home_km` ~ exponential
- `merchant_risk` ~ Beta(2, 6), `device_age_days` ~ exponential (integer days)
- `hour` from a daytime-heavy distribution; `is_foreign` and `card_present` ~ Bernoulli
- `transaction_id` is sequential; `customer_ref` is a SHA-256 derived opaque id
- `validate_privacy_safe()` runs before the frame is returned

The same seed always produces the same data.

## 9. How the fraud probability is generated

A logistic function over a weighted sum (`FraudLogitWeights` in `src/config.py`):

```
logit = intercept + w_amount*log1p(amount/median) + w_velocity*velocity_1h
      + w_distance*distance + w_merchant*merchant_risk + w_foreign*is_foreign
      + w_hour*odd_hour + w_card*card_present + w_device*log1p(age)/log1p(365)
p     = min(sigmoid(logit), fraud_probability_cap)      # cap default 0.95
is_fraud ~ Bernoulli(p)
```

Higher amount, velocity, merchant risk, distance, foreign flag and late-night hours
(23:00-04:59) raise risk; card-present and older devices lower it. With default
weights roughly 7% of transactions are labelled fraud.

## 10. Model training

```
python src/train.py
```

Pipeline: generate data, stratified split (60% train / 20% validation / 20% test),
fit `RandomForestClassifier` (200 trees, `class_weight="balanced_subsample"`),
score with `predict_proba()`, pick the threshold on the **validation** set, and
evaluate on the held-out **test** set. Output includes dataset size, class
distribution, threshold and strategy, classification report, precision, recall,
F1, ROC-AUC and PR-AUC. The artifact and `data_demo.csv` are written afterwards.

Options: `--n`, `--seed`, `--threshold`, `--threshold-strategy {fixed,f1,precision}`,
`--target-precision`, `--high-risk-threshold`.

Typical results with defaults (seed 42, fixed threshold 0.35, test set):
precision 0.28, recall 0.62, F1 0.38, ROC-AUC 0.86, PR-AUC 0.37 (identical on Linux and Windows runs). These describe
performance on synthetic data only.

## 11. Threshold strategies

| Strategy | Behaviour |
| --- | --- |
| `fixed` (default) | Uses `FRAUD_REVIEW_THRESHOLD` (default 0.35) |
| `f1` | Validation threshold that maximises F1 |
| `precision` | Smallest validation threshold reaching `--target-precision` (default 0.85); if none does, the highest available threshold; if no thresholds exist, the configured review threshold |

An explicit `--threshold` always overrides `--threshold-strategy`. The high-risk
threshold is raised to the review threshold if it would otherwise be lower.
A very demanding precision target may be unreachable, which yields a very high
threshold that flags almost nothing; that is the specified fallback.

Score-to-threshold comparisons live only in `ScoringConfig.band()` and
`ScoringConfig.decision()`.

## 12. Run the Flask API

Train first, then:

```
python api.py
```

The server listens on `http://127.0.0.1:5000`.

## 13. API endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | `{"status": "ok"}` |
| GET | `/config` | Current `review_threshold` and `high_risk_threshold` |
| POST | `/config` | Change runtime thresholds; HTTP 400 if outside 0-1 or high < review |
| POST | `/predict` | Score one transaction; all eight features required and numeric |
| POST | `/predict?threshold=0.5` | Same, with a one-request review threshold override (global config unchanged) |

Runtime changes via `POST /config` live in memory only and reset on restart.

## 14. Run Streamlit

```
streamlit run app.py
```

Controls: number of transactions, generator seed, review threshold slider and
high-risk slider (its minimum always equals the review threshold). The dashboard
only reads the artifact and never modifies it. It shows total, flagged and fraud
counts, precision, recall, F1 (all on synthetic data) and the top 25 flagged
transactions by score, with reason codes.

## 15. Run tests

```
pip install -r requirements-dev.txt
pytest tests/
```

## 16. Example API request

Start the API in one terminal (`python api.py`) and send requests from a second terminal.

**Windows PowerShell** (recommended on Windows):

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health

$body = '{"amount":950,"hour":3,"velocity_1h":6,"distance_from_home_km":420,"merchant_risk":0.8,"device_age_days":2,"is_foreign":1,"card_present":0}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType "application/json" -Body $body
```

One-request threshold override (global config is not changed):

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:5000/predict?threshold=0.99" -ContentType "application/json" -Body $body
```

**PowerShell with curl.exe** (PowerShell strips inner quotes, so use a file):

```powershell
Set-Content -Path tx.json -Value '{"amount":950,"hour":3,"velocity_1h":6,"distance_from_home_km":420,"merchant_risk":0.8,"device_age_days":2,"is_foreign":1,"card_present":0}'
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" --data "@tx.json"
```

**Linux / macOS / Git Bash:**

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
  -d '{"amount":950,"hour":3,"velocity_1h":6,"distance_from_home_km":420,"merchant_risk":0.8,"device_age_days":2,"is_foreign":1,"card_present":0}'
```

## 17. Example API response

```json
{
  "decision": "review",
  "reason_codes": ["Unusually high amount", "Unusually far from home", "Very new device"],
  "risk_band": "High",
  "risk_score": 0.8964,
  "threshold_used": 0.35
}
```

## 18. Responsible-use limitations

- Trained and evaluated only on synthetic data generated by the project's own formula; the model largely rediscovers that formula.
- Metrics do not predict real-world performance. This is **not** a validated fraud detection model and must not be used to make decisions about real people or payments.
- A high score supports human review; it does not prove a transaction is fraudulent.
- Precision is modest at the default threshold, so many flagged items would be false alarms.
- No fairness, drift, adversarial-robustness or regulatory analysis has been done.

## 19. Future improvements

- Probability calibration and cost-sensitive threshold selection
- SHAP-based explanations alongside the rule-based reason codes
- Authentication and rate limiting on the API; persisted configuration
- Drift monitoring and scheduled retraining
- Richer synthetic scenarios (fraud rings, seasonality)
- Docker packaging and CI for the test suite