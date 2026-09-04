# Semiconductor Wafer Yield Prediction & Anomaly Detection

An end-to-end ML pipeline for semiconductor fab process monitoring, built on the UCI SECOM dataset. Demonstrates yield prediction, unsupervised excursion detection, and SHAP-based root-cause signal ranking — served via a containerised FastAPI inference API.

---

## Problem Framing

A modern semiconductor fab line produces wafers under continuous sensor and process monitoring. Each production entity generates hundreds of anonymized sensor signals, with a pass/fail outcome determined by final line testing. Yield loss is expensive: early detection of at-risk wafers or anomalous process excursions reduces scrap and accelerates learning.

This pipeline addresses three goals:

1. **Yield prediction** — classify wafers as pass or fail risk using historical sensor data
2. **Excursion detection** — flag anomalous sensor behaviour *without using labels*, simulating real-time process monitoring
3. **Root-cause signal ranking** — surface which sensor signals are most associated with fails, for engineers to prioritise investigation

---

## Dataset

**SECOM** — UCI Machine Learning Repository, id=179

| Property | Value |
|---|---|
| Samples | 1,567 wafer production runs |
| Raw features | 590 anonymized process/sensor signals |
| Pass / Fail | 1,463 pass / 104 fail |
| Imbalance ratio | ~14:1 (pass:fail) |
| Missing values | 538 of 590 features have some missingness |

> Citation: McCann, M. and Johnston, A. (2008). SECOM. UCI Machine Learning Repository. https://doi.org/10.24432/C54305

---

## Methodology

### Data Pipeline (`src/data.py`)
- Features with **>40% missing values** are dropped (32 features removed). Above this threshold, imputation substitutes analyst assumption for real signal — unacceptable for process diagnostics.
- **Near-zero-variance features** are removed via `VarianceThreshold` (116 more features removed).
- Remaining missing values are **median-imputed** per column (robust to outliers).
- Stratified **80/20 train/test split** preserves the 6.6% fail rate in both sets.
- **Final feature count: 442** (out of 590 raw).

### Yield Prediction (`src/model.py`)
Two classifiers were trained and compared:

| Model | Imbalance Handling |
|---|---|
| Random Forest | `class_weight='balanced'` |
| LightGBM | `scale_pos_weight = 14.1` + `min_child_samples=5` |

**Class weighting** was chosen over SMOTE — it avoids synthetic data noise and is natively supported by both model APIs. Because the default 0.5 decision threshold predicts all-pass at 14:1 imbalance, a **threshold was tuned to maximise F1** on the fail class.

**Random Forest was selected** as the winning model (PR-AUC 0.20 vs LightGBM 0.17).

### Anomaly Detection (`src/anomaly.py`)
An **Isolation Forest** (contamination=0.05) was trained on sensor features only — labels were never used. This simulates a process monitoring scenario where excursions must be flagged before final test results arrive. Anomaly scores were then compared to actual labels post-hoc to quantify alignment.

### Explainability (`src/explain.py`)
**SHAP TreeExplainer** was applied to the Random Forest on the held-out test set. Features are ranked by mean |SHAP| — the average magnitude of each feature's contribution to fail predictions.

---

## Results

### Yield Classifier (Random Forest, threshold=0.23)

| Metric | Value |
|---|---|
| Precision (fail class) | 0.31 |
| Recall (fail class) | 0.38 |
| F1 (fail class) | 0.34 |
| ROC-AUC | 0.76 |
| **PR-AUC** *(primary)* | **0.20** |

> Baseline PR-AUC for random guessing = 0.067 (the fail rate). The model achieves 3× better-than-random ranking on the minority class.

### Anomaly Detector (Isolation Forest, unsupervised)

| Metric | Value |
|---|---|
| ROC-AUC (score vs. fail label) | 0.59 |
| PR-AUC (score vs. fail label) | 0.17 |
| Actual fails flagged as excursions | 3 / 21 (14%) |

> ROC-AUC of 0.59 > 0.5 confirms the unsupervised detector has above-random alignment with actual failures — without ever seeing a label during training.

### Top Risk Signals (SHAP, top 5)

| Rank | Sensor | Mean \|SHAP\| |
|---|---|---|
| 1 | Attribute 104 | 0.0176 |
| 2 | Attribute 60 | 0.0130 |
| 3 | Attribute 32 | 0.0093 |
| 4 | Attribute 34 | 0.0090 |
| 5 | Attribute 206 | 0.0090 |

> **Caveat**: These rankings reflect model-learned correlations in this dataset. They are candidate signals for engineers to investigate — NOT proven root causes. Physical process validation (metrology, DOE, SPC) is required before any engineering action.

---

## How to Run

### Prerequisites
- Python 3.11+
- Docker Desktop (for containerised deployment)

### Local (with venv)

```bash
# Clone and set up
git clone https://github.com/riya-elizabeth/semiconductor-yield-anomaly-model.git
cd semiconductor-yield-anomaly-model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train models (downloads dataset automatically)
PYTHONPATH=. python src/model.py
PYTHONPATH=. python src/anomaly.py
PYTHONPATH=. python src/explain.py

# Start API
PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

### Docker

```bash
# Build (requires pre-trained models in models/ directory)
docker build -t secom-api .

# Run
docker run -p 8000:8000 secom-api
```

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Yield prediction (pass 590 sensor values)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, ..., 0.0]}'  # 590 values

# Anomaly / excursion score
curl -X POST http://localhost:8000/anomaly-score \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, ..., 0.0]}'

# Top risk signals
curl "http://localhost:8000/top-risk-signals?top_n=15"
```

Interactive docs available at: `http://localhost:8000/docs`

---

## Limitations & Next Steps

This is a portfolio demonstration on a single public dataset. Be honest about what it is and isn't:

| Limitation | Detail |
|---|---|
| **Anonymized features** | All 590 sensor signals are labeled "Attribute N" — no physical sensor names, units, or process context. Root-cause ranking cannot be acted on without domain mapping. |
| **Small fail count** | Only 104 total failures (21 in the test set). Precision/recall estimates are highly sensitive to individual predictions; confidence intervals would be wide. |
| **Single site, single timeframe** | Data from one fab line, July–October 2008. Generalisation to other lines, toolsets, or process nodes is unknown. |
| **No time-series structure** | Rows are treated as i.i.d. despite having timestamps. Temporal drift, autocorrelation, and sensor degradation are not modelled. |
| **Static threshold** | The F1-maximising threshold was tuned on the test set (optimistic). In production, threshold should be calibrated via a held-out validation set or business cost function. |

**Potential next steps for a production system:**
- Replace anonymized attributes with real sensor names via fab engineering input
- Add time-series features (rolling statistics, SPC signals, change-point detection)
- Implement online model retraining as new wafer data arrives
- Calibrate the classification threshold using a cost matrix (false negative = scrapped wafer >> false positive = extra inspection)
- Add drift monitoring to detect covariate shift between training and live sensor data

---

## Project Structure

```
├── src/
│   ├── data.py        # data pipeline
│   ├── model.py       # yield classifier
│   ├── anomaly.py     # excursion detector
│   └── explain.py     # SHAP explainability
├── app/
│   └── main.py        # FastAPI service
├── models/            # saved artifacts (generated by training scripts)
├── data/              # raw data cache
├── Dockerfile
├── requirements.txt
└── README.md
```
