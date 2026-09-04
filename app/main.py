"""
FastAPI inference service for the SECOM wafer fab ML pipeline.

Endpoints:
  GET  /health            — liveness check
  POST /predict           — yield pass/fail prediction with fail probability
  POST /anomaly-score     — unsupervised excursion score and flag
  GET  /top-risk-signals  — precomputed SHAP-ranked sensor signals

All models are loaded once at startup from the models/ directory.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator

MODELS_DIR = Path(__file__).parent.parent / "models"
N_RAW_FEATURES = 590  # raw sensor readings expected from the client


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SensorReading(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def check_length(cls, v):
        if len(v) != N_RAW_FEATURES:
            raise ValueError(
                f"Expected {N_RAW_FEATURES} sensor values, got {len(v)}"
            )
        return v


class PredictResponse(BaseModel):
    pass_fail_prediction: str   # "pass" or "fail"
    fail_probability: float


class AnomalyResponse(BaseModel):
    anomaly_score: float        # higher = more anomalous
    is_excursion: bool


# ---------------------------------------------------------------------------
# App startup: load all models once
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["preprocessor"] = joblib.load(MODELS_DIR / "preprocessor.joblib")
    _state["keep_cols"] = joblib.load(MODELS_DIR / "keep_cols.joblib")
    _state["yield_model"] = joblib.load(MODELS_DIR / "yield_model.joblib")
    _state["yield_threshold"] = float(
        joblib.load(MODELS_DIR / "yield_threshold.joblib")
    )
    _state["anomaly_model"] = joblib.load(MODELS_DIR / "anomaly_model.joblib")
    _state["excursion_threshold"] = float(
        joblib.load(MODELS_DIR / "excursion_threshold.joblib")
    )
    with open(MODELS_DIR / "top_risk_signals.json") as f:
        _state["top_risk_signals"] = json.load(f)

    yield  # app runs here

    _state.clear()


app = FastAPI(
    title="SECOM Wafer Fab ML API",
    description=(
        "Yield prediction, excursion detection, and root-cause signal ranking "
        "for semiconductor wafer fabrication process monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper: apply the same preprocessing the training pipeline used
# ---------------------------------------------------------------------------

def _preprocess(raw_features: list[float]) -> np.ndarray:
    """Apply missingness drop + imputation + variance filter to a single row."""
    import pandas as pd

    keep_cols = _state["keep_cols"]
    preprocessor = _state["preprocessor"]

    # Build a DataFrame with all 590 column names the preprocessor expects
    col_names = [f"Attribute {i+1}" for i in range(N_RAW_FEATURES)]
    row_df = pd.DataFrame([raw_features], columns=col_names)

    # Select only the columns that survived missingness filtering at train time
    row_filtered = row_df[keep_cols]

    # Apply imputer + variance filter
    return preprocessor.transform(row_filtered)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(reading: SensorReading):
    try:
        X = _preprocess(reading.features)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    model = _state["yield_model"]
    threshold = _state["yield_threshold"]

    prob_fail = float(model.predict_proba(X)[0, 1])
    prediction = "fail" if prob_fail >= threshold else "pass"

    return PredictResponse(
        pass_fail_prediction=prediction,
        fail_probability=round(prob_fail, 4),
    )


@app.post("/anomaly-score", response_model=AnomalyResponse)
def anomaly_score(reading: SensorReading):
    try:
        X = _preprocess(reading.features)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    iso = _state["anomaly_model"]
    excursion_threshold = _state["excursion_threshold"]

    # decision_function: lower = more anomalous; invert so higher = more anomalous
    raw_score = float(iso.decision_function(X)[0])
    anomaly_score = round(-raw_score, 6)
    is_excursion = raw_score <= excursion_threshold

    return AnomalyResponse(
        anomaly_score=anomaly_score,
        is_excursion=bool(is_excursion),
    )


@app.get("/top-risk-signals")
def top_risk_signals(
    top_n: Annotated[int, Query(ge=1, le=20)] = 15
):
    signals = _state["top_risk_signals"][:top_n]
    return {
        "top_n": top_n,
        "caveat": (
            "These are sensor signals most correlated with fail predictions "
            "in the training data. They are NOT proven root causes — "
            "physical/process validation is required before engineering action."
        ),
        "signals": signals,
    }
