"""
Unsupervised excursion detection for SECOM wafer fab sensor data.

An Isolation Forest is trained on sensor features alone — labels are never
used during training. This simulates a real process monitoring scenario where
the line runs continuously and anomalous sensor behaviour is flagged without
waiting for final test results.

After training, anomaly scores are compared against actual fail labels to
quantify how well unsupervised excursion flags align with real defects.

contamination=0.05: assume ~5% of production runs are true excursions.
This is a prior; adjust per fab line historical data in production.
"""

from __future__ import annotations

import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
)
from src.data import load_and_prepare_data

MODELS_DIR = Path(__file__).parent.parent / "models"
CONTAMINATION = 0.05
RANDOM_STATE = 42


def train_and_save():
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data(
        save_preprocessor=False
    )
    MODELS_DIR.mkdir(exist_ok=True)

    # Fit on training features only — no labels used
    iso = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train)

    # decision_function: higher = more normal, lower = more anomalous
    train_scores = iso.decision_function(X_train)
    test_scores = iso.decision_function(X_test)

    # Invert so higher score = more anomalous (aligns with fail = positive)
    train_scores_inv = -train_scores
    test_scores_inv = -test_scores

    # Excursion flag: bottom contamination% of training scores as threshold
    excursion_threshold = np.percentile(train_scores, CONTAMINATION * 100)
    joblib.dump(float(excursion_threshold), MODELS_DIR / "excursion_threshold.joblib")

    # Evaluate alignment with actual fail labels (no peeking during training)
    test_flags = (test_scores <= excursion_threshold).astype(int)

    roc = roc_auc_score(y_test, test_scores_inv)
    pr_auc = average_precision_score(y_test, test_scores_inv)

    print(f"\n{'='*50}")
    print("Anomaly Detector — Alignment with Actual Fails")
    print(f"{'='*50}")
    print(f"ROC-AUC (anomaly score vs fail label): {roc:.4f}")
    print(f"PR-AUC  (anomaly score vs fail label): {pr_auc:.4f}")
    print(f"\nExcursion flag classification (threshold=p{int(CONTAMINATION*100)}):")
    print(classification_report(y_test, test_flags, target_names=["pass", "excursion"]))

    fail_excursion_overlap = (
        (test_flags == 1) & (y_test == 1)
    ).sum()
    total_excursions = test_flags.sum()
    total_fails = y_test.sum()
    print(f"Of {total_fails} actual fails, {fail_excursion_overlap} "
          f"({100*fail_excursion_overlap/max(total_fails,1):.0f}%) were flagged as excursions.")

    metrics = {
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr_auc, 4),
        "excursion_threshold_pct": int(CONTAMINATION * 100),
        "excursion_threshold_value": round(float(excursion_threshold), 6),
        "total_test_fails": int(total_fails),
        "fails_flagged_as_excursions": int(fail_excursion_overlap),
        "total_excursion_flags": int(total_excursions),
    }
    with open(MODELS_DIR / "anomaly_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(iso, MODELS_DIR / "anomaly_model.joblib")
    print("\nSaved: models/anomaly_model.joblib")
    print("Saved: models/excursion_threshold.joblib")
    print("Saved: models/anomaly_metrics.json")
    return iso, metrics


if __name__ == "__main__":
    train_and_save()
