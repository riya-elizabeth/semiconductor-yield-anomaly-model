"""
Root-cause signal ranking via SHAP feature importance.

Uses TreeExplainer (model-specific, fast) on the saved Random Forest
classifier to compute SHAP values on the held-out test set.

Features are ranked by mean |SHAP| — higher = stronger association with
the fail prediction.

> **Important caveat for engineers**: These rankings show which sensor
> signals are most *correlated* with wafer fails as learned by the model.
> They are NOT proven root causes. Physical process validation — metrology,
> DOE, SPC — is required before engineering action. Treat this as a
> prioritised list of candidate signals to investigate, not a causal finding.
"""

from __future__ import annotations

import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for server/CI environments
import matplotlib.pyplot as plt
from pathlib import Path

import shap
from src.data import load_and_prepare_data

MODELS_DIR = Path(__file__).parent.parent / "models"
TOP_N = 20


def compute_and_save():
    _, X_test, _, y_test, feature_names = load_and_prepare_data(
        save_preprocessor=False
    )

    model = joblib.load(MODELS_DIR / "yield_model.joblib")

    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP >= 0.40 returns ndarray of shape (n_samples, n_features, n_classes)
    # for multi-output models; older versions return a list. Handle both.
    if isinstance(shap_values, list):
        sv = shap_values[1]          # older SHAP: list[class0, class1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]    # newer SHAP: (samples, features, classes)
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    ranked_idx = np.argsort(mean_abs_shap)[::-1]

    top_features = [
        {
            "rank": i + 1,
            "feature": feature_names[int(idx)],
            "mean_abs_shap": round(float(mean_abs_shap[int(idx)]), 6),
        }
        for i, idx in enumerate(ranked_idx[:TOP_N])
    ]

    print(f"\nTop {TOP_N} sensor signals associated with wafer fails (by mean |SHAP|):")
    print(f"{'Rank':<6} {'Feature':<20} {'Mean |SHAP|'}")
    print("-" * 40)
    for row in top_features:
        print(f"{row['rank']:<6} {row['feature']:<20} {row['mean_abs_shap']:.6f}")

    print("""
---
> Caveat: Rankings reflect model-learned correlations in this dataset.
> They are candidate signals for engineers to investigate — NOT proven
> root causes. Physical process validation is the required next step.
---
""")

    with open(MODELS_DIR / "top_risk_signals.json", "w") as f:
        json.dump(top_features, f, indent=2)

    # Bar plot
    labels = [r["feature"] for r in top_features][::-1]
    values = [r["mean_abs_shap"] for r in top_features][::-1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(labels, values, color="#c0392b")
    ax.set_xlabel("Mean |SHAP value| (association with fail prediction)")
    ax.set_title(f"Top {TOP_N} Sensor Signals Associated with Wafer Fails\n"
                 "(Correlational — not proven root causes)")
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    fig.savefig(MODELS_DIR / "shap_top20.png", dpi=150)
    plt.close(fig)

    print(f"Saved: models/top_risk_signals.json")
    print(f"Saved: models/shap_top20.png")
    return top_features


if __name__ == "__main__":
    compute_and_save()
