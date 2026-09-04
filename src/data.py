"""
Data pipeline for the UCI SECOM semiconductor manufacturing dataset.

Each row is a wafer production entity with 590 anonymized process/sensor signals
and a binary pass/fail outcome from final line testing.

Pipeline steps:
1. Load dataset via ucimlrepo (id=179)
2. Report shape, missingness, and class imbalance
3. Drop features with > 40% missing values (above this threshold imputation is
   too speculative — the signal is more likely sensor dropout than informative)
4. Drop near-zero-variance features (no discriminative power)
5. Impute remaining missing values with column median (robust to outliers)
6. Stratified 80/20 train/test split
7. Save preprocessor pipeline to models/ for API reuse

Citation: McCann, M. and Johnston, A. (2008). SECOM.
UCI Machine Learning Repository. https://doi.org/10.24432/C54305
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MODELS_DIR = Path(__file__).parent.parent / "models"
MISSING_THRESHOLD = 0.40
VARIANCE_THRESHOLD = 0.0
TEST_SIZE = 0.20
RANDOM_STATE = 42


def _load_secom():
    """Load SECOM dataset, returning (X DataFrame, y Series).

    ucimlrepo returns SECOM with target_col=None so features/targets are not
    pre-split. We extract them manually from the combined 'original' DataFrame.
    Label encoding from the dataset: -1 = pass, 1 = fail.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        secom = fetch_ucirepo(id=179)
        df = secom.data.original
        y = df["class"]
        # Drop non-feature columns: label and timestamp
        X = df.drop(columns=["class", "timestamp"], errors="ignore")
        print("Loaded dataset via ucimlrepo.")
        return X, y
    except Exception as e:
        raise RuntimeError(
            f"ucimlrepo failed ({e}). Install with: pip install ucimlrepo"
        ) from e


def _report(X: pd.DataFrame, y: pd.Series) -> None:
    # SECOM label convention: -1 = pass, 1 = fail
    n_pass = (y == -1).sum()
    n_fail = (y == 1).sum()
    ratio = n_pass / n_fail if n_fail > 0 else float("inf")
    miss_pct = X.isnull().mean()
    print(f"\n{'='*50}")
    print(f"Raw dataset shape     : {X.shape}")
    print(f"Pass / Fail counts    : {n_pass} / {n_fail}")
    print(f"Imbalance ratio       : {ratio:.1f}:1 (pass:fail)")
    print(f"Features > 40% missing: {(miss_pct > MISSING_THRESHOLD).sum()}")
    print(f"Features any missing  : {(miss_pct > 0).sum()}")
    print(f"{'='*50}\n")


def load_and_prepare_data(save_preprocessor: bool = True):
    """
    Full data preparation pipeline.

    Returns
    -------
    X_train, X_test : np.ndarray  (after imputation + variance filtering)
    y_train, y_test : np.ndarray  (1 = pass, -1 = fail; kept as-is from SECOM)
    feature_names   : list[str]   columns surviving the pipeline
    """
    X_raw, y_raw = _load_secom()
    _report(X_raw, y_raw)

    # Recode to 0/1: SECOM uses -1=pass, 1=fail; we map fail→1 (positive class)
    y = (y_raw == 1).astype(int)

    # Drop high-missingness features
    miss_rate = X_raw.isnull().mean()
    keep_cols = miss_rate[miss_rate <= MISSING_THRESHOLD].index.tolist()
    X = X_raw[keep_cols].copy()
    print(f"After missingness drop : {X.shape[1]} features remain "
          f"(dropped {X_raw.shape[1] - X.shape[1]})")

    # Train/test split before fitting any transformers (no leakage)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Build sklearn pipeline: median imputation → variance filter
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("var_filter", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
    ])

    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    # Recover feature names after variance filter
    var_mask = preprocessor.named_steps["var_filter"].get_support()
    feature_names = [keep_cols[i] for i, keep in enumerate(var_mask) if keep]

    print(f"After variance filter  : {X_train.shape[1]} features remain")
    print(f"Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")
    print(f"Train fail rate: {y_train.mean():.3f}  |  Test fail rate: {y_test.mean():.3f}\n")

    if save_preprocessor:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
        joblib.dump(keep_cols, MODELS_DIR / "keep_cols.joblib")
        joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")
        print("Preprocessor saved to models/")

    return (
        X_train, X_test,
        y_train.values, y_test.values,
        feature_names,
    )


if __name__ == "__main__":
    load_and_prepare_data()
