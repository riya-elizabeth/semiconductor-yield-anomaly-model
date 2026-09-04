# Code Walkthrough — Semiconductor Yield & Anomaly Detection Pipeline

This document is a study guide. It explains **what the project is, why every decision was made, what every tool does, and how every line of code works** — from raw data to a live deployed API.

---

## 1. What Is This Project?

### The Real-World Problem

A semiconductor fab (fabrication plant) makes silicon wafers. Each wafer goes through hundreds of manufacturing steps, each monitored by sensors. At the end, the wafer either **passes** or **fails** a quality test.

The problem: by the time you know a wafer failed, it's already been through the entire expensive process. If you could **predict failure early** — using sensor readings from earlier in the process — you could intervene sooner and save money.

This project builds three things:
1. **A yield predictor** — given 590 sensor readings, predict pass or fail
2. **An excursion detector** — flag unusual sensor behaviour *without* using any labels (unsupervised)
3. **A root-cause ranker** — surface which sensors are most associated with failures

All three are then served via a **live REST API** deployed on Google Cloud.

### The Dataset

**SECOM** (Semiconductor Manufacturing) from the UCI Machine Learning Repository.

| Property | Value |
|---|---|
| Rows | 1,567 (one per wafer production run) |
| Columns | 590 sensor/process measurements + 1 label + 1 timestamp |
| Label | -1 = pass, 1 = fail (SECOM's convention) |
| Fail rate | 104 failures out of 1,567 = 6.6% |
| Imbalance | ~14 passes for every 1 fail |

The features are **anonymized** — named "Attribute 1" through "Attribute 590". In a real fab these would have physical names (temperature, pressure, flow rate, etc.). The anonymization is why root-cause ranking is "for investigation" rather than actionable on its own.

---

## 2. Architecture Overview

```
Raw Data (UCI SECOM)
        │
        ▼
┌─────────────────┐
│   src/data.py   │  ← Clean, impute, split, save preprocessor
└────────┬────────┘
         │  X_train, X_test, y_train, y_test
    ┌────┴─────────────────────────┐
    ▼                              ▼
┌──────────────┐          ┌───────────────┐
│ src/model.py │          │ src/anomaly.py│
│ (supervised) │          │ (unsupervised)│
└──────┬───────┘          └──────┬────────┘
       │                         │
       ▼                         ▼
yield_model.joblib        anomaly_model.joblib
yield_threshold.joblib    excursion_threshold.joblib
       │
       ▼
┌────────────────┐
│ src/explain.py │  ← SHAP values on yield model
└──────┬─────────┘
       │
       ▼
top_risk_signals.json
       │
       ▼
┌─────────────────┐
│  app/main.py    │  ← FastAPI loads all artifacts at startup
│  (FastAPI)      │
└──────┬──────────┘
       │
       ▼
GCP Cloud Run (live HTTPS API)
```

Everything flows **top to bottom once** (training). After that, the API serves predictions in real time by loading the saved artifacts.

---

## 3. Tools Used and Why

| Tool | What it is | Why we used it |
|---|---|---|
| **pandas** | DataFrame library | Reading, filtering, and manipulating tabular data |
| **numpy** | Array math library | Fast numerical operations on model inputs/outputs |
| **scikit-learn** | ML toolkit | Preprocessing pipeline, Random Forest, Isolation Forest, metrics |
| **LightGBM** | Gradient boosting library | Fast, strong tree model to compare against Random Forest |
| **SHAP** | Model explainability | Computes how much each feature contributed to each prediction |
| **joblib** | Serialization | Saves/loads Python objects (models, pipelines) to disk |
| **FastAPI** | Web framework | Builds the REST API endpoints |
| **Pydantic** | Data validation | Validates that API inputs have the right shape and types |
| **uvicorn** | Web server | Runs the FastAPI app (like gunicorn but async) |
| **Docker** | Containerization | Packages the app so it runs identically anywhere |
| **GCP Cloud Run** | Serverless hosting | Runs the Docker container at a public HTTPS URL, scales to zero |
| **ucimlrepo** | Dataset loader | Fetches the SECOM dataset directly from UCI |

---

## 4. File-by-File Code Walkthrough

---

### `src/data.py` — The Data Pipeline

This is the foundation. Every other script calls this one.

#### Constants at the top

```python
MODELS_DIR = Path(__file__).parent.parent / "models"
MISSING_THRESHOLD = 0.40
VARIANCE_THRESHOLD = 0.0
TEST_SIZE = 0.20
RANDOM_STATE = 42
```

- `Path(__file__).parent.parent / "models"` — `__file__` is the path to `data.py` itself. `.parent` goes up to `src/`, `.parent` again goes up to the project root, then we append `models/`. This makes the path work regardless of where you run the script from.
- `MISSING_THRESHOLD = 0.40` — any feature with >40% of values missing gets dropped entirely. Why 40%? Because if almost half your data for a feature is missing, any value you substitute (impute) is mostly made up. Below 40%, imputing with the median is reasonable.
- `RANDOM_STATE = 42` — fixes the random seed so results are reproducible. 42 is conventional, any number works.

#### `_load_secom()`

```python
def _load_secom():
    from ucimlrepo import fetch_ucirepo
    secom = fetch_ucirepo(id=179)
    df = secom.data.original
    y = df["class"]
    X = df.drop(columns=["class", "timestamp"], errors="ignore")
    return X, y
```

- `fetch_ucirepo(id=179)` downloads the SECOM dataset. The `id=179` is SECOM's permanent identifier in the UCI registry.
- **Why `.data.original` instead of `.data.features`?** The ucimlrepo package is supposed to split features and targets automatically, but SECOM's metadata doesn't declare which column is the target, so `features` and `targets` come back as `None`. We go to `original` (the raw combined DataFrame) and split manually.
- `df["class"]` is the label column (-1 or 1).
- `df.drop(columns=["class", "timestamp"])` removes the label and timestamp — we only want the 590 sensor readings as features.

#### `_report()`

```python
def _report(X, y):
    n_pass = (y == -1).sum()
    n_fail = (y == 1).sum()
    ratio = n_pass / n_fail
```

Just prints diagnostic information. Important: in SECOM, **-1 = pass and 1 = fail** (counterintuitive). This is the original dataset's convention.

#### `load_and_prepare_data()` — The Main Function

**Step 1: Recode labels**
```python
y = (y_raw == 1).astype(int)
```
Converts SECOM's (-1, 1) labels to (0, 1) that scikit-learn expects. `(y_raw == 1)` gives True/False → `.astype(int)` makes it 1/0. Result: 1 = fail (positive class), 0 = pass.

**Step 2: Drop high-missingness features**
```python
miss_rate = X_raw.isnull().mean()
keep_cols = miss_rate[miss_rate <= MISSING_THRESHOLD].index.tolist()
X = X_raw[keep_cols].copy()
```
- `isnull().mean()` gives a Series where each value is the fraction of nulls in that column (0.0 to 1.0).
- We keep only columns where that fraction is ≤ 0.40.
- Result: 590 → 558 features (dropped 32).

**Step 3: Train/test split BEFORE fitting transformers**
```python
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```
- `stratify=y` ensures both train and test have the same 6.6% fail rate. Without this, you might randomly get all the fails in training or all in test.
- **Critical: we split BEFORE imputing.** If we imputed first using the whole dataset, the test set values would influence the training imputation — that's data leakage, which artificially inflates performance.

**Step 4: Build the preprocessing pipeline**
```python
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("var_filter", VarianceThreshold(threshold=0.0)),
])
X_train = preprocessor.fit_transform(X_train_df)
X_test = preprocessor.transform(X_test_df)
```
- `Pipeline` chains steps so they apply in order.
- `SimpleImputer(strategy="median")` fills missing values with the column median. Median is better than mean for skewed sensor data because it's not pulled by outliers.
- `VarianceThreshold(threshold=0.0)` removes features where every value is identical (zero variance). These have no information.
- `fit_transform` on train — **learns** the median from training data, then applies it.
- `transform` on test — applies the **same medians learned from training** to test data. Never fit on test.
- Result: 558 → 442 features.

**Step 5: Save the preprocessor**
```python
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
joblib.dump(keep_cols, MODELS_DIR / "keep_cols.joblib")
joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")
```
We save three things:
- `preprocessor.joblib` — the fitted imputer + variance filter (knows the medians, knows which columns to keep)
- `keep_cols.joblib` — which of the original 590 columns survived the missingness filter
- `feature_names.joblib` — the final 442 column names after both filters

The API needs all three to process a new incoming request exactly the same way training processed the data.

---

### `src/model.py` — Yield Prediction

#### Why two models?

We train Random Forest and LightGBM and compare. In ML you rarely know which model will win — you try both and let the metrics decide. Here Random Forest won.

#### Why class_weight='balanced' instead of SMOTE?

**SMOTE** (Synthetic Minority Oversampling Technique) creates fake fail samples by interpolating between real ones. The problem: fake data introduces noise and can cause the model to learn patterns that don't exist in reality.

**class_weight='balanced'** instead tells the model: "when you make a mistake on a fail, penalize it 14× more than a mistake on a pass." This is mathematically equivalent to oversampling but without fake data.

#### `_best_threshold()`

```python
def _best_threshold(y_prob, y_true):
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred_t, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t
```

This is one of the most important functions in the project.

By default, scikit-learn classifiers predict class 1 (fail) if `probability >= 0.5`. But with 14:1 imbalance, the model almost never outputs a probability above 0.5 for fails — so it predicts "pass" for everything. Accuracy is 93% but you've detected zero failures.

This function tries 200 different thresholds from 0.01 to 0.99 and finds the one that maximises F1 score on the fail class. For Random Forest, the winning threshold was **0.23** — meaning "if the model says there's a 23% or higher chance of failure, flag it as fail."

#### Why PR-AUC as the primary metric?

- **Accuracy** is useless here. A model that always predicts "pass" gets 93.4% accuracy. Meaningless.
- **ROC-AUC** measures the model's ability to separate the two classes. It's useful but can look good even when the model fails on the minority class.
- **PR-AUC** (Precision-Recall AUC) measures how well the model performs specifically on the positive (fail) class. It's the right metric when one class is rare. Our Random Forest got **0.20 vs a random baseline of 0.067** — 3× better than random.

#### `train_random_forest()`

```python
rf = RandomForestClassifier(
    n_estimators=300,       # 300 trees in the forest
    class_weight="balanced", # penalize fail mistakes 14x more
    max_features="sqrt",    # each tree uses sqrt(442) ≈ 21 features
    random_state=42,
    n_jobs=-1,              # use all CPU cores
)
```

A Random Forest builds many decision trees on random subsets of data and features, then votes. `max_features="sqrt"` is standard — giving each tree a random subset of features makes the trees diverse, which improves the ensemble.

#### `train_lightgbm()`

```python
neg, pos = np.bincount(y_train)   # counts of 0s and 1s
spw = neg / pos                    # ~14 (14 passes per fail)
lgbm = lgb.LGBMClassifier(
    scale_pos_weight=spw,          # same idea as class_weight but LightGBM's version
    min_child_samples=5,           # default is 20, too high for 82 training fails
    subsample=0.8,                 # use 80% of rows per tree
    colsample_bytree=0.8,          # use 80% of features per tree
)
```

`min_child_samples=5` was a key fix. LightGBM's default requires at least 20 samples in each leaf node. With only 82 training fails, this was too restrictive — the model couldn't form any meaningful splits on the fail class.

#### Winner selection

```python
best = max(results, key=lambda r: r["pr_auc"])
```

Picks the model with the highest PR-AUC. Random Forest won (0.20 vs 0.17).

---

### `src/anomaly.py` — Excursion Detection

This is the **unsupervised** component. Labels are never used during training.

#### Why Isolation Forest?

Isolation Forest works by randomly partitioning data. The key insight: **anomalies are easier to isolate** — they require fewer random cuts to separate from the rest. Normal points cluster together and take many cuts to isolate.

It works well on high-dimensional data (442 features here) without needing distance metrics, which makes it fast and robust.

#### The training

```python
iso = IsolationForest(
    contamination=0.05,   # we assume ~5% of runs are true excursions
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_train)          # X_train only, no y_train
```

`contamination=0.05` is a prior assumption: we tell the model that about 5% of samples are anomalous. This sets where the decision boundary falls. In a real fab, you'd set this based on historical excursion rates.

#### Scoring and threshold

```python
train_scores = iso.decision_function(X_train)
excursion_threshold = np.percentile(train_scores, 5)  # bottom 5th percentile
```

`decision_function` returns a score where **lower = more anomalous**. We take the 5th percentile of training scores as the threshold — any new sample scoring below this is flagged as an excursion.

```python
train_scores_inv = -train_scores  # invert: higher = more anomalous
```

We invert the sign so the score is more intuitive (higher = worse) and so it aligns with ROC-AUC calculation (higher scores should correspond to positives/fails).

#### Post-hoc evaluation

```python
roc = roc_auc_score(y_test, test_scores_inv)  # 0.5875
```

We never used labels during training — but after training, we check: do samples the model flagged as anomalous tend to actually be fails? ROC-AUC of 0.59 > 0.5 (random), meaning yes, there is real alignment. The detector has above-random ability to find real failures without supervision.

---

### `src/explain.py` — Root-Cause Signal Ranking

#### What SHAP is

SHAP (SHapley Additive exPlanations) answers: **for this specific prediction, how much did each feature contribute?**

For example, if the model predicted 80% fail probability for a wafer, SHAP tells you: "Attribute 104 pushed the probability up by 0.12, Attribute 60 pushed it up by 0.08, Attribute 300 pushed it down by 0.03..."

#### TreeExplainer vs KernelExplainer

```python
explainer = shap.TreeExplainer(model)
```

`TreeExplainer` is model-specific — it uses the tree structure of Random Forest to compute exact SHAP values efficiently. `KernelExplainer` is model-agnostic but much slower (hours on this dataset). Always use TreeExplainer for tree models.

#### Getting the SHAP values

```python
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    sv = shap_values[1]        # older SHAP: class 1 = fail
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]  # newer SHAP: shape (samples, features, classes)
```

Different versions of the SHAP library return different formats. Newer versions return a 3D array `(n_samples, n_features, n_classes)`. We want class 1 (fail) so we take `[:, :, 1]`.

#### Ranking features

```python
mean_abs_shap = np.abs(sv).mean(axis=0)
ranked_idx = np.argsort(mean_abs_shap)[::-1]
```

- `np.abs(sv)` — absolute value (we care about magnitude, not direction)
- `.mean(axis=0)` — average across all test samples → one value per feature
- `np.argsort(...)[::-1]` — sort indices from highest to lowest

Result: `Attribute 104` has the highest average |SHAP| value (0.0176), meaning it contributes most to fail predictions on average across the test set.

#### The caveat

This is deliberately written into the output:
> "Rankings reflect model-learned correlations. NOT proven root causes."

This matters professionally. In a real fab, an engineer who saw "Attribute 104 is top-ranked" might shut down equipment or change a process. But correlation ≠ causation. The correct interpretation is: "investigate Attribute 104 first, but validate with physical experiments before acting."

---

### `app/main.py` — The FastAPI Service

#### Pydantic input validation

```python
class SensorReading(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def check_length(cls, v):
        if len(v) != 590:
            raise ValueError(f"Expected 590 sensor values, got {len(v)}")
        return v
```

Pydantic automatically:
- Checks that `features` is a list
- Converts each element to float (so integers work too)
- Runs `check_length` to enforce exactly 590 values

If any of these fail, FastAPI returns a 422 error automatically — you never even reach the model code.

#### Lifespan — loading models once at startup

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["yield_model"] = joblib.load(MODELS_DIR / "yield_model.joblib")
    # ... load everything else ...
    yield       # ← app runs here, handling requests
    _state.clear()
```

`lifespan` is a FastAPI pattern that runs setup code before the app starts and teardown after it stops. Loading models here (not inside each endpoint) means:
- **Models load once** at startup (~5 seconds)
- **Each request** is fast because models are already in memory

If you loaded the model inside `/predict`, every single request would spend seconds loading a 100MB file from disk. That would be terrible for latency.

#### `_preprocess()` — making sure API inputs match training

```python
def _preprocess(raw_features):
    col_names = [f"Attribute {i+1}" for i in range(590)]
    row_df = pd.DataFrame([raw_features], columns=col_names)
    row_filtered = row_df[keep_cols]          # drop same 32 columns as training
    return preprocessor.transform(row_filtered)  # impute + variance filter
```

This is the most important helper function in the API. It replicates **exactly** what `data.py` did during training:
1. Name the 590 columns the same way ("Attribute 1", "Attribute 2", ...)
2. Drop the same 32 high-missingness columns (`keep_cols`)
3. Apply the same imputer + variance filter (`preprocessor.transform`)

If you skip any of these steps, the model receives features in the wrong order or wrong shape and produces garbage predictions.

#### `/predict` endpoint

```python
@app.post("/predict", response_model=PredictResponse)
def predict(reading: SensorReading):
    X = _preprocess(reading.features)
    prob_fail = float(model.predict_proba(X)[0, 1])
    prediction = "fail" if prob_fail >= threshold else "pass"
    return PredictResponse(pass_fail_prediction=prediction, fail_probability=round(prob_fail, 4))
```

- `predict_proba(X)` returns a `(1, 2)` array — one row, two columns (prob of class 0, prob of class 1)
- `[0, 1]` gets the first row, second column = probability of class 1 (fail)
- Compare against the tuned threshold (0.23) not the default 0.5

#### `/anomaly-score` endpoint

```python
raw_score = float(iso.decision_function(X)[0])
anomaly_score = round(-raw_score, 6)   # invert: higher = more anomalous
is_excursion = raw_score <= excursion_threshold
```

The Isolation Forest's `decision_function` returns a negative number for anomalies. We invert it so the returned `anomaly_score` is positive and higher = more anomalous (more intuitive for users). `is_excursion` is True if the raw score fell below the 5th percentile threshold computed during training.

---

## 5. The Full Data Flow for One Prediction Request

When you call `POST /predict` with 590 sensor readings, here's exactly what happens:

```
1. FastAPI receives JSON → Pydantic validates 590 floats → SensorReading object

2. _preprocess(features):
   a. Create DataFrame with columns "Attribute 1"..."Attribute 590"
   b. Keep only the 558 columns that survived missingness filtering
   c. Apply preprocessor.transform():
      - SimpleImputer fills any NaN with training-set medians
      - VarianceThreshold drops 116 more columns
   d. Result: (1, 442) numpy array

3. model.predict_proba(X):
   - 300 Random Forest trees each vote
   - Average vote = probability of fail
   - e.g. output: [[0.8267, 0.1733]]

4. Compare 0.1733 against threshold 0.23:
   - 0.1733 < 0.23 → prediction = "pass"

5. Return JSON: {"pass_fail_prediction": "pass", "fail_probability": 0.1733}
```

---

## 6. Key Concepts to Know for Interviews

**Data leakage** — contaminating your model with information it wouldn't have in production. We avoided this by splitting before imputing.

**Class imbalance** — when one class is much rarer than the other (here 14:1). Naively training on imbalanced data produces models that ignore the minority class. Solutions: class weights, SMOTE, threshold tuning.

**Threshold tuning** — instead of always using 0.5 as the cutoff for binary classification, find the cutoff that optimises the metric you actually care about (here: F1 on the fail class).

**PR-AUC vs ROC-AUC** — both measure classifier quality, but PR-AUC is more informative when the dataset is imbalanced. ROC-AUC can look high even when the model is bad at finding the rare class.

**SHAP** — a principled way to explain model predictions. Each feature gets a score representing its contribution to a specific prediction. The theoretical foundation comes from game theory (Shapley values).

**Supervised vs unsupervised** — the yield model is supervised (uses labels to train). The anomaly detector is unsupervised (trains on features only, labels never seen). Both are valid — they answer different questions.

**Sklearn Pipeline** — chains preprocessing steps so they apply in sequence and you can call `fit_transform` / `transform` on the whole thing. Critical benefit: prevents data leakage when you fit only on training data.

**joblib** — like pickle but optimised for numpy arrays and sklearn objects. Standard way to save trained ML models to disk.

**FastAPI lifespan** — the recommended pattern for loading expensive resources (models, DB connections) once at startup rather than on every request.

---

## 7. Things That Were Harder Than Expected (and Why)

| Problem | What happened | How we fixed it |
|---|---|---|
| ucimlrepo returned None | Package doesn't auto-split SECOM because `target_col=None` in metadata | Used `.data.original` and split manually |
| Labels were inverted | SECOM uses -1=pass, 1=fail (backwards from what you'd expect) | Found it when fail rate showed as 93% instead of 6.6% |
| Both models predicted all-pass | Default 0.5 threshold predicts majority class when imbalance is 14:1 | Added `_best_threshold()` to tune the cutoff |
| SHAP returned 3D array | Newer SHAP version changed output format from list to ndarray | Added version check: `if shap_values.ndim == 3: sv = shap_values[:, :, 1]` |
| Docker image rejected by Cloud Run | Mac built ARM image, Cloud Run needs AMD64 | Rebuilt with `--platform linux/amd64` |
| LightGBM predicted all-pass even after threshold tuning | `min_child_samples=20` default too high for 82 training fails | Reduced to `min_child_samples=5` |
