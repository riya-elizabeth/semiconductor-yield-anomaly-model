I'm building a portfolio project to demonstrate fit for a "Staff AI Data Engineer – Wafer Fabrication" 
role at a semiconductor company. I need a complete, working, well-documented pipeline built end-to-end. 
I have a few hours, so prioritize a working, demoable result over exhaustive coverage.

DATASET
Use the UCI SECOM semiconductor manufacturing dataset (id=179). Load it with the `ucimlrepo` package:
    from ucimlrepo import fetch_ucirepo
    secom = fetch_ucirepo(id=179)
    X = secom.data.features
    y = secom.data.targets
If ucimlrepo isn't available or fails, fall back to downloading uci-secom.csv from Kaggle 
(paresh2047/uci-semcom) or scraping archive.ics.uci.edu/dataset/179/secom directly — document 
whichever path you used in the README.

BUSINESS FRAMING (use this language in docstrings, README, and comments)
This is a semiconductor fab line producing wafers under continuous sensor/process monitoring. 
Each row is a production entity with 590 anonymized process sensor signals and a pass/fail 
outcome from final line testing. The goals are: (1) predict yield/fail risk, (2) flag anomalous 
sensor behavior independent of the label ("excursion detection"), and (3) surface which sensor 
signals are most associated with fails, for engineers to investigate as candidate root causes — 
explicitly framed as correlation/association, not proven causation.

BUILD THE FOLLOWING

1. Data pipeline (src/data.py)
   - Load SECOM, report shape, missingness %, and class imbalance ratio
   - Drop or impute features with excessive missingness (document your threshold and choice)
   - Drop zero/near-zero-variance features
   - Train/test split with stratification (imbalance-aware)

2. Yield prediction model (src/model.py)
   - Train a classifier for pass/fail (try Random Forest and one gradient boosting model — 
     LightGBM if available, else sklearn GradientBoosting)
   - Handle class imbalance explicitly (class_weight='balanced' and/or SMOTE — pick one, justify it)
   - Evaluate with metrics appropriate for imbalance: precision/recall/F1 on the minority (fail) 
     class, ROC-AUC, PR-AUC — NOT plain accuracy as the headline metric
   - Save the trained model (joblib/pickle) to models/

3. Anomaly / excursion detector (src/anomaly.py)
   - Fit an Isolation Forest (or One-Class SVM) on the sensor features, unsupervised
   - Score every sample with an anomaly score
   - Report how well anomaly scores align with actual fails (without using the label to train) — 
     this is the "process monitoring / excursion detection" story
   - Save this model too

4. Root-cause / explainability (src/explain.py)
   - Use SHAP (or permutation importance if SHAP install is troublesome given time) on the 
     classifier to rank the top 15-20 sensor features most associated with fails
   - Output a ranked table/plot
   - Add a short markdown note explicitly caveating: these are correlational signals for 
     engineers to investigate, not proven root causes — physical/process validation would be 
     the next step

5. FastAPI service (app/main.py)
   - POST /predict — takes a JSON array of 590 sensor readings, returns {pass_fail_prediction, 
     fail_probability}
   - POST /anomaly-score — takes the same input, returns {anomaly_score, is_excursion} 
     (threshold-based flag)
   - GET /health — basic health check
   - GET /top-risk-signals — returns the top-N SHAP/importance features from step 4 (static, 
     precomputed)
   - Load models from models/ at startup, not per-request
   - Add request validation with Pydantic

6. Containerization
   - Dockerfile that builds and runs the FastAPI app (uvicorn), exposing port 8000
   - requirements.txt pinned to what's actually used
   - Confirm the container builds and runs locally: `docker build` then `docker run`, and 
     curl /health and /predict with a sample row to prove it works end-to-end

7. README.md
   - Problem framing (use the business framing above)
   - Dataset description + citation (McCann, M. and Johnston, A. (2008). SECOM. UCI Machine 
     Learning Repository. https://doi.org/10.24432/C54305)
   - Methodology summary (imbalance handling, model choice, anomaly approach, explainability)
   - Results: actual numbers from your run — precision/recall/F1/AUC for the classifier, and 
     however you quantified anomaly-detector alignment with real fails
   - How to run locally and via Docker
   - Explicit "Limitations & next steps" section: anonymized features (no real physical sensor 
     names), small fail count (104), single-site/single-timeframe data, no true production 
     time-series structure — be honest, don't oversell this as more than a portfolio 
     demonstration

CONSTRAINTS
- Everything must actually run — test each step, don't leave placeholder code
- Keep it to a single cohesive repo, sensible folder structure (src/, app/, models/, data/, 
  notebooks/ if you want an EDA notebook too)
- Prioritize finishing steps 1-6 working end-to-end over polishing any one step — if time runs 
  short, cut notebook EDA polish before cutting the Docker deployment
- At the end, give me 3 resume bullet variants (short/medium/strong, in the reverse-STAR 
  "result first" format) I can use, and confirm the exact numbers to cite so I don't misstate them