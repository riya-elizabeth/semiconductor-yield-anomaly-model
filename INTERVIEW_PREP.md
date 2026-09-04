# Interview Prep — Semiconductor Yield & Anomaly Detection Project

Questions interviewers will actually ask about this project, grouped by type.
For each question: the answer you should give, using the exact numbers from this project.

---

## "Walk me through the project" openers

These come first, almost always.

**Q: Tell me about this project.**

Know this cold. Suggested answer structure:
> "I built an end-to-end ML pipeline on semiconductor wafer data. The dataset has 590 sensor readings per wafer with a pass/fail outcome, heavily imbalanced at 14:1. I built three things: a yield classifier, an unsupervised excursion detector, and a SHAP explainability layer — then served all three through a FastAPI app deployed on GCP Cloud Run."

Never say "I built a model that predicts things." Always say what the data was, what the business problem was, and what you actually shipped.

---

## Data and Preprocessing Questions

**Q: How did you handle missing values? Why median and not mean?**
> Median is robust to outliers. Sensor data often has extreme readings from equipment malfunctions — those would pull the mean toward them. Median ignores them. Also, I dropped features with >40% missing first, because above that threshold imputation is more guesswork than recovery.

**Q: What is data leakage and did you avoid it here?**
> Data leakage is when information from your test set influences training. I avoided it by splitting first, then fitting the imputer only on training data. If I'd imputed the whole dataset before splitting, the test set's values would have shifted the training medians — the model would look better than it really is.

**Q: Why did you go from 590 to 442 features?**
> Two steps: dropped 32 features with >40% missingness, then dropped 116 more with near-zero variance. Zero-variance features are identical for every wafer — they carry no information and just add noise.

**Q: What is stratified splitting and why did you use it?**
> Stratified splitting preserves the class ratio in both train and test sets. With only 104 fails total, a random split might accidentally put 90 of them in training and only 14 in test — making evaluation unreliable. Stratification ensures both sets have the same ~6.6% fail rate.

---

## Imbalance and Metrics Questions

These are the most likely technical deep-dives.

**Q: How did you handle the 14:1 class imbalance?**
> Two ways: class_weight='balanced' during training, and threshold tuning after. class_weight tells the model to penalize missed failures 14x more than missed passes — mathematically equivalent to oversampling but without fake data. Threshold tuning found that 0.23 was the optimal cutoff, not the default 0.5.

**Q: Why not SMOTE?**
> SMOTE creates synthetic fail samples by interpolating between real ones. The risk is that those synthetic samples introduce patterns that don't exist in real production data. class_weight achieves the same goal — making the model sensitive to failures — without inventing data.

**Q: Why is accuracy a bad metric here?**
> A model that predicts "pass" for every single wafer gets 93.4% accuracy. It detects zero failures. Accuracy is useless when classes are imbalanced. The metric that matters is how well the model finds the rare class.

**Q: What's the difference between ROC-AUC and PR-AUC? Which did you use and why?**
> ROC-AUC measures the model's ability to separate classes across all thresholds. It can look deceptively good with imbalanced data because the large majority class dominates. PR-AUC focuses specifically on the minority class — precision and recall for fails only. With 14:1 imbalance, PR-AUC is the honest metric. Our random baseline PR-AUC was 0.067 (the fail rate); we got 0.20, which is 3x better than random.

**Q: What is threshold tuning and when would you use it in production?**
> Instead of always predicting "fail" at probability >= 0.5, I searched 200 thresholds from 0.01 to 0.99 and found the one that maximized F1 on the fail class. The winner was 0.23. In production you'd tune the threshold using a cost matrix — a missed failure might cost $10,000 in scrapped wafers, while a false alarm costs $500 in unnecessary inspection. The threshold should reflect that tradeoff, not just maximize F1.

---

## Model Questions

**Q: Why Random Forest over LightGBM here?**
> I trained both and compared on PR-AUC. Random Forest got 0.20, LightGBM got 0.17. LightGBM needed more tuning to work well with the small fail count (only 82 training fails) — its default min_child_samples=20 was too restrictive. RF won on this dataset so that's what we deployed.

**Q: What hyperparameters did you tune for Random Forest?**
> n_estimators=300 (more trees = more stable), max_features='sqrt' (each tree sees a random subset of features, which makes trees diverse and the ensemble stronger), class_weight='balanced'. I didn't do a full grid search given the time constraint — that would be the next step.

**Q: What is Isolation Forest and how does it detect anomalies?**
> Isolation Forest randomly partitions data using decision tree cuts. Anomalies are easier to isolate — they sit far from clusters and need fewer cuts to separate. Normal points are close to their neighbors and need many cuts. The anomaly score is based on how quickly a sample gets isolated. It works well on high-dimensional data without distance metrics.

**Q: Your anomaly detector got ROC-AUC 0.59. Is that good?**
> For unsupervised detection, yes — it's meaningful above-random performance (0.5 = random). The key point is that the labels were never seen during training. The model found anomalous sensor patterns purely from the data distribution, and those patterns correlate with real failures. That's the value: you can flag excursions in real time on the production line before final test results exist.

---

## SHAP / Explainability Questions

**Q: What is SHAP and why did you use it?**
> SHAP assigns each feature a contribution score for each individual prediction, based on game theory (Shapley values). It answers "how much did Attribute 104 contribute to this specific fail prediction?" I used TreeExplainer, which is optimized for tree-based models and runs in seconds instead of hours.

**Q: What does "mean absolute SHAP value" mean?**
> For each feature, you take the absolute SHAP value across all test samples and average them. High mean |SHAP| means that feature consistently moves predictions — either toward fail or toward pass — by a large amount. It's a global importance measure.

**Q: Why did you caveat the SHAP results as correlational?**
> Because correlation is not causation. The model learned that Attribute 104 is associated with failures in this dataset. But it could be a proxy — maybe Attribute 104 is correlated with some unmeasured physical cause. An engineer acting on this without process validation could change the wrong thing. The right use of SHAP here is to prioritize which sensors to investigate, not to prescribe action.

---

## API and Deployment Questions

**Q: Why FastAPI over Flask?**
> FastAPI has native Pydantic integration for input validation, automatic Swagger docs generation, and is async-native. Flask requires more boilerplate for the same functionality. FastAPI is also the current industry standard for ML inference APIs.

**Q: Why load models at startup instead of per request?**
> Loading a model from disk takes seconds. If you load it inside the endpoint function, every single request pays that cost. Loading once at startup means the model stays in memory and each request takes milliseconds. For a production API serving thousands of requests, this is the difference between a usable and an unusable service.

**Q: What is Docker and why containerize this?**
> Docker packages the app with all its dependencies into a single image. Without it, "it works on my machine" is a real problem — different Python versions, different package versions, different OS. With Docker, the image runs identically on your laptop, a colleague's machine, and GCP Cloud Run.

**Q: What is Cloud Run and why did you choose it?**
> Cloud Run is a serverless container platform on GCP. You give it a Docker image and it runs it at a public HTTPS URL, scales automatically with traffic, and scales to zero when idle (so you pay nothing when nobody's using it). It was the right choice here because the API doesn't get continuous traffic — it's a portfolio demo.

---

## Harder Questions (Senior / Staff Roles)

**Q: What would you do differently in a production system?**

Honest answers that show maturity:
- Real-time feature pipeline instead of batch CSV (Kafka or Pub/Sub for streaming sensor data)
- Model monitoring: detect when incoming sensor distributions drift from training data
- Proper threshold calibration using a business cost matrix, not just F1
- A/B testing framework before deploying a new model version
- Replace anonymized features with real sensor names via domain mapping
- Time-series features: rolling statistics, lag features, SPC (Statistical Process Control) signals

**Q: How would you retrain this model as new wafer data comes in?**
> Set up a scheduled retraining job (weekly or monthly). Monitor for distribution drift — if the incoming sensor data starts looking different from training data, trigger a retrain. Always evaluate the new model against the current one on a held-out validation set before swapping it in. Blue-green deployment to avoid downtime.

**Q: Your test set only has 21 failures. How confident are you in these metrics?**
> Not very, and that's an honest limitation I documented. With 21 samples, confidence intervals on precision and recall are wide — one misclassified wafer changes F1 by several points. This is why I used PR-AUC (based on probability rankings, not binary decisions) as the primary metric — it's more stable with small sample sizes. In production you'd need far more failure data before trusting specific numbers.

---

## Numbers to Have Ready at All Times

Never fumble these in an interview:

| What | Number |
|---|---|
| Dataset size | 1,567 wafers, 590 sensors |
| Fail count | 104 total (21 in test set) |
| Class imbalance | 14:1 pass:fail |
| Features after pipeline | 442 (from 590) |
| Decision threshold | 0.23 (tuned, not default 0.5) |
| RF PR-AUC | 0.20 (vs 0.067 random baseline = 3×) |
| RF ROC-AUC | 0.76 |
| RF F1 (fail class) | 0.34 |
| Anomaly detector ROC-AUC | 0.59 (unsupervised, no labels) |
| Top risk signal | Attribute 104 (mean \|SHAP\| = 0.018) |
| Live API | https://secom-api-170732699413.us-central1.run.app/docs |
