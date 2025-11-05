# 📚 Teacher Churn Prediction for Yandex Textbook

## 🚀 Overview

This repository contains a machine learning solution for predicting teacher churn on the **Yandex Textbook** educational platform.  
By analyzing weekly activity snapshots alongside behavioral embeddings, the model estimates the probability that a teacher will stop actively using the service on two distinct time horizons:

- **week_churn** — churn within the next week  
- **month_churn** — churn within the next month

The problem is framed as **binary classification** on tabular numerical features and embeddings, treated as **two independent targets**.  
All text data has been pre-processed into numerical representations by the organizers.

---

## 🧾 Repository Contents
- `task.pdf` — Original problem statement  
- `solution.ipynb` — Jupyter Notebook with full solution pipeline  
- `README.md` — Project documentation

---

## 🧠 Problem Description

Based on platform activity history and aggregated behavior metrics, the task is to predict churn probabilities for teachers in the test sample. The targets are provided only for training data.

### Provided Datasets
- **train.csv** — features + targets  
- **test.csv** — features only  
- **sample_submission.csv** — required submission format

Feature schema:
- `nid` — teacher identifier  
- `report_date` — snapshot date  
- `v_0 … v_N` — anonymized numerical features (activity aggregates, embeddings)  
- `week_churn`, `month_churn` — target labels (train only)

---

## 📏 Evaluation Metric

Performance is evaluated using **macro ROC-AUC**:

$$
\text{score}=\frac{1}{2}\left(\mathrm{ROC\text{-}AUC}\!\left(\text{week\char`_churn}\right)+\mathrm{ROC\text{-}AUC}\!\left(\text{month\char`_churn}\right)\right)
$$

Submission probabilities must be valid floats in `[0, 1]`.

---

## 📤 Submission Format

The submission file must contain exactly:

| nid | week_churn | month_churn |
|-----|------------|--------------|
| ... | ... | ... |

Requirements:
- row count & ordering must match `test.csv` exactly
- no NaNs / no extra columns
- probabilities only
- duplicate `nid` allowed if present in test set

---

## ✅ Rules

- Any standard ML models are allowed (trees, boosting, neural networks, etc.)
- External data is prohibited
- Randomness must be fixed for reproducibility
- Additional manual labeling is forbidden

---

## 🛠️ Solution Pipeline

The implementation in `solution.ipynb` includes:

### 1. Data Preprocessing
- Date parsing from `report_date`
- Temporal feature extraction:
  - month, week number, day, day of week
- Cyclical seasonal encoding
- Handling anonymized numeric features

### 2. Feature Engineering
- Seasonal behavior patterns
- Aggregation-based churn risk features
- Snapshot-based drift signals

### 3. Modeling Strategy
- Separate **LightGBM** models for each target
- Binary classification with probability outputs
- Ensemble predictions for robustness
- Random state fixing for reproducibility

### 4. Post-processing
- Probability calibration
- Validation against metric behavior

---

## 📈 Model Advantages

✔️ Handles two independent churn horizons  
✔️ Works with dense numeric embeddings  
✔️ Encodes temporal periodicity  
✔️ Robust to date distribution shifts  
✔️ Maintains reproducibility constraints

---

## 🏁 Final Output

Last submission was estimated with **0.951** ROC AUC.
Our team took the **7th** place among more than 50 teams.

---

## 📌 Acknowledgements

This repository was created within the context of the **Yandex Textbook** churn prediction challenge.

