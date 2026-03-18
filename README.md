# 🔬 Predictive Modeling for Cancer Risk Assessment (CRPM)

> **Triaging patients into Low / Medium / High cancer risk from 17 clinical and lifestyle risk factors — with a critical data leakage discovery at the core of the narrative**
>
> A multiclass classification system on a 2,000-patient synthetic cancer risk dataset. The project's defining moment is the discovery that `Overall_Risk_Score` is a leakage feature (caused near-100% accuracy), and the subsequent rigorous effort to build a genuinely predictive model — ultimately deploying a class-weighted XGBoost tuned by Optuna, served via a dual-mode Streamlit app.

---

<div align="center">

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%2B%20Optuna-green)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![uv](https://img.shields.io/badge/Package%20Manager-uv-7C3AED)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

> ⚕️ **Medical Disclaimer:** This system is for educational and research purposes only. It is not validated for clinical use and must not be used for medical risk assessment without appropriate validation, regulatory approval, and supervised clinical governance.

---

## 📊 Project Slides

> **Want the visual overview first?** The deck covers the leakage discovery, experiment progression, and system design in 12 slides.

👉 **[View the Project Presentation (PPTX)](https://docs.google.com/presentation/d/1RRkU4vUiAhrPyIq38v7rlvOqECDz0FEs/edit?usp=sharing&ouid=117459468470211543781&rtpof=true&sd=true)**

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Problem Statement](#1-problem-statement) |
| 2 | [Project Overview](#2-project-overview) |
| 3 | [Tech Stack](#3-tech-stack) |
| 4 | [High-Level Architecture](#4-high-level-architecture) |
| 5 | [Repository Structure](#5-repository-structure) |
| 6 | [Dataset & Features](#6-dataset--features) |
| 7 | [The Data Leakage Discovery](#7-the-data-leakage-discovery) |
| 8 | [Experiment Progression](#8-experiment-progression) |
| 9 | [Model Performance — Deployed Model](#9-model-performance--deployed-model) |
| 10 | [Streamlit Web Application](#10-streamlit-web-application) |
| 11 | [How to Replicate — Full Setup Guide](#11-how-to-replicate--full-setup-guide) |
| 12 | [Running the Application](#12-running-the-application) |
| 13 | [Business Applications & Other Domains](#13-business-applications--other-domains) |
| 14 | [How to Improve This Project](#14-how-to-improve-this-project) |
| 15 | [Troubleshooting](#15-troubleshooting) |
| 16 | [Glossary](#16-glossary) |

---

## 1. Problem Statement

### What problem are we solving?

Cancer remains one of the leading causes of mortality worldwide. Early identification of high-risk individuals — those most likely to develop cancer — enables proactive screening, lifestyle intervention, and targeted monitoring, dramatically improving outcomes. However, cancer risk is determined by the interaction of dozens of demographic, genetic, lifestyle, and environmental factors. Manually assessing this multidimensional risk profile is infeasible at scale.

The challenge in building a predictive risk model is threefold: the dataset is heavily imbalanced (High-risk patients are rare), the High-risk class is the most clinically important, and naive feature inclusion can produce deceptively high accuracy through data leakage.

Core objectives:

- 🔬 **Identify genuine risk factors** — distinguish which features are real predictors vs proxy variables or leakage sources
- 🎯 **Triclass prediction** — classify patients as Low, Medium, or High cancer risk from clinical and lifestyle inputs
- ⚖️ **Handle imbalance** — the High-risk class (102 patients, 5.1%) needs special treatment to avoid being dominated by the Medium class (78.7%)
- 🏥 **Honest evaluation** — detect and eliminate data leakage before reporting any performance metric

### What does CRPM answer?

> *"Given a patient's age, BMI, smoking history, genetic markers, dietary patterns, and environmental exposures — what is their cancer risk level: Low, Medium, or High?"*

---

## 2. Project Overview

| Aspect | Detail |
|--------|--------|
| **Dataset** | Synthetic cancer risk dataset (2,000 patients, 21 columns) |
| **Task** | Multiclass classification: Low / Medium / High cancer risk |
| **Class distribution** | Medium 1,574 (78.7%) · Low 324 (16.2%) · High 102 (5.1%) |
| **Features (final)** | 17 clinical/lifestyle features (after removing Patient_ID, Cancer_Type, Overall_Risk_Score) |
| **Critical design decision** | Overall_Risk_Score causes data leakage → removed from all model training |
| **Deployed model** | Class-weighted XGBClassifier tuned via Optuna (40 trials, macro-F1 objective) |
| **Best Optuna CV macro-F1** | 0.7213 (class-weighted XGBoost) |
| **Saved artifacts** | `model_xgb_new.pkl` · `final_xgb_class_weighted.pkl` · `label_encoder.pkl` · `feature_names.pkl` |
| **Streamlit modes** | Batch CSV upload + manual single patient input |
| **Python version** | 3.13 |
| **Package manager** | uv (pyproject.toml) |

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.13 | Core language |
| **Package manager** | `uv` + `pyproject.toml` | Dependency management (replaces pip) |
| **ML — Gradient Boosting** | XGBoost 3.1.1 | Primary classifier — class weights handle imbalance |
| **ML — Baselines** | RandomForestClassifier, LogisticRegression (sklearn) | Baseline comparisons |
| **Imbalance** | SMOTE (imbalanced-learn) | Synthetic minority oversampling |
| **HPT** | Optuna (TPE sampler) | Bayesian hyperparameter optimisation |
| **Data processing** | Pandas 2.3.3, NumPy 2.3.4 | Data loading, feature engineering, train/test splits |
| **Preprocessing** | StandardScaler, LabelEncoder (sklearn) | Feature scaling and label encoding |
| **Serialisation** | joblib 1.5.2 | Saves/loads all model artifacts |
| **Web UI** | Streamlit 1.51.0 | Dual-mode prediction interface (batch + manual) |
| **Visualisation** | Matplotlib, Seaborn | EDA and correlation analysis |

---

## 4. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│                                                                      │
│  cancer-risk-factors.csv  (2,000 rows · 21 columns)                  │
│         │                                                            │
│   Drop: Patient_ID  (identifier — no predictive value)               │
│   Drop: Cancer_Type (outcome label — data leakage)                  │
│   Drop: Overall_Risk_Score (computed from target — data leakage)    │
│                                                                      │
│   17 genuine risk factor features remain                             │
│   LabelEncoder: Low→? · Medium→? · High→?                           │
│   train_test_split(test_size=0.2, stratify=y, random_state=42)      │
│   SMOTE applied to training set only (for RF experiments)            │
└──────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                     EXPERIMENT LAYER                                 │
│                                                                      │
│  [LEAKAGE] RF + LogReg with Overall_Risk_Score → ~100%, 98%         │
│  RF without Overall_Risk_Score → 83%, macro F1 0.49 (High: 0.10)   │
│  SMOTE + RF → 84%, macro F1 0.64 (High: 0.38)                      │
│  Optuna RF (50 trials) → CV macro F1 0.959 / test High F1 0.29     │
│  XGBoost + SMOTE pipeline → 85%, macro F1 0.68 (High: 0.42)        │
│  XGBoost + Optuna (40 trials, High recall) → CV recall 0.734        │
│  Class-weighted XGBoost → High recall 0.80 / macro F1 0.53         │
│  ✅ Class-weighted XGBoost + Optuna (40 trials, macro F1)            │
│     → Best CV macro F1: 0.7213 → DEPLOYED                           │
└──────────────────────────────────────────────────────────────────────┘
                               │
              model_xgb_new.pkl + label_encoder.pkl + feature_names.pkl
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                      SERVING LAYER (app.py)                          │
│                                                                      │
│  Mode A: Upload CSV (batch)                                          │
│    preprocess_input() → model.predict() → le.inverse_transform()    │
│    Display: risk labels + per-class probabilities + download CSV     │
│                                                                      │
│  Mode B: Manual input (single patient)                               │
│    Sidebar: number_input per feature → predict → probability table  │
│    Highlight: P(High) >= 0.5 → st.warning (clinical follow-up)      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Repository Structure

```
Predictive-Modeling-for-Cancer-Risk-Assessment-Using-Machine-Learning/
│
├── Cancer_Risk_Prediction_(ML).ipynb     # Full experiment notebook (all 9 experiments)
│
├── cancer-risk-factors.csv               # Raw dataset (2,000 rows, 21 columns)
│
├── model_xgb_new.pkl                     # Deployed: class-weighted XGBoost (Optuna-tuned)
├── final_xgb_class_weighted.pkl          # Alternative: class-weighted XGBoost (baseline)
├── label_encoder.pkl                     # Fitted LabelEncoder for Risk_Level target
├── feature_names.pkl                     # Ordered list of 17 feature names
│
├── app.py                                # Streamlit app (batch + manual modes)
├── main.py                               # CLI stub (prints hello)
│
├── pyproject.toml                        # uv project config (Python 3.13, dependencies)
├── requirements.txt                      # Flat dependency list
└── uv.lock                               # Exact locked dependency versions
```

---

## 6. Dataset & Features

### Dataset

| Property | Detail |
|----------|--------|
| **Source** | Synthetic cancer risk factors dataset |
| **Rows** | 2,000 patient records |
| **Columns** | 21 total |
| **Target** | `Risk_Level` — Low / Medium / High |

### Class Distribution

```
Risk_Level
Medium    1574   (78.7%)   ← dominant class
Low        324   (16.2%)
High       102   ( 5.1%)   ← most clinically important; rarest
```

This is a severe three-way imbalance — Medium dominates, and High (the most clinically critical class) has only 102 examples. Without imbalance handling, models learn to predict Medium almost exclusively.

### Feature Set (17 genuine risk factors)

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Numeric | Patient age (years) |
| `Gender` | Binary int | 0 = Female, 1 = Male |
| `Smoking` | Numeric | Smoking intensity / pack-years |
| `Alcohol_Use` | Numeric | Alcohol consumption level |
| `Obesity` | Numeric | Obesity score |
| `Family_History` | Binary | 1 = family history of cancer |
| `Diet_Red_Meat` | Numeric | Red meat dietary intake |
| `Diet_Salted_Processed` | Numeric | Salted/processed food intake |
| `Fruit_Veg_Intake` | Numeric | Fruit and vegetable intake (protective) |
| `Physical_Activity` | Numeric | Physical activity level |
| `Air_Pollution` | Numeric | Environmental air pollution exposure |
| `Occupational_Hazards` | Numeric | Occupational carcinogen exposure |
| `BRCA_Mutation` | Binary | 1 = BRCA gene mutation present |
| `H_Pylori_Infection` | Binary | 1 = H. pylori infection (stomach cancer risk) |
| `Calcium_Intake` | Numeric | Calcium supplementation/diet level |
| `BMI` | Numeric | Body Mass Index |
| `Physical_Activity_Level` | Numeric | Physical activity intensity level |

### Columns Removed (and Why)

| Column | Reason for Removal |
|--------|-------------------|
| `Patient_ID` | Unique identifier — no predictive value |
| `Cancer_Type` | Outcome label correlated with risk level — would cause data leakage |
| `Overall_Risk_Score` | Computed directly from the target — severe data leakage (see Section 7) |

---

## 7. The Data Leakage Discovery

### The Smoking Gun

The first two models — Random Forest and Logistic Regression — included `Overall_Risk_Score` as a feature. They achieved near-perfect accuracy:

```
Random Forest (with Overall_Risk_Score):
  Accuracy: 1.00 (400/400 correct — 1 mistake out of 400)
  High: precision 1.00  recall 0.95  F1 0.97
  ...

Logistic Regression (with Overall_Risk_Score):
  Accuracy: 0.98
  High: precision 1.00  recall 0.80  F1 0.89
```

These results are implausibly good for a 5-class-heavy imbalanced multiclass problem on a 2,000-row dataset.

### The Diagnosis

`Overall_Risk_Score` is a **computed score derived from the same risk factors that determine the target label** — essentially a pre-computed version of what the model is trying to predict. Including it means the model "cheats" by reading the answer:

```
Cancer_Type + Risk factors → Overall_Risk_Score → Risk_Level

The model was learning: Overall_Risk_Score → Risk_Level
...instead of:           Risk factors → Risk_Level
```

### The Proof

After removing `Overall_Risk_Score`, a simple Random Forest dropped to 83% accuracy with catastrophic performance on the High class:

```
Random Forest (without Overall_Risk_Score):
  Accuracy: 0.83
  High: precision 1.00  recall 0.05  F1 0.10  ← misses 95% of High-risk patients
  Low:  precision 0.85  recall 0.34  F1 0.48
  Medium: precision 0.83  recall 0.99  F1 0.90
```

This collapse confirmed the leakage. The challenge was now real: build a model that genuinely distinguishes risk levels from the 17 actual risk factor features.

### Why Cancer_Type Was Also Dropped

`Cancer_Type` is an outcome category strongly correlated with risk level. Including it would let the model learn "Prostate → High risk" rather than learning from true biological and lifestyle drivers. It is not a predictive input — it is a grouping derived from the same clinical pathway as the target.

---

## 8. Experiment Progression

All experiments after Section 7 are conducted on the 17 genuine features with `Overall_Risk_Score` removed.

### Experiment Summary

| # | Model | Imbalance Strategy | Accuracy | High F1 | Macro F1 | Note |
|---|-------|-------------------|----------|---------|----------|------|
| 1 | RF | None (with leakage) | 1.00 | 0.97 | 0.99 | ⚠️ Leakage — invalid |
| 2 | LogReg | None (with leakage) | 0.98 | 0.89 | 0.95 | ⚠️ Leakage — invalid |
| 3 | RF | None | 0.83 | 0.10 | 0.49 | Real baseline — misses 95% of High |
| 4 | RF | SMOTE | 0.84 | 0.38 | 0.64 | Better but High recall still 0.35 |
| 5 | RF + Optuna (50 trials) | SMOTE (outside) | 0.83 | 0.29 | 0.61 | CV macro F1 0.959 but overfitted |
| 6 | XGBoost (baseline) | SMOTE inside pipeline | 0.85 | 0.42 | 0.68 | Best baseline so far |
| 7 | XGBoost + Optuna (40 trials) | SMOTE inside pipeline | — | — | CV recall 0.734 | Optimised for High recall |
| 8 | Class-weighted XGBoost | weights = {H:15.4, L:4.9, M:1.0} | 0.63 | 0.34 | 0.53 | High recall 0.80 but overall collapse |
| 9 ✅ | **Class-weighted XGB + Optuna** | **class_weight dict** | **—** | **—** | **CV 0.7213** | **Deployed** |

### Key Finding: Optuna RF Paradox

The Optuna-tuned Random Forest achieved an impressive macro F1 of **0.959 on the SMOTE-resampled training cross-validation** — but only **0.61 on the true test set**, with the High class F1 collapsing to 0.29. This illustrates the danger of evaluating on resampled validation data: the SMOTE-synthetic High-risk samples are easier to classify than real patients.

### Why Class-Weighted XGBoost Was Chosen

Class weights force the model to penalise misclassification of rare classes more severely. Computed as:

```python
classes, counts = np.unique(y_train_enc, return_counts=True)
class_weights = {cls: max(counts) / count for cls, count in zip(classes, counts)}
# Result: {High: 15.35, Low: 4.86, Medium: 1.0}
```

High-risk patients carry 15× the penalty weight of Medium-risk patients — preventing the model from ignoring them. Combined with Optuna's Bayesian HPT optimising macro-F1 across all three classes, this produced the best balanced performance.

### Optuna Hyperparameter Search (Deployed Model)

Best parameters from the `xgb_class_weighted_macroF1` study (40 trials, TPE sampler):

```python
{
    'n_estimators':  197,
    'max_depth':       9,
    'learning_rate': 0.0948,
    'subsample':     0.7322,
    'colsample_bytree': 0.8379,
    'gamma':         1.5757,
    'reg_alpha':     0.9263,
    'reg_lambda':    2.8072
}
Best CV macro-F1: 0.7213
```

---

## 9. Model Performance — Deployed Model

### Class-Weighted XGBoost (baseline, before Optuna)

The untuned class-weighted model showed the trade-off between High recall and overall accuracy:

```
Class-weighted XGBoost (baseline):
  Accuracy: 0.63
  High:   precision 0.21  recall 0.80  F1 0.34
  Low:    precision 0.42  recall 0.80  F1 0.55
  Medium: precision 0.92  recall 0.58  F1 0.71
  Class weights: {High: 15.35, Low: 4.86, Medium: 1.0}
```

High recall reaches 0.80 — catching 80% of all high-risk patients — but at significant cost to overall accuracy (0.63).

### Deployed Model: Class-Weighted XGBoost + Optuna

The Optuna-tuned variant (model_xgb_new.pkl) achieves a better macro-F1 balance across all three classes. Best CV macro-F1: **0.7213**.

### Baseline XGBoost + SMOTE Pipeline (for comparison)

```
XGBoost + SMOTE (inside pipeline):
  Accuracy: 0.85
  High:   precision 0.39  recall 0.45  F1 0.42
  Low:    precision 0.79  recall 0.68  F1 0.73
  Medium: precision 0.90  recall 0.92  F1 0.91
  Macro avg: precision 0.69  recall 0.68  F1 0.68
```

Better accuracy but lower High recall — misses 55% of high-risk patients.

### Three Saved Artifacts

| File | Model | Strategy | Use |
|------|-------|----------|-----|
| `model_xgb_new.pkl` | XGBoost | Class-weighted + Optuna (macro-F1) | **Deployed in app.py** |
| `final_xgb_class_weighted.pkl` | XGBoost | Class-weighted (baseline) | Comparison |
| `label_encoder.pkl` | LabelEncoder | Fitted on y_train | Required for `le.inverse_transform()` |
| `feature_names.pkl` | Python list | 17 feature names in correct order | Column validation at inference |

---

## 10. Streamlit Web Application

### Two Prediction Modes

#### Mode A — Upload CSV (Batch)

```
User uploads CSV with feature columns
    ↓
preprocess_input(df)
    ├── Identify missing columns → fill with 0
    ├── Reorder to FEATURE_NAMES (17 features)
    └── pd.to_numeric(errors='coerce').fillna(0)
    ↓
model.predict(X) → encoded class integers
    ↓
le.inverse_transform(preds) → ["Low", "Medium", "High", ...]
    ↓
predict_proba(X) → P(High), P(Low), P(Medium) per row
    ↓
Display: full table with predictions + probabilities + download button
```

#### Mode B — Manual Input (Single Patient)

```
Sidebar: 17 number_input widgets (one per feature, default = 0.0)
    ↓
st.sidebar.button("Predict")
    ↓
Prediction + probability table (sorted by probability desc)
    ↓
P(High) highlighted:
  >= 0.5 → st.warning("High risk probability >= 0.5 — consider clinical follow-up")
  < 0.5  → st.success("High risk probability below 0.5")
```

### Inference Preprocessing (`preprocess_input`)

```python
def preprocess_input(df):
    # Check for missing columns
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        st.warning(f"Filling {len(missing)} missing columns with zeros")
        for c in missing:
            df[c] = 0

    # Reorder to match training column order
    df = df[FEATURE_NAMES].copy()

    # Convert all to numeric, fill NaN with 0
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df
```

Note that the `StandardScaler` fitted during training is **not** applied at inference — the Streamlit app calls `model.predict()` directly on the raw numeric features. This works because XGBoost is not sensitive to feature scaling, but for scikit-learn models (LogReg, SVM) a saved scaler would be required.

### App Footer

```
Model: class-weighted XGBoost (tuned via Optuna).
Ensure uploaded CSV has the same feature columns used in training.
```

---

## 11. How to Replicate — Full Setup Guide

### Prerequisites

- Python 3.13
- `uv` package manager

---

### Step 1 — Clone and Install

```bash
git clone https://github.com/sahatanmoyofficial/Predictive-Modeling-for-Cancer-Risk-Assessment-Using-Machine-Learning.git
cd Predictive-Modeling-for-Cancer-Risk-Assessment-Using-Machine-Learning

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
uv pip install -r requirements.txt
```

---

### Step 2 — Verify Artifacts

```bash
ls *.pkl
# model_xgb_new.pkl
# final_xgb_class_weighted.pkl
# label_encoder.pkl
# feature_names.pkl
```

If missing, run the full notebook to regenerate:

```bash
jupyter notebook "Cancer_Risk_Prediction_(ML).ipynb"
# Run all cells — the last cells save the pkl files
```

---

### Step 3 — Launch Streamlit App

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

### Step 4 — Explore the Notebook

```bash
jupyter notebook "Cancer_Risk_Prediction_(ML).ipynb"
```

The notebook walks through all 9 experiments in order — the leakage discovery is in the `## Removing Overall_Risk_Score and retrying` section.

---

## 12. Running the Application

### Streamlit UI (Local)

```bash
source .venv/bin/activate
streamlit run app.py
```

### Batch Prediction (CSV Upload)

Prepare a CSV with the 17 feature columns in any order:

```csv
Age,Gender,Smoking,Alcohol_Use,Obesity,Family_History,Diet_Red_Meat,...
68,0,7,2,8,0,5,...
74,1,8,9,8,0,0,...
```

Upload via the Streamlit UI → download the results CSV with `Predicted_Risk_Level` and `prob_High`, `prob_Low`, `prob_Medium` columns added.

### Manual Single Patient

Use the sidebar sliders/inputs for all 17 features, then click **Predict**.

---

## 13. Business Applications & Other Domains

### Primary Use Case — Population-Level Cancer Risk Triage

> ⚕️ This system is for research and educational demonstration only. Clinical deployment requires regulatory validation, prospective clinical validation, and clinical governance oversight.

| Stakeholder | Value |
|-------------|-------|
| **Oncologists / GPs** | Automated pre-screening flags which patients warrant further investigation |
| **Public health programmes** | Stratify large populations by risk tier for resource-efficient screening allocation |
| **Occupational health** | Flag workers with high occupational + lifestyle risk combinations |
| **Insurance / actuarial** | Risk stratification for underwriting (with appropriate regulation) |
| **Research institutions** | Identify high-risk cohorts for longitudinal study recruitment |

### Same Architecture in Other Medical Domains

| Domain | Risk Classes | Key Analogous Features |
|--------|-------------|----------------------|
| **Cardiovascular disease** | Low/Med/High | BMI, smoking, cholesterol, blood pressure, age |
| **Diabetes risk** | Pre-diabetic / diabetic / normal | BMI, family history, glucose, activity level |
| **Stroke risk** | Low/Med/High | Age, hypertension, AF, smoking, cholesterol |
| **Mental health** | Low/Med/High | Lifestyle, genetics, trauma history, occupation |

---

## 14. How to Improve This Project

### 🧠 Model Improvements

| Area | Priority | Recommendation |
|------|----------|---------------|
| **Evaluate deployed model on test set** | 🔴 High | The notebook shows CV macro-F1 = 0.7213 but never prints the test classification report for the final Optuna model — add `classification_report(y_test_enc, final_model.predict(X_test))` |
| **SMOTE inside pipeline for XGBoost** | 🔴 High | Class-weighted XGBoost bypasses SMOTE entirely — try combining `class_weight` + `SMOTE` inside an ImbPipeline for comparison |
| **CatBoost or LightGBM** | 🟡 Medium | Both have built-in class weighting and often outperform XGBoost on small imbalanced datasets |
| **Threshold tuning** | 🟡 Medium | For the High class, find the optimal probability threshold that maximises recall at acceptable precision — don't rely on default 0.5 |
| **Feature importance** | 🟡 Medium | Add XGBoost feature importance or SHAP values — which of the 17 factors matter most? |
| **Cross-validation** | 🟡 Medium | Evaluate final model with StratifiedKFold to confirm CV macro-F1 holds across folds |

### 🏗️ Engineering Improvements

| Area | Recommendation |
|------|---------------|
| **Save StandardScaler** | If future experiments use sklearn models (LogReg, SVM), save the fitted scaler alongside the model |
| **Default sidebar values** | Manual mode defaults all 17 features to 0.0 — add sensible defaults (e.g. Age=50, BMI=25) for a more realistic demo |
| **Feature descriptions** | Add feature descriptions or tooltips to the Streamlit sidebar inputs for non-technical users |
| **Model metadata** | Store training date, dataset hash, and feature list alongside each `.pkl` |
| **Unit tests** | Test `preprocess_input()` for missing column handling and dtype coercion |

---

## 15. Troubleshooting

| Error / Symptom | Fix |
|----------------|-----|
| `FileNotFoundError: model_xgb_new.pkl` | Confirm all 4 pkl files are in the project root; retrain from notebook if missing |
| `ValueError: Missing columns` | Uploaded CSV must have at least the 17 feature columns (names case-sensitive) |
| `sklearn version mismatch` | Use Python 3.13 with versions from `uv.lock`; label_encoder.pkl is sklearn-version-sensitive |
| All predictions are "Medium" | Check that `Overall_Risk_Score` was not included — it would cause near-100% accuracy; check feature list in `feature_names.pkl` |
| `uv` command not found | Install: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Streamlit shows blank prediction | Click the "Predict" button in Mode B; output only appears after button press |
| Batch CSV probabilities all similar | Normal for a challenging imbalanced dataset — the model is appropriately uncertain; High probability > 0.5 is a strong signal |

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **Data leakage** | When a feature used in training encodes information about the target that would not be available at inference — producing artificially high accuracy that doesn't generalise |
| **Overall_Risk_Score** | A pre-computed composite score in this dataset derived from the same factors that determine Risk_Level — including it causes data leakage |
| **Cancer_Type** | The type of cancer diagnosed — an outcome label correlated with risk level, not a predictive input feature |
| **Multiclass classification** | Predicting one of 3+ discrete classes — here: Low, Medium, High cancer risk |
| **Class imbalance** | When one class is much more frequent (Medium: 78.7%) than others (High: 5.1%) — causes models to ignore rare classes |
| **SMOTE** | Synthetic Minority Oversampling Technique — generates synthetic minority class examples by interpolating between existing ones |
| **Class weights** | Per-class multipliers applied to the loss function — High-risk patients carry 15.35× the penalty weight here |
| **Optuna** | Bayesian hyperparameter optimisation framework using TPE sampler — proposes better parameters based on trial history |
| **TPE sampler** | Tree-structured Parzen Estimator — Optuna's default sampler for intelligent HPT |
| **Macro-F1** | F1 score averaged equally across all classes (regardless of class size) — appropriate for imbalanced multiclass problems |
| **LabelEncoder** | sklearn transformer encoding string labels (Low/Medium/High) as integers for model compatibility |
| **`feature_names.pkl`** | Serialised list of 17 feature names in the exact training column order — used to validate and reorder inference inputs |
| **`predict_proba`** | Returns the probability of each class for each input — used to generate calibrated risk scores |
| **uv** | Ultra-fast Python package manager — `pyproject.toml` specifies dependencies, `uv.lock` freezes exact versions |
| **BRCA_Mutation** | Mutation in the BRCA1 or BRCA2 gene — significantly increases risk of breast and ovarian cancer |
| **H. pylori** | Helicobacter pylori — a bacterial infection associated with increased stomach cancer risk |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⚕️ Medical Disclaimer

This project is for **educational and research purposes only**. The models and predictions generated are not validated for clinical use and must not be used for medical risk assessment, diagnosis, or treatment decisions. Any clinical application requires regulatory approval (e.g. FDA, CE marking), prospective clinical validation, and integration into a supervised clinical workflow with appropriate governance.

---

## 👤 Author

**Tanmoy Saha**
[linkedin.com/in/sahatanmoyofficial](https://linkedin.com/in/sahatanmoyofficial) | sahatanmoyofficial@gmail.com

---
