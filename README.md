# Capstone-project

# GHR Healthcare Clinician Retention Model

A machine learning project built for GHR Healthcare to predict whether a clinician will return for another placement assignment within 42 days of completing their current one. The model outputs a per-candidate retention likelihood score (0–100%) that recruiters can use to prioritize outreach and make more data-driven placement decisions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Objectives](#business-objectives)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Results](#results)
- [Scoring Function](#scoring-function)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

---

## Project Overview

GHR Healthcare places clinicians in contract roles across the country. High clinician turnover leads to increased recruiting costs, longer onboarding times, and staffing instability for the healthcare facilities they serve. This project addresses that problem by building a binary classification model that predicts retention at the placement level — each row in the dataset represents one placement, and the model predicts whether that clinician will begin a new placement within 42 days of that one ending.

The project follows the CRISP-DM methodology and went through four rounds of improvement covering feature engineering, model selection, hyperparameter tuning, and data leakage removal before arriving at the final model.

---

## Business Objectives

- Reduce clinician turnover by identifying likely returners before they go elsewhere
- Lower recruiting costs by retaining clinicians who already have experience with GHR
- Improve placement quality by keeping experienced clinicians in the pipeline
- Enable data-driven recruiter decisions through probability-based retention scores

---

## Dataset

The dataset contains **22,757 placement-level records** exported from GHR's Bullhorn ATS system. Each row represents one placement for one clinician.

- **Target variable:** `Retained_6_weeks` (1 if clinician returned within 42 days, 0 if not)
- **Class distribution:** 36% retained, 64% not retained
- **Train/test split:** 80/20 stratified

### Data Preparation

- Dates parsed and sorted by candidate and start date
- Retention label computed by checking whether the same clinician's next placement began within 42 days of the current placement ending
- Duplicate header rows removed
- The following columns were dropped due to leakage or low signal:
  - `status` — assigned after the placement ends, constitutes leakage
  - `jobOrder` — 4,600+ unique values generating high-cardinality noise
  - `customText9` — confirmed to be an internal record ID, not a real feature
  - `salary` — 99.7% of values were zero

---

## Feature Engineering

All features were computed without using any information from future placements to prevent data leakage. Candidate history features use only past rows for the same clinician.

| Feature | Description |
|---|---|
| `contract_length_days` | Duration of the current placement in days |
| `candidate_total_placements` | Total number of placements this clinician has had |
| `candidate_placement_rank` | Sequence number of this placement (1st, 2nd, 3rd...) |
| `candidate_avg_contract_days` | Clinician's average contract length across all placements |
| `candidate_retention_rate` | Rolling historical retention rate using only past placements. First-placement clinicians receive a sentinel value of -1 |
| `prev_gap_days` | Gap in days before this contract started. First-placement clinicians receive -1 |
| `is_first_placement` | Binary flag: 1 if this is the clinician's first placement, 0 otherwise |
| `hours_per_week` | Hours worked per week from customFloat18 |
| `same_location_as_prev` | Binary flag: 1 if clinician returned to the same location as previous assignment |
| `customText14_binary` | Binary encoding of a Yes/No field where Yes clinicians retain at 50% vs 33% |
| `employment_type_clean` | Cleaned version of employmentType (Travel, Local, Remote, PRN, Permanent, Other) |

### Features Excluded

- `start_year` — captures historical time trend that does not generalize to future predictions
- `start_dayofweek` — near-zero signal in placement data
- `start_month` / `start_quarter` — correlation with retention below 0.001 in data analysis

---

## Models

Five models were built across multiple iterations. All models use a scikit-learn Pipeline that applies median imputation and StandardScaler for numeric features, and most-frequent imputation with one-hot encoding for categorical features.

### Base Model: Logistic Regression

Selected for its simplicity and interpretability. Established the performance floor for comparison.

- Solver: lbfgs
- Max iterations: 2,000
- Class weight: balanced
- Regularization C: 0.1 (tuned via 5-fold GridSearchCV)

### Random Forest

Introduced as the first nonlinear model. Confirmed that tree-based approaches significantly outperform Logistic Regression on this data. Also used as the primary source of feature importance scores that guided feature engineering.

- n_estimators: 300
- min_samples_leaf: 2
- class_weight: balanced

### XGBoost (Final Recommended Model)

Gradient boosted decision tree ensemble that builds trees sequentially, with each new tree correcting the errors of the previous one. Tuned via 5-fold GridSearchCV.

- n_estimators: 200
- max_depth: 5
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 1.0
- scale_pos_weight: ~1.755 (neg/pos class ratio)

### CatBoost

Gradient boosting with ordered boosting and L2 leaf regularization. Tuned via RandomizedSearchCV with 15 iterations.

- iterations: 300
- depth: 8
- learning_rate: 0.1
- l2_leaf_reg: 3

### Gradient Boosting

scikit-learn's GradientBoostingClassifier included as an additional comparison point. Tuned via RandomizedSearchCV with 20 iterations.

- n_estimators: 200
- learning_rate: 0.1
- max_depth: 4
- min_samples_leaf: 5
- subsample: 0.8

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (Base) | 0.7597 | 0.6578 | 0.7048 | 0.6805 | 0.8144 |
| Random Forest | 0.7761 | 0.6593 | 0.7937 | 0.7203 | 0.8657 |
| Gradient Boosting (Tuned) | 0.8249 | 0.7386 | 0.8016 | 0.7669 | 0.9059 |
| CatBoost (Tuned) | 0.8302 | 0.7416 | 0.8185 | 0.7778 | 0.9120 |
| **XGBoost (Tuned)** | **0.8238** | **0.6933** | **0.9232** | **0.7919** | **0.9109** |

### Why XGBoost

XGBoost is recommended as the final model despite CatBoost having a marginally higher ROC-AUC (0.9120 vs 0.9109). In a staffing context, failing to identify a clinician who would have returned is more costly than flagging someone who does not return. XGBoost's recall of 0.923 means it correctly catches 92.3% of all actual returners, which is 10.5 percentage points higher than CatBoost.

### Top Predictors

Feature importance from the Random Forest and permutation importance from XGBoost consistently identified the same top predictors:

1. `candidate_total_placements` (importance: 0.353) — loyalty history is the single strongest signal
2. `candidate_avg_contract_days` — clinician's typical commitment pattern
3. `contract_length_days` — duration of current contract
4. `prev_gap_days` — how quickly the clinician returned after their last contract
5. `candidate_retention_rate` — rolling historical return rate

---

## Scoring Function

The `score_retention()` function allows anyone to upload a new placements CSV and get per-candidate retention likelihood scores without any manual feature engineering.

```python
results = score_retention()
```

### How It Works

1. Prompts the user to upload a placements CSV file
2. Checks for required columns and normalizes common column name variations
3. Automatically runs the full feature engineering pipeline
4. Applies the trained XGBoost model to generate a 0–100% retention likelihood per placement row
5. Aggregates scores at the candidate level (average across all placements)
6. Returns a ranked table of candidates with their overall Retention Likelihood %
7. Saves results to `retention_scores.csv`

### Required Columns

Only three columns are required at minimum:

- `candidate` — clinician name or ID
- `dateBegin` — placement start date
- `dateEnd` — placement end date

All other columns are optional. If they are missing the function fills them with safe defaults and notifies the user that predictions may be slightly less accurate.

### Output Format

| Candidate | Total_Placements | Retention_Likelihood_% |
|---|---|---|
| Jane Smith | 8 | 91.4 |
| John Doe | 3 | 67.2 |
| ... | ... | ... |

Candidates are ranked from highest to lowest retention likelihood. Scores can be interpreted as:

- **70% or above** — High priority, immediate outreach recommended
- **40–69%** — Medium priority, monitor and engage
- **Below 40%** — Low retention likelihood

---

## How to Run

This project was built and run in Google Colab. To run it yourself:

1. Open `GHR_Clinician_Retention_Model.ipynb` in Google Colab
2. Run all cells from top to bottom
3. When prompted in Cell 1, upload your placements CSV file
4. All feature engineering, model training, and evaluation will run automatically
5. To score a new dataset, call `score_retention()` in the final cell and upload a new CSV when prompted

### Running the Scoring Function on New Data

```python
# After running all training cells, call this to score new data
results = score_retention()

# results is a pandas DataFrame you can filter or sort
high_priority = results[results["Retention_Likelihood_%"] >= 70]
print(high_priority)
```

---

## Project Structure

```
GHR-Retention-Model/
│
├── GHR_Clinician_Retention_Model.ipynb   # Main notebook with all models and scoring function
│
├── reports/
│   ├── GHR_Modeling_Report.md            # Formal modeling report (base model + improvements)
│   └── GHR_Evaluation_Report.md          # CRISP-DM evaluation report
│
├── data/
│   └── CombinedAllPlacements.csv         # Training dataset (placement-level records)
│
└── README.md
```

---

## Requirements

The notebook runs in Google Colab which comes with most dependencies pre-installed. The following packages are used:

```
pandas
numpy
scikit-learn
xgboost
catboost
matplotlib
shap
faker          # only needed for the fake data generation cell
```

If running locally, install dependencies with:

```bash
pip install pandas numpy scikit-learn xgboost catboost matplotlib shap faker
```

---

## Authors

Sai Nuthalapati and Keaton Sandlin
Data Mining Project, April 2026
Supervisor: Donny Wright
