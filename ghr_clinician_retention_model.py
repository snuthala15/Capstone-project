from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================
# 1. LOAD & BASIC CLEANING
# ==========================

df = pd.read_csv("CombinedAllPlacements (1).csv", low_memory=False)
df.columns = df.columns.str.strip()

df = df[df["dateBegin"].astype(str).str.lower() != "datebegin"]
df = df[df["dateEnd"].astype(str).str.lower() != "dateend"]

df["candidate"] = df["candidate"].astype(str).str.strip()
df["dateBegin"] = df["dateBegin"].astype(str).str.strip()
df["dateEnd"]   = df["dateEnd"].astype(str).str.strip()

df["dateBegin"] = pd.to_datetime(df["dateBegin"], format="%m/%d/%y", errors="coerce")
df["dateEnd"]   = pd.to_datetime(df["dateEnd"], format="%m/%d/%y", errors="coerce")

df = df.dropna(subset=["candidate", "dateBegin", "dateEnd"])
df = df.drop(columns=["Unnamed: 18", "Unnamed: 19"], errors="ignore")

print("Rows after cleaning:", len(df))

df = df.sort_values(["candidate", "dateBegin"])
df["next_dateBegin"] = df.groupby("candidate")["dateBegin"].shift(-1)
df["next_candidate"] = df.groupby("candidate")["candidate"].shift(-1)
df["gap_days"] = (df["next_dateBegin"] - df["dateEnd"]).dt.days

df["Retained_6_weeks"] = (
    (df["candidate"] == df["next_candidate"]) &
    (df["gap_days"] >= 0) &
    (df["gap_days"] <= 42)
).astype(int)

print("Target distribution:")
print(df["Retained_6_weeks"].value_counts())

# ==========================
# 2. FEATURE ENGINEERING
# ==========================

df["contract_length_days"] = (df["dateEnd"] - df["dateBegin"]).dt.days

df["candidate_total_placements"]  = df.groupby("candidate")["candidate"].transform("count")
df["candidate_avg_contract_days"] = df.groupby("candidate")["contract_length_days"].transform("mean")
df["candidate_placement_rank"]    = df.groupby("candidate").cumcount() + 1
df["prev_gap_days"]               = df.groupby("candidate")["gap_days"].shift(1).fillna(-1)
df["is_first_placement"]          = (df["candidate_placement_rank"] == 1).astype(int)
df["hours_per_week"]              = pd.to_numeric(df["customFloat18"], errors="coerce")

valid_types = ["Travel", "Local", "Remote", "PRN", "Permanent"]
df["employment_type_clean"] = df["employmentType"].where(
    df["employmentType"].isin(valid_types), other="Other"
)

df["candidate_retention_rate"] = df.groupby("candidate")["Retained_6_weeks"].transform(
    lambda x: x.shift(1).expanding().mean()
).fillna(-1)

df["prev_location"] = df.groupby("candidate")["correlatedCustomText2"].shift(1)
df["same_location_as_prev"] = (df["correlatedCustomText2"] == df["prev_location"]).astype(int)
df["customText14_binary"] = (df["customText14"] == "Yes").astype(int)

# ==========================
# 3. PREPARE FEATURES
# ==========================

drop_cols = [
    "Retained_6_weeks",
    "next_dateBegin",
    "next_candidate",
    "gap_days",
    "prev_dateEnd",
    "id",
    "candidate",
    "dateBegin",
    "dateEnd",
    "salary",
    "salary_numeric",
    "status",
    "employmentType",
    "customText9",
    "customFloat18",
    "jobOrder",
    "customText14",
    "prev_location",
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
y = df["Retained_6_weeks"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nNumerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# ==========================
# 4. TRAIN / TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# 5. PREPROCESSING PIPELINE
# ==========================

all_missing_cols = [c for c in X.columns if X[c].isna().all()]
if all_missing_cols:
    X_train = X_train.drop(columns=all_missing_cols, errors="ignore")
    X_test  = X_test.drop(columns=all_missing_cols, errors="ignore")

categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ],
    remainder="drop"
)

print("Numerical features:", len(numerical_cols))
print("Categorical features:", len(categorical_cols))
print("Dropped all-missing columns:", all_missing_cols)

def evaluate_model(name, y_true, y_pred, y_proba):
    metrics = {
        "Accuracy":  round(accuracy_score(y_true, y_pred),  4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall":    round(recall_score(y_true, y_pred),    4),
        "F1":        round(f1_score(y_true, y_pred),        4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_proba),  4),
    }
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")
    print("  Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    return metrics

all_results = {}

# ==========================
# 6. LOGISTIC REGRESSION
# ==========================

log_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

param_grid_log = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__solver": ["lbfgs"]
}

grid_log = GridSearchCV(
    log_model,
    param_grid_log,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_log.fit(X_train, y_train)

print("\nBest Logistic Regression Parameters:", grid_log.best_params_)
print("Best Logistic Regression ROC-AUC (CV):", grid_log.best_score_)

best_log = grid_log.best_estimator_
y_pred_best = best_log.predict(X_test)
y_proba_best = best_log.predict_proba(X_test)[:, 1]

all_results["Logistic Regression"] = evaluate_model(
    "LOGISTIC REGRESSION (TUNED)", y_test, y_pred_best, y_proba_best
)

# ==========================
# 7. XGBOOST
# ==========================

pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / pos if pos > 0 else 1.0

xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    ))
])

xgb_model.fit(X_train, y_train)

param_grid_xgb = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.01, 0.1],
    "classifier__subsample": [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0]
}

grid_xgb = GridSearchCV(
    xgb_model,
    param_grid_xgb,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0
)

grid_xgb.fit(X_train, y_train)

print("\nBest XGBoost Parameters:", grid_xgb.best_params_)
print("Best XGBoost ROC-AUC (CV):", grid_xgb.best_score_)

best_xgb = grid_xgb.best_estimator_
y_pred_best = best_xgb.predict(X_test)
y_proba_best = best_xgb.predict_proba(X_test)[:, 1]

all_results["XGBoost (tuned)"] = evaluate_model(
    "XGBOOST (TUNED)", y_test, y_pred_best, y_proba_best
)

# ==========================
# 8. RANDOM FOREST
# ==========================

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

all_results["Random Forest"] = evaluate_model(
    "RANDOM FOREST", y_test, y_pred_rf, y_proba_rf
)

import matplotlib.pyplot as plt

importances = rf_model.named_steps["classifier"].feature_importances_
feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feat_imp.head(15))

feat_imp.head(15).plot(kind="barh")
plt.title("Top Important Features")
plt.show()

# ==========================
# 9. CATBOOST
# ==========================

!pip install catboost

from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV

cat_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", CatBoostClassifier(
        random_state=42,
        verbose=0
    ))
])

param_grid_cat = {
    "classifier__iterations": [200, 300, 500],
    "classifier__depth": [4, 6, 8],
    "classifier__learning_rate": [0.03, 0.05, 0.1],
    "classifier__l2_leaf_reg": [1, 3, 5]
}

random_cat = RandomizedSearchCV(
    cat_model,
    param_distributions=param_grid_cat,
    n_iter=15,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42
)

random_cat.fit(X_train, y_train)

print("Best CatBoost Parameters:", random_cat.best_params_)
print("Best CatBoost ROC-AUC:", random_cat.best_score_)

best_cat = random_cat.best_estimator_
y_pred_cat_best = best_cat.predict(X_test)
y_proba_cat_best = best_cat.predict_proba(X_test)[:, 1]

all_results["CatBoost (tuned)"] = evaluate_model(
    "CATBOOST (TUNED)", y_test, y_pred_cat_best, y_proba_cat_best
)

# ==========================
# 10. GRADIENT BOOSTING
# ==========================

from sklearn.ensemble import GradientBoostingClassifier

gb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(random_state=42))
])

param_grid_gb = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__learning_rate": [0.03, 0.05, 0.1],
    "classifier__max_depth": [2, 3, 4],
    "classifier__min_samples_leaf": [1, 3, 5],
    "classifier__subsample": [0.8, 1.0]
}

grid_gb = RandomizedSearchCV(
    gb_model,
    param_grid_gb,
    n_iter=20,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42
)

grid_gb.fit(X_train, y_train)

print("Best Gradient Boosting Parameters:", grid_gb.best_params_)
print("Best Gradient Boosting ROC-AUC:", grid_gb.best_score_)

best_gb = grid_gb.best_estimator_
y_pred_gb  = best_gb.predict(X_test)
y_proba_gb = best_gb.predict_proba(X_test)[:, 1]

all_results["Gradient Boosting (tuned)"] = evaluate_model(
    "GRADIENT BOOSTING (TUNED)", y_test, y_pred_gb, y_proba_gb
)

# ==========================
# 11. MODEL COMPARISON
# ==========================

summary_df = pd.DataFrame(all_results).T
summary_df = summary_df[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
summary_df = summary_df.sort_values("ROC-AUC", ascending=False)

print("\n" + "="*65)
print("  MODEL COMPARISON SUMMARY")
print("="*65)
print(summary_df.to_string())
print()
print(f"→ Best by ROC-AUC: {summary_df['ROC-AUC'].idxmax()}  ({summary_df['ROC-AUC'].max():.4f})")
print(f"→ Best by Recall:  {summary_df['Recall'].idxmax()}  ({summary_df['Recall'].max():.4f})")
print(f"→ Best by F1:      {summary_df['F1'].idxmax()}  ({summary_df['F1'].max():.4f})")

# ==========================
# 12. RETENTION SCORING
# ==========================

full_probs = best_xgb.predict_proba(X)[:, 1]

placement_scores = pd.DataFrame({
    "Candidate":         df["candidate"].values,
    "Contract_Number":   df["candidate_placement_rank"].values,
    "Placement_Score_%": (full_probs * 100).round(1)
})

overall_scores = (placement_scores
    .groupby("Candidate")
    .agg(
        Total_Placements=("Contract_Number", "max"),
        Overall_Score=("Placement_Score_%", "mean"),
        Times_in_Dataset=("Placement_Score_%", "count")
    )
    .round(1)
    .reset_index()
    .sort_values("Overall_Score", ascending=False)
)

print("Upload your candidates CSV file:")
cand_upload = files.upload()
cand_filename = list(cand_upload.keys())[0]
cand_df = pd.read_csv(cand_filename, low_memory=False)
cand_df.columns = cand_df.columns.str.strip()

cand_df = cand_df[cand_df["Name"].astype(str).str.lower() != "name"]
cand_names = cand_df["Name"].astype(str).str.strip().unique()

matched = overall_scores[overall_scores["Candidate"].isin(cand_names)].reset_index(drop=True)

print(f"\nCandidates in placements data: {len(overall_scores)}")
print(f"Candidates matched to candidates file: {len(matched)}")
print("\nTop 20 by Overall Retention Score:")
print(matched.head(20).to_string(index=False))

# ==========================
# 13. SCORE NEW DATASET
# ==========================

def score_retention():
    from google.colab import files
    import pandas as pd
    import numpy as np

    print("Please upload your placements CSV file:")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]

    df_new = pd.read_csv(filename, low_memory=False)
    df_new.columns = df_new.columns.str.strip()

    print(f"\nFile loaded: {filename}")
    print(f"Shape: {df_new.shape}")
    print(f"Columns found: {df_new.columns.tolist()}\n")

    rename_map = {
        "Candidate":        "candidate",
        "CANDIDATE":        "candidate",
        "DateBegin":        "dateBegin",
        "date_begin":       "dateBegin",
        "start_date":       "dateBegin",
        "DateEnd":          "dateEnd",
        "date_end":         "dateEnd",
        "end_date":         "dateEnd",
        "Employment Type":  "employmentType",
        "Employee Type":    "employeeType",
        "Hours Per Week":   "customFloat18",
        "Salary":           "salary",
        "Status":           "status",
        "ID":               "id",
        "Id":               "id",
    }
    df_new = df_new.rename(columns={k: v for k, v in rename_map.items() if k in df_new.columns})

    required = ["candidate", "dateBegin", "dateEnd"]
    missing = [c for c in required if c not in df_new.columns]
    if missing:
        print("ERROR: This dataset cannot be scored.")
        print(f"Missing required columns: {missing}")
        print("\nColumns found in your file:")
        for col in df_new.columns.tolist():
            print(f"  - {col}")
        return None

    df_new = df_new[df_new["dateBegin"].astype(str).str.lower() != "datebegin"]
    df_new = df_new[df_new["dateEnd"].astype(str).str.lower() != "dateend"]

    for fmt in ["%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"]:
        try:
            df_new["dateBegin"] = pd.to_datetime(df_new["dateBegin"], format=fmt, errors="coerce")
            df_new["dateEnd"]   = pd.to_datetime(df_new["dateEnd"],   format=fmt, errors="coerce")
            break
        except Exception:
            continue

    df_new["candidate"] = df_new["candidate"].astype(str).str.strip()
    df_new = df_new.dropna(subset=["candidate", "dateBegin", "dateEnd"])
    df_new = df_new.sort_values(["candidate", "dateBegin"]).reset_index(drop=True)

    if len(df_new) == 0:
        print("ERROR: No valid rows after cleaning.")
        return None

    print(f"Valid rows after cleaning: {len(df_new)}")

    df_new["next_dateBegin"]   = df_new.groupby("candidate")["dateBegin"].shift(-1)
    df_new["next_candidate"]   = df_new.groupby("candidate")["candidate"].shift(-1)
    df_new["gap_days"]         = (df_new["next_dateBegin"] - df_new["dateEnd"]).dt.days
    df_new["Retained_6_weeks"] = (
        (df_new["candidate"] == df_new["next_candidate"]) &
        (df_new["gap_days"] >= 0) &
        (df_new["gap_days"] <= 42)
    ).astype(int)

    df_new["contract_length_days"]        = (df_new["dateEnd"] - df_new["dateBegin"]).dt.days
    df_new["candidate_total_placements"]  = df_new.groupby("candidate")["candidate"].transform("count")
    df_new["candidate_avg_contract_days"] = df_new.groupby("candidate")["contract_length_days"].transform("mean")
    df_new["candidate_placement_rank"]    = df_new.groupby("candidate").cumcount() + 1
    df_new["prev_gap_days"]               = df_new.groupby("candidate")["gap_days"].shift(1).fillna(-1)
    df_new["is_first_placement"]          = (df_new["candidate_placement_rank"] == 1).astype(int)
    df_new["candidate_retention_rate"]    = df_new.groupby("candidate")["Retained_6_weeks"].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(-1)

    if "customFloat18" in df_new.columns:
        df_new["hours_per_week"] = pd.to_numeric(df_new["customFloat18"], errors="coerce")
    else:
        df_new["hours_per_week"] = np.nan

    if "correlatedCustomText2" in df_new.columns:
        df_new["prev_location"]         = df_new.groupby("candidate")["correlatedCustomText2"].shift(1)
        df_new["same_location_as_prev"] = (df_new["correlatedCustomText2"] == df_new["prev_location"]).astype(int)
    else:
        df_new["same_location_as_prev"] = 0

    if "customText14" in df_new.columns:
        df_new["customText14_binary"] = (df_new["customText14"] == "Yes").astype(int)
    else:
        df_new["customText14_binary"] = 0

    valid_types = ["Travel", "Local", "Remote", "PRN", "Permanent"]
    if "employmentType" in df_new.columns:
        df_new["employment_type_clean"] = df_new["employmentType"].where(
            df_new["employmentType"].isin(valid_types), other="Other"
        )
    else:
        df_new["employment_type_clean"] = "Other"

    optional_cols = {
        "customFloat18":         "hours_per_week",
        "correlatedCustomText2": "same location as previous",
        "customText14":          "Yes/No retention flag",
        "employmentType":        "employment type",
        "correlatedCustomText1": "division",
        "correlatedCustomText5": "team",
        "customText1":           "specialty",
        "customText2":           "unit",
        "employeeType":          "employee type",
    }
    missing_optional = [v for k, v in optional_cols.items() if k not in df_new.columns]
    if missing_optional:
        print("Note: These optional columns were not found — predictions may be less accurate:")
        for m in missing_optional:
            print(f"  - {m}")

    always_drop = [
        "Retained_6_weeks", "next_dateBegin", "next_candidate", "gap_days",
        "prev_dateEnd", "id", "candidate", "dateBegin", "dateEnd",
        "salary", "salary_numeric", "status", "employmentType",
        "customText9", "customFloat18", "jobOrder", "customText14", "prev_location"
    ]
    X_new = df_new.drop(columns=[c for c in always_drop if c in df_new.columns], errors="ignore")
    X_new = X_new.drop(columns=[c for c in X_new.columns if X_new[c].isna().all()], errors="ignore")

    print("\nScoring...")
    try:
        probs = best_xgb.predict_proba(X_new)[:, 1]
    except Exception as e:
        print(f"\nERROR: Could not score the dataset.")
        print(f"Details: {e}")
        print("Columns the model received:")
        print(X_new.columns.tolist())
        return None

    output = pd.DataFrame({
        "Candidate":       df_new["candidate"].values,
        "Contract_Number": df_new["candidate_placement_rank"].values,
        "Score":           (probs * 100).round(1)
    })

    final = (output
        .groupby("Candidate")
        .agg(
            Total_Placements=("Contract_Number", "max"),
            Overall_Score=("Score", "mean"),
        )
        .round(1)
        .reset_index()
        .sort_values("Overall_Score", ascending=False)
        .rename(columns={"Overall_Score": "Retention_Likelihood_%"})
        .reset_index(drop=True)
    )

    print(f"\nCandidates scored: {len(final)}")
    print("\n" + "="*50)
    print("    CANDIDATE RETENTION LIKELIHOOD SCORES")
    print("="*50)
    print(final.to_string(index=False))

    final.to_csv("retention_scores.csv", index=False)
    print("\nSaved to retention_scores.csv")

    return final
