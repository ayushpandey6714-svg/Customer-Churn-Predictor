"""
CUSTOMER CHURN PREDICTION - TRAINING SCRIPT
============================================
Churn = the customer left the company (cancelled the service).
Goal: predict in advance which customer will churn, so the company
can give them an offer to retain them.

This is a BINARY CLASSIFICATION problem (the answer is one of only 2: Churn=1 or No churn=0).

Flow:
  1. Load + clean data
  2. Train/Test split  (FIRST of all, to prevent data leakage)
  3. Preprocessing Pipeline (missing values, scaling, encoding)
  4. Compare multiple models with Cross-Validation
  5. Hyperparameter tuning of the best model
  6. Choose a threshold (inside the train data only)
  7. Final evaluation on the test set (only ONCE)
  8. Save the model (app.py will use it)

Run:  python train.py
"""

# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------
import json                      # to save metadata (metrics, threshold)
import warnings                  # to silence unnecessary warnings
import joblib                    # to save/load the trained model to/from a file
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # mode for saving graphs without a screen (safe on servers/Colab)
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,            # split data into train/test
    StratifiedKFold,             # cross-validation; keeps the churn ratio the same in every fold
    cross_validate,              # CV with multiple metrics at once
    cross_val_predict,           # out-of-fold predictions through CV
    RandomizedSearchCV,          # hyperparameter tuning (tries random combinations)
)
from sklearn.compose import ColumnTransformer      # different preprocessing for different columns
from sklearn.pipeline import Pipeline              # chain steps together
from sklearn.impute import SimpleImputer           # fill missing values
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, precision_recall_curve,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42                # same number = same result every time (reproducible)

# ---------------------------------------------------------------
# STEP 1: LOAD + CLEAN DATA
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: Load + clean data")
print("=" * 60)

df = pd.read_csv("data/churn.csv")                 # read the CSV into a DataFrame (table)
print("Shape:", df.shape)                          # (rows, columns)

# TotalCharges should be numeric, but in this dataset some entries are a blank space " ",
# so pandas made it text (object). errors="coerce" = anything that can't become a number
# turns into NaN (missing). These are the new customers with tenure=0.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# customerID is just a unique ID; the model can learn nothing from it -> drop it.
# (If we kept it, the model would memorize IDs = overfitting)
df = df.drop(columns=["customerID"])

# Convert the target from text to number: Yes -> 1 (churn), No -> 0
df["Churn"] = (df["Churn"] == "Yes").astype(int)

print("\nChurn distribution (1 = churn):")
print(df["Churn"].value_counts(normalize=True).round(3))
# ~26.5% churn: this is CLASS IMBALANCE. So looking only at accuracy is WRONG
# (if you predict "No churn" for everyone you still get ~73% accuracy!). We look at Recall/F1/ROC-AUC.

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ---------------------------------------------------------------
# STEP 2: Separate X and y + TRAIN/TEST SPLIT
# ---------------------------------------------------------------
# X = features (input columns), y = target (what we want to predict)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# stratify=y -> the churn ratio stays the same in both train and test.
# test_size=0.2 -> 20% of the data is locked in a "vault" for testing. It opens only on exam day.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------
# STEP 3: PREPROCESSING PIPELINE
# ---------------------------------------------------------------
# Divide columns into 2 types: numeric and categorical
numeric_cols = X_train.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
print("\nNumeric cols    :", numeric_cols)
print("Categorical cols:", categorical_cols)

# Numeric pipeline:
#   1) SimpleImputer(median): fill missing numbers with the median (less affected by outliers)
#   2) StandardScaler: (x - mean) / std. This brings all columns to the same scale
#      (tenure 0-72 vs TotalCharges 0-8000 -> the scale difference confuses LogisticRegression)
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical pipeline:
#   1) SimpleImputer(most_frequent): fill missing text with the most common value
#   2) OneHotEncoder: splits "Contract" = Month-to-month / One year / Two year into
#      3 separate 0/1 columns. (A model doesn't understand text, only numbers)
#      handle_unknown="ignore" -> if a new category shows up in the app, it won't crash
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# ColumnTransformer: numeric cols get numeric_pipe, categorical cols get categorical_pipe
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

# WHY PIPELINE? (to avoid DATA LEAKAGE - an interview favourite)
# When a Pipeline is used inside cross-validation, in every fold the scaler/imputer is fit
# ONLY on that fold's train part. If we had scaled the entire data first,
# information from the validation data (mean/std) would leak into train -> the score would look falsely good.

# ---------------------------------------------------------------
# STEP 4: COMPARE MODELS (Cross-Validation)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Compare models (5-fold Cross-Validation on TRAIN only)")
print("=" * 60)

# class_weight="balanced": gives the minority class (churn) more importance, to handle imbalance
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

# StratifiedKFold: splits the train data into 5 parts; train on 4, validate on 1, 5 times
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = ["roc_auc", "f1", "recall", "precision", "accuracy"]

results = []
for name, model in models.items():
    # Attach the preprocessing pipeline to every model
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    results.append({
        "Model": name,
        "ROC-AUC": scores["test_roc_auc"].mean(),
        "F1": scores["test_f1"].mean(),
        "Recall": scores["test_recall"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Accuracy": scores["test_accuracy"].mean(),
    })

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).round(4)
print(results_df.to_string(index=False))
results_df.to_csv("models/model_comparison.csv", index=False)

best_name = results_df.iloc[0]["Model"]
print(f"\nBest model (by ROC-AUC): {best_name}")

# ---------------------------------------------------------------
# STEP 5: HYPERPARAMETER TUNING (of the best model)
# ---------------------------------------------------------------
# Hyperparameter = settings we give the model BEFORE it learns
# (like the depth of a tree). Parameter = what the model learns from data itself (like coefficients).
print("\n" + "=" * 60)
print(f"STEP 5: Hyperparameter tuning -> {best_name}")
print("=" * 60)

param_grids = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},   # smaller C = more regularization
    "Decision Tree": {"model__max_depth": [3, 4, 5, 6, 8],
                      "model__min_samples_leaf": [5, 10, 20, 50]},
    "Random Forest": {"model__n_estimators": [200, 300, 500],
                      "model__max_depth": [5, 8, 12, None],
                      "model__min_samples_leaf": [1, 3, 5, 10]},
    "Gradient Boosting": {"model__n_estimators": [100, 200, 300],
                          "model__learning_rate": [0.03, 0.05, 0.1],
                          "model__max_depth": [2, 3, 4],
                          "model__subsample": [0.7, 0.85, 1.0]},
}

base_pipe = Pipeline([("prep", preprocessor), ("model", models[best_name])])
search = RandomizedSearchCV(
    base_pipe,
    param_distributions=param_grids[best_name],
    n_iter=10,                  # try 10 random combinations (saves time)
    scoring="roc_auc",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
search.fit(X_train, y_train)    # tuning ONLY on train data
print("Best params :", search.best_params_)
print("Best CV AUC :", round(search.best_score_, 4))

final_pipe = search.best_estimator_     # tuned model (already refit on the full train data)

# ---------------------------------------------------------------
# STEP 6: CHOOSING THE THRESHOLD
# ---------------------------------------------------------------
# The model gives a "probability" (like 0.43). Default rule: >= 0.5 -> churn.
# But in business, missing a churner (False Negative) is expensive, so we can lower the
# threshold to catch more churners. We choose the threshold by maximizing F1,
# but on the TRAIN out-of-fold predictions (we haven't touched the test set -> no leakage).
print("\n" + "=" * 60)
print("STEP 6: Threshold selection (on train out-of-fold predictions)")
print("=" * 60)

oof_proba = cross_val_predict(final_pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof_proba)
f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)   # F1 = 2PR/(P+R)
best_threshold = float(thr[np.argmax(f1_scores)])
print(f"Chosen threshold: {best_threshold:.3f}  (instead of the default 0.5)")

# ---------------------------------------------------------------
# STEP 7: FINAL EVALUATION (TEST SET - only once!)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: Final evaluation on TEST set")
print("=" * 60)

test_proba = final_pipe.predict_proba(X_test)[:, 1]          # probability of churn
test_pred = (test_proba >= best_threshold).astype(int)       # apply threshold for a 0/1 decision

metrics = {
    "accuracy": accuracy_score(y_test, test_pred),
    "precision": precision_score(y_test, test_pred),
    "recall": recall_score(y_test, test_pred),
    "f1": f1_score(y_test, test_pred),
    "roc_auc": roc_auc_score(y_test, test_proba),
}
for k, v in metrics.items():
    print(f"{k:10s}: {v:.4f}")

print("\nClassification report:")
print(classification_report(y_test, test_pred, target_names=["No churn", "Churn"]))

# Confusion matrix:
#   [[TN FP]    TN = correctly said "won't leave"     FP = false alarm (wouldn't have left, but we said churn)
#    [FN TP]]   FN = MISS (left, but we didn't catch it)  TP = correctly caught churner
print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))

# ---------------------------------------------------------------
# SAVE GRAPHS
# ---------------------------------------------------------------
ConfusionMatrixDisplay.from_predictions(y_test, test_pred,
                                        display_labels=["No churn", "Churn"], cmap="Blues")
plt.title("Confusion Matrix (Test)")
plt.savefig("models/confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

RocCurveDisplay.from_predictions(y_test, test_proba)
plt.title("ROC Curve (Test)")
plt.savefig("models/roc_curve.png", dpi=120, bbox_inches="tight")
plt.close()

# Feature importance: which features influenced the prediction the most
prep_fitted = final_pipe.named_steps["prep"]
feature_names = prep_fitted.get_feature_names_out()
mdl = final_pipe.named_steps["model"]
if hasattr(mdl, "feature_importances_"):            # Tree-based models
    importances = mdl.feature_importances_
elif hasattr(mdl, "coef_"):                         # Logistic Regression: absolute value of coefficients
    importances = np.abs(mdl.coef_[0])
else:
    importances = None

if importances is not None:
    imp = (pd.Series(importances, index=feature_names).sort_values(ascending=False).head(12)[::-1])
    imp.index = [n.split("__", 1)[1] for n in imp.index]   # "num__tenure" -> "tenure"
    imp.plot(kind="barh", figsize=(7, 5), color="steelblue")
    plt.title("Top 12 Feature Importances")
    plt.savefig("models/feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------
# STEP 8: SAVE THE MODEL
# ---------------------------------------------------------------
# Save the whole pipeline (preprocessing + model) in one file. This way the exact same
# preprocessing is applied automatically to new data in the app. No need to save the scaler/encoder separately.
joblib.dump(final_pipe, "models/churn_pipeline.joblib")

# Options for the app's dropdowns and metadata, saved as JSON
meta = {
    "best_model": best_name,
    "best_params": {k: (v if not isinstance(v, np.generic) else v.item())
                    for k, v in search.best_params_.items()},
    "threshold": best_threshold,
    "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
    "categorical_options": {c: sorted(X_train[c].dropna().unique().tolist()) for c in categorical_cols},
    "numeric_ranges": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in numeric_cols},
    "feature_columns": X_train.columns.tolist(),
}
with open("models/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved: models/churn_pipeline.joblib, models/meta.json, graphs, model_comparison.csv")
print("Now run the app:  streamlit run app.py")
