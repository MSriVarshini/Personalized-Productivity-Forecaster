"""
Trains both models for the Personalized Productivity Forecaster:
  1. Regression: predict focus_hours (continuous)
  2. Classification: predict high_productivity_day (binary)

Saves trained models + feature list + metrics to models/artifacts.pkl
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("/home/claude/productivity_forecaster/data/daily_log.csv")

FEATURES = [
    "is_weekend", "is_exam_week", "sleep_hours", "social_media_hours",
    "study_app_hours", "class_hours", "energy_self_log",
    "prev_day_focus_hours", "rolling_3day_focus_avg",
]

X = df[FEATURES]
y_reg = df["focus_hours"]
y_clf = df["high_productivity_day"]


split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
y_clf_train, y_clf_test = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {"regression": {}, "classification": {}}


reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42),
}

best_reg_name, best_reg_model, best_reg_r2 = None, None, -np.inf
for name, model in reg_models.items():
    model.fit(X_train_scaled, y_reg_train)
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_reg_test, preds)
    r2 = r2_score(y_reg_test, preds)
    results["regression"][name] = {"MAE": round(mae, 3), "R2": round(r2, 3)}
    if r2 > best_reg_r2:
        best_reg_name, best_reg_model, best_reg_r2 = name, model, r2


clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
}

best_clf_name, best_clf_model, best_clf_f1 = None, None, -np.inf
for name, model in clf_models.items():
    model.fit(X_train_scaled, y_clf_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_clf_test, preds)
    prec = precision_score(y_clf_test, preds, zero_division=0)
    rec = recall_score(y_clf_test, preds, zero_division=0)
    f1 = f1_score(y_clf_test, preds, zero_division=0)
    results["classification"][name] = {
        "Accuracy": round(acc, 3), "Precision": round(prec, 3),
        "Recall": round(rec, 3), "F1": round(f1, 3),
    }
    if f1 > best_clf_f1:
        best_clf_name, best_clf_model, best_clf_f1 = name, model, f1

feature_importance = None
if hasattr(best_reg_model, "feature_importances_"):
    feature_importance = dict(zip(FEATURES, best_reg_model.feature_importances_.round(4)))

artifacts = {
    "features": FEATURES,
    "scaler": scaler,
    "best_regressor": best_reg_model,
    "best_regressor_name": best_reg_name,
    "best_classifier": best_clf_model,
    "best_classifier_name": best_clf_name,
    "results": results,
    "feature_importance": feature_importance,
    "focus_hours_mean": float(df["focus_hours"].mean()),
}

with open("/home/claude/productivity_forecaster/models/artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("=== Regression results (test set) ===")
for name, m in results["regression"].items():
    print(f"  {name}: MAE={m['MAE']}, R2={m['R2']}")
print(f"  Best: {best_reg_name}")

print("\n=== Classification results (test set) ===")
for name, m in results["classification"].items():
    print(f"  {name}: Acc={m['Accuracy']}, Prec={m['Precision']}, Rec={m['Recall']}, F1={m['F1']}")
print(f"  Best: {best_clf_name}")

if feature_importance:
    print("\n=== Feature importance (best regressor) ===")
    for feat, imp in sorted(feature_importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp}")
