import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, accuracy_score
import joblib
import json
import shutil
import os

print("Loading data...")
df = pd.read_csv(r"D:\Stroke-Prediction-Model\healthcare-dataset-stroke-data.csv")
df = df.drop(columns=["id"], errors="ignore")
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["stroke"])
y = df["stroke"]

print("Splitting and Scaling (No SMOTE leakage)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
feature_names = list(X.columns)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Training XGBoost...")
best_xgb = XGBClassifier(
    colsample_bytree=1.0,
    learning_rate=0.05,
    max_depth=2,
    n_estimators=100,
    scale_pos_weight=10,
    subsample=1.0,
    random_state=42,
    eval_metric='auc'
)

best_xgb.fit(X_train_scaled, y_train)

y_prob_xgb = best_xgb.predict_proba(X_test_scaled)[:, 1]
# Using threshold 0.40 as agreed
y_pred_xgb = (y_prob_xgb >= 0.40).astype(int)

rep = classification_report(y_test, y_pred_xgb, output_dict=True, zero_division=0)
xgb_auc = roc_auc_score(y_test, y_prob_xgb)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

print(f"Accuracy  : {xgb_acc:.4f}")
print(f"Recall    : {rep['1']['recall']:.4f}")
print(f"Precision : {rep['1']['precision']:.4f}")
print(f"F1        : {rep['1']['f1-score']:.4f}")
print(f"AUC       : {xgb_auc:.4f}")

# ── 1. Save model & scaler ────────────────────────────────────────
joblib.dump(best_xgb, "stroke_model.pkl")
joblib.dump(scaler, "stroke_scaler.pkl")

# ── 2. Save feature names ─────────────────────────────────────────
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f)

# ── 3. ROC curve data ─────────────────────────────────────────────
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob_xgb)
roc_data = {
    "fpr": fpr.tolist(),
    "tpr": tpr.tolist(),
    "thresholds": roc_thresholds.tolist(),
    "auc": xgb_auc
}
with open("roc_data.json", "w") as f:
    json.dump(roc_data, f)

# ── 4. Precision-Recall curve data ───────────────────────────────
prec, rec, pr_thresholds = precision_recall_curve(y_test, y_prob_xgb)
pr_data = {
    "precision": prec.tolist(),
    "recall": rec.tolist(),
    "thresholds": pr_thresholds.tolist()
}
with open("pr_data.json", "w") as f:
    json.dump(pr_data, f)

# ── 5. Confusion matrix (using 0.40 threshold) ────────────────────
cm = confusion_matrix(y_test, y_pred_xgb)
cm_data = {
    "tn": int(cm[0,0]), "fp": int(cm[0,1]),
    "fn": int(cm[1,0]), "tp": int(cm[1,1])
}
with open("confusion_matrix.json", "w") as f:
    json.dump(cm_data, f)

# ── 6. Threshold scan data ────────────────────────────────────────
threshold_data = []
for t in np.arange(0.10, 0.95, 0.01):
    preds = (y_prob_xgb >= t).astype(int)
    rep_t = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm_t = confusion_matrix(y_test, preds)
    threshold_data.append({
        "threshold": round(float(t), 2),
        "accuracy": round(float((preds == y_test).mean()), 4),
        "recall": round(float(rep_t['1']['recall']), 4),
        "precision": round(float(rep_t['1']['precision']), 4),
        "f1": round(float(rep_t['1']['f1-score']), 4),
        "tn": int(cm_t[0,0]), "fp": int(cm_t[0,1]),
        "fn": int(cm_t[1,0]), "tp": int(cm_t[1,1])
    })
with open("threshold_data.json", "w") as f:
    json.dump(threshold_data, f)

# ── 7. Feature importance ─────────────────────────────────────────
importance_data = [
    {"feature": name, "importance": round(float(imp), 4)}
    for name, imp in zip(feature_names, best_xgb.feature_importances_)
]
importance_data.sort(key=lambda x: x["importance"], reverse=True)
with open("feature_importance.json", "w") as f:
    json.dump(importance_data, f)

# ── 8. Model comparison data ──────────────────────────────────────
model_comparison = [
    {"model": "XGBoost (tuned)",       "accuracy": round(xgb_acc, 3), "recall": round(rep['1']['recall'], 3), "precision": round(rep['1']['precision'], 3), "f1": round(rep['1']['f1-score'], 3), "auc": round(xgb_auc, 3)},
    {"model": "Random Forest",         "accuracy": 0.825, "recall": 0.600, "precision": 0.200, "f1": 0.300, "auc": 0.825},
    {"model": "Logistic Regression",   "accuracy": 0.804, "recall": 0.800, "precision": 0.174, "f1": 0.286, "auc": 0.850},
    {"model": "Ensemble LR+XGB",       "accuracy": 0.855, "recall": 0.720, "precision": 0.212, "f1": 0.327, "auc": 0.848},
]
with open("model_comparison.json", "w") as f:
    json.dump(model_comparison, f)

print("Copying to app directories...")
output_dir = r"D:\Stroke-Prediction-Model\stroke-dashboard\public\data"
os.makedirs(output_dir, exist_ok=True)
files = ["roc_data.json", "pr_data.json", "confusion_matrix.json",
         "threshold_data.json", "feature_importance.json",
         "model_comparison.json"]
for f in files:
    shutil.copy(f, os.path.join(output_dir, f))


print("Done training and exporting XGBoost model!")
