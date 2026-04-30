# Data Handling
import pandas as pd
import numpy as np

# Visualization (even if EDA done, keep ready)
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Handling Imbalance
from imblearn.over_sampling import SMOTE

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

print("Hello")

df = pd.read_csv("D:\Stroke-Prediction-Model\healthcare-dataset-stroke-data.csv")

df.head()

print("Shape of dataset:", df.shape)

df.columns

df.info()

df.isnull().sum()

df["stroke"].value_counts()

df.drop("id", axis=1, inplace=True)

df.columns

df.isnull().sum()

df["bmi"].fillna(df["bmi"].median(), inplace=True)

df.isnull().sum()

df.isnull().sum()

df["bmi"] = df["bmi"].fillna(df["bmi"].median())

df.isnull().sum()

X = df.drop("stroke", axis=1)
y = df["stroke"]

X.shape
y.shape

print("X shape:", X.shape)
print("y shape:", y.shape)

X.select_dtypes(include="object").columns

X = pd.get_dummies(X, drop_first=True)

print("New shape of X:", X.shape)

X.select_dtypes(include="object").columns

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

import pandas as pd
pd.Series(y_train).value_counts()

pd.Series(y_test).value_counts()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print("Logistic Regression Results")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest Results")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))



rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

# Train improved RF
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("=== Improved Random Forest Results ===\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob_rf))

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("=== XGBoost Results ===\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob_xgb))

import numpy as np

neg, pos = np.bincount(y_train)
print("Negative class:", neg)
print("Positive class:", pos)

scale_weight = neg / pos
print("scale_pos_weight:", scale_weight)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    class_weight='balanced',
    random_state=42
)

svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

print("=== SVM Results ===\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

print("ROC-AUC:", roc_auc_score(y_test, y_prob_svm))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

knn_model = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance'   # helps with imbalance
)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
y_prob_knn = knn_model.predict_proba(X_test)[:, 1]

print("=== KNN Results ===\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

print("ROC-AUC:", roc_auc_score(y_test, y_prob_knn))

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

lgb_model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

print("=== LightGBM Results ===\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lgb))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lgb))

print("ROC-AUC:", roc_auc_score(y_test, y_prob_lgb))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    'C': [0.01, 0.1, 1, 5, 10, 50],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']   # supports both l1 and l2
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    LogisticRegression(max_iter=2000),
    param_grid,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)

best_log = grid.best_estimator_

y_pred_best = best_log.predict(X_test)
y_prob_best = best_log.predict_proba(X_test)[:, 1]

print("=== Optimized Logistic Regression ===")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob_best))

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_prob))

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

model = GaussianNB()

# Accuracy
accuracy = cross_val_score(model, X, y, cv=10, scoring='accuracy')
print("Average Accuracy:", accuracy.mean())

# Precision
precision = cross_val_score(model, X, y, cv=10, scoring='precision')
print("Average Precision:", precision.mean())

# Recall
recall = cross_val_score(model, X, y, cv=10, scoring='recall')
print("Average Recall:", recall.mean())

# F1
f1 = cross_val_score(model, X, y, cv=10, scoring='f1')
print("Average F1:", f1.mean())

auc = cross_val_score(model, X, y, cv=10, scoring='roc_auc')
print("Average AUC:", auc.mean())

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(class_weight='balanced', max_iter=1000)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs']
}

grid_lr = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=1000),
    param_grid,
    cv=10,
    scoring='roc_auc'
)

grid_lr.fit(X, y)

print("Best Parameters:", grid_lr.best_params_)
print("Best AUC:", grid_lr.best_score_)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced', random_state=42)

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid_rf,
    cv=10,
    scoring='roc_auc',
    n_jobs=-1
)

grid_rf.fit(X, y)

print("Best Parameters:", grid_rf.best_params_)
print("Best AUC:", grid_rf.best_score_)

from sklearn.model_selection import cross_val_score

best_rf = grid_rf.best_estimator_
best_lr = grid_lr.best_estimator_

print("RF Accuracy:", cross_val_score(best_rf, X, y, cv=10, scoring='accuracy').mean())
print("RF Recall:", cross_val_score(best_rf, X, y, cv=10, scoring='recall').mean())
print("RF F1:", cross_val_score(best_rf, X, y, cv=10, scoring='f1').mean())
print("RF AUC:", cross_val_score(best_rf, X, y, cv=10, scoring='roc_auc').mean())

import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

best_results = []

for t in np.arange(0.3, 0.8, 0.01):
    y_pred_custom = (y_prob_best > t).astype(int)
    
    acc = accuracy_score(y_test, y_pred_custom)
    prec = precision_score(y_test, y_pred_custom)
    rec = recall_score(y_test, y_pred_custom)
    f1 = f1_score(y_test, y_pred_custom)
    
    best_results.append((t, acc, prec, rec, f1))

# sort by F1 (balanced metric)
best_results = sorted(best_results, key=lambda x: x[4], reverse=True)

best_results[:10]

print("Length y_test:", len(y_test))
print("Length y_prob_best:", len(y_prob_best))

Length y_test: 1533
Length y_prob_best: 1022

print("Length y_test:", len(y_test))
print("Length y_prob_best:", len(y_prob_best))

print("Length y_test:", len(y_test))
print("Length y_prob_best:", len(y_prob_best))

y_prob_best = best_log.predict_proba(X_test)[:, 1]

print(len(y_test))
print(len(y_prob_best))

import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

for t in np.arange(0.3, 0.8, 0.05):
    y_pred_custom = (y_prob_best > t).astype(int)
    
    acc = accuracy_score(y_test, y_pred_custom)
    prec = precision_score(y_test, y_pred_custom)
    rec = recall_score(y_test, y_pred_custom)
    f1 = f1_score(y_test, y_pred_custom)
    
    print(f"Threshold: {t:.2f}")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    print("-----")

print("Min prob:", y_prob_best.min())
print("Max prob:", y_prob_best.max())
print("Mean prob:", y_prob_best.mean())

import numpy as np
print(np.unique(y_pred_custom))

# Show what parameters your grid search found
print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)

from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Get probabilities from your already-trained best_log model
y_prob_best = best_log.predict_proba(X_test)[:, 1]

# Find optimal threshold by maximizing F1
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_best)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
best_thresh = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold: {best_thresh:.3f}")

# Try a range of thresholds so you can SEE the tradeoff
print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Recall':>8} {'Precision':>10} {'F1':>6}")
print("-" * 50)
for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    preds = (y_prob_best >= t).astype(int)
    rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, preds)
    rec = rep['1']['recall']
    pre = rep['1']['precision']
    f1  = rep['1']['f1-score']
    print(f"{t:>10.2f} {acc:>10.3f} {rec:>8.3f} {pre:>10.3f} {f1:>6.3f}")

# Apply the best threshold
y_pred_tuned = (y_prob_best >= best_thresh).astype(int)
print(f"\n=== LR with Optimal Threshold ({best_thresh:.3f}) ===")
print(classification_report(y_test, y_pred_tuned, zero_division=0))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_best))
print("Accuracy:", accuracy_score(y_test, y_pred_tuned))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                             accuracy_score, precision_recall_curve)
from imblearn.over_sampling import SMOTE

# ── 1. Load & encode ──────────────────────────────────────────────
df = pd.read_csv("D:\Stroke-Prediction-Model\healthcare-dataset-stroke-data.csv")
df = df.drop(columns=["id"], errors="ignore")
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["stroke"])
y = df["stroke"]

# ── 2. Split FIRST — before anything else ────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # keeps 5% stroke ratio in both splits
)

print("Train class dist:", y_train.value_counts().to_dict())
print("Test  class dist:", y_test.value_counts().to_dict())

# ── 3. Scale ONLY on training data ───────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)   # fit+transform train
X_test_scaled  = scaler.transform(X_test_raw)        # transform only test

# ── 4. Apply SMOTE ONLY on scaled training data ──────────────────
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)

print("After SMOTE:", pd.Series(y_train_bal).value_counts().to_dict())

# ── 5. Train Logistic Regression with Grid Search ─────────────────
lr = LogisticRegression(max_iter=1000, solver='liblinear')

param_grid = {
    'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    'penalty': ['l1', 'l2'],
    'class_weight': [None, 'balanced']
}

grid = GridSearchCV(
    lr, param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train_bal, y_train_bal)

best_lr = grid.best_estimator_
print("\nBest params:", grid.best_params_)
print("Best CV AUC:", grid.best_score_)

# ── 6. Evaluate on untouched test set ────────────────────────────
y_prob = best_lr.predict_proba(X_test_scaled)[:, 1]
y_pred = best_lr.predict(X_test_scaled)

print("\n=== Default threshold (0.5) ===")
print(classification_report(y_test, y_pred, zero_division=0))
print("ROC-AUC :", roc_auc_score(y_test, y_prob))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ── 7. Threshold scan ─────────────────────────────────────────────
print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Recall':>8} {'Precision':>10} {'F1':>6}")
print("-" * 50)
for t in np.arange(0.10, 0.65, 0.05):
    p = (y_prob >= t).astype(int)
    rep = classification_report(y_test, p, output_dict=True, zero_division=0)
    print(f"{t:>10.2f} "
          f"{accuracy_score(y_test,p):>10.3f} "
          f"{rep['1']['recall']:>8.3f} "
          f"{rep['1']['precision']:>10.3f} "
          f"{rep['1']['f1-score']:>6.3f}")

# Extend threshold search beyond 0.60 to find where recall starts dropping
print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Recall':>8} {'Precision':>10} {'F1':>6}")
print("-" * 50)
for t in [0.55, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.75, 0.80]:
    preds = (y_prob_best >= t).astype(int)
    rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, preds)
    rec = rep['1']['recall']
    pre = rep['1']['precision']
    f1  = rep['1']['f1-score']
    print(f"{t:>10.2f} {acc:>10.3f} {rec:>8.3f} {pre:>10.3f} {f1:>6.3f}")

print(len(y_test), len(y_prob_best))

y_prob_best = best_model.predict_proba(X_test)[:, 1]

# Verify sizes match now
print(len(y_test), len(y_prob_best))  # should both show 1022

# ── 8. Extended threshold search ─────────────────────────────────
y_prob_best = best_lr.predict_proba(X_test_scaled)[:, 1]  # ← this fixes the mismatch

print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Recall':>8} {'Precision':>10} {'F1':>6}")
print("-" * 50)
for t in [0.55, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.75, 0.80]:
    preds = (y_prob_best >= t).astype(int)
    rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
    print(f"{t:>10.2f} "
          f"{accuracy_score(y_test, preds):>10.3f} "
          f"{rep['1']['recall']:>8.3f} "
          f"{rep['1']['precision']:>10.3f} "
          f"{rep['1']['f1-score']:>6.3f}")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# XGBoost handles imbalance better
xgb = XGBClassifier(
    scale_pos_weight=19,   # ratio of negative/positive (95/5 ≈ 19)
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric='auc'
)
xgb.fit(X_train_scaled, y_train)   # no SMOTE needed, use original train

y_prob_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
print("XGB ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

xgb = XGBClassifier(
    random_state=42,
    eval_metric='auc',
    use_label_encoder=False
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [10, 15, 19],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_xgb = GridSearchCV(
    xgb, param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Use original (non-SMOTE) training data — XGB handles imbalance via scale_pos_weight
grid_xgb.fit(X_train_scaled, y_train)

print("Best params:", grid_xgb.best_params_)
print("Best CV AUC:", grid_xgb.best_score_)

y_prob_xgb = grid_xgb.best_estimator_.predict_proba(X_test_scaled)[:, 1]
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

# Full threshold scan for tuned XGBoost
best_xgb = grid_xgb.best_estimator_
y_prob_xgb = best_xgb.predict_proba(X_test_scaled)[:, 1]

print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Recall':>8} {'Precision':>10} {'F1':>6}")
print("-" * 50)
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    preds = (y_prob_xgb >= t).astype(int)
    rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
    print(f"{t:>10.2f} "
          f"{accuracy_score(y_test, preds):>10.3f} "
          f"{rep['1']['recall']:>8.3f} "
          f"{rep['1']['precision']:>10.3f} "
          f"{rep['1']['f1-score']:>6.3f}")

print("\nROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    class_weight='balanced',
    random_state=42
)

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2']
}

grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_rf.fit(X_train_scaled, y_train)  # no SMOTE, using class_weight='balanced' instead

y_prob_rf = grid_rf.best_estimator_.predict_proba(X_test_scaled)[:, 1]
print("Best params:", grid_rf.best_params_)
print("RF ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

# Average LR + XGBoost probabilities
y_prob_ensemble = (y_prob * 0.5) + (y_prob_xgb * 0.5)  # y_prob = your LR probs

print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Recall':>8} {'Precision':>10} {'F1':>6}")
print("-" * 50)
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    preds = (y_prob_ensemble >= t).astype(int)
    rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
    print(f"{t:>10.2f} "
          f"{accuracy_score(y_test, preds):>10.3f} "
          f"{rep['1']['recall']:>8.3f} "
          f"{rep['1']['precision']:>10.3f} "
          f"{rep['1']['f1-score']:>6.3f}")

print("\nEnsemble ROC-AUC:", roc_auc_score(y_test, y_prob_ensemble))

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

grid_xgb2 = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='auc'),
    {
        'n_estimators': [100, 200, 300],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'scale_pos_weight': [1],  # 1 because SMOTE already balanced it
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
)

grid_xgb2.fit(X_train_bal, y_train_bal)  # SMOTE balanced data

y_prob_xgb2 = grid_xgb2.best_estimator_.predict_proba(X_test_scaled)[:, 1]
print("Best params:", grid_xgb2.best_params_)
print("SMOTE XGB ROC-AUC:", roc_auc_score(y_test, y_prob_xgb2))

print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Recall':>8} {'Precision':>10} {'F1':>6}")
print("-" * 50)
for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    preds = (y_prob_xgb2 >= t).astype(int)
    rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
    print(f"{t:>10.2f} "
          f"{accuracy_score(y_test, preds):>10.3f} "
          f"{rep['1']['recall']:>8.3f} "
          f"{rep['1']['precision']:>10.3f} "
          f"{rep['1']['f1-score']:>6.3f}")

# LR best was at threshold 0.60
y_final_lr = (y_prob >= 0.60).astype(int)
rep_lr = classification_report(y_test, y_final_lr, output_dict=True, zero_division=0)

print("=== FINAL LR SUMMARY ===")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")
print(f"Recall    : {rep_lr['1']['recall']:.3f}")
print(f"Precision : {rep_lr['1']['precision']:.3f}")
print(f"F1        : {rep_lr['1']['f1-score']:.3f}")
print(f"Accuracy  : {accuracy_score(y_test, y_final_lr):.3f}")

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# ── WRONG WAY: SMOTE before split ────────────────────────────────
df = pd.read_csv("D:\Stroke-Prediction-Model\healthcare-dataset-stroke-data.csv")
df = df.drop(columns=["id"], errors="ignore")
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["stroke"])
y = df["stroke"]

# SMOTE on full data FIRST (wrong)
sm = SMOTE(random_state=42)
X_leaked, y_leaked = sm.fit_resample(X, y)

# Then split
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_leaked, y_leaked, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_l = scaler.fit_transform(X_train_l)
X_test_l  = scaler.transform(X_test_l)

print("Test set stroke rate (WRONG way):", y_test_l.mean().round(3))  # will show ~0.50
print("Test set stroke rate (YOUR way) : ~0.05 (real world)\n")

# Test on multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    "XGBoost"             : XGBClassifier(n_estimators=100, max_depth=2, random_state=42, eval_metric='auc')
}

print(f"{'Model':<22} {'Accuracy':>10} {'Recall':>8} {'Precision':>10} {'F1':>8} {'AUC':>8}")
print("-" * 70)
for name, model in models.items():
    model.fit(X_train_l, y_train_l)
    y_pred = model.predict(X_test_l)
    y_prob = model.predict_proba(X_test_l)[:, 1]
    rep = classification_report(y_test_l, y_pred, output_dict=True, zero_division=0)
    print(f"{name:<22} "
          f"{accuracy_score(y_test_l, y_pred):>10.3f} "
          f"{rep['1']['recall']:>8.3f} "
          f"{rep['1']['precision']:>10.3f} "
          f"{rep['1']['f1-score']:>8.3f} "
          f"{roc_auc_score(y_test_l, y_prob):>8.3f}")

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}

grid_rf_leaked = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid,
    cv=5,
    scoring='accuracy',  # optimizing for accuracy this time
    n_jobs=-1,
    verbose=1
)

grid_rf_leaked.fit(X_train_l, y_train_l)

best_rf_l = grid_rf_leaked.best_estimator_
y_pred_l  = best_rf_l.predict(X_test_l)
y_prob_l  = best_rf_l.predict_proba(X_test_l)[:, 1]

rep = classification_report(y_test_l, y_pred_l, output_dict=True, zero_division=0)

print("Best params:", grid_rf_leaked.best_params_)
print(f"\nAccuracy  : {accuracy_score(y_test_l, y_pred_l):.4f}")
print(f"Recall    : {rep['1']['recall']:.4f}")
print(f"Precision : {rep['1']['precision']:.4f}")
print(f"F1        : {rep['1']['f1-score']:.4f}")
print(f"AUC       : {roc_auc_score(y_test_l, y_prob_l):.4f}")

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report)

# ── 1. Save model & scaler ────────────────────────────────────────
joblib.dump(grid_xgb.best_estimator_, "stroke_model.pkl")
joblib.dump(scaler, "stroke_scaler.pkl")
print("✅ Model and scaler saved")

# ── 2. Save feature names ─────────────────────────────────────────
feature_names = list(X.columns)
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f)
print("✅ Feature names saved")

# ── 3. ROC curve data ─────────────────────────────────────────────
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob_xgb)
roc_data = {
    "fpr": fpr.tolist(),
    "tpr": tpr.tolist(),
    "thresholds": roc_thresholds.tolist(),
    "auc": roc_auc_score(y_test, y_prob_xgb)
}
with open("roc_data.json", "w") as f:
    json.dump(roc_data, f)
print("✅ ROC data saved")

# ── 4. Precision-Recall curve data ───────────────────────────────
prec, rec, pr_thresholds = precision_recall_curve(y_test, y_prob_xgb)
pr_data = {
    "precision": prec.tolist(),
    "recall": rec.tolist(),
    "thresholds": pr_thresholds.tolist()
}
with open("pr_data.json", "w") as f:
    json.dump(pr_data, f)
print("✅ PR curve data saved")

# ── 5. Confusion matrix ───────────────────────────────────────────
y_final = (y_prob_xgb >= 0.40).astype(int)
cm = confusion_matrix(y_test, y_final)
cm_data = {
    "tn": int(cm[0,0]), "fp": int(cm[0,1]),
    "fn": int(cm[1,0]), "tp": int(cm[1,1])
}
with open("confusion_matrix.json", "w") as f:
    json.dump(cm_data, f)
print("✅ Confusion matrix saved")

# ── 6. Threshold scan data ────────────────────────────────────────
threshold_data = []
for t in np.arange(0.10, 0.95, 0.01):
    preds = (y_prob_xgb >= t).astype(int)
    rep = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm_t = confusion_matrix(y_test, preds)
    threshold_data.append({
        "threshold": round(float(t), 2),
        "accuracy": round(float((preds == y_test).mean()), 4),
        "recall": round(float(rep['1']['recall']), 4),
        "precision": round(float(rep['1']['precision']), 4),
        "f1": round(float(rep['1']['f1-score']), 4),
        "tn": int(cm_t[0,0]), "fp": int(cm_t[0,1]),
        "fn": int(cm_t[1,0]), "tp": int(cm_t[1,1])
    })
with open("threshold_data.json", "w") as f:
    json.dump(threshold_data, f)
print("✅ Threshold data saved")

# ── 7. Feature importance ─────────────────────────────────────────
importance_data = [
    {"feature": name, "importance": round(float(imp), 4)}
    for name, imp in zip(feature_names, grid_xgb.best_estimator_.feature_importances_)
]
importance_data.sort(key=lambda x: x["importance"], reverse=True)
with open("feature_importance.json", "w") as f:
    json.dump(importance_data, f)
print("✅ Feature importance saved")

# ── 8. Model comparison data ──────────────────────────────────────
model_comparison = [
    {"model": "XGBoost (tuned)",       "accuracy": 0.840, "recall": 0.720, "precision": 0.194, "f1": 0.305, "auc": 0.851},
    {"model": "Logistic Regression",   "accuracy": 0.804, "recall": 0.800, "precision": 0.174, "f1": 0.286, "auc": 0.850},
    {"model": "XGBoost (untuned)",     "accuracy": 0.818, "recall": 0.700, "precision": 0.180, "f1": 0.289, "auc": 0.818},
    {"model": "Random Forest",         "accuracy": 0.825, "recall": 0.600, "precision": 0.200, "f1": 0.300, "auc": 0.825},
    {"model": "Ensemble LR+XGB",       "accuracy": 0.855, "recall": 0.720, "precision": 0.212, "f1": 0.327, "auc": 0.848},
    {"model": "XGBoost+SMOTE",         "accuracy": 0.926, "recall": 0.160, "precision": 0.190, "f1": 0.174, "auc": 0.768},
]
with open("model_comparison.json", "w") as f:
    json.dump(model_comparison, f)
print("✅ Model comparison saved")

# ── 9. Raw dataset sample for demographics ────────────────────────
df_orig = pd.read_csv("D:\Stroke-Prediction-Model\healthcare-dataset-stroke-data.csv")
df_orig["bmi"] = df_orig["bmi"].fillna(df_orig["bmi"].median())
df_orig.to_csv("stroke_data_clean.csv", index=False)
print("✅ Clean dataset saved")

print("\n🎉 All files saved! Check your project folder.")

# ── FINAL MODEL VERIFICATION ──────────────────────────────────────
print("\n" + "="*60)
print("       FINAL MODEL VERIFICATION — XGBoost (tuned)")
print("="*60)

# Reload to confirm saved model works
loaded_model  = joblib.load("stroke_model.pkl")
loaded_scaler = joblib.load("stroke_scaler.pkl")

# Predict
y_prob_final = loaded_model.predict_proba(X_test_scaled)[:, 1]
y_pred_final = (y_prob_final >= 0.40).astype(int)

# Metrics
rep = classification_report(y_test, y_pred_final, output_dict=True, zero_division=0)
cm  = confusion_matrix(y_test, y_pred_final)

print(f"\n  Threshold  : 0.40")
print(f"  AUC        : {roc_auc_score(y_test, y_prob_final):.4f}")
print(f"  Accuracy   : {accuracy_score(y_test, y_pred_final):.4f}")
print(f"  Recall     : {rep['1']['recall']:.4f}")
print(f"  Precision  : {rep['1']['precision']:.4f}")
print(f"  F1         : {rep['1']['f1-score']:.4f}")

print(f"\n  Confusion Matrix:")
print(f"  ┌─────────────────────────┐")
print(f"  │ TN: {cm[0,0]:>4}  FP: {cm[0,1]:>4}  │")
print(f"  │ FN: {cm[1,0]:>4}  TP: {cm[1,1]:>4}  │")
print(f"  └─────────────────────────┘")

print(f"\n  Full Classification Report:")
print(classification_report(y_test, y_pred_final, zero_division=0))

# Verify saved files exist
import os
files = ["stroke_model.pkl", "stroke_scaler.pkl", "feature_names.json",
         "roc_data.json", "pr_data.json", "confusion_matrix.json",
         "threshold_data.json", "feature_importance.json",
         "model_comparison.json", "stroke_data_clean.csv"]

print("  Saved Files:")
all_good = True
for f in files:
    exists = os.path.exists(f)
    status = "✅" if exists else "❌"
    size   = f"{os.path.getsize(f)/1024:.1f} KB" if exists else "MISSING"
    print(f"  {status} {f:<35} {size}")
    if not exists:
        all_good = False

print("\n" + "="*60)
print("  ✅ ALL GOOD — Ready to build dashboard!" if all_good else "  ❌ Some files missing — rerun Step 1")
print("="*60)

# Run this in your notebook or manually copy
import shutil, os

output_dir = r"D:\Stroke-Prediction-Model\stroke-dashboard\public\data"
os.makedirs(output_dir, exist_ok=True)

files = ["roc_data.json", "pr_data.json", "confusion_matrix.json",
         "threshold_data.json", "feature_importance.json",
         "model_comparison.json", "stroke_data_clean.csv"]

for f in files:
    shutil.copy(f, os.path.join(output_dir, f))
    print(f"✅ Copied {f}")

