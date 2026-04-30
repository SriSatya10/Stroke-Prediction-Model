import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# Load data exactly as in notebook
df = pd.read_csv(r"D:\Stroke-Prediction-Model\healthcare-dataset-stroke-data.csv")
df = df.drop(columns=["id"], errors="ignore")
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["stroke"])
y = df["stroke"]

# SMOTE on full data (leaked, as per notebook)
sm = SMOTE(random_state=42)
X_leaked, y_leaked = sm.fit_resample(X, y)

# Split
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_leaked, y_leaked, test_size=0.20, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_l = scaler.fit_transform(X_train_l)
X_test_l  = scaler.transform(X_test_l)

# Best RF Model params as outputted in the notebook
best_rf_l = RandomForestClassifier(
    criterion='entropy',
    max_depth=20,
    max_features='sqrt',
    min_samples_leaf=1,
    n_estimators=500,
    random_state=42,
    class_weight='balanced'
)

best_rf_l.fit(X_train_l, y_train_l)

# Get predicted probabilities
y_prob_l = best_rf_l.predict_proba(X_test_l)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_l, y_prob_l)
roc_auc = auc(fpr, tpr)

# Plot nicely colored ROC curve
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Random Forest (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='#d62728', lw=2, linestyle='--', label='Random Classifier')

# Fill the area under the curve
plt.fill_between(fpr, tpr, alpha=0.1, color='#1f77b4')

plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18, pad=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)

# Save plot to file
plt.tight_layout()
plt.savefig(r"D:\Stroke-Prediction-Model\notebooks\rf_leaked_roc_curve.png", dpi=300)
print("Saved ROC Curve Image successfully")
