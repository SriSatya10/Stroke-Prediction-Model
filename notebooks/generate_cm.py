import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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
y_pred_l = best_rf_l.predict(X_test_l)

# Create Confusion Matrix
cm = confusion_matrix(y_test_l, y_pred_l)

# Plot nicely colored confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Stroke', 'Stroke'],
            yticklabels=['No Stroke', 'Stroke'],
            annot_kws={"size": 16})
plt.title('Random Forest Confusion Matrix', fontsize=18, pad=15)
plt.ylabel('Actual', fontsize=14)
plt.xlabel('Predicted', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save plot to file
plt.tight_layout()
plt.savefig(r"D:\Stroke-Prediction-Model\notebooks\rf_leaked_confusion_matrix.png", dpi=300)
print("Saved Confusion Matrix Image successfully")
