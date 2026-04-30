import joblib
import numpy as np
import json
import shap
import warnings
warnings.filterwarnings('ignore')

model = joblib.load('stroke_model.pkl')
scaler = joblib.load('stroke_scaler.pkl')
with open('feature_names.json') as f:
    feature_names = json.load(f)

explainer = shap.TreeExplainer(model)
data = {
  "age": 72,
  "hypertension": 1,
  "heart_disease": 0,
  "avg_glucose_level": 227,
  "bmi": 57.6,
  "gender_Male": 0,
  "gender_Other": 0,
  "ever_married_Yes": 1,
  "work_type_Never_worked": 0,
  "work_type_Private": 1,
  "work_type_Self-employed": 0,
  "work_type_children": 0,
  "Residence_type_Urban": 1,
  "smoking_status_formerly smoked": 0,
  "smoking_status_never smoked": 1,
  "smoking_status_smokes": 0
}

features = np.array([data[f] for f in feature_names]).reshape(1, -1)
features_scaled = scaler.transform(features)
shap_vals = explainer.shap_values(features_scaled)

if isinstance(shap_vals, list):
    shap_class1 = shap_vals[1][0]
else:
    shap_class1 = shap_vals[0, :, 1] if len(shap_vals.shape) == 3 else shap_vals[0]

for name, val in zip(feature_names, shap_class1):
    print(f"{name}: {round(float(val), 4)}")
