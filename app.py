from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json, numpy as np
import shap

app = Flask(__name__)
CORS(app)

model  = joblib.load("stroke_model.pkl")
scaler = joblib.load("stroke_scaler.pkl")
with open("feature_names.json") as f:
    feature_names = json.load(f)

explainer = shap.TreeExplainer(model)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = np.array([data[f] for f in feature_names]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0][1]
    
    if prob < 0.3:   risk = "Low"
    elif prob < 0.5: risk = "Medium"
    elif prob < 0.7: risk = "High"
    else:            risk = "Critical"
    
    shap_vals = explainer.shap_values(features_scaled)
    if isinstance(shap_vals, list):
        shap_class1 = shap_vals[1][0]
    else:
        shap_class1 = shap_vals[0, :, 1] if len(shap_vals.shape) == 3 else shap_vals[0]
        
    contributing_factors = [
        {"feature": name, "shap_value": round(float(val), 4)} 
        for name, val in zip(feature_names, shap_class1)
    ]
    contributing_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    
    # Generate natural language explanation
    top_pos = [f for f in contributing_factors if f["shap_value"] > 0]
    top_neg = [f for f in contributing_factors if f["shap_value"] < 0]
    
    explanation = f"The model evaluated your inputs and calculated a {round(float(prob)*100, 1)}% probability of stroke risk. "
    if top_pos:
        f_name = top_pos[0]['feature'].replace('gender_', '').replace('work_type_', '').replace('smoking_status_', '').replace('_', ' ').capitalize()
        explanation += f"Your {f_name} is the most significant factor increasing this risk. "
    if top_neg:
        f_name = top_neg[0]['feature'].replace('gender_', '').replace('work_type_', '').replace('smoking_status_', '').replace('_', ' ').capitalize()
        explanation += f"Conversely, your {f_name} provides the strongest protective effect against this risk."
    
    return jsonify({
        "probability": round(float(prob), 4), 
        "risk": risk,
        "contributing_factors": contributing_factors,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)