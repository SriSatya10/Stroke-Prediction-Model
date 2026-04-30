import json
import urllib.request

payload = {
  "age": 80,
  "hypertension": 1,
  "heart_disease": 1,
  "avg_glucose_level": 200,
  "bmi": 40,
  "gender_Male": 1,
  "gender_Other": 0,
  "ever_married_Yes": 1,
  "work_type_Never_worked": 0,
  "work_type_Private": 1,
  "work_type_Self-employed": 0,
  "work_type_children": 0,
  "Residence_type_Urban": 1,
  "smoking_status_formerly smoked": 0,
  "smoking_status_never smoked": 0,
  "smoking_status_smokes": 1
}

req = urllib.request.Request('http://127.0.0.1:5000/predict')
req.add_header('Content-Type', 'application/json')
response = urllib.request.urlopen(req, json.dumps(payload).encode('utf-8'))
data = json.loads(response.read())

print("Probability:", data["probability"])
for f in data["contributing_factors"]:
    print(f"{f['feature']}: {f['shap_value']}")
