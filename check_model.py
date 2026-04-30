import joblib
model = joblib.load('stroke_model.pkl')
print(type(model))
