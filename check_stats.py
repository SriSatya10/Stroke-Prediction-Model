import pandas as pd

df = pd.read_csv("d:/Stroke-Prediction-Model/healthcare-dataset-stroke-data.csv")
print("Stroke rate by smoking status:")
print(df.groupby('smoking_status')['stroke'].mean())
print("\nAverage Age by smoking status:")
print(df.groupby('smoking_status')['age'].mean())

print("\nStroke rate by Hypertension:")
print(df.groupby('hypertension')['stroke'].mean())
print("\nAverage Age by Hypertension:")
print(df.groupby('hypertension')['age'].mean())
