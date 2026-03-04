import pandas as pd, joblib
from sklearn.metrics import classification_report, confusion_matrix
from src.feature_extraction import extract_features

model = joblib.load("models/phishing_model.pkl")
test = pd.read_csv("data/processed/test.csv")
X_test = extract_features(test)
y_test = test["label"]

pred = model.predict(X_test)
print(classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
import joblib, pandas as pd
from src.feature_extraction import extract_features

model = joblib.load("models/phishing_model.pkl")

while True:
    url = input("Enter a URL (or 'exit'): ").strip()
    if url.lower() == "exit":
        break
    X = extract_features(pd.DataFrame([[url]], columns=["url"]))
    pred = model.predict(X)[0]
    print("🚨 Phishing" if pred else "✅ Legitimate")
