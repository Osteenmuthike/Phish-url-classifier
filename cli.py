# src/cli.py
import sys
import joblib
import pandas as pd
from src.feature_extraction import extract_features

MODEL_PATH = "models/phishing_model.pkl"

def predict_url(url):
    model = joblib.load(MODEL_PATH)
    feats = extract_features(url)
    X = pd.DataFrame([feats])
    pred = model.predict(X)[0]
    return "Phishing" if int(pred) == 1 else "Legitimate"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter URL to check: ").strip()
    print(predict_url(url))
