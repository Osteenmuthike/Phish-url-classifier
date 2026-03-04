import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/samples/sample_urls.csv")
if df['label'].dtype == object:
    df['label'] = df['label'].str.lower().map({
        'phishing': 1, 'phish': 1, '1': 1,
        'legit': 0, 'benign': 0, '0': 0
    })

train, test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
print("✅ train/test files created.")
