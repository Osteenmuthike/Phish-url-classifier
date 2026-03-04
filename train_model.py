# src/train_model.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join("data", "processed", "train.csv")
MODEL_PATH = os.path.join("models", "phishing_model.pkl")

print(f"[INFO] Loading dataset from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)

if "label" not in df.columns:
    raise ValueError("Dataset must include 'label' column")

# Separate features and target
X = df.drop(columns=["label"])
y = df["label"]

print(f"[INFO] Dataset shape: {X.shape}, Labels: {y.shape}")
print("[INFO] Splitting data ...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("[INFO] Training RandomForest model ...")
model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"[SUCCESS] Model saved to {MODEL_PATH}")
