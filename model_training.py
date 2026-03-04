import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Path to cleaned dataset
DATA_PATH = "data\\phishing_legit_dataset_clean.csv"  # relative to project root

# === Fix path to make 'src' importable ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_extraction import extract_features

# === Load dataset ===
data_path = "data/phishing_legit_dataset.csv"
df = pd.read_csv(data_path)

# === Extract features ===
print("Extracting features...")

feature_rows = []
labels = []

for _, row in df.iterrows():
    try:
        url = row["url"]
        label = int(row["label"])
        features = extract_features(url)
        feature_rows.append([
            features["url_length"],
            features["hostname_length"],
            features["path_length"],
            features["num_dots"],
            features["num_hyphens"],
            features["num_digits"],
            features["has_ip"],
            features["https_present"],
            features["num_subdomains"]
        ])
        labels.append(label)
    except Exception as e:
        print(f"Skipping {url}: {e}")

X = pd.DataFrame(feature_rows, columns=[
    "url_length", "hostname_length", "path_length", "num_dots",
    "num_hyphens", "num_digits", "has_ip", "https_present", "num_subdomains"
])
y = labels

# === Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
print("Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\nModel evaluation:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# === Save model ===
joblib.dump(model, "src/model.pkl")
print("\n✅ Model retrained and saved successfully as src/model.pkl")
