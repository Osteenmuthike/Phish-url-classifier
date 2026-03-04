import pandas as pd
import os
from feature_extraction import extract_features

# Paths
RAW_PATH = os.path.join("data", "raw", "openphish_labelled.csv")
PROCESSED_PATH = os.path.join("data", "processed", "train.csv")

print(f"[INFO] Loading raw dataset from: {RAW_PATH}")
df = pd.read_csv(RAW_PATH)

# Ensure expected columns exist
if "url" not in df.columns or "label" not in df.columns:
    raise ValueError("❌ CSV must have 'url' and 'label' columns.")

processed = []

print("[INFO] Extracting features...")
for i, row in df.iterrows():
    url = row["url"]
    label = row["label"]
    try:
        features = extract_features(url)
        features["label"] = label
        processed.append(features)
    except Exception as e:
        print(f"[WARN] Failed to process URL {url}: {e}")

# Convert to DataFrame
df_processed = pd.DataFrame(processed)
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
df_processed.to_csv(PROCESSED_PATH, index=False)

print(f"[SUCCESS] Saved processed dataset to: {PROCESSED_PATH}")
print(f"✅ Total samples: {len(df_processed)}")
print(f"✅ Feature columns: {list(df_processed.columns)}")
