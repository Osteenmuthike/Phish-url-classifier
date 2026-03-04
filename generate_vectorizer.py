import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Try to find your dataset
dataset_path = None
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith(".csv"):
            dataset_path = os.path.join(root, f)
            break

if not dataset_path:
    raise FileNotFoundError("No dataset CSV found in /data folder. Please add your URL dataset first.")

print(f"[INFO] Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

# Ensure it has a 'url' column (adjust if necessary)
if 'url' not in df.columns:
    # Try common alternatives
    for alt in ['URL', 'Url', 'links', 'domain']:
        if alt in df.columns:
            df.rename(columns={alt: 'url'}, inplace=True)
            break

if 'url' not in df.columns:
    raise KeyError("Dataset must contain a column named 'url'.")

# Create and fit a TF-IDF Vectorizer
print("[INFO] Fitting TF-IDF vectorizer on URLs...")
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 5),
    lowercase=True,
    max_features=5000
)

vectorizer.fit(df['url'].astype(str))

# Save the vectorizer
vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
with open(vec_path, "wb") as f:
    pickle.dump(vectorizer, f)

print(f"[SUCCESS] Vectorizer saved at: {vec_path}")
