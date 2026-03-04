# scripts/add_legit_urls.py
import pandas as pd
from pathlib import Path
import os

phish_path = Path("data/raw/openphish_labelled.csv")
legit_raw = Path("data/raw/top1m_legit.csv")
out = Path("data/processed/train.csv")

phish_df = pd.read_csv(phish_path)
if "url" not in phish_df.columns:
    phish_df.columns = ["url"]

phish_df["label"] = 1

legit_df = pd.read_csv(legit_raw, header=None, names=["url"], on_bad_lines="skip", engine="python")
legit_df["label"] = 0

# If legit smaller, sample with replacement
if len(legit_df) < len(phish_df):
    legit_df = legit_df.sample(n=len(phish_df), replace=True, random_state=42)
else:
    legit_df = legit_df.sample(n=len(phish_df), random_state=42)

combined = pd.concat([phish_df[["url","label"]], legit_df[["url","label"]]], ignore_index=True).sample(frac=1, random_state=42)
os.makedirs(out.parent, exist_ok=True)
combined.to_csv(out, index=False)
print("✅ Combined dataset created:", out)
print(combined['label'].value_counts())
