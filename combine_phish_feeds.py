# scripts/combine_phish_feeds.py
import pandas as pd
from pathlib import Path

files = [
    "data/raw/openphish_labelled.csv",
    "data/raw/openphish_live.csv",
    "data/raw/phishtank_labeled.csv"
]

dfs = []
for f in files:
    try:
        dfs.append(pd.read_csv(f))
        print(f"Loaded {f}")
    except FileNotFoundError:
        pass

df = pd.concat(dfs, ignore_index=True)
df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
df["label"] = 1
out = Path("data/processed/openphish_combined.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("Final phishing URLs:", len(df))
print("Saved ->", out)
