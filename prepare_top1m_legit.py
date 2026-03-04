# scripts/prepare_top1m_legit.py
import pandas as pd
from pathlib import Path

IN = Path("data/processed/top1m_https_clean.csv")
OUT = Path("data/processed/top1m_legit.csv")

print(f"[INFO] Loading {IN} ...")
df = pd.read_csv(IN)

if 'url' not in df.columns:
    df.columns = ['url']

df["url"] = df["url"].astype(str).str.strip()
df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
df["label"] = 0  # legitimate

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"[INFO] Legitimate URLs saved to {OUT}")
print(f"[INFO] Total legitimate URLs: {len(df)}")
