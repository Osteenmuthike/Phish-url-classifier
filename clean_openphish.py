# scripts/clean_openphish.py
import pandas as pd
from urllib.parse import urlparse, urlunparse
from pathlib import Path

IN = Path("data/raw/openphish_live.csv")
OUT = Path("data/raw/openphish_labelled.csv")

df = pd.read_csv(IN)
def clean(u):
    try:
        p = urlparse(u.strip())
        return urlunparse((p.scheme or "http", p.netloc.lower(), p.path or "/", "", "", ""))
    except:
        return u.strip()

df["url"] = df.iloc[:,0].astype(str).map(clean)
df["label"] = 1
df = df[["url","label"]].drop_duplicates().reset_index(drop=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"[SUCCESS] cleaned -> {OUT} ({len(df)} URLs)")
