# scripts/fetch_openphish_live.py
import requests
import pandas as pd
from pathlib import Path

URL = "https://openphish.com/feed.txt"
OUT = Path("data/raw/openphish_live.csv")

print(f"Fetching from {URL} ...")
resp = requests.get(URL, timeout=10)
resp.raise_for_status()

urls = [u.strip() for u in resp.text.splitlines() if u.strip()]
df = pd.DataFrame({"url": urls, "label": 1})
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Saved {len(df)} phishing URLs -> {OUT}")
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
