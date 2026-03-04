# scripts/generate_legit_sample.py
import pandas as pd
from pathlib import Path

raw = Path("data/raw/top1m_raw.csv")
out = Path("data/raw/top1m_legit.csv")

if not raw.exists():
    raise SystemExit("Put top1m_raw.csv into data/raw/ first (one domain per line or rank,domain).")

df = pd.read_csv(raw, header=None, names=["rank", "domain"], on_bad_lines="skip", engine="python")
df["url"] = "https://" + df["domain"].astype(str)
df["url"].head(2000).to_csv(out, index=False, header=False)
print(f"[SUCCESS] Wrote {out}")
