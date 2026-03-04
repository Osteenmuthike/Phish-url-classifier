#!/usr/bin/env python3
"""
Train a phishing-URL classifier and save a single sklearn Pipeline to models/model.pkl.

Behavior:
- If data/processed/train.csv exists, load it and train.
- Otherwise, attempt to build a balanced dataset by sampling:
    - legit URLs from data/processed/top1m_https_clean.csv (label=0)
    - phishing URLs from data/raw/openphish_labeled.csv (or data/raw/openphish.txt) (label=1)
  If phishing feed is missing, falls back to a synthetic phishing generator (for demos).
- Uses src/feature_extraction.extract_features(...) to compute features for URLs.
- Trains a RandomForest inside a Pipeline (imputer -> scaler -> classifier).
- Saves pipeline to models/model.pkl and metadata to models/model_metadata.json

Usage:
    python scripts/train_and_save_model.py
Options:
    --train_csv PATH    : path to processed train CSV (default: data/processed/train.csv)
    --out_model PATH    : output model path (default: models/model.pkl)
    --test_size FLOAT   : local validation test fraction if constructing dataset (default: 0.2)
    --seed INT          : random seed (default: 42)
"""
import argparse
from pathlib import Path
import json
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# Ensure repo src is importable
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from feature_extraction import extract_features  # should be in src/
except Exception as e:
    raise ImportError("Could not import src/feature_extraction.py. Ensure it exists.") from e

warnings.filterwarnings("ignore")


def load_or_build_dataset(train_csv_path: Path, top1m_path: Path, phish_path: Path, n_each: int, seed: int, test_size: float):
    """
    Returns: X_train_df, X_test_df, y_train, y_test, used_dataframe (combined)
    """
    if train_csv_path.exists():
        print(f"✅ Loading existing processed train CSV: {train_csv_path}")
        df = pd.read_csv(train_csv_path)
        if 'label' not in df.columns or 'url' not in df.columns:
            raise ValueError("train.csv must contain 'url' and 'label' columns")
        # if dataset is already small and includes test, we will split here for local validation
        X = df['url'].tolist()
        y = df['label'].astype(int).values
        X_train_urls, X_test_urls, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
        return X_train_urls, X_test_urls, y_train, y_test, df

    # Otherwise build balanced dataset from top1m + phishing feed
    print("ℹ️  No processed train.csv found — attempting to build dataset from feeds.")
    if not top1m_path.exists():
        raise FileNotFoundError(f"Top-1M cleaned file not found: {top1m_path}")

    legit_df = pd.read_csv(top1m_path)
    if 'url' not in legit_df.columns:
        # attempt to treat first column as url
        legit_df.columns = ['url'] + list(legit_df.columns[1:])
    legit_df['url'] = legit_df['url'].astype(str).str.strip()
    legit_df = legit_df.drop_duplicates(subset='url')
    legit_df['label'] = 0
    print(f"Loaded {len(legit_df)} legit URLs from {top1m_path}")

    # load phishing feed if available
    phish_df = None
    if phish_path.exists():
        print(f"Loading phishing feed from: {phish_path}")
        if phish_path.suffix.lower() in ['.txt', '.list']:
            urls = [line.strip() for line in phish_path.read_text(encoding='utf8').splitlines() if line.strip()]
            phish_df = pd.DataFrame({'url': urls})
        else:
            phish_df = pd.read_csv(phish_path)
        if 'url' not in phish_df.columns:
            phish_df.columns = ['url'] + list(phish_df.columns[1:])
        phish_df['url'] = phish_df['url'].astype(str).str.strip()
        phish_df = phish_df.drop_duplicates(subset='url')
        phish_df['label'] = 1
        print(f"Loaded {len(phish_df)} phishing URLs from {phish_path}")
    else:
        print("⚠️ Phishing feed not found. Generating synthetic phishing samples for demo.")
        # generate synthetic phishing-like URLs
        templates = [
            "http://{brand}-secure-login.com/",
            "http://{brand}.verify-account.net/",
            "http://secure-{brand}-account.co/",
            "http://{brand}-login-update.org/",
            "http://{brand}confirm-login.com/"
        ]
        brands = ["paypal", "bankofamerica", "chase", "amazon", "apple", "google", "microsoft", "facebook", "netflix", "dropbox"]
        phish_urls = []
        for i in range(max(n_each * 2, 200)):
            t = random.choice(templates)
            b = random.choice(brands)
            phish_urls.append(t.format(brand=b) + (f"?id={random.randint(100,9999)}"))
        phish_df = pd.DataFrame({'url': phish_urls})
        phish_df['label'] = 1
        print(f"Generated {len(phish_df)} synthetic phishing URLs.")

    # sample n_each from each
    if len(legit_df) < n_each:
        raise ValueError(f"Not enough legit URLs to sample {n_each} (have {len(legit_df)})")
    if len(phish_df) < n_each:
        print(f"Warning: phishing feed has {len(phish_df)} rows < requested {n_each}. Will sample with replacement.")
        phish_sample = phish_df.sample(n=n_each, replace=True, random_state=seed).reset_index(drop=True)
    else:
        phish_sample = phish_df.sample(n=n_each, random_state=seed).reset_index(drop=True)

    legit_sample = legit_df.sample(n=n_each, random_state=seed).reset_index(drop=True)

    combined = pd.concat([legit_sample[['url','label']], phish_sample[['url','label']]], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Built combined dataset: {len(combined)} rows ({n_each} legit + {n_each} phish)")

    # split into train/test
    X = combined['url'].tolist()
    y = combined['label'].values
    X_train_urls, X_test_urls, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    return X_train_urls, X_test_urls, y_train, y_test, combined


def compute_feature_dataframe(url_list):
    """
    Uses src.feature_extraction.extract_features to compute features.
    Handles either a DataFrame-returning function or list-of-dicts.
    Returns a pandas DataFrame of numeric features (one row per URL).
    """
    print("Extracting features for", len(url_list), "URLs (this may take a moment)...")
    # extract_features may accept a list or DataFrame; handle both
    try:
        # try passing a list first
        feats = extract_features(url_list)
    except Exception:
        # fallback: pass DataFrame
        feats = extract_features(pd.DataFrame({'url': url_list}))

    # feats might be a list-of-dicts or a DataFrame
    if isinstance(feats, list):
        df_feats = pd.DataFrame(feats)
    elif isinstance(feats, pd.DataFrame):
        df_feats = feats.copy()
    else:
        # try to convert to DataFrame
        df_feats = pd.DataFrame(feats)

    # drop any non-numeric columns (like 'url') if present
    non_numeric_cols = [c for c in df_feats.columns if not pd.api.types.is_numeric_dtype(df_feats[c])]
    if 'url' in non_numeric_cols:
        non_numeric_cols.remove('url')  # keep URL if present but we'll drop it below
    # We will drop 'url' column if present
    if 'url' in df_feats.columns:
        df_feats = df_feats.drop(columns=['url'])
    # Fill NaNs
    df_feats = df_feats.fillna(0)
    return df_feats


def main(args):
    start_time = time.time()
    out_dir = Path(args.out_model).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = Path(args.train_csv)
    top1m = Path("data/processed/top1m_https_clean.csv")
    phish_feed = Path("data/raw/openphish_labeled.csv")  # default path; can be txt or csv

    X_train_urls, X_test_urls, y_train, y_test, combined_df = load_or_build_dataset(
        train_csv, top1m, phish_feed, args.n_each, args.seed, args.test_size
    )

    # extract features for train and test
    X_train_df = compute_feature_dataframe(X_train_urls)
    X_test_df = compute_feature_dataframe(X_test_urls)

    print("Feature columns:", list(X_train_df.columns))

    # build pipeline: imputer -> scaler -> RF
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, random_state=args.seed))
    ])

    print("Training classifier...")
    pipeline.fit(X_train_df, y_train)

    print("Evaluating on local test set...")
    y_pred = pipeline.predict(X_test_df)
    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test_df)[:, 1]
    except Exception:
        pass

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4))
    try:
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        if auc is not None:
            print("ROC-AUC:", round(auc, 4))
    except Exception:
        pass
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save pipeline
    model_path = Path(args.out_model)
    joblib.dump(pipeline, model_path)
    print(f"\n✅ Saved pipeline to: {model_path}")

    # Save a metadata JSON
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "train_rows": len(X_train_urls),
        "test_rows": len(X_test_urls),
        "n_estimators": args.n_estimators,
        "random_seed": args.seed,
        "model_file": str(model_path),
        "feature_columns": list(X_train_df.columns),
        "notes": "Trained with RandomForest inside sklearn Pipeline"
    }
    meta_path = model_path.parent / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print("Saved metadata to:", meta_path)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/processed/train.csv", help="Optional processed train CSV")
    parser.add_argument("--out_model", default="models/model.pkl", help="Output model pipeline path")
    parser.add_argument("--n_each", type=int, default=125, help="Number of legit and phishing samples to build (if train_csv missing)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Local validation set fraction")
    parser.add_argument("--n_estimators", type=int, default=200, help="RandomForest n_estimators")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
