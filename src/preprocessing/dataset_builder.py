import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .text_cleaning import clean_text, normalize_label
from .feature_extraction import extract_all_features, IMPORTANT_FEATURES

DATASET_CONFIGS = {
    "twitter_filtered": "twitter_filtered.csv",
    "twitter_unfiltered": "twitter_unfiltered.csv",
    "complete_filtered": "complete_filtered.csv",
    "complete_unfiltered": "complete_unfiltered.csv",
}


def load_dataset_config(config_name, data_dir="data/raw"):
    if config_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}")

    path = Path(data_dir) / DATASET_CONFIGS[config_name]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Place CSVs in data/raw/")

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")

    # standardize column names
    col_map = {}
    for col in df.columns:
        if col.lower() in ["content", "text", "tweet", "message", "post"]:
            col_map[col] = "content"
            break
    for col in df.columns:
        if col.lower() in ["label", "class", "verdict", "misinfo", "is_misinfo"]:
            col_map[col] = "label"
            break

    if col_map:
        df = df.rename(columns=col_map)

    if "content" not in df.columns:
        raise ValueError("No text column found in dataset")
    if "label" not in df.columns:
        print("No label column found, defaulting all to 0")
        df["label"] = 0

    # clean up
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip() != ""]
    df = df.drop_duplicates(subset=["content"])
    df["label"] = df["label"].apply(normalize_label)

    print(f"After cleaning: {len(df)} rows | class dist: {df['label'].value_counts().to_dict()}")
    return df


def _process_row(row):
    content = row.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return None

    cleaned = clean_text(content)
    if not cleaned:
        return None

    features = extract_all_features(cleaned)
    features["content"] = cleaned
    features["label"] = int(row.get("label", 0))
    features["source"] = row.get("source", "unknown")
    features["media"] = row.get("media", "unknown")

    return features


def preprocess_and_split(df, balance=True, test_size=0.15, val_size=0.1, random_state=42):
    print(f"Processing {len(df)} rows...")
    rows = df.to_dict(orient="records")

    processed = []
    for row in tqdm(rows, desc="Extracting features"):
        result = _process_row(row)
        if result is not None:
            processed.append(result)

    print(f"Valid rows after processing: {len(processed)}")
    proc_df = pd.DataFrame(processed)

    if balance:
        proc_df = balance_dataset(proc_df)

    # split: first carve out test, then split remainder into train/val
    train_val, test = train_test_split(
        proc_df, test_size=test_size, stratify=proc_df["label"], random_state=random_state
    )
    actual_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=actual_val_size, stratify=train_val["label"], random_state=random_state
    )

    print(f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name} class dist: {split['label'].value_counts().to_dict()}")

    return train, val, test


def balance_dataset(df, synthetic_data_dir="synthetic_data"):
    counts = df["label"].value_counts()
    if len(counts) < 2:
        return df

    minority = counts.idxmin()
    majority = counts.idxmax()
    imbalance = counts[majority] - counts[minority]

    if imbalance <= 0:
        return df

    print(f"Balancing dataset: minority class={minority}, need {imbalance} more samples")

    # try loading synthetic samples first
    cache = Path(synthetic_data_dir) / f"label_{minority}_samples.json"
    synthetic_rows = []

    if cache.exists():
        try:
            with open(cache) as f:
                all_synth = json.load(f)
            samples = all_synth[:imbalance]
            for s in samples:
                content = s.get("content") or s.get("Content", "")
                if content:
                    synthetic_rows.append({"content": content, "label": int(minority), "source": "synthetic", "media": "synthetic"})
            print(f"Loaded {len(synthetic_rows)} synthetic samples from cache")
        except Exception as e:
            print(f"Could not load synthetic samples: {e}")

    if not synthetic_rows:
        # fallback: oversample with replacement
        print("No synthetic data found, oversampling minority class instead")
        minority_df = df[df["label"] == minority]
        oversampled = minority_df.sample(n=imbalance, replace=True, random_state=42)
        df = pd.concat([df, oversampled], ignore_index=True)
    else:
        synth_df = pd.DataFrame(synthetic_rows)
        df = pd.concat([df, synth_df], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balanced distribution: {df['label'].value_counts().to_dict()}")
    return df


def create_prompt(content, features=None, use_enhanced=False):
    base = "[INST] Analyze the following text and determine if it contains misinformation about refugees or migrants. "

    if use_enhanced and features:
        ctx = ""
        pol = features.get("sentiment_polarity")
        if pol is not None:
            desc = "negative" if pol < -0.2 else ("positive" if pol > 0.2 else "neutral")
            ctx += f"Sentiment: {desc} (score: {pol:.2f}). "
        wc = features.get("word_count")
        if wc:
            ctx += f"Length: {wc} words. "
        if ctx:
            base += "Context: " + ctx

    return base + f"Text: {content} [/INST]"


def save_splits(train_df, val_df, test_df, output_dir, config_name):
    out = Path(output_dir) / config_name
    out.mkdir(parents=True, exist_ok=True)

    train_df.to_json(out / "train.json", orient="records")
    val_df.to_json(out / "val.json", orient="records")
    test_df.to_json(out / "test.json", orient="records")

    print(f"Saved splits to {out}/")
