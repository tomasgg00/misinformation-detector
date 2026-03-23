#!/usr/bin/env python
"""Preprocess dataset configurations for training."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_dataset_config, preprocess_and_split, augment_difficult_samples, save_splits
from src.config import DATASET_CONFIGS


def run(config_name, balance=True, augment=True, output_dir="data/processed", data_dir="data/raw"):
    print(f"\n{'='*50}")
    print(f"Processing: {config_name}")
    print(f"{'='*50}")

    df = load_dataset_config(config_name, data_dir=data_dir)
    train, val, test = preprocess_and_split(df, balance=balance)

    if augment:
        print("Augmenting training set...")
        train = augment_difficult_samples(train)

    save_splits(train, val, test, output_dir, config_name)
    print(f"Done: {config_name}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess misinformation datasets")
    parser.add_argument("--config", choices=list(DATASET_CONFIGS.keys()) + ["all"], default="all")
    parser.add_argument("--balance", action="store_true", default=True)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()

    configs = list(DATASET_CONFIGS.keys()) if args.config == "all" else [args.config]

    for cfg in configs:
        run(cfg, balance=args.balance, augment=not args.no_augment,
            output_dir=args.output_dir, data_dir=args.data_dir)

    print("\nAll done!")


if __name__ == "__main__":
    main()
