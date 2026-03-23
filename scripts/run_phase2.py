#!/usr/bin/env python
"""Phase 2: Distillation — teacher → ModernBERT-large student."""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import DATASET_CONFIGS
from src.distillation import generate_rationales, load_teacher_model, run_distillation, mix_validation_data
from src.distillation.hp_search import run_distillation_hp_search


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Knowledge distillation")
    parser.add_argument("--teacher-path", required=True, help="Path to fine-tuned teacher model")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), default="twitter_filtered")
    parser.add_argument("--hp-search", action="store_true")
    parser.add_argument("--domain-adapt", action="store_true", help="Mix URXD data into training")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output-dir", default="results/phase2")
    args = parser.parse_args()

    processed_dir = Path("data/processed") / args.dataset
    if not (processed_dir / "train.json").exists():
        print("Processed data not found. Run run_preprocessing.py first.")
        sys.exit(1)

    train_df = pd.read_json(processed_dir / "train.json")
    val_df = pd.read_json(processed_dir / "val.json")

    if args.domain_adapt:
        urxd_path = Path("data/processed/urxd/train.json")
        if urxd_path.exists():
            urxd_df = pd.read_json(urxd_path)
            train_df = mix_validation_data(train_df, urxd_df, mix_ratio=0.3)
        else:
            print("URXD data not found, skipping domain adaptation")

    # generate or load rationales
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    rationale_path = out_dir / "rationales.json"
    cache_path = out_dir / "rationales_cache.json"

    teacher_model, teacher_tokenizer = load_teacher_model(args.teacher_path, bits=8)
    texts = train_df["content"].tolist()
    labels = train_df["label"].tolist()

    rationale_data = generate_rationales(
        teacher_model, teacher_tokenizer,
        texts, labels,
        output_path=str(rationale_path),
        cache_path=str(cache_path),
    )

    del teacher_model
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.hp_search:
        best_params = run_distillation_hp_search(
            rationale_data,
            args.teacher_path,
            output_dir=str(out_dir / "hp_search"),
            n_trials=15,
        )
    else:
        best_params = {
            "alpha": args.alpha,
            "temperature": args.temperature,
            "epochs": args.epochs,
        }

    metrics = run_distillation(
        rationale_data,
        args.teacher_path,
        output_dir=str(out_dir / "distilled_model"),
        config=best_params,
    )

    print(f"\nDistillation complete. Final metrics: {metrics}")


if __name__ == "__main__":
    main()
