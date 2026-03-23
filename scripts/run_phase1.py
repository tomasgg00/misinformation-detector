#!/usr/bin/env python
"""Phase 1: Fine-tune a model on a dataset configuration."""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datasets import Dataset
from src.config import DATASET_CONFIGS, MODEL_CONFIGS, get_results_dir
from src.preprocessing import create_prompt
from src.training import train_model


MODEL_MAP = {
    "llama": "src.models.llama.LlamaDetector",
    "gemma": "src.models.gemma.GemmaDetector",
    "modernbert": "src.models.modernbert.ModernBERTDetector",
    "deepseek": "src.models.deepseek.DeepSeekDetector",
}


def load_detector(model_name):
    module_path, cls_name = MODEL_MAP[model_name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def df_to_hf_dataset(df, tokenizer, use_enhanced=False, max_length=512):
    records = []
    for _, row in df.iterrows():
        features = {k: row.get(k) for k in ["sentiment_polarity", "word_count", "avg_word_length", "verb_ratio"]}
        prompt = create_prompt(row["content"], features=features, use_enhanced=use_enhanced)
        enc = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length")
        records.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": int(row["label"]),
        })
    return Dataset.from_list(records)


def run(model_name, dataset_config, env="default", hp_search=False, use_enhanced=False):
    print(f"\nTraining {model_name} on {dataset_config} (env={env})")

    processed_dir = Path("data/processed") / dataset_config
    if not (processed_dir / "train.json").exists():
        print(f"Processed data not found at {processed_dir}. Run run_preprocessing.py first.")
        return

    train_df = pd.read_json(processed_dir / "train.json")
    val_df = pd.read_json(processed_dir / "val.json")

    DetectorClass = load_detector(model_name)
    cfg = MODEL_CONFIGS[model_name]

    if hp_search:
        from src.training import run_optuna_search
        results_dir = get_results_dir(model_name, dataset_config)
        best_params = run_optuna_search(
            DetectorClass, None, None,  # datasets passed inside
            output_dir=results_dir / "hp_search",
            n_trials=5 if env == "dev" else 10,
        )
        print(f"Best HP: {best_params}")
    else:
        best_params = {}

    detector = DetectorClass(
        model_name=cfg["model_name"],
        lora_r=best_params.get("lora_r", cfg["lora_r"]),
        lora_alpha=best_params.get("lora_alpha", cfg["lora_alpha"]),
    )
    detector.load()

    train_dataset = df_to_hf_dataset(train_df, detector.tokenizer, use_enhanced=use_enhanced)
    val_dataset = df_to_hf_dataset(val_df, detector.tokenizer, use_enhanced=use_enhanced)

    results_dir = get_results_dir(model_name, dataset_config)
    model_out = Path("models") / f"{model_name}_{dataset_config}"

    train_cfg = {"epochs": 2 if env == "dev" else 5}
    train_cfg.update(best_params)

    metrics = train_model(
        detector.model, detector.tokenizer,
        train_dataset, val_dataset,
        output_dir=str(model_out),
        config=train_cfg,
    )

    detector.save(model_out)
    print(f"Model saved to {model_out}")
    print(f"Metrics: {metrics}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Fine-tune transformer models")
    parser.add_argument("--model", choices=list(MODEL_MAP.keys()) + ["all"], default="llama")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()) + ["all"], default="twitter_filtered")
    parser.add_argument("--env", choices=["dev", "default", "prod"], default="default")
    parser.add_argument("--hp-search", action="store_true")
    parser.add_argument("--no-enhanced-prompt", action="store_true")
    args = parser.parse_args()

    models = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]
    datasets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]

    for m in models:
        for d in datasets:
            run(m, d, env=args.env, hp_search=args.hp_search, use_enhanced=not args.no_enhanced_prompt)


if __name__ == "__main__":
    main()
