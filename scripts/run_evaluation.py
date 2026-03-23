#!/usr/bin/env python
"""Evaluate a saved model on a test/validation set."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.evaluation import evaluate_model, run_shap_analysis, analyze_errors


MODEL_TYPE_MAP = {
    "llama": "src.models.llama.LlamaDetector",
    "gemma": "src.models.gemma.GemmaDetector",
    "modernbert": "src.models.modernbert.ModernBERTDetector",
    "deepseek": "src.models.deepseek.DeepSeekDetector",
    "distilled": "src.models.modernbert.ModernBERTDetector",
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", choices=list(MODEL_TYPE_MAP.keys()), default="modernbert")
    parser.add_argument("--dataset", default="urxd", help="Dataset name (folder in data/processed/)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--shap", action="store_true")
    parser.add_argument("--error-analysis", action="store_true")
    parser.add_argument("--output-dir", default="results/evaluation")
    args = parser.parse_args()

    # load data
    data_path = Path("data/processed") / args.dataset / f"{args.split}.json"
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        sys.exit(1)

    df = pd.read_json(data_path)
    print(f"Loaded {len(df)} {args.split} samples")

    # load model
    module_path, cls_name = MODEL_TYPE_MAP[args.model_type].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    DetectorClass = getattr(mod, cls_name)

    detector = DetectorClass()
    detector.load_for_inference(args.model_path)

    out_dir = Path(args.output_dir) / args.model_type / args.dataset
    metrics = evaluate_model(detector, df, output_dir=str(out_dir), config_name=args.split)

    if args.shap:
        print("\nRunning SHAP analysis...")
        run_shap_analysis(
            detector.model, detector.tokenizer,
            texts=df["content"].tolist(),
            labels=df["label"].tolist(),
            output_dir=str(out_dir / "shap"),
            n_samples=30,
        )

    if args.error_analysis:
        print("\nRunning error analysis...")
        # need to re-run predictions to get per-sample results
        y_true, y_pred, y_proba, texts = [], [], [], []
        for _, row in df.iterrows():
            res = detector.predict(row["content"])
            y_true.append(int(row["label"]))
            y_pred.append(res["label"])
            y_proba.append(res["class_probabilities"]["TRUE (Factual)"])
            texts.append(row["content"])

        analyze_errors(y_true, y_pred, texts, probabilities=y_proba, output_dir=str(out_dir / "errors"))


if __name__ == "__main__":
    main()
