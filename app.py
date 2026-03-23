#!/usr/bin/env python
"""
Unified CLI for the refugee misinformation detection pipeline.

Usage:
    python app.py preprocess --config twitter_filtered
    python app.py train --model llama --dataset twitter_filtered
    python app.py distill --teacher-path models/llama_twitter_filtered --dataset twitter_filtered
    python app.py evaluate --model-path models/distilled_model --dataset urxd --shap
    python app.py pipeline --model llama --dataset twitter_filtered
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATASET_CONFIGS, MODEL_CONFIGS


BANNER = """
╔══════════════════════════════════════════════════════════╗
║   Refugee Misinformation Detector — CBS Master Thesis    ║
║   Boll & Gonçalves, 2025                                 ║
╚══════════════════════════════════════════════════════════╝
"""


def cmd_preprocess(args):
    from scripts.run_preprocessing import run, main as _main
    sys.argv = ["preprocess"]
    if args.config:
        sys.argv += ["--config", args.config]
    if args.all_configs:
        sys.argv += ["--config", "all"]
    _main()


def cmd_train(args):
    import scripts.run_phase1 as p1
    p1.run(
        model_name=args.model,
        dataset_config=args.dataset,
        env=args.env,
        hp_search=args.hp_search,
        use_enhanced=args.enhanced_prompt,
    )


def cmd_distill(args):
    import subprocess
    cmd = [sys.executable, "scripts/run_phase2.py",
           "--teacher-path", args.teacher_path,
           "--dataset", args.dataset]
    if args.hp_search:
        cmd.append("--hp-search")
    if args.domain_adapt:
        cmd.append("--domain-adapt")
    subprocess.run(cmd, check=True)


def cmd_evaluate(args):
    import subprocess
    cmd = [sys.executable, "scripts/run_evaluation.py",
           "--model-path", args.model_path,
           "--model-type", args.model_type,
           "--dataset", args.dataset]
    if args.shap:
        cmd.append("--shap")
    if args.error_analysis:
        cmd.append("--error-analysis")
    subprocess.run(cmd, check=True)


def cmd_pipeline(args):
    print(f"Running full pipeline: {args.model} on {args.dataset}")

    # 1. preprocess
    from src.preprocessing import load_dataset_config, preprocess_and_split, save_splits
    print("\n[1/4] Preprocessing...")
    df = load_dataset_config(args.dataset)
    train, val, test = preprocess_and_split(df, balance=True)
    save_splits(train, val, test, "data/processed", args.dataset)

    # 2. phase 1
    print("\n[2/4] Training Phase 1...")
    import scripts.run_phase1 as p1
    p1.run(args.model, args.dataset, env=args.env)

    # 3. phase 2 distillation
    print("\n[3/4] Distillation...")
    model_path = f"models/{args.model}_{args.dataset}"
    import subprocess
    subprocess.run([
        sys.executable, "scripts/run_phase2.py",
        "--teacher-path", model_path,
        "--dataset", args.dataset,
        "--domain-adapt",
    ], check=True)

    # 4. evaluate
    print("\n[4/4] Evaluating distilled model...")
    subprocess.run([
        sys.executable, "scripts/run_evaluation.py",
        "--model-path", f"results/phase2/{args.dataset}/distilled_model/student_model",
        "--model-type", "distilled",
        "--dataset", args.dataset,
        "--error-analysis",
    ], check=True)

    print("\nPipeline complete.")


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="Refugee Misinformation Detection Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # preprocess
    p = sub.add_parser("preprocess", help="Preprocess dataset(s)")
    p.add_argument("--config", choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--all-configs", action="store_true")

    # train
    p = sub.add_parser("train", help="Phase 1: Fine-tune a model")
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="llama")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), default="twitter_filtered")
    p.add_argument("--env", choices=["dev", "default", "prod"], default="default")
    p.add_argument("--hp-search", action="store_true")
    p.add_argument("--enhanced-prompt", action="store_true")

    # distill
    p = sub.add_parser("distill", help="Phase 2: Knowledge distillation")
    p.add_argument("--teacher-path", required=True)
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), default="twitter_filtered")
    p.add_argument("--hp-search", action="store_true")
    p.add_argument("--domain-adapt", action="store_true")

    # evaluate
    p = sub.add_parser("evaluate", help="Evaluate a saved model")
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-type", default="modernbert")
    p.add_argument("--dataset", default="urxd")
    p.add_argument("--shap", action="store_true")
    p.add_argument("--error-analysis", action="store_true")

    # pipeline
    p = sub.add_parser("pipeline", help="Run full pipeline end-to-end")
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="llama")
    p.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), default="twitter_filtered")
    p.add_argument("--env", choices=["dev", "default", "prod"], default="default")

    args = parser.parse_args()

    dispatch = {
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "distill": cmd_distill,
        "evaluate": cmd_evaluate,
        "pipeline": cmd_pipeline,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
