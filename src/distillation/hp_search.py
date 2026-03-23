import os
import json
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 15 trials takes a while but Optuna is smart about exploring the space
DEFAULT_N_TRIALS = 15


def run_distillation_hp_search(rationale_data, teacher_model_path, output_dir, n_trials=DEFAULT_N_TRIALS):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.3, 0.9)
        temperature = trial.suggest_float("temperature", 2.0, 8.0)
        lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
        lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
        use_focal = trial.suggest_categorical("use_focal_loss", [True, False])
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

        config = {
            "alpha": alpha,
            "temperature": temperature,
            "lr": lr,
            "lora_r": lora_r,
            "lora_alpha": lora_r * 2,
            "use_focal_loss": use_focal,
            "label_smoothing": label_smoothing,
            "epochs": 3,  # fast trials
            "batch_size": 4,
            "grad_accum": 4,
        }

        try:
            from src.distillation.distill import run_distillation
            metrics = run_distillation(
                rationale_data,
                teacher_model_path,
                output_dir=str(out / f"trial_{trial.number}"),
                config=config,
            )
            return metrics.get("eval_f1", 0.0)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest distillation params (f1={study.best_value:.4f}):")
    for k, v in best.items():
        print(f"  {k}: {v}")

    results = {
        "best_value": study.best_value,
        "best_params": best,
        "n_trials": n_trials,
    }
    with open(out / "hp_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best
