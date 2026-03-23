import os
import json
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optuna_search(model_class, train_dataset, val_dataset, output_dir, n_trials=10):
    """
    Quick Optuna search over LoRA and training hyperparams.
    # 2 epochs per trial is enough to get a signal on what works
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
        lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
        lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        batch_size = trial.suggest_categorical("batch_size", [2, 4])
        grad_accum = trial.suggest_categorical("grad_accum", [4, 8])
        use_enhanced = trial.suggest_categorical("use_enhanced_prompt", [True, False])

        try:
            detector = model_class(lora_r=lora_r, lora_alpha=lora_alpha)
            detector.load()

            from src.training.trainer import train_model
            metrics = train_model(
                detector.model,
                detector.tokenizer,
                train_dataset,
                val_dataset,
                output_dir=str(out / f"trial_{trial.number}"),
                config={
                    "lr": lr,
                    "batch_size": batch_size,
                    "epochs": 2,  # fast trials
                    "grad_accum": grad_accum,
                },
            )

            return metrics.get("eval_f1", 0.0)

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    print(f"Best params (f1={study.best_value:.4f}): {best}")

    with open(out / "best_params.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": best}, f, indent=2)

    return best
