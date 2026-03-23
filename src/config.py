import os
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
SYNTHETIC_DATA_DIR = ROOT_DIR / "synthetic_data"

# make sure dirs exist
for d in [DATA_DIR / "raw", PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, SYNTHETIC_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASET_CONFIGS = {
    "twitter_filtered": "twitter_filtered.csv",
    "twitter_unfiltered": "twitter_unfiltered.csv",
    "complete_filtered": "complete_filtered.csv",
    "complete_unfiltered": "complete_unfiltered.csv",
}

MODEL_CONFIGS = {
    "llama": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "gemma": {
        "model_name": "google/gemma-2-9b-it",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "modernbert": {
        "model_name": "answerdotai/ModernBERT-large",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["query", "key", "value"],
    },
    "deepseek": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
}

TRAINING_DEFAULTS = {
    "lr": 2e-5,
    "batch_size": 4,
    "epochs": 5,
    "grad_accum": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "eval_steps": 50,
    "save_steps": 50,
}

DISTILLATION_DEFAULTS = {
    "alpha": 0.7,           # weight on soft labels
    "temperature": 4.0,     # distillation temperature
    "lr": 3e-5,
    "epochs": 8,
    "batch_size": 4,
    "grad_accum": 4,
    "use_focal_loss": True,
    "label_smoothing": 0.1,
    "early_stopping_patience": 5,
}

# these 7 came out as most important in feature importance analysis
IMPORTANT_FEATURES = [
    "sentiment_polarity",
    "char_count",
    "avg_word_length",
    "verb_ratio",
    "word_count",
    "type_token_ratio",
    "readability_score",
]


def get_model_config(model_name):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]


def get_results_dir(model_name, dataset_config):
    if model_name == "distilled":
        d = RESULTS_DIR / "phase2" / dataset_config
    else:
        d = RESULTS_DIR / "phase1" / model_name / dataset_config
    d.mkdir(parents=True, exist_ok=True)
    return d
