import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve, auc
)


def compute_full_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=1)
    prec_cls, rec_cls, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=1)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    auc_roc = 0.0
    if y_proba is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_proba)
        except Exception:
            pass

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
        # per-class
        "precision_misinfo": float(prec_cls[0]) if len(prec_cls) > 0 else 0.0,
        "recall_misinfo": float(rec_cls[0]) if len(rec_cls) > 0 else 0.0,
        "f1_misinfo": float(f1_cls[0]) if len(f1_cls) > 0 else 0.0,
        "precision_factual": float(prec_cls[1]) if len(prec_cls) > 1 else 0.0,
        "recall_factual": float(rec_cls[1]) if len(rec_cls) > 1 else 0.0,
        "f1_factual": float(f1_cls[1]) if len(f1_cls) > 1 else 0.0,
        # confusion matrix breakdown
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Misinformation", "Factual"],
        yticklabels=["Misinformation", "Factual"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_proba, output_path, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_model(model, dataset_df, output_dir, config_name="eval", use_enhanced=False):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_proba = [], [], []

    print(f"Running inference on {len(dataset_df)} samples...")
    for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        text = row["content"]
        features = {k: row.get(k) for k in ["sentiment_polarity", "word_count", "avg_word_length"]}

        result = model.predict(text, features=features, use_enhanced=use_enhanced)
        y_true.append(int(row["label"]))
        y_pred.append(result["label"])
        y_proba.append(result["class_probabilities"]["TRUE (Factual)"])

    metrics = compute_full_metrics(y_true, y_pred, y_proba)

    print(f"\n{'='*40}")
    print(f"Results for {config_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"{'='*40}")

    with open(out / f"{config_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(y_true, y_pred, out / f"{config_name}_confusion_matrix.png", title=config_name)
    plot_roc_curve(y_true, y_proba, out / f"{config_name}_roc_curve.png", title=config_name)

    return metrics
