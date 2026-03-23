import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_errors(y_true, y_pred, texts, probabilities=None, output_dir=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fp_mask = (y_pred == 1) & (y_true == 0)  # predicted factual, actually misinfo
    fn_mask = (y_pred == 0) & (y_true == 1)  # predicted misinfo, actually factual
    correct_mask = y_pred == y_true

    fp_texts = [texts[i] for i in np.where(fp_mask)[0]]
    fn_texts = [texts[i] for i in np.where(fn_mask)[0]]

    fp_samples = []
    for i in np.where(fp_mask)[0]:
        entry = {
            "text": texts[i],
            "true_label": int(y_true[i]),
            "predicted": int(y_pred[i]),
            "confidence": float(probabilities[i]) if probabilities is not None else None,
        }
        fp_samples.append(entry)

    fn_samples = []
    for i in np.where(fn_mask)[0]:
        entry = {
            "text": texts[i],
            "true_label": int(y_true[i]),
            "predicted": int(y_pred[i]),
            "confidence": float(probabilities[i]) if probabilities is not None else None,
        }
        fn_samples.append(entry)

    # basic stats
    def avg_len(text_list):
        if not text_list:
            return 0
        return np.mean([len(t.split()) for t in text_list])

    fp_stats = {
        "count": int(fp_mask.sum()),
        "avg_word_count": avg_len(fp_texts),
        "rate": float(fp_mask.sum() / max(1, len(y_true))),
    }
    fn_stats = {
        "count": int(fn_mask.sum()),
        "avg_word_count": avg_len(fn_texts),
        "rate": float(fn_mask.sum() / max(1, len(y_true))),
    }

    print(f"False positives: {fp_stats['count']} ({fp_stats['rate']:.2%})")
    print(f"False negatives: {fn_stats['count']} ({fn_stats['rate']:.2%})")
    print(f"Avg length FP: {fp_stats['avg_word_count']:.1f} words")
    print(f"Avg length FN: {fn_stats['avg_word_count']:.1f} words")

    results = {
        "fp_samples": fp_samples,
        "fn_samples": fn_samples,
        "fp_stats": fp_stats,
        "fn_stats": fn_stats,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "error_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
        save_error_examples(fp_samples, fn_samples, out)

    return results


def save_error_examples(fp_samples, fn_samples, output_dir, n_examples=20):
    out = Path(output_dir) / "error_examples.txt"

    with open(out, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("FALSE POSITIVES (predicted factual, actually misinformation)\n")
        f.write("=" * 60 + "\n\n")
        for i, s in enumerate(fp_samples[:n_examples]):
            conf = f" (conf: {s['confidence']:.3f})" if s["confidence"] else ""
            f.write(f"[{i+1}]{conf}\n{s['text']}\n\n")

        f.write("=" * 60 + "\n")
        f.write("FALSE NEGATIVES (predicted misinformation, actually factual)\n")
        f.write("=" * 60 + "\n\n")
        for i, s in enumerate(fn_samples[:n_examples]):
            conf = f" (conf: {s['confidence']:.3f})" if s["confidence"] else ""
            f.write(f"[{i+1}]{conf}\n{s['text']}\n\n")

    print(f"Error examples saved to {out}")
