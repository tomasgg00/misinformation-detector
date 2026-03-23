import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# SHAP with transformers can be slow, keep n_samples small for interactive use
try:
    import shap
    _shap_ok = True
except ImportError:
    _shap_ok = False
    print("shap not installed — run: pip install shap")


def run_shap_analysis(model, tokenizer, texts, labels, output_dir, n_samples=50):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not _shap_ok:
        print("SHAP not available, skipping analysis")
        return {}

    import torch
    from transformers import pipeline

    # sample a manageable subset
    idx = np.random.choice(len(texts), min(n_samples, len(texts)), replace=False)
    sample_texts = [texts[i] for i in idx]
    sample_labels = [labels[i] for i in idx]

    print(f"Running SHAP on {len(sample_texts)} samples...")

    try:
        # try text classification pipeline approach
        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1,
        )

        explainer = shap.Explainer(pipe)
        shap_values = explainer(sample_texts)

        # save summary plot
        shap.plots.bar(shap_values[:, :, 0], show=False)
        plt.savefig(out / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        # extract top tokens by mean absolute SHAP value
        # this part is approximate since token indices vary per example
        token_importance = {}
        for i, sv in enumerate(shap_values):
            for j, token in enumerate(sv.data):
                token = str(token).strip()
                if len(token) < 2:
                    continue
                val = float(abs(sv.values[j, 0]))
                token_importance[token] = token_importance.get(token, 0) + val

        sorted_tokens = sorted(token_importance.items(), key=lambda x: x[1], reverse=True)
        top_misinfo = sorted_tokens[:20]

        results = {
            "top_tokens_misinfo": top_misinfo,
            "n_samples_analyzed": len(sample_texts),
        }

        with open(out / "shap_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"Top misinfo tokens: {[t[0] for t in top_misinfo[:10]]}")
        return results

    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return {}


def get_feature_contributions(feature_values, predictions, feature_names):
    """
    Quick correlation between feature values and prediction confidence.
    Not proper SHAP for features — just pearson correlation as a proxy.
    """
    import pandas as pd

    if not feature_values or not predictions:
        return {}

    df = pd.DataFrame(feature_values)
    df["pred_proba"] = predictions

    contributions = {}
    for feat in feature_names:
        if feat in df.columns:
            corr = df[feat].corr(df["pred_proba"])
            contributions[feat] = round(float(corr), 4) if not np.isnan(corr) else 0.0

    return contributions
