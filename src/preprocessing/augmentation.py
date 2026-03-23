import random
import pandas as pd
import numpy as np


def _word_dropout(text, p=0.15):
    words = text.split()
    if len(words) <= 3:
        return text
    kept = [w for w in words if random.random() > p]
    return " ".join(kept) if kept else text


def _word_swap(text, p=0.1):
    words = text.split()
    for i in range(len(words) - 1):
        if random.random() < p:
            words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def augment_difficult_samples(df, n_samples=None, random_state=42):
    """
    Augment minority class samples to help with class imbalance.
    Uses simple word dropout and word swap — nothing fancy but works okay.
    """
    random.seed(random_state)

    counts = df["label"].value_counts()
    if len(counts) < 2:
        return df

    minority_label = counts.idxmin()
    majority_count = counts.max()
    minority_count = counts.min()

    if n_samples is None:
        n_samples = majority_count - minority_count

    if n_samples <= 0:
        return df

    minority_df = df[df["label"] == minority_label]
    augmented_rows = []

    for _ in range(n_samples):
        row = minority_df.sample(1, random_state=random.randint(0, 9999)).iloc[0].to_dict()
        text = row.get("content", "")

        # randomly pick augmentation method
        if random.random() < 0.5:
            row["content"] = _word_dropout(text)
        else:
            row["content"] = _word_swap(text)

        augmented_rows.append(row)

    aug_df = pd.DataFrame(augmented_rows)
    result = pd.concat([df, aug_df], ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Augmented {n_samples} samples for class {minority_label}. "
          f"New distribution: {result['label'].value_counts().to_dict()}")

    return result
