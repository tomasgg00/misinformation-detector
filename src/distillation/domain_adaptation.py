import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def mix_validation_data(train_df, val_df, mix_ratio=0.2, random_state=42):
    """
    Mix some URXD validation samples into the training data.
    Helps the model generalize to real-world refugee-related posts.
    # mix in some URXD samples to help the model adapt to the target domain
    """
    n_mix = int(len(val_df) * mix_ratio)
    if n_mix == 0:
        print("mix_ratio too small, nothing to mix in")
        return train_df

    mixed_in = val_df.sample(n=n_mix, random_state=random_state)
    result = pd.concat([train_df, mixed_in], ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Mixed in {n_mix} URXD samples. New train size: {len(result)}")
    print(f"New class distribution: {result['label'].value_counts().to_dict()}")

    return result


def create_domain_adapted_splits(lmd_train, lmd_val, urxd_df, mix_ratio=0.3, random_state=42):
    """
    Build final train/val splits for distillation with domain adaptation.

    Training: LMD train + mix_ratio of URXD
    Validation: remaining URXD (held out properly for evaluation)
    """
    n_urxd_train = int(len(urxd_df) * mix_ratio)
    urxd_train = urxd_df.sample(n=n_urxd_train, random_state=random_state)
    urxd_val = urxd_df.drop(urxd_train.index)

    adapted_train = pd.concat([lmd_train, urxd_train], ignore_index=True)
    adapted_train = adapted_train.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Domain-adapted train: {len(adapted_train)} samples "
          f"({len(lmd_train)} LMD + {len(urxd_train)} URXD)")
    print(f"URXD validation: {len(urxd_val)} samples")
    print(f"Train class dist: {adapted_train['label'].value_counts().to_dict()}")
    print(f"Val class dist: {urxd_val['label'].value_counts().to_dict()}")

    return adapted_train, urxd_val
