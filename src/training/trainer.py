import torch
import torch.nn as nn
import numpy as np
import os
import json
from pathlib import Path
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=1)
    prec_cls, rec_cls, f1_cls, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=1)

    result = {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }

    # per-class breakdown — useful to track misinfo recall specifically
    if len(f1_cls) >= 2:
        result["f1_misinfo"] = f1_cls[0]
        result["f1_factual"] = f1_cls[1]
        result["recall_misinfo"] = rec_cls[0]
        result["recall_factual"] = rec_cls[1]

    return result


class MisinformationTrainer(Trainer):
    """Custom trainer with class-weighted loss to handle imbalance."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, dtype=torch.float).to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def train_model(model, tokenizer, train_dataset, val_dataset, output_dir, config=None):
    cfg = {
        "lr": 2e-5,
        "batch_size": 4,
        "epochs": 5,
        "grad_accum": 4,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "eval_steps": 50,
        "save_steps": 50,
    }
    if config:
        cfg.update(config)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # compute class weights from training labels
    labels = [ex["label"] for ex in train_dataset]
    counts = np.bincount(labels)
    total = len(labels)
    weights = [total / (len(counts) * c) for c in counts]
    print(f"Class weights: {weights}")

    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        evaluation_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = MisinformationTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")
    trainer.train()

    metrics = trainer.evaluate()
    print(f"Final val metrics: {metrics}")

    # save metrics
    with open(out / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
