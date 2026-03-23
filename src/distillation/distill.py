import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datasets import Dataset
from transformers import (
    Trainer, TrainingArguments, EarlyStoppingCallback,
    AutoModelForSequenceClassification, AutoTokenizer,
)
from peft import LoraConfig, get_peft_model

DEFAULT_ALPHA = 0.7       # weight for soft labels vs hard labels
DEFAULT_TEMP = 4.0        # distillation temperature
DEFAULT_LR = 3e-5
STUDENT_MODEL = "answerdotai/ModernBERT-large"


def focal_loss(logits, labels, gamma=2.0):
    # helps when misinformation class is underrepresented
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


class DistillationTrainer(Trainer):

    def __init__(self, *args, teacher_logits=None, alpha=DEFAULT_ALPHA,
                 temperature=DEFAULT_TEMP, use_focal_loss=True, label_smoothing=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_logits = teacher_logits  # pre-computed, keyed by index
        self.alpha = alpha
        self.temperature = temperature
        self.use_focal_loss = use_focal_loss
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        idx = inputs.pop("idx", None)  # index into teacher_logits

        outputs = model(**inputs)
        student_logits = outputs.logits

        # hard loss (on true labels)
        if self.use_focal_loss:
            hard_loss = focal_loss(student_logits, labels)
        else:
            hard_loss = F.cross_entropy(student_logits, labels, label_smoothing=self.label_smoothing)

        # soft loss (KL divergence against teacher distribution)
        soft_loss = torch.tensor(0.0, device=student_logits.device)
        if self.teacher_logits is not None and idx is not None:
            t_logits = torch.stack([
                torch.tensor(self.teacher_logits[int(i)], dtype=torch.float32)
                for i in idx
            ]).to(student_logits.device)

            soft_targets = F.softmax(t_logits / self.temperature, dim=-1)
            soft_preds = F.log_softmax(student_logits / self.temperature, dim=-1)
            soft_loss = F.kl_div(soft_preds, soft_targets, reduction="batchmean") * (self.temperature ** 2)

        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return (total_loss, outputs) if return_outputs else total_loss


def prepare_student_model(model_name=STUDENT_MODEL, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    print(f"Loading student model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["query", "key", "value"],
    )
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Student trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


def _build_dataset(rationale_data, tokenizer, teacher_model=None, max_length=512, use_rationale=True):
    """Tokenize + optionally get teacher logits."""
    records = []
    teacher_logits_map = {}

    for i, item in enumerate(rationale_data):
        text = item["text"]
        rationale = item.get("rationale", "")
        label = item["true_label"]

        # optionally concatenate rationale as extra context for student
        if use_rationale and rationale:
            input_text = f"{text} [SEP] {rationale[:200]}"  # truncate rationale
        else:
            input_text = text

        enc = tokenizer(input_text, truncation=True, max_length=max_length, padding="max_length")
        records.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": label,
            "idx": i,
        })

        # get teacher logits if teacher model provided
        if teacher_model is not None:
            with torch.no_grad():
                t_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
                t_inputs = {k: v.to(teacher_model.device) for k, v in t_inputs.items()}
                t_out = teacher_model(**t_inputs)
                teacher_logits_map[i] = t_out.logits[0].cpu().tolist()

    dataset = Dataset.from_list(records)
    return dataset, teacher_logits_map if teacher_model else None


def run_distillation(rationale_data, teacher_model_path, output_dir, config=None):
    cfg = {
        "alpha": DEFAULT_ALPHA,
        "temperature": DEFAULT_TEMP,
        "lr": DEFAULT_LR,
        "epochs": 8,
        "batch_size": 4,
        "grad_accum": 4,
        "use_focal_loss": True,
        "label_smoothing": 0.1,
        "early_stopping_patience": 5,
        "lora_r": 16,
        "lora_alpha": 32,
    }
    if config:
        cfg.update(config)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # load student
    student_model, student_tokenizer = prepare_student_model(
        lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"]
    )

    # split rationale data into train/val (80/20)
    n = len(rationale_data)
    n_train = int(0.8 * n)
    train_data = rationale_data[:n_train]
    val_data = rationale_data[n_train:]

    print("Loading teacher to get logits...")
    from src.distillation.rationale_generator import load_teacher_model
    teacher_model, _ = load_teacher_model(teacher_model_path, bits=8)

    train_dataset, teacher_logits = _build_dataset(train_data, student_tokenizer, teacher_model=teacher_model)
    val_dataset, _ = _build_dataset(val_data, student_tokenizer, teacher_model=None)

    # free teacher model memory
    del teacher_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=25,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    from src.training.trainer import compute_metrics
    trainer = DistillationTrainer(
        model=student_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=student_tokenizer,
        compute_metrics=compute_metrics,
        teacher_logits=teacher_logits,
        alpha=cfg["alpha"],
        temperature=cfg["temperature"],
        use_focal_loss=cfg["use_focal_loss"],
        label_smoothing=cfg["label_smoothing"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )

    print("Starting distillation training...")
    trainer.train()
    metrics = trainer.evaluate()
    print(f"Distillation results: {metrics}")

    student_model.save_pretrained(out / "student_model")
    student_tokenizer.save_pretrained(out / "student_model")

    with open(out / "distillation_metrics.json", "w") as f:
        json.dump({**cfg, **metrics}, f, indent=2)

    return metrics
