"""
Gaslight Guard — RoBERTa-base Fine-tuning Script
=================================================
Data sources:
  1. tweet_eval / irony  (HuggingFace)  →  Sarcastic (1) + Sincere (0)
  2. Curated hand-labeled corpus        →  Passive-Aggressive (2) + Gaslighting (3)
                                            + extra Sincere/Sarcastic

Labels: 0=Sincere | 1=Sarcastic | 2=Passive-Aggressive | 3=Gaslighting
Model:  roberta-base  →  saved to ./model/gaslight-guard/

Usage:
    python train.py                  # full training run
    python train.py --epochs 5       # custom epochs
    python train.py --dry-run        # sanity check, 1 step only
"""

import argparse
import os
import sys
import random
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Local curated data
sys.path.insert(0, str(Path(__file__).parent))
from data.curated_data import get_curated_data, LABEL2ID, ID2LABEL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME    = "roberta-base"
OUTPUT_DIR    = Path(__file__).parent / "model" / "gaslight-guard"
MAX_LEN       = 128
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
SEED          = 42
NUM_LABELS    = 4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_tweet_eval_irony():
    """
    Download tweet_eval irony split from HuggingFace datasets.
    label 0 (not ironic) → Sincere (0)
    label 1 (ironic)     → Sarcastic (1)
    Returns list of (text, label_id)
    """
    try:
        from datasets import load_dataset
        print("  Downloading tweet_eval/irony from HuggingFace…")
        ds = load_dataset("tweet_eval", "irony")
        data = []
        for split in ("train", "validation", "test"):
            if split not in ds:
                continue
            for row in ds[split]:
                lbl = 1 if row["label"] == 1 else 0  # ironic→Sarcastic, else→Sincere
                data.append((row["text"], lbl))
        print(f"  tweet_eval/irony loaded: {len(data)} examples")
        return data
    except Exception as e:
        print(f"  [WARN] Could not load tweet_eval: {e}")
        print("  Falling back to curated-only training.")
        return []


def load_all_data(dry_run: bool = False):
    """Combine tweet_eval + curated data into balanced train/val splits."""
    tweet_data = load_tweet_eval_irony()
    curated    = get_curated_data()

    combined = tweet_data + curated

    # Balance: count per class, cap at max_per_class
    from collections import Counter
    counts = Counter(lbl for _, lbl in combined)
    print(f"\n  Raw class distribution: {dict(counts)}")

    # Cap majority classes (Sincere & Sarcastic from tweet_eval can dominate)
    max_per_class = min(max(counts.values()), 1200)
    balanced = []
    class_seen = Counter()
    random.shuffle(combined)
    for text, lbl in combined:
        if class_seen[lbl] < max_per_class:
            balanced.append((text, lbl))
            class_seen[lbl] += 1

    counts2 = Counter(lbl for _, lbl in balanced)
    print(f"  Balanced class distribution: {dict(counts2)}")
    print(f"  Total training examples: {len(balanced)}\n")

    if dry_run:
        balanced = balanced[:64]

    texts  = [t for t, _ in balanced]
    labels = [l for _, l in balanced]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.12, random_state=SEED, stratify=labels
    )
    return train_texts, val_texts, train_labels, val_labels


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class GaslightDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def compute_metrics(preds, labels):
    pred_ids = np.argmax(preds, axis=1)
    report = classification_report(
        labels, pred_ids,
        target_names=list(ID2LABEL.values()),
        output_dict=True,
        zero_division=0,
    )
    acc = report["accuracy"]
    return acc, report


def train(epochs: int = 4, dry_run: bool = False):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if dry_run:
        print("*** DRY-RUN MODE: 1 step only ***\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading data…")
    train_texts, val_texts, train_labels, val_labels = load_all_data(dry_run=dry_run)

    # ── Tokenizer & Model ─────────────────────────────────────────────────
    print(f"Loading tokenizer & model: {MODEL_NAME}")
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # ── Datasets & Loaders ────────────────────────────────────────────────
    train_ds = GaslightDataset(train_texts, train_labels, tokenizer)
    val_ds   = GaslightDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps   = len(train_loader) * epochs
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_acc = 0.0
    history = []

    print(f"\nStarting training: {epochs} epoch(s), {len(train_ds)} train / {len(val_ds)} val\n")
    print("─" * 60)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss    = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 20 == 0:
                print(f"  Epoch {epoch} | step {step:>4}/{len(train_loader)} | loss {loss.item():.4f}")

            if dry_run:
                break  # one step only

        avg_train_loss = total_loss / (step + 1)

        # ── Validate ──
        model.eval()
        all_preds, all_labels_list = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch  = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                all_preds.append(outputs.logits.cpu().numpy())
                all_labels_list.extend(batch["labels"].cpu().numpy())

        all_preds = np.vstack(all_preds)
        val_acc, report = compute_metrics(all_preds, all_labels_list)
        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch}/{epochs} — train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f} | val_acc: {val_acc:.4f}")
        for lbl_name in ID2LABEL.values():
            if lbl_name in report:
                r = report[lbl_name]
                print(f"  {lbl_name:<22} P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  support={r['support']}")
        print("─" * 60)

        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_acc": val_acc})

        if val_acc > best_val_acc and not dry_run:
            best_val_acc = val_acc
            print(f"  ✅ New best val_acc={val_acc:.4f}. Saving model…")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

        if dry_run:
            break

    # ── Save regardless on dry-run (for quick testing) ──────────────────
    if dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

    # ── Save training metadata ─────────────────────────────────────────────
    metadata = {
        "model_name": MODEL_NAME,
        "num_labels": NUM_LABELS,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "best_val_acc": best_val_acc if not dry_run else None,
        "epochs_trained": epochs,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "history": history,
    }
    with open(OUTPUT_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n🎉 Training complete. Model saved to: {OUTPUT_DIR}")
    if not dry_run:
        print(f"   Best validation accuracy: {best_val_acc:.4f}")

    return best_val_acc


# ---------------------------------------------------------------------------
# Evaluation on held-out stress-test samples
# ---------------------------------------------------------------------------

def evaluate_stress_test():
    """Quick evaluation of the saved model against known-difficult sarcasm samples."""
    from data.curated_data import LABEL2ID, ID2LABEL

    STRESS = [
        ("Oh wow, you're absolutely brilliant for doing that.", 1),
        ("What a fantastic idea, I'm sure it'll work out great.", 1),
        ("Sure, because that totally makes sense.", 1),
        ("Oh, clearly you know best about everything.", 1),
        ("Great job breaking everything again.", 1),
        ("I never said that. You're making things up.", 3),
        ("You're being way too sensitive about this.", 3),
        ("Fine. Whatever you think is best.", 2),
        ("Must be nice to never have to worry about things like that.", 2),
        ("I really appreciate you taking the time to explain that.", 0),
    ]

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizerFast.from_pretrained(OUTPUT_DIR)
    model     = RobertaForSequenceClassification.from_pretrained(OUTPUT_DIR)
    model.to(device)
    model.eval()

    correct = 0
    print("\n── Stress Test Results ──────────────────────────────────────────")
    for text, expected_id in STRESS:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        pred_id = logits.argmax(-1).item()
        ok = "✅" if pred_id == expected_id else "❌"
        if pred_id == expected_id:
            correct += 1
        print(f"  {ok}  [{ID2LABEL[expected_id]:<22} → {ID2LABEL[pred_id]:<22}]  {text[:60]}")

    acc = correct / len(STRESS)
    print(f"\n  Accuracy: {correct}/{len(STRESS)} = {acc:.0%}  {'(PASS)' if acc >= 0.70 else '(FAIL)'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gaslight Guard classifier")
    parser.add_argument("--epochs",   type=int,  default=4,     help="Number of training epochs (default: 4)")
    parser.add_argument("--dry-run",  action="store_true",       help="Run one step only (sanity check)")
    parser.add_argument("--eval-only",action="store_true",       help="Skip training; only run stress test on saved model")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_stress_test()
    else:
        train(epochs=args.epochs, dry_run=args.dry_run)
        if not args.dry_run:
            print("\n── Running post-training stress test… ──")
            evaluate_stress_test()
