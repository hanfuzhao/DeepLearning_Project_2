"""
Evaluate the fine-tuned Gaslight Guard model on the held-out test set.

Usage:
    python evaluate.py
    python evaluate.py --model-dir ../../model/gaslight-guard
    python evaluate.py --output results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from test_cases import get_test_data

LABEL_NAMES = ["Sincere", "Sarcastic", "Passive-Aggressive", "Gaslighting"]
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "model" / "gaslight-guard"
MAX_LEN = 128


def load_model(model_dir: Path):
    print(f"Loading model from: {model_dir}")
    tokenizer = RobertaTokenizerFast.from_pretrained(str(model_dir))
    model = RobertaForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    return tokenizer, model


def predict(texts, tokenizer, model, device, batch_size=16):
    all_preds = []
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
    return all_preds, all_probs


def run_evaluation(model_dir: Path, output_path: Path | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer, model = load_model(model_dir)
    model.to(device)

    test_data = get_test_data()
    texts      = [t for t, _, _ in test_data]
    true_ids   = [l for _, l, _ in test_data]
    true_names = [n for _, _, n in test_data]

    print(f"Test set: {len(texts)} examples\n")

    pred_ids, pred_probs = predict(texts, tokenizer, model, device)

    # ── Per-class report ──────────────────────────────────────────────────────
    report = classification_report(
        true_ids, pred_ids,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print("=" * 65)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 65)
    print(f"\n{'Label':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 65)
    for name in LABEL_NAMES:
        r = report[name]
        flag = "  [!]" if r["f1-score"] < 0.4 else "     "
        print(f"{flag}{name:<19} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1-score']:>10.3f} {int(r['support']):>10}")
    print("-" * 65)
    print(f"{'Macro avg':<24} {report['macro avg']['precision']:>10.3f} {report['macro avg']['recall']:>10.3f} {report['macro avg']['f1-score']:>10.3f}")
    print(f"{'Accuracy':<24} {'':>10} {'':>10} {report['accuracy']:>10.3f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(true_ids, pred_ids)
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    header = f"{'':>24}" + "".join(f"{n[:8]:>12}" for n in LABEL_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:<22}" + "".join(f"{v:>12}" for v in row))

    # ── Per-example detail ────────────────────────────────────────────────────
    print("\n\nPer-example results:")
    print("-" * 90)
    correct = 0
    per_example = []
    for i, (text, true_id, true_name) in enumerate(test_data):
        pred_id   = pred_ids[i]
        pred_name = LABEL_NAMES[pred_id]
        conf      = pred_probs[i][pred_id]
        ok        = pred_id == true_id
        if ok:
            correct += 1
        marker = "OK " if ok else "ERR"
        print(f"  [{marker}] true={true_name:<22} pred={pred_name:<22} conf={conf:.2f}  {text[:55]}")
        per_example.append({
            "text": text,
            "true_label": true_name,
            "pred_label": pred_name,
            "confidence": round(float(conf), 4),
            "correct": ok,
        })

    overall_acc = correct / len(test_data)
    macro_f1    = report["macro avg"]["f1-score"]
    print(f"\nOverall accuracy : {correct}/{len(test_data)} = {overall_acc:.1%}")
    print(f"Macro F1         : {macro_f1:.3f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if output_path:
        result = {
            "overall_accuracy": round(overall_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": {
                name: {
                    "precision": round(report[name]["precision"], 4),
                    "recall":    round(report[name]["recall"],    4),
                    "f1":        round(report[name]["f1-score"],  4),
                    "support":   int(report[name]["support"]),
                }
                for name in LABEL_NAMES
            },
            "confusion_matrix": cm.tolist(),
            "per_example": per_example,
        }
        output_path.write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to: {output_path}")

    return overall_acc, macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gaslight Guard on held-out test set")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output",    type=Path, default=None, help="Save results as JSON (e.g. results.json)")
    args = parser.parse_args()

    if not (args.model_dir / "config.json").exists():
        print(f"[ERROR] No model found at {args.model_dir}")
        print("  Run 'python train.py' first.")
        sys.exit(1)

    run_evaluation(args.model_dir, args.output)
