"""
Sarcasm Detection — Main Entry Point.

Provides a command-line interface to all pipeline stages:

    python main.py setup          Download data and build TF-IDF features.
    python main.py train          Train all three models and evaluate on test set.
    python main.py experiment     Run training-size sensitivity analysis.
    python main.py predict TEXT   Run inference with the best model (DistilBERT).
    python main.py compare        Print the saved model comparison table.

For a one-command full run from scratch, use setup.py instead.
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ─────────────────────────────────────────────────────────────────────────────


def cmd_setup() -> None:
    """Download the dataset and build TF-IDF feature matrices."""
    from scripts.make_dataset import prepare_dataset
    from scripts.build_features import build_features

    logger.info("Step 1 / 2 — Preparing dataset …")
    prepare_dataset()
    logger.info("Step 2 / 2 — Building TF-IDF features …")
    build_features()
    logger.info("Setup complete.  Run `python main.py train` next.")


def cmd_train() -> None:
    """Train all three models and print a comparison table."""
    import pandas as pd
    from scripts.model import train_all_models

    processed_dir = os.path.join("data", "processed")
    for split in ("train.csv", "val.csv", "test.csv"):
        path = os.path.join(processed_dir, split)
        if not os.path.exists(path):
            logger.error(
                "Missing '%s'. Run `python main.py setup` first.", path
            )
            sys.exit(1)

    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    results = train_all_models(train, val, test)

    print("\n" + "=" * 55)
    print("  TEST-SET RESULTS")
    print("=" * 55)
    col_w = 14
    print(f"{'Model':<22}", end="")
    for m in ("accuracy", "precision", "recall", "f1", "auc"):
        print(f"{m.upper():>{col_w}}", end="")
    print()
    print("-" * 55)
    for model_name, metrics in results.items():
        print(f"{model_name:<22}", end="")
        for m in ("accuracy", "precision", "recall", "f1", "auc"):
            print(f"{metrics[m]:>{col_w}.4f}", end="")
        print()
    print("=" * 55)


def cmd_experiment() -> None:
    """Run the training-set size sensitivity analysis."""
    from scripts.experiment import run_sensitivity_experiment

    logger.info("Starting sensitivity analysis … (this may take a while)")
    results = run_sensitivity_experiment()
    logger.info("Experiment complete.  Results saved to data/outputs/experiment/")


def cmd_predict(text: str) -> None:
    """Run inference on a single raw text string using the best saved model.

    Falls back to TF-IDF + LR if DistilBERT has not been trained yet.

    Args:
        text: Raw headline or social-media post to classify.
    """
    from scripts.model import DistilBERTModel, TFIDFClassifierModel, MODELS_DIR
    from scripts.make_dataset import clean_text
    import joblib
    import scipy.sparse as sp

    bert_path = os.path.join(MODELS_DIR, "distilbert")
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_lr.joblib")

    cleaned = clean_text(text)

    if os.path.exists(bert_path):
        model = DistilBERTModel.load(bert_path)
        prob = float(model.predict_proba([cleaned])[0])
        model_used = "DistilBERT"
    elif os.path.exists(tfidf_path):
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
        clf = TFIDFClassifierModel.load(tfidf_path)
        X = vectorizer.transform([cleaned])
        prob = float(clf.predict_proba(X)[0])
        model_used = "TF-IDF + LR"
    else:
        logger.error("No trained model found.  Run `python main.py train` first.")
        sys.exit(1)

    label = "SARCASTIC" if prob >= 0.5 else "NOT SARCASTIC"
    confidence = prob if prob >= 0.5 else 1 - prob

    print(f"\n  Model:       {model_used}")
    print(f"  Input:       {text!r}")
    print(f"  Prediction:  {label}")
    print(f"  Confidence:  {confidence:.1%}")
    print(f"  P(sarcastic): {prob:.4f}\n")


def cmd_compare() -> None:
    """Print the saved model comparison CSV to stdout."""
    import pandas as pd

    path = os.path.join("data", "outputs", "model_comparison.csv")
    if not os.path.exists(path):
        logger.error("No comparison file found.  Run `python main.py train` first.")
        sys.exit(1)
    df = pd.read_csv(path, index_col=0)
    print("\n" + df.to_string() + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI wiring
# ─────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Sarcasm Detection — Modeling Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("setup", help="Download data and build features")
    subparsers.add_parser("train", help="Train all three models")
    subparsers.add_parser("experiment", help="Run sensitivity analysis")
    subparsers.add_parser("compare", help="Print saved model comparison table")

    predict_parser = subparsers.add_parser("predict", help="Classify a text snippet")
    predict_parser.add_argument("text", type=str, help="Text to classify")

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "setup": cmd_setup,
        "train": cmd_train,
        "experiment": cmd_experiment,
        "compare": cmd_compare,
    }

    if args.command in dispatch:
        dispatch[args.command]()
    elif args.command == "predict":
        cmd_predict(args.text)
