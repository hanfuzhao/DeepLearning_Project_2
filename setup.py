"""
Full pipeline setup script.

Runs every stage in sequence so results can be reproduced from scratch:
    1. Download and prepare the sarcasm headlines dataset.
    2. Build TF-IDF feature matrices.
    3. Train all three models (Naive Baseline, TF-IDF + LR, DistilBERT).
    4. Run the training-size sensitivity experiment.

Usage:
    python setup.py

Expected runtime (CPU-only): ~30–60 minutes, dominated by DistilBERT fine-tuning.
With a GPU the DistilBERT stage drops to ~5–10 minutes.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Execute the full modeling pipeline end-to-end."""
    import pandas as pd
    from scripts.make_dataset import prepare_dataset
    from scripts.build_features import build_features
    from scripts.model import train_all_models
    from scripts.experiment import run_sensitivity_experiment

    logger.info("━━━  Step 1 / 4 — Dataset preparation  ━━━")
    prepare_dataset()

    logger.info("━━━  Step 2 / 4 — Feature engineering  ━━━")
    build_features()

    logger.info("━━━  Step 3 / 4 — Model training & evaluation  ━━━")
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")
    results = train_all_models(train, val, test)

    logger.info("━━━  Step 4 / 4 — Sensitivity experiment  ━━━")
    run_sensitivity_experiment()

    logger.info("━━━  All steps complete!  ━━━")
    logger.info("Model artifacts  → models/")
    logger.info("Evaluation plots → data/outputs/")
    logger.info("Experiment plots → data/outputs/experiment/")

    # Print final summary
    print("\n" + "=" * 55)
    print("  FINAL TEST-SET RESULTS")
    print("=" * 55)
    for name, m in results.items():
        print(f"  {name:<22}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
