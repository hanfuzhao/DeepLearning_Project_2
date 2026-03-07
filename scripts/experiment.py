"""
Experiment: Training Set Size Sensitivity Analysis.

Motivation
----------
In real-world cyberbullying detection systems, labelled data is expensive to
collect and often scarce.  Understanding how each model's performance degrades
as the labelled pool shrinks informs:
    (a) how much annotation effort is truly necessary before deploying,
    (b) which model architecture is most data-efficient, and
    (c) whether the dataset is large enough to saturate deep learning capacity.

Experimental Plan
-----------------
Each of the three models (Naive Baseline, TF-IDF + LR, DistilBERT) is trained
on stratified subsamples of the training set at fractions:

    10%, 20%, 30%, 50%, 75%, 100%

The validation set is used exclusively for DistilBERT early stopping; it is
NOT used to pick the best fraction.  All final evaluations are performed on
the FIXED, held-out test set so that results across fractions are comparable.

DistilBERT is trained for 1 epoch per fraction to keep the experiment
computationally tractable (full fine-tuning per fraction would require ~6×
the training budget of the normal run).  This is a deliberate trade-off noted
in the interpretation section.

Results
-------
Outputs are saved to data/outputs/experiment/:
    sensitivity_results.csv   — numeric results table
    sensitivity_f1.png        — F1 vs. training fraction line chart
    sensitivity_accuracy.png  — accuracy vs. training fraction line chart
"""

import logging
import os
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from scripts.model import (
    MODELS_DIR,
    OUTPUTS_DIR,
    DistilBERTModel,
    NaiveBaselineModel,
    TFIDFClassifierModel,
)

logger = logging.getLogger(__name__)

FRACTIONS: List[float] = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
EXPERIMENT_DIR = os.path.join(OUTPUTS_DIR, "experiment")
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Subsampling helpers
# ─────────────────────────────────────────────────────────────────────────────


def _subsample_sparse(
    X: sp.csr_matrix,
    y: np.ndarray,
    fraction: float,
    random_state: int = RANDOM_STATE,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Return a stratified subsample of a sparse feature matrix.

    Args:
        X: Full sparse feature matrix.
        y: Corresponding labels.
        fraction: Proportion of data to retain (0 < fraction ≤ 1).
        random_state: Reproducibility seed.

    Returns:
        Subsampled (X_sub, y_sub) pair.
    """
    if fraction >= 1.0:
        return X, y

    # train_test_split returns the *complement* first; we keep the 'test' half
    _, X_sub, _, y_sub = train_test_split(
        X,
        y,
        test_size=fraction,
        stratify=y,
        random_state=random_state,
    )
    return X_sub, y_sub


def _subsample_texts(
    texts: List[str],
    labels: List[int],
    fraction: float,
    random_state: int = RANDOM_STATE,
) -> Tuple[List[str], List[int]]:
    """Return a stratified subsample of a text list.

    Args:
        texts: Full list of raw headline strings.
        labels: Corresponding binary labels.
        fraction: Proportion of data to retain.
        random_state: Reproducibility seed.

    Returns:
        Subsampled (texts_sub, labels_sub) pair.
    """
    if fraction >= 1.0:
        return texts, labels

    labels_arr = np.array(labels)
    indices = np.arange(len(texts))
    _, idx_sub, _, _ = train_test_split(
        indices,
        labels_arr,
        test_size=fraction,
        stratify=labels_arr,
        random_state=random_state,
    )
    return [texts[i] for i in idx_sub], labels_arr[idx_sub].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


def _plot_sensitivity(df: pd.DataFrame, metric: str) -> None:
    """Line chart of a given metric vs. training set fraction.

    Args:
        df: Results DataFrame with columns: model, fraction, <metric>.
        metric: Column name to plot on the y-axis ('f1' or 'accuracy').
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = {"Naive Baseline": "s", "TF-IDF + LR": "o", "DistilBERT": "^"}
    colors = {"Naive Baseline": "#4C72B0", "TF-IDF + LR": "#DD8452", "DistilBERT": "#55A868"}

    for model_name, group in df.groupby("model"):
        group_sorted = group.sort_values("fraction")
        ax.plot(
            group_sorted["fraction"] * 100,
            group_sorted[metric],
            marker=markers.get(model_name, "o"),
            color=colors.get(model_name, None),
            linewidth=2,
            markersize=7,
            label=model_name,
        )

    ax.set_xlabel("Training Data Used (%)", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(
        f"Sensitivity Analysis: {metric.upper()} vs. Training Set Size",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_xticks([int(f * 100) for f in FRACTIONS])
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = os.path.join(EXPERIMENT_DIR, f"sensitivity_{metric}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Sensitivity plot (%s) saved to '%s'.", metric, save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────────────────────────────────────


def run_sensitivity_experiment() -> pd.DataFrame:
    """Run the training-size sensitivity experiment for all three models.

    Trains each model at each fraction of the training set and evaluates on
    the fixed test set.  Results are saved as a CSV and visualised as line
    charts.

    Returns:
        DataFrame with columns: model, fraction, n_samples, accuracy, precision,
        recall, f1, auc.
    """
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    processed_dir = os.path.join("data", "processed")
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    X_train_tfidf = sp.load_npz(os.path.join(processed_dir, "X_train_tfidf.npz"))
    X_test_tfidf = sp.load_npz(os.path.join(processed_dir, "X_test_tfidf.npz"))

    y_train = train["label"].values
    y_test = test["label"].values
    train_texts = train["text"].tolist()
    test_texts = test["text"].tolist()
    val_texts = val["text"].tolist()
    val_labels = val["label"].tolist()

    records: List[Dict] = []

    for fraction in FRACTIONS:
        pct_label = f"{int(fraction * 100)}%"
        logger.info("─── Training fraction: %s ───", pct_label)

        # ── Naive Baseline ──────────────────────────────────────────────────
        X_sub, y_sub = _subsample_sparse(X_train_tfidf, y_train, fraction)
        naive = NaiveBaselineModel()
        naive.fit(X_sub, y_sub)
        m = naive.evaluate(X_test_tfidf, y_test, f"naive@{pct_label}")
        records.append(
            {"model": "Naive Baseline", "fraction": fraction, "n_samples": len(y_sub), **m}
        )

        # ── TF-IDF + Logistic Regression ────────────────────────────────────
        tfidf_clf = TFIDFClassifierModel()
        # Fewer CV folds at small data sizes to avoid empty folds
        cv_folds = 3 if fraction <= 0.2 else 5
        tfidf_clf.fit(X_sub, y_sub, cv=cv_folds)
        m = tfidf_clf.evaluate(X_test_tfidf, y_test, f"tfidf_lr@{pct_label}")
        records.append(
            {"model": "TF-IDF + LR", "fraction": fraction, "n_samples": len(y_sub), **m}
        )

        # ── DistilBERT (1 epoch per fraction for tractability) ──────────────
        sub_texts, sub_labels = _subsample_texts(train_texts, y_train.tolist(), fraction)
        bert = DistilBERTModel(
            model_dir=os.path.join(EXPERIMENT_DIR, f"bert_{int(fraction * 100)}pct")
        )
        bert.fit(
            sub_texts,
            sub_labels,
            val_texts,
            val_labels,
            num_epochs=1,
            batch_size=32,
        )
        m = bert.evaluate(test_texts, y_test, f"distilbert@{pct_label}")
        records.append(
            {"model": "DistilBERT", "fraction": fraction, "n_samples": len(sub_labels), **m}
        )

    results_df = pd.DataFrame(records)
    csv_path = os.path.join(EXPERIMENT_DIR, "sensitivity_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info("Sensitivity results saved to '%s'.", csv_path)

    for metric in ("f1", "accuracy"):
        _plot_sensitivity(results_df, metric)

    logger.info("\nSensitivity analysis complete.\n%s", results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    run_sensitivity_experiment()
