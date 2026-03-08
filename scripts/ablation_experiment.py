"""
Experiment: Preprocessing Pipeline & N-gram Ablation Study.

Motivation
----------
Sarcasm in news headlines relies on stylistic signals that standard NLP
preprocessing can destroy.  Specifically:

  - Capitalisation carries emphasis ("Oh SURE that worked"):
    lowercasing may erase a discriminative signal.
  - Punctuation marks tone and rhetorical intent ("Really?!"):
    stripping all non-alpha characters may remove these cues.
  - URLs are structural noise for meaning but their presence may correlate
    with topic (Onion vs HuffPost): removing them could hurt or help.

Additionally, the choice of TF-IDF n-gram range determines whether the model
can capture multi-word irony cues ("totally normal", "for once in his life"):
  - Unigrams only (1,1): fastest, ignores phrase-level signals.
  - Unigrams + bigrams (1,2): current default.
  - Unigrams through trigrams (1,3): richer but sparser.

Experimental Plan
-----------------
Two ablation groups are run:

  Group A — Preprocessing Variants (5 configs, TF-IDF ngram=(1,2) fixed)
    full_pipeline       : current production pipeline (control)
    no_lowercase        : skip lowercasing step
    keep_punctuation    : retain !, ?, ., , — only strip digits and symbols
    no_url_removal      : skip URL stripping step
    raw_text            : only whitespace normalisation, no other cleaning

  Group B — N-gram Variants (3 configs, full preprocessing fixed)
    ngram_unigram       : ngram_range=(1,1)
    ngram_bigram        : ngram_range=(1,2)  ← same as full_pipeline, reference
    ngram_trigram       : ngram_range=(1,3)

All variants are evaluated on the FIXED held-out test set so results are
directly comparable.  DistilBERT is intentionally excluded: its WordPiece
tokenizer performs its own internal normalisation (lowercasing via
DistilBertTokenizerFast's do_lower_case=True), making text preprocessing
largely redundant and masking the effect we are trying to measure.  This is
itself a finding noted in the interpretation section of the report.

Results
-------
Outputs are saved to data/outputs/experiment/ablation/:
    ablation_results.csv        — numeric results table (all variants × metrics)
    ablation_f1_bar.png         — grouped bar chart: F1 per variant per model
    ablation_metrics_heatmap.png — heatmap: metric × variant for TF-IDF + LR
"""

import logging
import os
import re
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from scripts.model import NaiveBaselineModel, OUTPUTS_DIR, TFIDFClassifierModel
from scripts.make_dataset import load_raw_data, RAW_DATA_PATH, split_data
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

ABLATION_DIR = os.path.join(OUTPUTS_DIR, "experiment", "ablation")
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing variant definitions
# ─────────────────────────────────────────────────────────────────────────────

def _clean_full_pipeline(text: str) -> str:
    """Current production pipeline (control variant).

    Steps: lowercase → strip URLs → remove non-alpha except apostrophes
           → collapse whitespace.
    """
    text = str(text).lower()                              # step 1: lowercase
    text = re.sub(r"http\S+|www\S+", "", text)            # step 2: URLs
    text = re.sub(r"[^a-z\s']", " ", text)               # step 3: non-alpha
    text = re.sub(r"\s+", " ", text).strip()              # step 4: whitespace
    return text


def _clean_no_lowercase(text: str) -> str:
    """Skip lowercasing; preserve original casing.

    Hypothesis: sarcastic headlines may use ALL-CAPS for ironic emphasis
    (e.g. 'SHOCKING: Man Does Thing').  Lowercasing destroys this signal.
    """
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)            # step 2: URLs
    text = re.sub(r"[^a-zA-Z\s']", " ", text)            # step 3: non-alpha (case-aware)
    text = re.sub(r"\s+", " ", text).strip()              # step 4: whitespace
    return text


def _clean_keep_punctuation(text: str) -> str:
    """Retain key punctuation marks (!, ?, ., ,) after lowercasing.

    Hypothesis: punctuation encodes tone — "Really?!" versus "Really."
    carries different ironic weight that unigram TF-IDF could exploit via
    punctuation-augmented tokens or character n-grams.
    """
    text = str(text).lower()                              # step 1: lowercase
    text = re.sub(r"http\S+|www\S+", "", text)            # step 2: URLs
    # Keep letters, spaces, apostrophes, and sentence-level punctuation
    text = re.sub(r"[^a-z\s'!?,.]", " ", text)           # step 3: relaxed
    text = re.sub(r"\s+", " ", text).strip()              # step 4: whitespace
    return text


def _clean_no_url_removal(text: str) -> str:
    """Skip URL removal; treat URLs as regular tokens.

    Hypothesis: URL domains (e.g. 'theonion.com' versus 'huffpost.com')
    may leak source identity, but URL sub-tokens could also be noise.
    This variant measures the net effect.
    """
    text = str(text).lower()                              # step 1: lowercase
    # step 2 SKIPPED — URLs retained
    text = re.sub(r"[^a-z\s']", " ", text)               # step 3: non-alpha
    text = re.sub(r"\s+", " ", text).strip()              # step 4: whitespace
    return text


def _clean_raw_text(text: str) -> str:
    """Minimal cleaning: only collapse whitespace.

    Serves as an upper bound on how much information the raw string contains
    and a lower bound on what structured preprocessing buys us.
    """
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()              # step 4 only
    return text


# Map variant name → cleaning function
PREPROCESSING_VARIANTS: Dict[str, Callable[[str], str]] = {
    "full_pipeline":    _clean_full_pipeline,
    "no_lowercase":     _clean_no_lowercase,
    "keep_punctuation": _clean_keep_punctuation,
    "no_url_removal":   _clean_no_url_removal,
    "raw_text":         _clean_raw_text,
}

# N-gram variants: applied with full_pipeline cleaning
NGRAM_VARIANTS: Dict[str, Tuple[int, int]] = {
    "ngram_unigram":  (1, 1),
    "ngram_bigram":   (1, 2),   # matches full_pipeline — reference point
    "ngram_trigram":  (1, 3),
}


# ─────────────────────────────────────────────────────────────────────────────
# Feature building helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_tfidf_features(
    train_texts: List[str],
    val_texts: List[str],
    test_texts: List[str],
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Fit a TF-IDF vectorizer on train_texts and transform all three splits.

    Uses the same hyperparameters as build_features.py to ensure a fair
    comparison: only the cleaning function and/or ngram_range differs.

    Args:
        train_texts: Cleaned training headlines.
        val_texts:   Cleaned validation headlines (not used for fitting).
        test_texts:  Cleaned test headlines (fixed evaluation set).
        ngram_range: Token n-gram range passed to TfidfVectorizer.

    Returns:
        Tuple (X_train, X_val, X_test) of sparse TF-IDF matrices.
    """
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        min_df=2,
    )
    X_train = vectorizer.fit_transform(pd.Series(train_texts).fillna(""))
    X_val   = vectorizer.transform(pd.Series(val_texts).fillna(""))
    X_test  = vectorizer.transform(pd.Series(test_texts).fillna(""))
    logger.info(
        "TF-IDF fitted (ngram=%s): vocab=%d | X_train=%s | X_test=%s",
        ngram_range, len(vectorizer.vocabulary_), X_train.shape, X_test.shape,
    )
    return X_train, X_val, X_test


# ─────────────────────────────────────────────────────────────────────────────
# Single-variant runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_variant(
    variant_name: str,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    clean_fn: Callable[[str], str],
    ngram_range: Tuple[int, int],
) -> List[Dict]:
    """Train and evaluate Naive Baseline + TF-IDF+LR for one ablation variant.

    Args:
        variant_name: String identifier for this config (used in output rows).
        train_raw:    Training DataFrame with 'headline' and 'is_sarcastic' columns
                      (raw, before any cleaning).
        val_raw:      Validation DataFrame (same schema).
        test_raw:     Test DataFrame (same schema).
        clean_fn:     Cleaning function to apply to headline text.
        ngram_range:  TF-IDF n-gram range tuple.

    Returns:
        List of result dicts, one per model evaluated.
    """
    # Apply cleaning function
    train_texts = train_raw["headline"].apply(clean_fn).tolist()
    val_texts   = val_raw["headline"].apply(clean_fn).tolist()
    test_texts  = test_raw["headline"].apply(clean_fn).tolist()

    y_train = train_raw["is_sarcastic"].values.astype(int)
    y_test  = test_raw["is_sarcastic"].values.astype(int)

    # Build features
    X_train, X_val, X_test = _build_tfidf_features(
        train_texts, val_texts, test_texts, ngram_range=ngram_range
    )

    records = []

    # ── Naive Baseline ────────────────────────────────────────────────────────
    # Note: NaiveBaseline is deterministic given class distribution, which is
    # identical across variants (same splits, same labels). Its scores act as
    # a sanity-check anchor — they SHOULD be constant across variants.
    naive = NaiveBaselineModel()
    naive.fit(X_train, y_train)
    m = naive.evaluate(X_test, y_test, f"naive|{variant_name}")
    records.append({
        "variant": variant_name,
        "model": "Naive Baseline",
        **m,
    })

    # ── TF-IDF + Logistic Regression ─────────────────────────────────────────
    tfidf_clf = TFIDFClassifierModel()
    tfidf_clf.fit(X_train, y_train, cv=5)
    m = tfidf_clf.evaluate(X_test, y_test, f"tfidf_lr|{variant_name}")
    records.append({
        "variant": variant_name,
        "model": "TF-IDF + LR",
        **m,
    })

    logger.info("Variant '%s' complete.", variant_name)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_f1_bar(df: pd.DataFrame) -> None:
    """Grouped bar chart: F1 per variant, grouped by model.

    Variants are sorted descending by TF-IDF+LR F1 so the best preprocessing
    choice is immediately visible.

    Args:
        df: Results DataFrame with columns: variant, model, f1.
    """
    tfidf_order = (
        df[df["model"] == "TF-IDF + LR"]
        .sort_values("f1", ascending=False)["variant"]
        .tolist()
    )

    models = df["model"].unique().tolist()
    x = np.arange(len(tfidf_order))
    width = 0.35
    colors = {"Naive Baseline": "#4C72B0", "TF-IDF + LR": "#DD8452"}

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, model_name in enumerate(models):
        subset = df[df["model"] == model_name].set_index("variant")
        values = [subset.loc[v, "f1"] if v in subset.index else 0.0
                  for v in tfidf_order]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=model_name,
            color=colors.get(model_name, "#999"),
            alpha=0.85,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(tfidf_order, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(
        "Ablation Study: F1 by Preprocessing / N-gram Variant",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.axhline(
        df[(df["model"] == "TF-IDF + LR") & (df["variant"] == "full_pipeline")]["f1"].values[0],
        color="#DD8452", linestyle="--", linewidth=1.2, alpha=0.6,
        label="full_pipeline baseline (TF-IDF + LR)",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path = os.path.join(ABLATION_DIR, "ablation_f1_bar.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("F1 bar chart saved to '%s'.", save_path)


def _plot_metrics_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of all metrics × all variants for TF-IDF + LR only.

    Gives a compact view of whether preprocessing trade-offs affect
    precision/recall asymmetrically (relevant for a cyberbullying setting
    where false negatives may be more costly than false positives).

    Args:
        df: Full results DataFrame.
    """
    metrics_cols = ["accuracy", "precision", "recall", "f1", "auc"]
    tfidf_df = df[df["model"] == "TF-IDF + LR"].copy()
    tfidf_df = tfidf_df.sort_values("f1", ascending=False)

    pivot = tfidf_df.set_index("variant")[metrics_cols]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot,
        annot=True, fmt=".3f",
        cmap="YlGnBu",
        vmin=0.5, vmax=1.0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "TF-IDF + LR: Metrics Heatmap by Ablation Variant\n"
        "(sorted by F1 descending)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Variant", fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(ABLATION_DIR, "ablation_metrics_heatmap.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Metrics heatmap saved to '%s'.", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_experiment(raw_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Run the full preprocessing and n-gram ablation study.

    Loads raw (uncleaned) data, applies each variant's cleaning function,
    rebuilds TF-IDF features from scratch, trains and evaluates both
    non-BERT models on the fixed test set.

    Why load raw data?
        The processed CSVs in data/processed/ have already had the production
        clean_text() applied. To isolate the effect of individual preprocessing
        steps we must start from the original scraped headlines.

    Args:
        raw_path: Path to the raw JSONL dataset.

    Returns:
        DataFrame with all ablation results across variants and models.
    """
    os.makedirs(ABLATION_DIR, exist_ok=True)

    # ── Load raw (uncleaned) data ─────────────────────────────────────────────
    logger.info("Loading raw data from '%s'.", raw_path)
    df_raw = load_raw_data(raw_path)
    # Validate expected columns
    assert "headline" in df_raw.columns and "is_sarcastic" in df_raw.columns, (
        "Raw data must have 'headline' and 'is_sarcastic' columns. "
        f"Found: {df_raw.columns.tolist()}"
    )
    df_raw = df_raw[["headline", "is_sarcastic"]].dropna().drop_duplicates(
        subset=["headline"]
    ).reset_index(drop=True)
    logger.info("Raw records after dedup: %d", len(df_raw))

    # ── Stratified split (mirrors prepare_dataset() proportions exactly) ─────
    # We split on the raw data to ensure all variants see the same train/val/test
    # indices — the only thing that changes between variants is the cleaning fn.
    df_raw["label"] = df_raw["is_sarcastic"].astype(int)
    train_raw, val_raw, test_raw = split_data(df_raw.rename(
        columns={"headline": "text"}
    ))
    # Re-attach 'headline' column for apply() calls below
    train_raw = train_raw.rename(columns={"text": "headline"})
    val_raw   = val_raw.rename(columns={"text": "headline"})
    test_raw  = test_raw.rename(columns={"text": "headline"})
    # Ensure is_sarcastic column is present (split_data uses 'label')
    train_raw["is_sarcastic"] = train_raw["label"]
    val_raw["is_sarcastic"]   = val_raw["label"]
    test_raw["is_sarcastic"]  = test_raw["label"]

    logger.info(
        "Raw splits → train: %d | val: %d | test: %d",
        len(train_raw), len(val_raw), len(test_raw),
    )

    all_records: List[Dict] = []

    # ── Group A: Preprocessing Variants (n-gram=(1,2) fixed) ─────────────────
    logger.info("=" * 60)
    logger.info("GROUP A: Preprocessing Variants  [ngram=(1,2) fixed]")
    logger.info("=" * 60)
    for variant_name, clean_fn in PREPROCESSING_VARIANTS.items():
        logger.info("── Variant: %s", variant_name)
        records = _run_one_variant(
            variant_name=variant_name,
            train_raw=train_raw,
            val_raw=val_raw,
            test_raw=test_raw,
            clean_fn=clean_fn,
            ngram_range=(1, 2),
        )
        for r in records:
            r["group"] = "preprocessing"
        all_records.extend(records)

    # ── Group B: N-gram Variants (full_pipeline cleaning fixed) ──────────────
    logger.info("=" * 60)
    logger.info("GROUP B: N-gram Variants  [full_pipeline cleaning fixed]")
    logger.info("=" * 60)
    for variant_name, ngram_range in NGRAM_VARIANTS.items():
        # Skip ngram_bigram if full_pipeline already covers it (same result)
        if variant_name == "ngram_bigram":
            logger.info(
                "Skipping ngram_bigram — identical to full_pipeline in Group A."
            )
            # Copy full_pipeline result and re-label as ngram_bigram for completeness
            fp_rows = [r for r in all_records if r["variant"] == "full_pipeline"]
            for r in fp_rows:
                relabeled = {**r, "variant": "ngram_bigram", "group": "ngram"}
                all_records.append(relabeled)
            continue

        logger.info("── Variant: %s  (ngram_range=%s)", variant_name, ngram_range)
        records = _run_one_variant(
            variant_name=variant_name,
            train_raw=train_raw,
            val_raw=val_raw,
            test_raw=test_raw,
            clean_fn=_clean_full_pipeline,
            ngram_range=ngram_range,
        )
        for r in records:
            r["group"] = "ngram"
        all_records.extend(records)

    # ── Compile results ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_records)
    # Round for readability
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc"]
    results_df[metric_cols] = results_df[metric_cols].round(4)

    csv_path = os.path.join(ABLATION_DIR, "ablation_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info("Ablation results saved to '%s'.", csv_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_f1_bar(results_df)
    _plot_metrics_heatmap(results_df)

    # ── Console summary ───────────────────────────────────────────────────────
    summary = (
        results_df[results_df["model"] == "TF-IDF + LR"]
        [["group", "variant", "f1", "accuracy", "auc"]]
        .sort_values(["group", "f1"], ascending=[True, False])
    )
    logger.info(
        "\nAblation Summary (TF-IDF + LR only):\n%s", summary.to_string(index=False)
    )

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap Significance Test: ngram_unigram vs ngram_bigram
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_f1_diff(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 10_000,
    random_state: int = RANDOM_STATE,
) -> Dict:
    """Paired bootstrap test for the difference in F1 between two models.

    Resamples the test set with replacement n_bootstrap times and computes
    the F1 difference (model_b - model_a) on each resample. The p-value is
    the proportion of resamples where the difference is <= 0 (one-tailed:
    H1: bigram F1 > unigram F1).

    Args:
        y_true:      Ground-truth binary labels on the test set.
        y_pred_a:    Predictions from model A (ngram_unigram).
        y_pred_b:    Predictions from model B (ngram_bigram).
        n_bootstrap: Number of bootstrap resamples.
        random_state: Seed for reproducibility.

    Returns:
        Dict with keys: f1_a, f1_b, observed_diff, p_value, ci_lower, ci_upper.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    observed_f1_a = f1_score(y_true, y_pred_a, zero_division=0)
    observed_f1_b = f1_score(y_true, y_pred_b, zero_division=0)
    observed_diff = observed_f1_b - observed_f1_a

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        f1_a_boot = f1_score(y_true[idx], y_pred_a[idx], zero_division=0)
        f1_b_boot = f1_score(y_true[idx], y_pred_b[idx], zero_division=0)
        diffs[i] = f1_b_boot - f1_a_boot

    # One-tailed p-value: proportion of resamples where bigram <= unigram
    p_value = float(np.mean(diffs <= 0))

    # 95% confidence interval for the difference
    ci_lower, ci_upper = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    return {
        "f1_unigram":    round(observed_f1_a, 4),
        "f1_bigram":     round(observed_f1_b, 4),
        "observed_diff": round(observed_diff, 4),
        "p_value":       round(p_value, 4),
        "ci_lower":      round(ci_lower, 4),
        "ci_upper":      round(ci_upper, 4),
        "n_bootstrap":   n_bootstrap,
        "significant":   p_value < 0.05,
    }


def _plot_bootstrap_dist(diffs: np.ndarray, observed_diff: float, p_value: float) -> None:
    """Histogram of bootstrap F1 differences with observed value annotated.

    Args:
        diffs:         Array of (bigram F1 - unigram F1) from each resample.
        observed_diff: The actual observed difference on the full test set.
        p_value:       Computed one-tailed p-value.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(diffs, bins=60, color="#DD8452", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(observed_diff, color="#2d2d2d", linewidth=2.0, linestyle="-",
               label=f"Observed Δ = {observed_diff:.4f}")
    ax.axvline(0, color="#C44E52", linewidth=1.5, linestyle="--", label="Δ = 0 (null)")

    ci_lower, ci_upper = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
    ax.axvspan(ci_lower, ci_upper, alpha=0.12, color="#4C72B0", label="95% CI")

    ax.set_xlabel("F1 Difference (bigram − unigram)", fontsize=12)
    ax.set_ylabel("Bootstrap Frequency", fontsize=12)
    ax.set_title(
        f"Bootstrap Significance Test: bigram vs unigram F1\n"
        f"p = {p_value:.4f}  |  {'Significant at α=0.05' if p_value < 0.05 else 'Not significant at α=0.05'}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path = os.path.join(ABLATION_DIR, "bootstrap_ngram_significance.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Bootstrap plot saved to '%s'.", save_path)


def run_ngram_significance_test(raw_path: str = RAW_DATA_PATH) -> Dict:
    """Re-train unigram and bigram variants and run a bootstrap significance test.

    Independently re-runs only the two n-gram variants of interest on the same
    fixed test split used in the main ablation, then tests whether the observed
    F1 improvement from unigram → bigram is statistically reliable.

    Args:
        raw_path: Path to the raw JSONL dataset.

    Returns:
        Dict with bootstrap test statistics.
    """
    os.makedirs(ABLATION_DIR, exist_ok=True)

    # ── Reconstruct the same splits as the main ablation ─────────────────────
    logger.info("Loading raw data for significance test.")
    df_raw = load_raw_data(raw_path)
    df_raw = df_raw[["headline", "is_sarcastic"]].dropna().drop_duplicates(
        subset=["headline"]
    ).reset_index(drop=True)

    df_raw["label"] = df_raw["is_sarcastic"].astype(int)
    train_raw, val_raw, test_raw = split_data(df_raw.rename(columns={"headline": "text"}))
    train_raw = train_raw.rename(columns={"text": "headline"})
    val_raw   = val_raw.rename(columns={"text": "headline"})
    test_raw  = test_raw.rename(columns={"text": "headline"})
    train_raw["is_sarcastic"] = train_raw["label"]
    val_raw["is_sarcastic"]   = val_raw["label"]
    test_raw["is_sarcastic"]  = test_raw["label"]

    y_test = test_raw["is_sarcastic"].values.astype(int)

    predictions = {}
    for variant_name, ngram_range in [("ngram_unigram", (1, 1)), ("ngram_bigram", (1, 2))]:
        logger.info("── Training %s for significance test.", variant_name)

        train_texts = train_raw["headline"].apply(_clean_full_pipeline).tolist()
        val_texts   = val_raw["headline"].apply(_clean_full_pipeline).tolist()
        test_texts  = test_raw["headline"].apply(_clean_full_pipeline).tolist()

        X_train, _, X_test = _build_tfidf_features(
            train_texts, val_texts, test_texts, ngram_range=ngram_range
        )
        clf = TFIDFClassifierModel()
        clf.fit(X_train, train_raw["is_sarcastic"].values.astype(int), cv=5)
        predictions[variant_name] = clf.predict(X_test)

    # ── Bootstrap test ────────────────────────────────────────────────────────
    logger.info("Running bootstrap test (n=10,000 resamples)…")
    results = _bootstrap_f1_diff(
        y_true=y_test,
        y_pred_a=predictions["ngram_unigram"],
        y_pred_b=predictions["ngram_bigram"],
    )

    # Recompute diffs array for plotting (same seed → identical)
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_test)
    diffs = []
    for _ in range(10_000):
        idx = rng.integers(0, n, size=n)
        f1_b = f1_score(y_test[idx], predictions["ngram_bigram"][idx], zero_division=0)
        f1_a = f1_score(y_test[idx], predictions["ngram_unigram"][idx], zero_division=0)
        diffs.append(f1_b - f1_a)
    diffs = np.array(diffs)
    _plot_bootstrap_dist(diffs, results["observed_diff"], results["p_value"])

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = os.path.join(ABLATION_DIR, "bootstrap_significance_results.csv")
    pd.DataFrame([results]).to_csv(results_path, index=False)

    logger.info(
        "\nBootstrap Significance Test Results:\n"
        "  unigram F1:    %.4f\n"
        "  bigram  F1:    %.4f\n"
        "  Δ observed:    %.4f\n"
        "  95%% CI:       [%.4f, %.4f]\n"
        "  p-value:       %.4f\n"
        "  Significant:   %s",
        results["f1_unigram"], results["f1_bigram"],
        results["observed_diff"],
        results["ci_lower"], results["ci_upper"],
        results["p_value"],
        results["significant"],
    )
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Run full ablation + significance test
    run_ablation_experiment()
    run_ngram_significance_test()