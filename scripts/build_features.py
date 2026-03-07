"""
Feature engineering script for classical ML models.

Fits a TF-IDF vectorizer on the training split and transforms all three
splits (train / val / test). The resulting sparse matrices and the fitted
vectorizer are saved to disk for use by the classical ML model.

TF-IDF design choices:
    - Unigrams + bigrams (ngram_range=(1,2)) capture both single-word signals
      (e.g., "ridiculous") and two-word irony cues (e.g., "totally normal").
    - sublinear_tf=True dampens the effect of high term frequencies, which
      reduces the dominance of common filler words.
    - min_df=2 removes tokens that appear in fewer than 2 documents, cutting
      noise from typos and rare proper nouns.
    - max_features=50_000 limits memory usage while retaining a rich vocabulary.
"""

import logging
import os
from typing import Tuple

import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")

TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2


def load_splits(
    processed_dir: str = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train / val / test CSV splits from disk.

    Args:
        processed_dir: Directory containing train.csv, val.csv, test.csv.

    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    logger.info(
        "Loaded splits → train: %d | val: %d | test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def fit_tfidf(
    train: pd.DataFrame,
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on training text.

    Args:
        train: Training DataFrame with a 'text' column.
        max_features: Maximum number of token features.
        ngram_range: Lower and upper boundary of n-gram sizes.
        min_df: Minimum document frequency for a token to be included.

    Returns:
        Fitted TfidfVectorizer instance.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        min_df=min_df,
    )
    vectorizer.fit(train["text"].fillna(""))
    logger.info(
        "TF-IDF fitted: vocabulary size = %d", len(vectorizer.vocabulary_)
    )
    return vectorizer


def transform_splits(
    vectorizer: TfidfVectorizer,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Apply a fitted vectorizer to all three splits.

    Args:
        vectorizer: A fitted TfidfVectorizer.
        train, val, test: DataFrames with a 'text' column.

    Returns:
        Tuple of sparse matrices (X_train, X_val, X_test).
    """
    X_train = vectorizer.transform(train["text"].fillna(""))
    X_val = vectorizer.transform(val["text"].fillna(""))
    X_test = vectorizer.transform(test["text"].fillna(""))
    logger.info(
        "TF-IDF transform done → X_train: %s | X_val: %s | X_test: %s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )
    return X_train, X_val, X_test


def save_features(
    X_train: sp.csr_matrix,
    X_val: sp.csr_matrix,
    X_test: sp.csr_matrix,
    vectorizer: TfidfVectorizer,
    processed_dir: str = PROCESSED_DATA_DIR,
    models_dir: str = MODELS_DIR,
) -> None:
    """Persist sparse feature matrices and the fitted vectorizer to disk.

    Args:
        X_train, X_val, X_test: Sparse feature matrices.
        vectorizer: Fitted TfidfVectorizer to save.
        processed_dir: Directory for .npz feature files.
        models_dir: Directory for the vectorizer artifact.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    sp.save_npz(os.path.join(processed_dir, "X_train_tfidf.npz"), X_train)
    sp.save_npz(os.path.join(processed_dir, "X_val_tfidf.npz"), X_val)
    sp.save_npz(os.path.join(processed_dir, "X_test_tfidf.npz"), X_test)

    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vectorizer_path)

    logger.info("Features saved to '%s', vectorizer to '%s'.", processed_dir, vectorizer_path)


def build_features(
    processed_dir: str = PROCESSED_DATA_DIR,
    models_dir: str = MODELS_DIR,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, TfidfVectorizer]:
    """End-to-end TF-IDF feature building pipeline.

    Args:
        processed_dir: Directory with CSV splits (input) and .npz files (output).
        models_dir: Directory where the vectorizer artifact is saved.

    Returns:
        Tuple of (X_train, X_val, X_test, vectorizer).
    """
    train, val, test = load_splits(processed_dir)
    vectorizer = fit_tfidf(train)
    X_train, X_val, X_test = transform_splits(vectorizer, train, val, test)
    save_features(X_train, X_val, X_test, vectorizer, processed_dir, models_dir)
    return X_train, X_val, X_test, vectorizer


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    build_features()
