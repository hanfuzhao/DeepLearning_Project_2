"""
Dataset preparation script for sarcasm detection.

Downloads the News Headlines Dataset for Sarcasm Detection (Misra, 2019) and
produces clean train / validation / test splits.

Dataset source:
    https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection

Citation:
    Misra, R., & Arora, P. (2019). Sarcasm Detection using News Headlines Dataset.
    AI Open, 1, 1-9.

This dataset contains ~28,000 news headlines labeled as sarcastic (from The Onion)
or non-sarcastic (from HuffPost). In the context of this project, sarcasm detection
serves as a proxy for detecting subtly hostile or ironic language patterns that
frequently appear in cyberbullying and online harassment.
"""

import json
import logging
import os
import re
from typing import Tuple

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DATASET_URL = (
    "https://raw.githubusercontent.com/rishabhmisra/"
    "News-Headlines-Dataset-For-Sarcasm-Detection/master/"
    "Sarcasm_Headlines_Dataset.json"
)
RAW_DATA_PATH = os.path.join("data", "raw", "sarcasm_headlines.json")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def download_dataset(url: str = DATASET_URL, save_path: str = RAW_DATA_PATH) -> None:
    """Download the raw sarcasm headlines JSONL file from GitHub.

    Args:
        url: Direct download URL for the dataset.
        save_path: Local path where the raw file will be saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        logger.info("Raw dataset already exists at '%s', skipping download.", save_path)
        return

    logger.info("Downloading dataset from %s …", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write(response.text)
    logger.info("Dataset saved to '%s'.", save_path)


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Parse the JSONL raw data file into a DataFrame.

    Args:
        path: Path to the JSONL file (one JSON object per line).

    Returns:
        DataFrame with columns: headline, is_sarcastic, article_link.
    """
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    logger.info("Loaded %d records from '%s'.", len(df), path)
    return df


def clean_text(text: str) -> str:
    """Normalise a headline string for downstream modeling.

    Steps:
        1. Lowercase.
        2. Strip URLs (present in some scraped headlines).
        3. Remove non-alphabetic characters except apostrophes (preserves
           contractions that carry sentiment signal).
        4. Collapse whitespace.

    Args:
        text: Raw headline string.

    Returns:
        Cleaned string.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split (70 / 15 / 15 by default).

    Args:
        df: Full DataFrame with 'label' column for stratification.
        test_size: Proportion of data reserved for the test set.
        val_size: Proportion of data reserved for the validation set.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train, val, test) DataFrames.
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    adjusted_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val["label"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def prepare_dataset(
    raw_path: str = RAW_DATA_PATH,
    processed_dir: str = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end pipeline: download → clean → split → save.

    Args:
        raw_path: Where to store / read the raw JSONL file.
        processed_dir: Directory for the processed CSV splits.

    Returns:
        Tuple of (train, val, test) DataFrames with columns: text, label.
    """
    download_dataset(save_path=raw_path)

    df_raw = load_raw_data(raw_path)
    df = df_raw.rename(columns={"headline": "text", "is_sarcastic": "label"})
    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].apply(clean_text)
    df["label"] = df["label"].astype(int)

    # Remove duplicates and empty strings produced by cleaning
    n_before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info("Removed %d duplicate headlines.", n_before - len(df))
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    train, val, test = split_data(df)

    os.makedirs(processed_dir, exist_ok=True)
    train.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    logger.info(
        "Splits saved → train: %d | val: %d | test: %d",
        len(train),
        len(val),
        len(test),
    )
    label_dist = train["label"].value_counts(normalize=True)
    logger.info("Train label distribution:\n%s", label_dist.to_string())

    return train, val, test


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    prepare_dataset()
