"""
Model definitions and training orchestration for sarcasm detection.

Three model tiers are implemented as independent classes, all sharing the
BaseSarcasmModel interface so evaluation code is never duplicated:

    NaiveBaselineModel     — majority-class classifier (lower-bound reference)
    TFIDFClassifierModel   — TF-IDF sparse features + Logistic Regression
    DistilBERTModel        — fine-tuned DistilBERT for sequence classification

Common interface for every model:
    .fit(...)            train the model
    .predict(X)          return binary predictions (0 = not sarcastic, 1 = sarcastic)
    .predict_proba(X)    return P(sarcastic) as float array
    .evaluate(X, y)      return metrics dict {accuracy, precision, recall, f1, auc}
    .save(path)          persist trained artifact to disk
    .load(path)          (classmethod) restore from disk

Model artifacts saved at:
    models/naive_baseline.joblib
    models/tfidf_lr.joblib
    models/distilbert/

Usage:
    python main.py train
    python scripts/model.py          (runs train_all_models directly)
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

MODELS_DIR = "models"
OUTPUTS_DIR = os.path.join("data", "outputs")


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────────────────────────────────────


class BaseSarcasmModel(ABC):
    """Abstract interface and shared utilities for all sarcasm detection models."""

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        """Train the model on the provided data."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return binary class predictions (0 or 1)."""

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return P(sarcastic) for each sample as a float array in [0, 1]."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model artifact(s) to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseSarcasmModel":
        """Restore a model from a saved artifact on disk."""

    def evaluate(
        self,
        X,
        y_true: np.ndarray,
        split_name: str = "test",
    ) -> Dict[str, float]:
        """Compute standard classification metrics and log them.

        Args:
            X: Feature matrix (sparse) or list of texts, depending on model type.
            y_true: Ground-truth binary labels.
            split_name: Label shown in the log line, e.g. 'train', 'val', 'test'.

        Returns:
            Dict with keys: accuracy, precision, recall, f1, auc.
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_true, y_prob)),
        }

        logger.info(
            "[%s | %s] acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
            type(self).__name__,
            split_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["auc"],
        )
        return metrics

    def plot_confusion_matrix(
        self,
        X,
        y_true: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
    ) -> None:
        """Render and optionally save a confusion matrix heatmap.

        Args:
            X: Feature input passed to predict().
            y_true: Ground-truth labels.
            title: Plot title string.
            save_path: If given, the figure is saved at this path as a PNG.
        """
        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Sarcastic", "Sarcastic"],
            yticklabels=["Not Sarcastic", "Sarcastic"],
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info("Confusion matrix saved to '%s'.", save_path)
        plt.close()

    def error_analysis(
        self,
        X,
        y_true: np.ndarray,
        texts: List[str],
        n: int = 5,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Surface the n most confidently wrong predictions for error analysis.

        Samples are ranked by confidence in the incorrect direction, exposing
        the model's systematic blind spots.

        Args:
            X: Feature input passed to predict() and predict_proba().
            y_true: Ground-truth binary labels.
            texts: Original raw text strings, same order as X / y_true.
            n: Number of errors to return.
            save_path: Optional CSV path for the error table.

        Returns:
            DataFrame with columns: text, true_label, predicted_label, confidence.
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)

        error_mask = y_pred != y_true
        error_df = pd.DataFrame(
            {
                "text": [texts[i] for i in range(len(texts)) if error_mask[i]],
                "true_label": y_true[error_mask],
                "predicted_label": y_pred[error_mask],
                "confidence": y_prob[error_mask],
            }
        )
        error_df["confidence_in_wrong"] = error_df.apply(
            lambda row: row["confidence"]
            if row["predicted_label"] == 1
            else 1.0 - row["confidence"],
            axis=1,
        )
        top_errors = (
            error_df.sort_values("confidence_in_wrong", ascending=False)
            .head(n)
            .drop(columns=["confidence_in_wrong"])
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            top_errors.to_csv(save_path, index=False)
            logger.info("Error analysis saved to '%s'.", save_path)

        return top_errors


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Naive Baseline
# ─────────────────────────────────────────────────────────────────────────────


class NaiveBaselineModel(BaseSarcasmModel):
    """Majority-class classifier used as the naive performance floor.

    Always predicts the most frequent class observed during training.
    Any useful model must outperform this baseline.
    """

    def __init__(self) -> None:
        self._dummy = DummyClassifier(strategy="most_frequent", random_state=42)

    def fit(self, X, y: np.ndarray) -> None:
        """Fit the dummy classifier.

        Args:
            X: Training features (ignored by DummyClassifier).
            y: Binary training labels.
        """
        self._dummy.fit(X, y)
        majority = int(np.bincount(y).argmax())
        logger.info(
            "NaiveBaselineModel — majority class: %d  (%.1f%% of train)",
            majority,
            100.0 * np.mean(y == majority),
        )

    def predict(self, X) -> np.ndarray:
        return self._dummy.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self._dummy.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self._dummy, path)
        logger.info("NaiveBaselineModel saved to '%s'.", path)

    @classmethod
    def load(cls, path: str) -> "NaiveBaselineModel":
        instance = cls()
        instance._dummy = joblib.load(path)
        return instance


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — TF-IDF + Logistic Regression (Classical ML)
# ─────────────────────────────────────────────────────────────────────────────


class TFIDFClassifierModel(BaseSarcasmModel):
    """Classical ML model: TF-IDF sparse features + Logistic Regression.

    Hyperparameters (regularisation strength C, penalty type) are selected via
    grid-search cross-validation on the training set, optimising binary F1.

    Logistic Regression is preferred over tree-based alternatives because its
    coefficients are directly interpretable as feature weights — the magnitude
    and sign of each n-gram weight reveals how strongly that phrase pushes the
    prediction toward sarcasm, which is valuable for understanding cyberbullying
    language patterns.
    """

    def __init__(self) -> None:
        self._clf: Optional[LogisticRegression] = None
        self._best_params: Optional[Dict] = None

    def fit(
        self,
        X_train: sp.csr_matrix,
        y_train: np.ndarray,
        cv: int = 5,
    ) -> None:
        """Grid-search LR hyperparameters and refit the best estimator.

        Args:
            X_train: TF-IDF sparse feature matrix for the training set.
            y_train: Binary training labels.
            cv: Number of stratified cross-validation folds.
        """
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "max_iter": [1000],
        }
        base_clf = LogisticRegression(random_state=42, class_weight="balanced")
        grid = GridSearchCV(
            base_clf, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)

        self._clf = grid.best_estimator_
        self._best_params = grid.best_params_
        logger.info(
            "TFIDFClassifierModel best params: %s  |  CV F1 = %.4f",
            self._best_params,
            grid.best_score_,
        )

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        return self._clf.predict(X)

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        return self._clf.predict_proba(X)[:, 1]

    def top_features(self, vectorizer, n: int = 20) -> pd.DataFrame:
        """Return the n tokens most associated with each class.

        Args:
            vectorizer: The fitted TfidfVectorizer used to build features.
            n: Number of top features per class.

        Returns:
            DataFrame with columns: sarcastic_tokens, not_sarcastic_tokens.
        """
        coef = self._clf.coef_[0]
        vocab = np.array(vectorizer.get_feature_names_out())
        return pd.DataFrame(
            {
                "sarcastic_tokens": vocab[np.argsort(coef)[-n:][::-1]],
                "not_sarcastic_tokens": vocab[np.argsort(coef)[:n]],
            }
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"clf": self._clf, "best_params": self._best_params}, path)
        logger.info("TFIDFClassifierModel saved to '%s'.", path)

    @classmethod
    def load(cls, path: str) -> "TFIDFClassifierModel":
        instance = cls()
        data = joblib.load(path)
        instance._clf = data["clf"]
        instance._best_params = data["best_params"]
        return instance


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — DistilBERT (Deep Learning)
# ─────────────────────────────────────────────────────────────────────────────


class _SarcasmDataset(Dataset):
    """Internal PyTorch Dataset wrapper for tokenized text samples."""

    def __init__(self, encodings: Dict, labels: List[int]) -> None:
        self._encodings = encodings
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Dict:
        item = {key: torch.tensor(val[idx]) for key, val in self._encodings.items()}
        item["labels"] = torch.tensor(self._labels[idx], dtype=torch.long)
        return item


def _compute_hf_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy and binary F1 for the HuggingFace Trainer callback."""
    import evaluate as hf_evaluate

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = hf_evaluate.load("accuracy").compute(predictions=predictions, references=labels)
    f1 = hf_evaluate.load("f1").compute(
        predictions=predictions, references=labels, average="binary"
    )
    return {**acc, **f1}


class DistilBERTModel(BaseSarcasmModel):
    """Fine-tuned DistilBERT for sarcasm sequence classification.

    DistilBERT (Sanh et al., 2019) retains ~97% of BERT's language understanding
    at 40% fewer parameters and 60% faster inference, making it practical for
    both training and web-app deployment.

    Fine-tuning strategy:
        - All transformer layers unfrozen (full fine-tuning).
        - LR warm-up over ~10% of training steps stabilises early gradients.
        - Early stopping (patience = 2 epochs) on validation F1.

    Reference:
        Sanh et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
    """

    PRETRAINED_NAME = "distilbert-base-uncased"
    MAX_SEQ_LENGTH = 128
    INFERENCE_BATCH_SIZE = 64

    def __init__(
        self,
        model_dir: str = os.path.join(MODELS_DIR, "distilbert"),
    ) -> None:
        self.model_dir = model_dir
        self._tokenizer: Optional[DistilBertTokenizerFast] = None
        self._model: Optional[DistilBertForSequenceClassification] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _tokenize(self, texts: List[str]) -> Dict:
        """Tokenise a list of strings with truncation and padding."""
        return self._tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.MAX_SEQ_LENGTH,
        )

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        num_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
    ) -> None:
        """Fine-tune DistilBERT with early stopping on validation F1.

        Args:
            train_texts: Raw training headline strings.
            train_labels: Binary labels (0/1) for the training set.
            val_texts: Raw validation headline strings.
            val_labels: Binary labels for the validation set.
            num_epochs: Maximum training epochs.
            batch_size: Per-device batch size for training and evaluation.
            learning_rate: Peak AdamW learning rate.
            warmup_ratio: Fraction of total steps used for LR warm-up.
            weight_decay: L2 regularisation strength for AdamW.
        """
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(self.PRETRAINED_NAME)
        self._model = DistilBertForSequenceClassification.from_pretrained(
            self.PRETRAINED_NAME, num_labels=2
        )

        train_enc = self._tokenize(train_texts)
        val_enc = self._tokenize(val_texts)
        train_dataset = _SarcasmDataset(train_enc, train_labels)
        val_dataset = _SarcasmDataset(val_enc, val_labels)

        steps_per_epoch = max(1, len(train_texts) // batch_size)
        warmup_steps = max(1, int(warmup_ratio * steps_per_epoch * num_epochs))

        os.makedirs(self.model_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=100,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=_compute_hf_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        self._model = trainer.model
        self.save(self.model_dir)

    def predict(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Return binary sarcasm predictions for a list of texts.

        Args:
            texts: List or Series of raw headline strings.
        """
        return (self.predict_proba(texts) >= 0.5).astype(int)

    def predict_proba(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Return P(sarcastic) for each input text, processed in fixed-size batches.

        Args:
            texts: List or Series of raw headline strings.

        Returns:
            Float array of probabilities in [0, 1].
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        self._model.eval()
        self._model.to(self._device)

        all_probs: List[np.ndarray] = []
        for start in range(0, len(texts), self.INFERENCE_BATCH_SIZE):
            batch = texts[start : start + self.INFERENCE_BATCH_SIZE]
            enc = self._tokenize(batch)
            input_ids = torch.tensor(enc["input_ids"]).to(self._device)
            attention_mask = torch.tensor(enc["attention_mask"]).to(self._device)

            with torch.no_grad():
                logits = self._model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
            all_probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())

        return np.concatenate(all_probs)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        logger.info("DistilBERTModel saved to '%s'.", path)

    @classmethod
    def load(cls, path: str) -> "DistilBERTModel":
        instance = cls(model_dir=path)
        instance._tokenizer = DistilBertTokenizerFast.from_pretrained(path)
        instance._model = DistilBertForSequenceClassification.from_pretrained(path)
        return instance


# ─────────────────────────────────────────────────────────────────────────────
# Training orchestration
# ─────────────────────────────────────────────────────────────────────────────


def _plot_model_comparison(results: Dict[str, Dict[str, float]]) -> None:
    """Bar chart comparing all three models across five evaluation metrics."""
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, (name, color) in enumerate(zip(results.keys(), colors)):
        values = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Test Set Performance", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    save_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Model comparison chart saved to '%s'.", save_path)


def train_all_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Train all three models and evaluate them on the held-out test set.

    Assumes TF-IDF feature matrices and the vectorizer have already been built
    by scripts/build_features.py.

    Args:
        train: Training DataFrame with columns 'text' and 'label'.
        val:   Validation DataFrame with columns 'text' and 'label'.
        test:  Test DataFrame with columns 'text' and 'label'.

    Returns:
        Dict mapping model name → test-set metrics dict.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    processed_dir = os.path.join("data", "processed")
    X_train_tfidf = sp.load_npz(os.path.join(processed_dir, "X_train_tfidf.npz"))
    X_val_tfidf = sp.load_npz(os.path.join(processed_dir, "X_val_tfidf.npz"))
    X_test_tfidf = sp.load_npz(os.path.join(processed_dir, "X_test_tfidf.npz"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values
    test_texts = test["text"].tolist()

    results: Dict[str, Dict[str, float]] = {}

    # ── 1. Naive Baseline ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Model 1 / 3 — Naive Baseline")
    logger.info("=" * 60)
    naive = NaiveBaselineModel()
    naive.fit(X_train_tfidf, y_train)
    naive.save(os.path.join(MODELS_DIR, "naive_baseline.joblib"))
    naive.plot_confusion_matrix(
        X_test_tfidf, y_test,
        title="Naive Baseline — Confusion Matrix",
        save_path=os.path.join(OUTPUTS_DIR, "cm_naive_baseline.png"),
    )
    naive.error_analysis(
        X_test_tfidf, y_test, test_texts, n=5,
        save_path=os.path.join(OUTPUTS_DIR, "errors_naive_baseline.csv"),
    )
    results["Naive Baseline"] = naive.evaluate(X_test_tfidf, y_test, "test")

    # ── 2. TF-IDF + Logistic Regression ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Model 2 / 3 — TF-IDF + Logistic Regression")
    logger.info("=" * 60)
    tfidf_clf = TFIDFClassifierModel()
    tfidf_clf.fit(X_train_tfidf, y_train, cv=5)
    tfidf_clf.save(os.path.join(MODELS_DIR, "tfidf_lr.joblib"))
    tfidf_clf.plot_confusion_matrix(
        X_test_tfidf, y_test,
        title="TF-IDF + Logistic Regression — Confusion Matrix",
        save_path=os.path.join(OUTPUTS_DIR, "cm_tfidf_lr.png"),
    )
    tfidf_clf.error_analysis(
        X_test_tfidf, y_test, test_texts, n=5,
        save_path=os.path.join(OUTPUTS_DIR, "errors_tfidf_lr.csv"),
    )
    tfidf_clf.top_features(vectorizer, n=20).to_csv(
        os.path.join(OUTPUTS_DIR, "top_tfidf_features.csv"), index=False
    )
    results["TF-IDF + LR"] = tfidf_clf.evaluate(X_test_tfidf, y_test, "test")

    # ── 3. DistilBERT ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Model 3 / 3 — DistilBERT (fine-tuning)")
    logger.info("=" * 60)
    bert = DistilBERTModel()
    bert.fit(
        train["text"].tolist(), y_train.tolist(),
        val["text"].tolist(), y_val.tolist(),
    )
    bert.plot_confusion_matrix(
        test_texts, y_test,
        title="DistilBERT — Confusion Matrix",
        save_path=os.path.join(OUTPUTS_DIR, "cm_distilbert.png"),
    )
    bert.error_analysis(
        test_texts, y_test, test_texts, n=5,
        save_path=os.path.join(OUTPUTS_DIR, "errors_distilbert.csv"),
    )
    results["DistilBERT"] = bert.evaluate(test_texts, y_test, "test")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(results).T.round(4)
    summary_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"))
    logger.info("\nFinal test-set results:\n%s", summary_df.to_string())
    _plot_model_comparison(results)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    _train = pd.read_csv("data/processed/train.csv")
    _val = pd.read_csv("data/processed/val.csv")
    _test = pd.read_csv("data/processed/test.csv")
    train_all_models(_train, _val, _test)
