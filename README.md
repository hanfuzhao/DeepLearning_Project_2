# Sarcasm Detection for Cyberbullying Analysis

> **Team:** Jaideep, Hanfu, Kening  
> **Module:** 2 — NLP  
> **Task:** Binary classification — detect sarcastic language as a signal for cyberbullying and online harassment

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Where Each Model Lives](#3-where-each-model-lives)
4. [Model Descriptions](#4-model-descriptions)
5. [How to Run](#5-how-to-run)
6. [Dataset](#6-dataset)
7. [Results](#7-results)
8. [Experiment](#8-experiment-training-size-sensitivity-analysis)
9. [Output Files](#9-output-files)
10. [Reference](#10-reference)

---

## 1. Project Overview

Cyberbullying frequently employs sarcasm and irony as cover — insults disguised as compliments are harder to flag with keyword-based filters. This project builds and evaluates three progressively powerful NLP models that detect sarcastic language, providing a foundation for automated cyberbullying detection systems.

**Three-tier modeling approach (all implemented and documented):**

| # | Model | Type | Test F1 | Test AUC |
|---|-------|------|---------|----------|
| 1 | Majority-class classifier | Naive Baseline | 0.000 | 0.500 |
| 2 | TF-IDF + Logistic Regression | Classical ML | 0.860 | 0.938 |
| 3 | Fine-tuned DistilBERT | Deep Learning | **0.916** | **0.977** |

---

## 2. Repository Structure

```
540-project-2/
│
├── README.md                   ← you are here
├── requirements.txt            ← all Python dependencies
├── setup.py                    ← one-command full pipeline (data → train → experiment)
├── main.py                     ← CLI: setup / train / experiment / predict / compare
│
├── scripts/                    ← pipeline and utility scripts
│   ├── make_dataset.py         ← downloads and preprocesses the sarcasm dataset
│   ├── build_features.py       ← builds TF-IDF feature matrices
│   ├── model.py                ← ★ ALL THREE MODELS defined here (see Section 3)
│   └── experiment.py           ← training-size sensitivity analysis
│
├── models/                     ← trained model artifacts
│   ├── naive_baseline.joblib   ← Model 1: Naive Baseline
│   ├── tfidf_lr.joblib         ← Model 2: TF-IDF + Logistic Regression
│   ├── tfidf_vectorizer.joblib ← TF-IDF vectorizer (required by Model 2)
│   └── distilbert/             ← Model 3: fine-tuned DistilBERT weights
│       ├── model.safetensors
│       ├── config.json
│       └── tokenizer.json
│
├── data/
│   ├── raw/                    ← raw downloaded JSONL dataset
│   ├── processed/              ← train/val/test CSV splits + TF-IDF .npz matrices
│   └── outputs/                ← confusion matrices, comparison charts, error CSVs
│
├── notebooks/                  ← exploratory analysis notebooks
│   └── EDA.ipynb               ← dataset exploration: class balance, text length, top n-grams
│
└── .gitignore
```

---

## 3. Where Each Model Lives

| Model | Source Code | Trained Artifact |
|-------|-------------|-----------------|
| **Model 1: Naive Baseline** | `scripts/model.py` → class `NaiveBaselineModel` | `models/naive_baseline.joblib` |
| **Model 2: Classical ML** | `scripts/model.py` → class `TFIDFClassifierModel` | `models/tfidf_lr.joblib` |
| **Model 3: Deep Learning** | `scripts/model.py` → class `DistilBERTModel` | `models/distilbert/` |

All three classes share a common interface (`fit`, `predict`, `predict_proba`, `evaluate`, `save`, `load`) defined in the abstract base class `BaseSarcasmModel` at the top of `scripts/model.py`.

---

## 4. Model Descriptions

### Model 1 — Naive Baseline (`NaiveBaselineModel`)
Always predicts the most frequent class in the training set (non-sarcastic, 52.4%). Acts as a lower-bound sanity check — any real model must beat it. Achieves 52.4% accuracy and F1 = 0.

### Model 2 — TF-IDF + Logistic Regression (`TFIDFClassifierModel`)
- **Features:** TF-IDF with unigrams + bigrams, top 50,000 tokens, sublinear TF scaling
- **Classifier:** Logistic Regression with grid-search over C ∈ {0.01, 0.1, 1, 10} and penalty ∈ {L1, L2}; 5-fold cross-validation; optimises binary F1
- **Why LR:** Coefficients are directly interpretable — the weight of each n-gram reveals how strongly it signals sarcasm
- **Result:** 86.6% accuracy, F1 = 0.860

### Model 3 — Fine-tuned DistilBERT (`DistilBERTModel`)
- **Base model:** `distilbert-base-uncased` (66M parameters, 60% faster than BERT, retains ~97% of BERT performance)
- **Fine-tuning:** All layers unfrozen; AdamW optimizer; LR warm-up; early stopping on validation F1 (patience = 2)
- **Why DistilBERT:** Understands full sentence context and word order — critical for sarcasm which relies on tone and structure, not just keywords
- **Result:** 92.1% accuracy, F1 = 0.916, AUC = 0.977

---

## 5. How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

---

### Option A — Full pipeline from scratch (recommended)

```bash
python setup.py
```

Runs all four stages automatically:
1. Downloads the dataset (~28k headlines)
2. Builds TF-IDF feature matrices
3. Trains all three models and evaluates on test set
4. Runs the sensitivity experiment

**Expected runtime:** ~30–60 min on CPU | ~10–15 min with a GPU

---

### Option B — Run stages individually

```bash
python main.py setup        # download data + build TF-IDF features
python main.py train        # train all three models
python main.py experiment   # run sensitivity analysis
python main.py compare      # print results table
```

---

### Option C — Run inference on any text

```bash
python main.py predict "scientists discover the cure for everything, no big deal"
```

Output:
```
  Model:        DistilBERT
  Input:        'scientists discover the cure for everything, no big deal'
  Prediction:   SARCASTIC
  Confidence:   97.2%
  P(sarcastic): 0.9723
```

---

### Option D — Use a model directly in Python

```python
from scripts.model import DistilBERTModel, TFIDFClassifierModel, NaiveBaselineModel

# Load the best model (DistilBERT)
model = DistilBERTModel.load("models/distilbert")
prob = model.predict_proba(["your text here"])[0]
label = "SARCASTIC" if prob >= 0.5 else "NOT SARCASTIC"

# Load classical ML model
import joblib
clf = TFIDFClassifierModel.load("models/tfidf_lr.joblib")
vec = joblib.load("models/tfidf_vectorizer.joblib")
X = vec.transform(["your text here"])
print(clf.predict(X))
```

---

## 6. Dataset

**News Headlines Dataset for Sarcasm Detection** (Misra & Arora, 2019)

| Property | Value |
|----------|-------|
| Total samples | 28,619 headlines |
| Sarcastic (label=1) | 13,634 — sourced from *The Onion* |
| Non-sarcastic (label=0) | 14,985 — sourced from *HuffPost* |
| Class balance | 47.6% / 52.4% (near-balanced) |
| Download | Automatic on first run via `python main.py setup` |

**Train / Val / Test split (stratified):**

| Split | Samples |
|-------|---------|
| Train | 19,933 (70%) |
| Validation | 4,272 (15%) |
| Test | 4,272 (15%) |

---

## 7. Results

### Test-set performance

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Naive Baseline | 52.4% | 0.000 | 0.000 | 0.000 | 0.500 |
| TF-IDF + LR | 86.6% | 85.6% | 86.3% | 0.860 | 0.938 |
| **DistilBERT** | **92.1%** | **92.7%** | **90.5%** | **0.916** | **0.977** |

Confusion matrices and a comparison bar chart are saved in `data/outputs/` after running `python main.py train`.

---

## 8. Experiment: Training-Size Sensitivity Analysis

**Question:** How much labelled data does each model actually need?

Each model is trained on 10%, 20%, 30%, 50%, 75%, and 100% of the training set, then evaluated on the fixed test set.

**Run the experiment:**
```bash
python main.py experiment
```

Results saved to `data/outputs/experiment/`:
- `sensitivity_results.csv` — numeric table
- `sensitivity_f1.png` — F1 vs. training fraction
- `sensitivity_accuracy.png` — accuracy vs. training fraction

---

## 9. Output Files

All outputs are generated automatically after `python main.py train`:

| File | Description |
|------|-------------|
| `data/outputs/cm_naive_baseline.png` | Confusion matrix — Naive Baseline |
| `data/outputs/cm_tfidf_lr.png` | Confusion matrix — TF-IDF + LR |
| `data/outputs/cm_distilbert.png` | Confusion matrix — DistilBERT |
| `data/outputs/model_comparison.csv` | Numeric results for all three models |
| `data/outputs/model_comparison.png` | Bar chart comparing all metrics |
| `data/outputs/top_tfidf_features.csv` | Top 20 n-grams most associated with sarcasm |
| `data/outputs/errors_naive_baseline.csv` | 5 representative mispredictions — Naive Baseline |
| `data/outputs/errors_tfidf_lr.csv` | 5 representative mispredictions — TF-IDF + LR |
| `data/outputs/errors_distilbert.csv` | 5 representative mispredictions — DistilBERT |
| `data/outputs/experiment/sensitivity_f1.png` | Experiment: F1 vs. training size |
| `data/outputs/experiment/sensitivity_accuracy.png` | Experiment: Accuracy vs. training size |

---

## 10. Reference

Misra, R., & Arora, P. (2019). Sarcasm Detection using News Headlines Dataset. *AI Open*, 1, 1–9.  
Dataset: https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108*.
