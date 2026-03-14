# Test Set — Gaslight Guard

Held-out evaluation data, **completely separate** from training data.

## Structure

```
test_set/
├── test_cases.py   # 60 labeled examples (15 per class), all unseen during training
├── evaluate.py     # evaluation script: loads model, prints metrics + confusion matrix
└── README.md
```

## Test Set Composition

| Label              | Count |
|--------------------|-------|
| Sincere            | 15    |
| Sarcastic          | 15    |
| Passive-Aggressive | 15    |
| Gaslighting        | 15    |
| **Total**          | **60**|

All examples were written independently of `curated_data.py` and are not present in
`tweet_eval/irony` to avoid data leakage.

## Run Evaluation

```bash
# from the backend/ directory, with venv activated
cd backend
source venv/bin/activate

python data/test_set/evaluate.py
# with JSON output
python data/test_set/evaluate.py --output data/test_set/results.json
```

## Output

- Per-class Precision / Recall / F1
- Macro-averaged F1
- Confusion matrix
- Per-example pass/fail list
- Optional `results.json` for the evaluation notebook
