# ml/data/ — Dataset Documentation

This folder holds downloaded datasets. Everything under `raw/` is **gitignored**
(see root `.gitignore`) because these are large, publicly re-downloadable files —
only this README and the pipeline scripts that produce/consume them are tracked.

## Reproducing the data

```powershell
python ml\scripts\download_imdb.py
python ml\scripts\download_sst2.py
python ml\scripts\inspect_data.py
```

Both download scripts are idempotent — safe to re-run without re-downloading
if the data is already present.

## 1. IMDb Large Movie Review Dataset (primary)

- **Source:** https://ai.stanford.edu/~amaas/data/sentiment/
- **Citation:**
  Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).
  *Learning Word Vectors for Sentiment Analysis.* Proceedings of the 49th Annual
  Meeting of the Association for Computational Linguistics (ACL 2011).
- **License / usage:** Distributed by Stanford for research use. No redistribution
  restrictions stated beyond standard academic citation. We do not redistribute
  the raw files in this repository — the download script fetches directly from
  the original source.
- **Shape:** 25,000 train (12,500 pos / 12,500 neg) + 25,000 test
  (12,500 pos / 12,500 neg) + 50,000 unlabeled reviews.
- **Labels:** `0 = negative`, `1 = positive`. Binary only — no neutral class,
  which is a documented limitation of this dataset (see project `README.md`
  → Limitations, added in later phases).
- **Layout after download:**
  ```
  ml/data/raw/aclImdb/
  ├── train/{pos,neg,unsup}/*.txt
  └── test/{pos,neg}/*.txt
  ```

## 2. SST-2 (Stanford Sentiment Treebank, binary)

- **Source:** https://huggingface.co/datasets/stanfordnlp/sst2
- **Citation:**
  Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., &
  Potts, C. (2013). *Recursive Deep Models for Semantic Compositionality Over
  a Sentiment Treebank.* Proceedings of EMNLP 2013.
- **License / usage:** See the dataset card on Hugging Face for current terms;
  used here for research/educational purposes only, as a secondary
  generalization check (train on IMDb-derived models, evaluate transfer to SST-2).
- **Important note:** the public `test` split has **no real labels**
  (`label = -1` placeholder — reserved for the GLUE benchmark leaderboard).
  We use the `validation` split as our actual held-out generalization check,
  not `test`. The download script prints a warning about this so it isn't
  silently misused later.
- **Layout after download:**
  ```
  ml/data/raw/sst2/
  ├── train.csv
  ├── validation.csv
  └── test.csv   (unlabeled — do not use for evaluation)
  ```

## 3. Optional domain-specific review dataset

Not yet selected — deferred until aspect-based sentiment work (later ML phase).
When chosen (from Amazon/Yelp/Kaggle review data), its license and attribution
will be documented here and in `docs/resources.md` before use. No web scraping
will be performed — only downloadable, appropriately licensed datasets.

## Data quality inspection

Run `python ml/scripts/inspect_data.py` after both downloads. It checks, for
each dataset/split:

- row/file counts vs. expected shape
- duplicate texts (within-split AND cross-split)
- missing/empty texts
- class balance
- word-count distribution (min/max/mean/median)

Results are saved to `reports/results/data_inspection_summary.json`.

### Actual measured results (run on 2026-09-07)

**IMDb:**
- 50,000 total rows (25,000 train / 25,000 test), perfectly balanced classes
  in both splits (12,500 / 12,500 each) — no class-imbalance handling needed.
- 0 missing/empty texts.
- Word count: min 4, max 2,470, mean 231.2, median 173.0 — wide range; relevant
  for the DistilBERT comparison model later, which truncates at 512 tokens.
- Duplicate texts: 418 total — 96 within train, 199 within test,
  **123 cross-split (train ∩ test)**.

**SST-2:**
- train: 67,349 rows, moderately imbalanced (37,569 positive / 29,780 negative,
  ~56/44) — noted for methodology, not severe enough to need resampling.
- validation: 872 rows, balanced (444 / 428) — used as our generalization
  check (not `test`, which is unlabeled as noted above).
- 371 duplicate texts within train (not cross-checked against validation/test
  since those splits are tiny and this isn't the primary training dataset).

### Decision: IMDb cross-split duplicate handling (data leakage)

123 IMDb reviews are byte-identical between `train` and `test`. This is data
leakage: a model could effectively memorize a training example and "recognize"
it verbatim at evaluation time, inflating reported accuracy.

**Decision:** the official `test` split is left **unmodified**, so results
remain comparable to published IMDb benchmarks using the same split. The 123
overlapping rows are instead removed from `train` during preprocessing
(Phase 2, `ml/preprocessing/`). This is implemented, not just documented —
see `ml/preprocessing/` once Phase 2 begins.

Within-split duplicates (96 in train, 199 in test) are **not** removed — they
don't cause leakage, only a mild bias toward whatever opinion happens to be
repeated, which is an accepted characteristic of this dataset and is left
intact for comparability with prior work using the same corpus.

## Known considerations for preprocessing (Phase 2)

- **Do not strip negations.** Standard stop-word removal often deletes "not",
  "never", "no" — these flip sentiment meaning ("not good" ≠ "good") and must
  be preserved through cleaning.
- IMDb reviews contain HTML line breaks (`<br />`) that need explicit removal —
  they are not naturally clean text.
- Class balance is exactly 50/50 in IMDb by construction; SST-2 balance will
  be measured, not assumed.