# ml/preprocessing/ — Preprocessing Documentation

## Pipeline

```
raw text (ml/data/raw/)
    │
    ▼
remove_html()            strip <br />, unescape entities
    │
    ▼
[classical only] lowercase
    │
    ▼
cap_repeated_punctuation()   "!!!!!!!" -> "!!!"  (emphasis kept, length capped)
    │
    ▼
normalize_whitespace()
    │
    ▼
cleaned text (ml/data/processed/)
```

Run:
```powershell
python ml\preprocessing\build_splits.py
```
**Requires** `ml/scripts/download_imdb.py` and `ml/scripts/download_sst2.py` to
have already been run (Phase 1).

## Design decisions

### 1. No stopword removal, ever

Standard NLP pipelines often strip stopwords ("the", "is", "not", "a"...).
We deliberately don't. Words like **"not"**, **"never"**, **"no"**, **"n't"**
carry sentiment-flipping meaning — removing them turns "not good" into "good",
which is a straightforward correctness bug, not just an accuracy hit.
`test_clean_text.py` has explicit tests (`TestNegationPreservation`) asserting
this never regresses.

TF-IDF's own `min_df`/`max_df`/IDF weighting (configured in Phase 3) already
down-weights uninformative high-frequency words without needing a separate
deletion step.

### 2. Two cleaning modes, not one

| | `clean_for_classical()` | `clean_for_transformer()` |
|---|---|---|
| HTML removal | ✅ | ✅ |
| Whitespace normalization | ✅ | ✅ |
| Lowercasing | ✅ | ❌ (case is signal for subword tokenizers) |
| Punctuation stripped | ❌ (left for TF-IDF's tokenizer to ignore) | ❌ (left for the model's own tokenizer) |

Both columns are written to every processed CSV (`text_clean_classical`,
`text_clean_transformer`) so Phase 3 and Phase 4/5 scripts each read the
column meant for their model type, rather than re-deriving cleaning logic
independently and risking drift between them.

### 3. Cross-split leakage fix (carried over from Phase 1)

123 IMDb reviews were found to be byte-identical between the official
`train` and `test` splits (see `ml/data/README.md`). `build_splits.py`:

- Leaves `test` **completely unmodified** — results stay comparable to
  published IMDb benchmarks using the same split.
- Drops any `train` row whose raw text exactly matches a `test` row, **before**
  the train/validation split, so neither train nor validation can contain a
  memorized copy of a test example.
- Prints a warning if the removed-row count doesn't match Phase 1's measured
  123 — a canary in case the raw data changes underneath the pipeline.

### 4. Train/validation split

The official IMDb release only ships train/test — no validation split. We
carve **10%** off the (leakage-fixed) train set, **stratified by label** so
class balance (50/50) is preserved in both resulting sets, with
`random_state=42` fixed and documented so every run is reproducible.

### 5. SST-2 handling

Only `train` and `validation` are converted — the public `test` split is
unlabeled (`label=-1` placeholder, see `ml/data/README.md`) and is
intentionally excluded from `build_splits.py` entirely, not just flagged, so
it's structurally impossible to accidentally train or evaluate against it.

### 6. Repeated-punctuation capping

`"amazing!!!!!!!!!"` → `"amazing!!!"`. The emphasis signal (exclamation marks
correlate with sentiment intensity) is kept, but runs are capped at 3 so a
single outlier review with excessive punctuation can't disproportionately
affect length-based features later.

## Output files (`ml/data/processed/`, gitignored — regenerate via the script)

| File | Rows (typical) | Notes |
|---|---|---|
| `imdb_train.csv` | ~22,388 | leakage-fixed, 90% of adjusted train |
| `imdb_val.csv` | ~2,488 | leakage-fixed, 10% of adjusted train, stratified |
| `imdb_test.csv` | 25,000 | official, **unmodified** |
| `sst2_train.csv` | 67,349 | secondary dataset |
| `sst2_val.csv` | 872 | used as SST-2's actual held-out eval (not its `test`) |

Each CSV has columns: `id`, `text_raw`, `label`, `text_clean_classical`,
`text_clean_transformer`, `word_count`.

## Testing

```powershell
pytest ml\preprocessing\test_clean_text.py -v
```

20 tests, covering negation preservation (the most important group), HTML
removal, whitespace normalization, punctuation capping, classical-vs-
transformer mode differences, and edge cases (empty/None/whitespace-only
input). All 20 currently pass.
