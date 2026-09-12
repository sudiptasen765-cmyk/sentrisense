"""
ml/preprocessing/build_splits.py

Loads raw IMDb and SST-2 data, applies the cross-split leakage fix decided
in Phase 1 (ml/data/README.md), cleans text, and writes reproducible
train/validation/test splits to ml/data/processed/.

Usage:
    python ml/preprocessing/build_splits.py

Must be run AFTER:
    python ml/scripts/download_imdb.py
    python ml/scripts/download_sst2.py
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_text import clean_for_classical, clean_for_transformer  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMDB_RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "aclImdb"
SST2_RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "sst2"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"

RANDOM_STATE = 42  # fixed for reproducibility — documented, never changed silently
VAL_FRACTION = 0.10  # carved out of the official IMDb train split


def load_imdb_split(split: str) -> "pd.DataFrame":
    

    rows = []
    for label_name, label_val in (("pos", 1), ("neg", 0)):
        folder = IMDB_RAW_DIR / split / label_name
        for fp in sorted(folder.glob("*.txt")):
            rows.append(
                {
                    "id": f"imdb_{split}_{label_name}_{fp.stem}",
                    "text_raw": fp.read_text(encoding="utf-8", errors="replace"),
                    "label": label_val,
                }
            )
    return pd.DataFrame(rows)


def build_imdb_splits() -> None:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if not IMDB_RAW_DIR.exists():
        print("[error] IMDb raw data not found. Run ml/scripts/download_imdb.py first.", file=sys.stderr)
        sys.exit(1)

    print("[imdb] loading raw train/test...")
    train_df = load_imdb_split("train")
    test_df = load_imdb_split("test")
    print(f"[imdb] loaded {len(train_df)} train, {len(test_df)} test rows")

    # --- Leakage fix (decision documented in ml/data/README.md, Phase 1) ---
    # The official test split is left untouched for benchmark comparability.
    # Any train row whose text is byte-identical to a test row is dropped
    # from train, so no model can memorize-then-recognize a test example.
    test_texts = set(test_df["text_raw"])
    before = len(train_df)
    train_df = train_df[~train_df["text_raw"].isin(test_texts)].reset_index(drop=True)
    removed = before - len(train_df)
    print(f"[imdb] removed {removed} train rows overlapping with test (data leakage fix)")
    if removed == 0:
        print(
            "[imdb][warning] Expected ~123 rows removed based on Phase 1 inspection; "
            "got 0. Verify raw data wasn't re-downloaded/changed.",
            file=sys.stderr,
        )

    # --- Train/validation split (stratified, so class balance is preserved) ---
    train_split, val_split = train_test_split(
        train_df,
        test_size=VAL_FRACTION,
        stratify=train_df["label"],
        random_state=RANDOM_STATE,
    )
    print(f"[imdb] train/val split: {len(train_split)} train, {len(val_split)} val")

    # --- Clean text (both modes, so Phase 3 classical models and the Phase 4/5
    # transformer comparison can each use the column they need without
    # re-running cleaning independently and risking drift between them) ---
    for df in (train_split, val_split, test_df):
        df["text_clean_classical"] = df["text_raw"].apply(clean_for_classical)
        df["text_clean_transformer"] = df["text_raw"].apply(clean_for_transformer)
        df["word_count"] = df["text_raw"].str.split().str.len()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_split.to_csv(PROCESSED_DIR / "imdb_train.csv", index=False)
    val_split.to_csv(PROCESSED_DIR / "imdb_val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "imdb_test.csv", index=False)

    print(f"[imdb] wrote imdb_train.csv ({len(train_split)}), "
          f"imdb_val.csv ({len(val_split)}), imdb_test.csv ({len(test_df)})")


def build_sst2_splits() -> None:
    import pandas as pd

    if not SST2_RAW_DIR.exists():
        print("[error] SST-2 raw data not found. Run ml/scripts/download_sst2.py first.", file=sys.stderr)
        sys.exit(1)

    print("[sst2] loading raw train/validation...")
    # Only train + validation: the public 'test' split is unlabeled
    # (documented in ml/data/README.md) and is intentionally excluded here
    # so it can never accidentally be used as if it had real labels.
    for split_name in ("train", "validation"):
        src = SST2_RAW_DIR / f"{split_name}.csv"
        df = pd.read_csv(src)
        text_col = "sentence" if "sentence" in df.columns else df.columns[0]
        df = df.rename(columns={text_col: "text_raw"})

        df["text_clean_classical"] = df["text_raw"].apply(clean_for_classical)
        df["text_clean_transformer"] = df["text_raw"].apply(clean_for_transformer)
        df["word_count"] = df["text_raw"].astype(str).str.split().str.len()

        out_name = "sst2_train.csv" if split_name == "train" else "sst2_val.csv"
        df.to_csv(PROCESSED_DIR / out_name, index=False)
        print(f"[sst2] wrote {out_name} ({len(df)} rows)")


if __name__ == "__main__":
    try:
        import pandas  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        print("[error] pandas/scikit-learn not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    build_imdb_splits()
    build_sst2_splits()
    print("\n[done] All processed splits written to ml/data/processed/")
