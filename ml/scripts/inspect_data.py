"""
ml/scripts/inspect_data.py

First-pass data quality inspection for IMDb and SST-2, run after both
download scripts. Checks:
  - row/file counts
  - missing values
  - duplicate texts
  - class balance
  - text length distribution (word count)

Writes a machine-readable summary to reports/results/data_inspection_summary.json
and prints a human-readable version to the console.

Usage:
    python ml/scripts/inspect_data.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMDB_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "aclImdb"
SST2_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "sst2"
REPORT_PATH = PROJECT_ROOT / "reports" / "results" / "data_inspection_summary.json"


def inspect_imdb() -> dict:
    if not IMDB_DIR.exists():
        return {"status": "missing", "note": "Run download_imdb.py first."}

    import pandas as pd

    rows = []
    file_count = 0
    for split in ("train", "test"):
        for label_name, label_val in (("pos", 1), ("neg", 0)):
            folder = IMDB_DIR / split / label_name
            if not folder.exists():
                continue
            for fp in folder.glob("*.txt"):
                text = fp.read_text(encoding="utf-8", errors="replace")
                rows.append({"split": split, "label": label_val, "text": text})
                file_count += 1
                if file_count % 5000 == 0:
                    print(f"  ...read {file_count} files so far", flush=True)

    print(f"  finished reading {file_count} files, building summary...", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        return {"status": "empty", "note": "No files found under expected structure."}

    df["word_count"] = df["text"].str.split().str.len()
    duplicates = int(df.duplicated(subset=["text"]).sum())
    missing = int(df["text"].isna().sum() + (df["text"].str.strip() == "").sum())

    # Cross-split duplicate check: reviews appearing in BOTH train and test are
    # data leakage — a model could memorize a training review and "recognize"
    # it verbatim at test time, inflating reported accuracy. This is distinct
    # from ordinary within-split duplicates, which just slightly bias class
    # weighting but don't invalidate evaluation.
    train_texts = set(df[df["split"] == "train"]["text"])
    test_texts = set(df[df["split"] == "test"]["text"])
    cross_split_duplicates = len(train_texts & test_texts)

    within_train_dupes = int(df[df["split"] == "train"].duplicated(subset=["text"]).sum())
    within_test_dupes = int(df[df["split"] == "test"].duplicated(subset=["text"]).sum())

    summary = {
        "status": "ok",
        "total_rows": len(df),
        "by_split": df.groupby("split").size().to_dict(),
        "class_balance": {
            split: df[df["split"] == split]["label"].value_counts().to_dict()
            for split in df["split"].unique()
        },
        "duplicate_texts_total": duplicates,
        "duplicate_texts_within_train": within_train_dupes,
        "duplicate_texts_within_test": within_test_dupes,
        "duplicate_texts_cross_train_test_LEAKAGE": cross_split_duplicates,
        "empty_or_missing_texts": missing,
        "word_count_stats": {
            "min": int(df["word_count"].min()),
            "max": int(df["word_count"].max()),
            "mean": round(float(df["word_count"].mean()), 1),
            "median": float(df["word_count"].median()),
        },
    }
    return summary


def inspect_sst2() -> dict:
    if not SST2_DIR.exists():
        return {"status": "missing", "note": "Run download_sst2.py first."}

    import pandas as pd

    result = {"status": "ok", "splits": {}}
    for csv_path in sorted(SST2_DIR.glob("*.csv")):
        split_name = csv_path.stem
        df = pd.read_csv(csv_path)

        text_col = "sentence" if "sentence" in df.columns else df.columns[0]
        df["word_count"] = df[text_col].astype(str).str.split().str.len()

        duplicates = int(df.duplicated(subset=[text_col]).sum())
        missing = int(df[text_col].isna().sum())

        split_summary = {
            "rows": len(df),
            "duplicate_texts": duplicates,
            "missing_texts": missing,
            "word_count_stats": {
                "min": int(df["word_count"].min()),
                "max": int(df["word_count"].max()),
                "mean": round(float(df["word_count"].mean()), 1),
            },
        }
        if "label" in df.columns:
            split_summary["class_balance"] = df["label"].value_counts().to_dict()

        result["splits"][split_name] = split_summary

    return result


def main() -> None:
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        print(
            "[error] pandas isn't installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Inspecting IMDb...")
    imdb_summary = inspect_imdb()
    print("Inspecting SST-2...")
    sst2_summary = inspect_sst2()

    full_report = {"imdb": imdb_summary, "sst2": sst2_summary}

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(full_report, indent=2, default=str), encoding="utf-8")

    print("\n=== IMDb ===")
    print(json.dumps(imdb_summary, indent=2, default=str))
    print("\n=== SST-2 ===")
    print(json.dumps(sst2_summary, indent=2, default=str))
    print(f"\n[saved] Full report -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    for name, summary in (("IMDb", imdb_summary), ("SST-2", sst2_summary)):
        if summary.get("status") != "ok":
            print(
                f"\n[warning] {name} inspection incomplete: {summary.get('note', summary.get('status'))}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()