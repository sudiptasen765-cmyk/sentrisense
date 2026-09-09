"""
ml/scripts/download_sst2.py

Downloads the Stanford Sentiment Treebank (SST-2) via the Hugging Face
`datasets` library and exports train/validation/test splits to CSV for
easy loading alongside the IMDb data in later ML phases.

Source: https://huggingface.co/datasets/stanfordnlp/sst2

Usage:
    python ml/scripts/download_sst2.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "sst2"


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "[error] The 'datasets' package isn't installed. "
            "Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[download] stanfordnlp/sst2 via Hugging Face datasets")
    try:
        dataset = load_dataset("stanfordnlp/sst2")
    except Exception as exc:
        print(f"[error] Could not load dataset: {exc}", file=sys.stderr)
        print(
            "If this is a network error, check your connection. If Hugging Face "
            "requires authentication for this dataset in the future, run "
            "`huggingface-cli login` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n--- SST-2 dataset summary ---")
    for split_name, split_data in dataset.items():
        out_path = OUT_DIR / f"{split_name}.csv"
        split_data.to_csv(str(out_path), index=False)
        print(f"  {split_name:<12} : {len(split_data):>6} rows  -> {out_path.relative_to(PROJECT_ROOT)}")

    # Note: SST-2's official test split has no public labels (label = -1),
    # since it's used for the GLUE benchmark leaderboard. We use train/validation
    # for our own evaluation and flag this explicitly rather than silently
    # treating -1 as a valid class.
    if "test" in dataset:
        sample_labels = set(dataset["test"]["label"][:20])
        if sample_labels == {-1}:
            print(
                "\n[note] The 'test' split has no public labels (label=-1 placeholder, "
                "reserved for the GLUE leaderboard). Use 'validation' as the held-out "
                "generalization check instead — this will be documented in ml/data/README.md."
            )


if __name__ == "__main__":
    main()
