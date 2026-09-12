"""
ml/scripts/download_imdb.py

Downloads and extracts the Stanford IMDb Large Movie Review Dataset.

Source: https://ai.stanford.edu/~amaas/data/sentiment/
Citation: Maas et al. (2011), "Learning Word Vectors for Sentiment Analysis", ACL.

Usage:
    python ml/scripts/download_imdb.py

Idempotent: if the archive or extracted folder already exists, the
corresponding step is skipped rather than re-downloaded/re-extracted.
"""

import hashlib
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

# ---------------------------------------------------------------------------
DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# NOTE: Stanford's dataset page does not publish an official checksum, so we
# can't assert a known-good hash here. Instead we compute and print SHA1 after
# download so you can visually confirm no re-download gives a different value
# across machines/runs (a cheap corruption sanity check, not authenticity proof).

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / "aclImdb_v1.tar.gz"
EXTRACTED_DIR = RAW_DIR / "aclImdb"
# ---------------------------------------------------------------------------


def sha1_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def report_progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size)
    mb_done = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    print(f"\r  downloading: {pct:3d}%  ({mb_done:6.1f} / {mb_total:6.1f} MB)", end="")


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists():
        print(f"[skip] Archive already present: {ARCHIVE_PATH}")
        return

    print(f"[download] {DATASET_URL}")
    print(f"           -> {ARCHIVE_PATH}")
    print("Note: this file is ~80MB; large but not huge.")
    try:
        urlretrieve(DATASET_URL, ARCHIVE_PATH, reporthook=report_progress)
        print()
        print(f"[checksum] SHA1: {sha1_of(ARCHIVE_PATH)}")
        print("           (record this in ml/data/README.md if you want a reproducibility reference)")
    except Exception as exc:
        if ARCHIVE_PATH.exists():
            ARCHIVE_PATH.unlink()  # don't leave a partial/corrupt file behind
        print(f"\n[error] Download failed: {exc}", file=sys.stderr)
        print(
            "If ai.stanford.edu is unreachable, check your network/firewall, "
            "or manually download the file from the URL above and place it at:\n"
            f"  {ARCHIVE_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)


def extract() -> None:
    if EXTRACTED_DIR.exists():
        print(f"[skip] Already extracted: {EXTRACTED_DIR}")
        return

    print(f"[extract] {ARCHIVE_PATH} -> {RAW_DIR}")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        # Guard against path traversal in the archive (defensive, even for a
        # trusted source — never blindly trust tar.extractall on downloaded data).
        def is_within_directory(directory: Path, target: Path) -> bool:
            return directory.resolve() in target.resolve().parents or directory.resolve() == target.resolve()

        for member in tar.getmembers():
            member_path = RAW_DIR / member.name
            if not is_within_directory(RAW_DIR, member_path):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        tar.extractall(RAW_DIR)

    print("[done] Extraction complete.")


def summarize() -> None:
    train_pos = list((EXTRACTED_DIR / "train" / "pos").glob("*.txt"))
    train_neg = list((EXTRACTED_DIR / "train" / "neg").glob("*.txt"))
    test_pos = list((EXTRACTED_DIR / "test" / "pos").glob("*.txt"))
    test_neg = list((EXTRACTED_DIR / "test" / "neg").glob("*.txt"))
    train_unsup = list((EXTRACTED_DIR / "train" / "unsup").glob("*.txt")) if (EXTRACTED_DIR / "train" / "unsup").exists() else []

    print("\n--- IMDb dataset summary ---")
    print(f"  train/pos    : {len(train_pos):>6} files")
    print(f"  train/neg    : {len(train_neg):>6} files")
    print(f"  train/unsup  : {len(train_unsup):>6} files")
    print(f"  test/pos     : {len(test_pos):>6} files")
    print(f"  test/neg     : {len(test_neg):>6} files")
    print(f"  Expected     : 12500 / 12500 / 50000 / 12500 / 12500")

    if len(train_pos) != 12500 or len(train_neg) != 12500 or len(test_pos) != 12500 or len(test_neg) != 12500:
        print(
            "\n[warning] File counts don't match the expected dataset shape. "
            "The archive may be incomplete or a different version — verify manually "
            "before proceeding to preprocessing.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    download()
    extract()
    summarize()
