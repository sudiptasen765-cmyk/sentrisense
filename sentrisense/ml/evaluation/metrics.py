"""
ml/evaluation/metrics.py

Shared evaluation utilities used by every model training script (Phase 3
baseline, Phase 4 comparison candidates) so the eventual model comparison
table (docs/research, reports/results) is computed the same way for every
model — no risk of one model's "accuracy" being measured differently from
another's.
"""

import time
from pathlib import Path
from statistics import mean, median


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Standard classification metrics for a binary sentiment task."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )

    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
    }


def measure_inference_latency(predict_fn, samples, n_repeats: int = 200, warmup: int = 10) -> dict:
    """Measures single-sample inference latency, simulating a real API call
    (one text in, one prediction out) rather than batch throughput.

    predict_fn: callable taking ONE text string, returning a prediction.
    samples: list of text strings to draw from (cycled if n_repeats > len(samples)).
    warmup: number of untimed calls first, so first-call overhead (e.g. lazy
            imports, cache population) doesn't skew the measurement — this is
            deliberately DISTINCT from cold-start latency, which is measured
            separately at the Lambda level in Phase 24, not here.
    """
    if not samples:
        raise ValueError("samples must be non-empty")

    for i in range(warmup):
        predict_fn(samples[i % len(samples)])

    timings_ms = []
    for i in range(n_repeats):
        text = samples[i % len(samples)]
        start = time.perf_counter()
        predict_fn(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed_ms)

    timings_sorted = sorted(timings_ms)
    p95_idx = int(len(timings_sorted) * 0.95)

    return {
        "n_measured": n_repeats,
        "mean_ms": round(mean(timings_ms), 3),
        "median_ms": round(median(timings_ms), 3),
        "p95_ms": round(timings_sorted[min(p95_idx, len(timings_sorted) - 1)], 3),
        "min_ms": round(min(timings_ms), 3),
        "max_ms": round(max(timings_ms), 3),
    }


def get_model_size_mb(path) -> float:
    """Size of a serialized model file on disk, in MB. Takes the direct file
    size — for models split across multiple files, sum them before calling,
    or extend this to accept a directory."""
    path = Path(path)
    if path.is_dir():
        total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        total_bytes = path.stat().st_size
    return round(total_bytes / (1024 * 1024), 3)
