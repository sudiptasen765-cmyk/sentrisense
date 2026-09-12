"""
ml/evaluation/compare_models.py

Final Phase 4 comparison: Logistic Regression (Phase 3) vs. Linear SVM
(Phase 4A) vs. fine-tuned DistilBERT (Phase 4B).

Fairness fix: train_baseline.py and train_svm.py deliberately never touched
imdb_test.csv (by design — see their own docstrings: "Test is reserved for
final deployment-model evaluation once all three models have been
compared"). This script is that reservation being used. It loads the saved
Phase 3/4A artifacts and evaluates them on imdb_test.csv for the very first
time here, so their numbers are finally comparable to DistilBERT's
already-reported test metrics (train_transformer.py's Phase 4B protocol
evaluated on test from the start, per that phase's own spec).

Latency caveat (documented, not hidden): Logistic Regression and SVM
latency below is measured live, on THIS machine, right now. DistilBERT's
latency is pulled from its saved report, measured on a Google Colab CPU
during Phase 4B training (no local GPU was available for that phase — see
reports/results/model3_distilbert_metrics.json). These are not the same
physical machine. The ~150-200x latency gap between DistilBERT and the
classical models is large enough that this caveat does not change the
qualitative conclusion, but the exact millisecond figures for DistilBERT
should not be treated as precisely measured on this machine.

Usage:
    python ml/evaluation/compare_models.py

Requires:
    - ml/training/train_baseline.py already run (produces tfidf_vectorizer.joblib, logreg_model.joblib)
    - ml/training/train_svm.py already run (produces svm_model.joblib)
    - ml/training/train_transformer.py (or its Colab equivalent) already run
      (produces reports/results/model3_distilbert_metrics.json)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute_classification_metrics, get_model_size_mb, measure_inference_latency  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "v1.0"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
LOGREG_PATH = MODEL_DIR / "logreg_model.joblib"
SVM_PATH = MODEL_DIR / "svm_model.joblib"
DISTILBERT_REPORT_PATH = RESULTS_DIR / "model3_distilbert_metrics.json"

OUT_REPORT = RESULTS_DIR / "final_comparison.json"
OUT_TABLE_MD = PROJECT_ROOT / "docs" / "model_comparison.md"

N_LATENCY_SAMPLES = 200  # matches the sample count used in Phase 3/4A latency measurements


def evaluate_classical_model(name, model, vectorizer, X_test_text, y_test, model_path, extra_size_paths=None):
    """Runs a saved classical (TF-IDF-based) model against the test set for
    the first time, computing metrics and latency exactly like train_baseline.py
    and train_svm.py did for validation — same functions, same conventions,
    just pointed at test now that all three models are ready to be compared."""
    print(f"\n[{name}] transforming test set with the shared vectorizer...")
    X_test = vectorizer.transform(X_test_text)

    print(f"[{name}] predicting on test set (first time touching test for this model)...")
    preds = model.predict(X_test)
    test_metrics = compute_classification_metrics(y_test, preds)
    print(f"[{name}] test_acc={test_metrics['accuracy']}  test_f1={test_metrics['f1']}")

    print(f"[{name}] measuring live CPU inference latency on this machine...")
    test_samples = X_test_text.tolist()

    def predict_single(text: str) -> int:
        vec = vectorizer.transform([text])
        return int(model.predict(vec)[0])

    latency = measure_inference_latency(predict_single, test_samples, n_repeats=N_LATENCY_SAMPLES)
    print(f"[{name}] mean={latency['mean_ms']}ms  p95={latency['p95_ms']}ms")

    size_paths = [VECTORIZER_PATH, model_path] + (extra_size_paths or [])
    total_size_mb = round(sum(get_model_size_mb(p) for p in size_paths), 3)

    return {
        "model_name": name,
        "test_metrics": test_metrics,
        "test_latency_ms": latency,
        "model_size_mb": total_size_mb,
        "latency_measured_on": "local machine (this run)",
    }


def load_distilbert_summary():
    if not DISTILBERT_REPORT_PATH.exists():
        print(
            f"[error] DistilBERT report not found at {DISTILBERT_REPORT_PATH}. "
            "Run Phase 4B (train_transformer.py or its Colab equivalent) first.",
            file=sys.stderr,
        )
        sys.exit(1)

    report = json.loads(DISTILBERT_REPORT_PATH.read_text(encoding="utf-8"))
    return {
        "model_name": "distilbert_finetuned",
        "test_metrics": report["final_test_metrics"],
        "test_latency_ms": report["inference_latency_cpu_val_samples"],
        "model_size_mb": report["disk_size_mb"],
        "latency_measured_on": "Google Colab CPU during Phase 4B training (NOT this machine)",
        "gpu_latency_ms_secondary": report.get("inference_latency_gpu_val_samples_SECONDARY_ONLY"),
        "parameter_count": report.get("parameter_count"),
        "training_time_s": report.get("training_time_s"),
    }


def build_verdict(logreg_summary, svm_summary, distilbert_summary) -> dict:
    """Purely descriptive comparison logic, no fabricated thresholds —
    reports the actual measured deltas and lets the numbers speak."""
    baseline_f1 = logreg_summary["test_metrics"]["f1"]
    svm_f1 = svm_summary["test_metrics"]["f1"]
    distilbert_f1 = distilbert_summary["test_metrics"]["f1"]

    distilbert_gain_over_baseline_f1 = round(distilbert_f1 - baseline_f1, 4)
    svm_delta_over_baseline_f1 = round(svm_f1 - baseline_f1, 4)

    size_ratio = round(distilbert_summary["model_size_mb"] / logreg_summary["model_size_mb"], 1)
    latency_ratio = round(
        distilbert_summary["test_latency_ms"]["mean_ms"] / logreg_summary["test_latency_ms"]["mean_ms"], 1
    )

    return {
        "svm_vs_logreg_f1_delta": svm_delta_over_baseline_f1,
        "svm_beats_baseline": svm_delta_over_baseline_f1 > 0,
        "distilbert_vs_logreg_f1_gain": distilbert_gain_over_baseline_f1,
        "distilbert_size_multiple_vs_logreg": size_ratio,
        "distilbert_cpu_latency_multiple_vs_logreg": latency_ratio,
        "summary": (
            f"Linear SVM {'improves on' if svm_delta_over_baseline_f1 > 0 else 'does not improve on'} "
            f"the Logistic Regression baseline (F1 delta: {svm_delta_over_baseline_f1:+.4f}), at effectively "
            f"the same size and latency. DistilBERT gains {distilbert_gain_over_baseline_f1:+.4f} F1 over the "
            f"baseline, but at {size_ratio}x the model size and roughly {latency_ratio}x the CPU inference "
            "latency (caveat: DistilBERT's latency was measured on a different machine than the classical "
            "models' — see latency_measured_on fields). For a cost-sensitive, low-latency AWS Lambda "
            "deployment, this is a real accuracy-vs-efficiency tradeoff, not a clear-cut win either way — "
            "the right choice depends on whether the accuracy gain matters enough to the product to justify "
            "materially slower and more expensive inference."
        ),
    }


def main() -> None:
    try:
        import joblib
        import pandas as pd
    except ImportError as exc:
        print(f"[error] Missing dependency: {exc}. Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    test_path = PROCESSED_DIR / "imdb_test.csv"
    if not test_path.exists():
        print("[error] imdb_test.csv not found. Run ml/preprocessing/build_splits.py first.", file=sys.stderr)
        sys.exit(1)

    for p, label in [
        (VECTORIZER_PATH, "Phase 3 vectorizer"),
        (LOGREG_PATH, "Phase 3 Logistic Regression model"),
        (SVM_PATH, "Phase 4A SVM model"),
    ]:
        if not p.exists():
            print(f"[error] {label} not found at {p}. Run its training script first.", file=sys.stderr)
            sys.exit(1)

    print("[load] reading imdb_test.csv (first and only use of test in this comparison run)...")
    test_df = pd.read_csv(test_path)
    X_test_text = test_df["text_clean_classical"].fillna("")
    y_test = test_df["label"]
    print(f"[load] test={len(test_df)} rows")

    print("[load] loading shared TF-IDF vectorizer, Logistic Regression, SVM models...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    logreg_model = joblib.load(LOGREG_PATH)
    svm_model = joblib.load(SVM_PATH)

    logreg_summary = evaluate_classical_model(
        "tfidf_logreg", logreg_model, vectorizer, X_test_text, y_test, LOGREG_PATH
    )
    svm_summary = evaluate_classical_model(
        "tfidf_svm", svm_model, vectorizer, X_test_text, y_test, SVM_PATH
    )

    print("\n[distilbert] loading already-computed test metrics from Phase 4B report (not re-running inference)...")
    distilbert_summary = load_distilbert_summary()
    print(
        f"[distilbert] test_acc={distilbert_summary['test_metrics']['accuracy']}  "
        f"test_f1={distilbert_summary['test_metrics']['f1']}"
    )

    verdict = build_verdict(logreg_summary, svm_summary, distilbert_summary)

    print("\n" + "=" * 78)
    print(f"{'Model':<20}{'Test Acc':>10}{'Test F1':>10}{'Size (MB)':>12}{'Mean Lat (ms)':>16}")
    print("-" * 78)
    for s in (logreg_summary, svm_summary, distilbert_summary):
        print(
            f"{s['model_name']:<20}"
            f"{s['test_metrics']['accuracy']:>10.4f}"
            f"{s['test_metrics']['f1']:>10.4f}"
            f"{s['model_size_mb']:>12.3f}"
            f"{s['test_latency_ms']['mean_ms']:>16.3f}"
        )
    print("=" * 78)
    print(f"\n{verdict['summary']}\n")

    report = {
        "models": {
            "tfidf_logreg": logreg_summary,
            "tfidf_svm": svm_summary,
            "distilbert_finetuned": distilbert_summary,
        },
        "verdict": verdict,
        "note": "Logistic Regression and SVM test metrics/latency computed here, for the first "
                "time this test set has been used for either model. DistilBERT's metrics are "
                "pulled from its own Phase 4B report (already test-evaluated there per that "
                "phase's protocol) rather than re-run, to avoid reloading the large model here.",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] Full comparison report -> {OUT_REPORT.relative_to(PROJECT_ROOT)}")

    # Lightweight markdown table for docs/model_comparison.md — hand-editable
    # afterward for narrative, but the numbers here are the real, measured ones.
    md_lines = [
        "# Phase 4 Model Comparison\n",
        "All three models evaluated on the same held-out `imdb_test.csv`.\n",
        "| Model | Test Accuracy | Test F1 | Size (MB) | Mean CPU Latency (ms) | Latency measured on |",
        "|---|---|---|---|---|---|",
    ]
    for s in (logreg_summary, svm_summary, distilbert_summary):
        md_lines.append(
            f"| {s['model_name']} | {s['test_metrics']['accuracy']:.4f} | {s['test_metrics']['f1']:.4f} | "
            f"{s['model_size_mb']:.3f} | {s['test_latency_ms']['mean_ms']:.3f} | {s['latency_measured_on']} |"
        )
    md_lines.append(f"\n## Verdict\n\n{verdict['summary']}\n")

    OUT_TABLE_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[saved] Markdown comparison table -> {OUT_TABLE_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()