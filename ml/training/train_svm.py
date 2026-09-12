"""
ml/training/train_svm.py

Trains Model 2 (TF-IDF + Linear SVM) — Phase 4A comparison candidate against
the locked Phase 3 baseline (TF-IDF + Logistic Regression).

Fairness note: this script does NOT refit a new TF-IDF vectorizer. It loads
the exact fitted vectorizer artifact produced by train_baseline.py
(ml/models/v1.0/tfidf_vectorizer.joblib) so the SVM sees an identical feature
space to the baseline — the strongest possible guarantee of an apples-to-apples
comparison, stronger than merely matching the config numbers (50k vocab, 1-2
grams) independently.

Hyperparameter tuning is done manually against the dedicated validation set
(ml/data/processed/imdb_val.csv), NOT via cross-validation on train and NOT
touching imdb_test.csv at all — same protocol as Phase 3. Test is reserved
for the final Phase 4 model-comparison evaluation once Logistic Regression,
SVM, and DistilBERT have all been benchmarked.

Calibration note: LinearSVC has no predict_proba. If the eventual API needs
confidence scores, CalibratedClassifierCV is fit as a SEPARATE, clearly
labeled step below (see calibrate_and_report()) so its added latency
overhead never contaminates the core SVM benchmark numbers used for the
Logistic Regression vs. SVM vs. DistilBERT comparison table.

Usage:
    python ml/training/train_svm.py

Requires:
    - ml/preprocessing/build_splits.py to have been run (Phase 2)
    - ml/training/train_baseline.py to have been run (Phase 3, produces the
      vectorizer this script loads)
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metrics import compute_classification_metrics, get_model_size_mb, measure_inference_latency  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "v1.0"
REPORT_OUT = PROJECT_ROOT / "reports" / "results" / "model2_tfidf_svm_metrics.json"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"  # Phase 3 artifact, reused as-is
SVM_MODEL_OUT = MODEL_DIR / "svm_model.joblib"
CALIBRATED_MODEL_OUT = MODEL_DIR / "svm_model_calibrated.joblib"

RANDOM_STATE = 42  # matches Phase 3 — never changed independently, for comparability

# Mirrors the Phase 3 grid shape/spirit; LinearSVC's C plays an analogous
# regularization role to LogisticRegression's C.
C_GRID = [0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0]


def load_vectorizer():
    if not VECTORIZER_PATH.exists():
        print(
            f"[error] Phase 3 vectorizer not found at {VECTORIZER_PATH}. "
            "Run ml/training/train_baseline.py first — Phase 4A intentionally "
            "reuses that exact fitted vectorizer rather than refitting one.",
            file=sys.stderr,
        )
        sys.exit(1)

    import joblib

    print(f"[load] reusing Phase 3 fitted TF-IDF vectorizer -> {VECTORIZER_PATH.relative_to(PROJECT_ROOT)}")
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"[load] vocab size={len(vectorizer.vocabulary_)} (should match Phase 3 exactly)")
    return vectorizer


def tune_svm(X_train, y_train, X_val, y_val):
    from sklearn.svm import LinearSVC

    print(f"\n[tune] evaluating C in {C_GRID} against validation set...")
    best_c = None
    best_f1 = -1.0
    best_model = None
    tuning_results = []

    for c in C_GRID:
        t0 = time.perf_counter()
        clf = LinearSVC(C=c, max_iter=5000, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t0

        val_preds = clf.predict(X_val)
        val_metrics = compute_classification_metrics(y_val, val_preds)
        tuning_results.append({"C": c, "train_time_s": round(train_time_s, 1), **val_metrics})

        print(
            f"  C={c:<6} train_time={train_time_s:5.1f}s  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_c = c
            best_model = clf

    print(f"\n[selected] C={best_c} (best validation F1={best_f1:.4f})")
    return best_c, best_model, tuning_results


def calibrate_and_report(best_model, X_train, y_train, X_val, y_val, vectorizer, val_samples):
    """Separate, clearly-labeled calibration step. Fits CalibratedClassifierCV
    on top of the tuned LinearSVC so the API can expose confidence scores.
    Reported latency here is NEVER used in the core Logistic Regression vs.
    SVM vs. DistilBERT comparison table — only the uncalibrated numbers from
    main() are, per the Phase 4A "don't distort the core benchmark" rule.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    print("\n[calibrate] fitting CalibratedClassifierCV on top of the tuned LinearSVC (separate step)...")
    base_clf = LinearSVC(C=best_model.C, max_iter=5000, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    calibrated = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    calibration_time_s = time.perf_counter() - t0

    val_preds = calibrated.predict(X_val)
    calibrated_metrics = compute_classification_metrics(y_val, val_preds)
    print(
        f"[calibrate] done in {calibration_time_s:.1f}s  "
        f"val_acc={calibrated_metrics['accuracy']:.4f}  val_f1={calibrated_metrics['f1']:.4f}"
    )

    def predict_single_calibrated(text: str):
        vec = vectorizer.transform([text])
        return calibrated.predict_proba(vec)[0].tolist()

    calibrated_latency = measure_inference_latency(predict_single_calibrated, val_samples, n_repeats=200)
    print(
        f"[calibrate] calibrated inference (with proba) mean={calibrated_latency['mean_ms']}ms  "
        f"p95={calibrated_latency['p95_ms']}ms  "
        "(overhead vs. uncalibrated SVM — expected and documented, not a benchmark regression)"
    )

    import joblib

    joblib.dump(calibrated, CALIBRATED_MODEL_OUT)
    calibrated_size_mb = get_model_size_mb(CALIBRATED_MODEL_OUT)
    print(f"[calibrate] saved -> {CALIBRATED_MODEL_OUT.relative_to(PROJECT_ROOT)} ({calibrated_size_mb}MB)")

    return {
        "note": "Separate step, added ONLY to supply confidence scores for the API. "
                "Its latency/size overhead is documented here and must NOT be substituted "
                "for the core (uncalibrated) SVM numbers in the model comparison table.",
        "calibration_method": "sigmoid",
        "calibration_cv_folds": 5,
        "calibration_fit_time_s": round(calibration_time_s, 1),
        "validation_metrics": calibrated_metrics,
        "inference_latency_with_proba": calibrated_latency,
        "model_size_mb": calibrated_size_mb,
    }


def main() -> None:
    try:
        import joblib  # noqa: F401
        import pandas as pd
        from sklearn.svm import LinearSVC  # noqa: F401
    except ImportError as exc:
        print(f"[error] Missing dependency: {exc}. Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    train_path = PROCESSED_DIR / "imdb_train.csv"
    val_path = PROCESSED_DIR / "imdb_val.csv"
    if not train_path.exists() or not val_path.exists():
        print(
            "[error] Processed splits not found. Run ml/preprocessing/build_splits.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[load] reading processed IMDb train/val splits (same split as Phase 3)...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    print(f"[load] train={len(train_df)} rows, val={len(val_df)} rows")

    X_train_text = train_df["text_clean_classical"].fillna("")
    y_train = train_df["label"]
    X_val_text = val_df["text_clean_classical"].fillna("")
    y_val = val_df["label"]

    vectorizer = load_vectorizer()

    print("[vectorize] transforming train/val with the reused Phase 3 vectorizer (no refit)...")
    t0 = time.perf_counter()
    X_train = vectorizer.transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    vectorize_time_s = time.perf_counter() - t0
    print(f"[vectorize] done in {vectorize_time_s:.1f}s")

    best_c, best_model, tuning_results = tune_svm(X_train, y_train, X_val, y_val)

    print("\n[measure] inference latency on validation samples (uncalibrated SVM only)...")
    val_samples = X_val_text.tolist()

    def predict_single(text: str) -> int:
        vec = vectorizer.transform([text])
        return int(best_model.predict(vec)[0])

    latency = measure_inference_latency(predict_single, val_samples, n_repeats=200)
    print(f"[measure] mean={latency['mean_ms']}ms  p95={latency['p95_ms']}ms")

    print("[serialize] saving SVM model (vectorizer is the shared Phase 3 artifact, not re-saved)...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, SVM_MODEL_OUT)

    # Size comparison uses vectorizer + classifier together, same convention as
    # Phase 3's combined_size_mb, so model-size numbers are comparable apples-to-apples.
    combined_size_mb = get_model_size_mb(VECTORIZER_PATH) + get_model_size_mb(SVM_MODEL_OUT)
    print(
        f"[serialize] vectorizer={get_model_size_mb(VECTORIZER_PATH)}MB (shared/reused), "
        f"svm_model={get_model_size_mb(SVM_MODEL_OUT)}MB, total={round(combined_size_mb, 3)}MB"
    )

    final_val_metrics = compute_classification_metrics(y_val, best_model.predict(X_val))

    calibration_report = calibrate_and_report(
        best_model, X_train, y_train, X_val, y_val, vectorizer, val_samples
    )

    report = {
        "model_name": "tfidf_svm",
        "model_version": "v1.0",
        "vectorizer_reused_from": "ml/models/v1.0/tfidf_vectorizer.joblib (Phase 3, not refit)",
        "hyperparameter_grid_tested": C_GRID,
        "selected_hyperparameters": {"C": best_c, "max_iter": 5000},
        "tuning_results_all_candidates": tuning_results,
        "final_validation_metrics": final_val_metrics,
        "inference_latency_val_samples_uncalibrated": latency,
        "model_size_mb": {
            "vectorizer_shared": get_model_size_mb(VECTORIZER_PATH),
            "classifier": get_model_size_mb(SVM_MODEL_OUT),
            "total": round(combined_size_mb, 3),
        },
        "vectorize_transform_time_s": round(vectorize_time_s, 1),
        "calibration_separate_step": calibration_report,
        "note": "Evaluated against validation set only. imdb_test.csv was NOT "
                "used and remains untouched for final model-comparison evaluation. "
                "Uses the exact Phase 3 fitted vectorizer (no refit) for a strict "
                "apples-to-apples feature comparison against the Logistic Regression baseline.",
    }

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[saved] Full report -> {REPORT_OUT.relative_to(PROJECT_ROOT)}")
    print(f"[saved] SVM model -> {SVM_MODEL_OUT.relative_to(PROJECT_ROOT)}")
    print(f"[saved] Calibrated SVM model -> {CALIBRATED_MODEL_OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()