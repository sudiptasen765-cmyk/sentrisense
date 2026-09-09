"""
ml/training/train_baseline.py

Trains Model 1 (TF-IDF + Logistic Regression) — the baseline classical model.

Hyperparameter tuning is done manually against the dedicated validation set
(ml/data/processed/imdb_val.csv), NOT via cross-validation on train and NOT
touching imdb_test.csv at all. Test is reserved for final deployment-model
evaluation once all three models (Phase 3 + Phase 4) have been compared.

Usage:
    python ml/training/train_baseline.py

Requires ml/preprocessing/build_splits.py to have been run first.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metrics import compute_classification_metrics, get_model_size_mb, measure_inference_latency  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODEL_OUT_DIR = PROJECT_ROOT / "ml" / "models" / "v1.0"
REPORT_OUT = PROJECT_ROOT / "reports" / "results" / "model1_tfidf_logreg_metrics.json"

RANDOM_STATE = 42

# Extended after the initial run showed monotonically increasing validation
# accuracy all the way to C=10 (the prior upper bound) — that means the true
# optimum wasn't necessarily inside the original range. Extending further so
# we select based on an actual peak/plateau, not an artifact of where the
# search happened to stop.
C_GRID = [0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0]


def main() -> None:
    try:
        import joblib
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
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

    print("[load] reading processed IMDb train/val splits...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    print(f"[load] train={len(train_df)} rows, val={len(val_df)} rows")

    X_train_text = train_df["text_clean_classical"].fillna("")
    y_train = train_df["label"]
    X_val_text = val_df["text_clean_classical"].fillna("")
    y_val = val_df["label"]

    print("[vectorize] fitting TF-IDF on train only (val/test never seen during fit)...")
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )
    t0 = time.perf_counter()
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    vectorize_time_s = time.perf_counter() - t0
    print(f"[vectorize] done in {vectorize_time_s:.1f}s, vocab size={len(vectorizer.vocabulary_)}")

    print(f"\n[tune] evaluating C in {C_GRID} against validation set...")
    best_c = None
    best_f1 = -1.0
    best_model = None
    tuning_results = []

    for c in C_GRID:
        t0 = time.perf_counter()
        clf = LogisticRegression(C=c, max_iter=1000, random_state=RANDOM_STATE)
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

    print("[measure] inference latency on validation samples...")
    val_samples = X_val_text.tolist()

    def predict_single(text: str) -> int:
        vec = vectorizer.transform([text])
        return int(best_model.predict(vec)[0])

    latency = measure_inference_latency(predict_single, val_samples, n_repeats=200)
    print(f"[measure] mean={latency['mean_ms']}ms  p95={latency['p95_ms']}ms")

    print("[serialize] saving vectorizer + model...")
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = MODEL_OUT_DIR / "tfidf_vectorizer.joblib"
    model_path = MODEL_OUT_DIR / "logreg_model.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(best_model, model_path)

    combined_size_mb = get_model_size_mb(vectorizer_path) + get_model_size_mb(model_path)
    print(f"[serialize] vectorizer={get_model_size_mb(vectorizer_path)}MB, "
          f"model={get_model_size_mb(model_path)}MB, total={round(combined_size_mb, 3)}MB")

    final_val_metrics = compute_classification_metrics(y_val, best_model.predict(X_val))

    report = {
        "model_name": "tfidf_logreg",
        "model_version": "v1.0",
        "hyperparameter_grid_tested": C_GRID,
        "selected_hyperparameters": {"C": best_c, "max_iter": 1000},
        "vectorizer_config": {
            "max_features": 50_000,
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.9,
            "sublinear_tf": True,
            "vocab_size_actual": len(vectorizer.vocabulary_),
        },
        "tuning_results_all_candidates": tuning_results,
        "final_validation_metrics": final_val_metrics,
        "inference_latency_val_samples": latency,
        "model_size_mb": {
            "vectorizer": get_model_size_mb(vectorizer_path),
            "classifier": get_model_size_mb(model_path),
            "total": round(combined_size_mb, 3),
        },
        "vectorize_fit_transform_time_s": round(vectorize_time_s, 1),
        "note": "Evaluated against validation set only. imdb_test.csv was NOT "
                "used and remains untouched for final model-comparison evaluation.",
    }

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[saved] Full report -> {REPORT_OUT.relative_to(PROJECT_ROOT)}")
    print(f"[saved] Model files -> {MODEL_OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()