"""
test_inference_local.py

Local test harness for backend/lambda/analyze/inference.py's dual-model
logic, BEFORE any AWS infrastructure (S3/Lambda) exists.

inference.py normally downloads models from S3 on first use. Since S3
doesn't exist yet at this point in Phase 6, this script monkey-patches
inference.py's module-level caches directly with locally-loaded models
(bypassing the S3 download step entirely) so we can verify the actual
prediction logic -- tensor handling, label mapping, confidence calc,
model dispatch -- works correctly end to end for BOTH models.

Run from the project root (D:\\sentrisense) with the venv activated:
    python test_inference_local.py

Requires env vars to be set (dummy values are fine since we bypass the
S3 download path) -- this script sets them itself before importing
inference.py.
"""

import os
import sys

# --- Set dummy env vars BEFORE importing inference.py, since the module
# reads them at import time via os.environ[...] ---
os.environ.setdefault("MODEL_BUCKET", "unused-local-test")
os.environ.setdefault("VECTORIZER_KEY", "unused")
os.environ.setdefault("LOGREG_MODEL_KEY", "unused")
os.environ.setdefault("DISTILBERT_ONNX_KEY", "unused")
os.environ.setdefault("DISTILBERT_TOKENIZER_PREFIX", "unused")

sys.path.insert(0, "backend/lambda/analyze")
import inference  # noqa: E402

import joblib
import onnxruntime as ort
from transformers import AutoTokenizer

LOGREG_DIR = "ml/models/v1.0"
DISTILBERT_TOKENIZER_DIR = "ml/models/distilbert/v1.0"
DISTILBERT_ONNX_PATH = "ml/models/distilbert/v1.0_onnx/model_quantized.onnx"


def preload_local_models():
    """Bypasses inference.py's S3 download logic by populating its
    module-level caches directly from local files."""
    print("[preload] loading LogReg artifacts from local disk (bypassing S3)...")
    inference._vectorizer = joblib.load(os.path.join(LOGREG_DIR, "tfidf_vectorizer.joblib"))
    inference._logreg_model = joblib.load(os.path.join(LOGREG_DIR, "logreg_model.joblib"))
    print("[preload] LogReg OK")

    print("[preload] loading DistilBERT ONNX artifacts from local disk (bypassing S3)...")
    inference._onnx_session = ort.InferenceSession(
        DISTILBERT_ONNX_PATH, providers=["CPUExecutionProvider"]
    )
    inference._tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_TOKENIZER_DIR)
    print("[preload] DistilBERT OK")


def main():
    preload_local_models()

    test_cases = [
        ("This movie was fantastic. I loved every minute of it.", "positive"),
        ("This movie was boring, disappointing and painfully slow.", "negative"),
    ]

    print("\n" + "=" * 78)
    all_passed = True

    for model_name in ("logreg", "distilbert"):
        print(f"\n--- model={model_name} ---")
        for text, expected in test_cases:
            result = inference.predict_sentiment(text, model=model_name)
            correct = result["sentiment"] == expected
            all_passed = all_passed and correct
            status = "PASS" if correct else "FAIL"
            print(f"[{status}] expected={expected:<9} got={result['sentiment']:<9} "
                  f"confidence={result['confidence']:<8} model_tag={result['model']}")

    print("\n--- invalid model parameter handling ---")
    try:
        inference.predict_sentiment("some text", model="not-a-real-model")
        print("[FAIL] expected ValueError for invalid model, but none was raised")
        all_passed = False
    except ValueError as exc:
        print(f"[PASS] correctly raised ValueError: {exc}")

    print("\n" + "=" * 78)
    if all_passed:
        print("[result] All local inference checks PASSED for both models.")
    else:
        print("[result] SOME CHECKS FAILED -- resolve before deploying to Lambda.")


if __name__ == "__main__":
    main()