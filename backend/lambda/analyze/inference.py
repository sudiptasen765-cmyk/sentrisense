"""
backend/lambda/analyze/inference.py

Dual-model sentiment inference for SentriSense Lambda.

Supports two models, selected via the "model" request parameter:
  - "logreg"     (default) -- TF-IDF + Logistic Regression, Phase 3 locked
                  baseline. ~2.24 MB, sub-millisecond CPU inference.
  - "distilbert" -- Fine-tuned DistilBERT, ONNX INT8 quantized, Phase 5
                  optimized deployment artifact. ~64.3 MB, ~223ms CPU
                  inference (measured locally, see docs/deployment.md).

Both models are downloaded from S3 on first use per Lambda execution
environment (cold start) and cached in /tmp + module-level globals for
warm invocations. Only the model actually requested is downloaded/loaded
-- a logreg-only request never pays the cost of loading DistilBERT, and
vice versa.

Label mapping (confirmed against ml/preprocessing/build_splits.py and
verified with test_onnx_local.py against real IMDb-style reviews):
  0 = negative
  1 = positive
"""

import os
import numpy as np
import boto3
import joblib

# ---------------------------------------------------------------------------
# Config (all via environment variables -- see docs/deployment.md)
# ---------------------------------------------------------------------------
MODEL_BUCKET = os.environ["MODEL_BUCKET"]

# LogReg artifacts
VECTORIZER_KEY = os.environ["VECTORIZER_KEY"]
LOGREG_MODEL_KEY = os.environ["LOGREG_MODEL_KEY"]

# DistilBERT ONNX artifacts
DISTILBERT_ONNX_KEY = os.environ["DISTILBERT_ONNX_KEY"]
DISTILBERT_TOKENIZER_PREFIX = os.environ["DISTILBERT_TOKENIZER_PREFIX"]  # S3 "folder" holding tokenizer.json etc.

MAX_LENGTH = 512
LABEL_MAP = {0: "negative", 1: "positive"}

TMP_VECTORIZER_PATH = "/tmp/tfidf_vectorizer.joblib"
TMP_LOGREG_PATH = "/tmp/logreg_model.joblib"
TMP_ONNX_PATH = "/tmp/model_quantized.onnx"
TMP_TOKENIZER_DIR = "/tmp/distilbert_tokenizer"

s3_client = boto3.client("s3")

# Module-level caches -- persist across warm Lambda invocations
_vectorizer = None
_logreg_model = None
_onnx_session = None
_tokenizer = None


def _download_if_missing(bucket: str, key: str, local_path: str) -> None:
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        s3_client.download_file(bucket, key, local_path)


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


# ---------------------------------------------------------------------------
# LogReg path
# ---------------------------------------------------------------------------
def _load_logreg():
    global _vectorizer, _logreg_model

    if _vectorizer is None or _logreg_model is None:
        _download_if_missing(MODEL_BUCKET, VECTORIZER_KEY, TMP_VECTORIZER_PATH)
        _download_if_missing(MODEL_BUCKET, LOGREG_MODEL_KEY, TMP_LOGREG_PATH)
        _vectorizer = joblib.load(TMP_VECTORIZER_PATH)
        _logreg_model = joblib.load(TMP_LOGREG_PATH)

    return _vectorizer, _logreg_model


def _predict_logreg(text: str) -> dict:
    vectorizer, model = _load_logreg()

    vec = vectorizer.transform([text])
    proba = model.predict_proba(vec)[0]  # [P(negative), P(positive)]
    pred_label = int(model.predict(vec)[0])

    return {
        "sentiment": LABEL_MAP[pred_label],
        "confidence": round(float(proba[pred_label]), 4),
        "model": "tfidf-logreg-v1.0",
    }


# ---------------------------------------------------------------------------
# DistilBERT ONNX path
# ---------------------------------------------------------------------------
def _load_distilbert():
    global _onnx_session, _tokenizer

    if _onnx_session is None:
        _download_if_missing(MODEL_BUCKET, DISTILBERT_ONNX_KEY, TMP_ONNX_PATH)
        # onnxruntime is imported lazily so a logreg-only Lambda invocation
        # never pays the import cost of a dependency it doesn't need.
        import onnxruntime as ort
        _onnx_session = ort.InferenceSession(TMP_ONNX_PATH, providers=["CPUExecutionProvider"])

    if _tokenizer is None:
        os.makedirs(TMP_TOKENIZER_DIR, exist_ok=True)
        for fname in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            _download_if_missing(
                MODEL_BUCKET,
                f"{DISTILBERT_TOKENIZER_PREFIX.rstrip('/')}/{fname}",
                os.path.join(TMP_TOKENIZER_DIR, fname),
            )
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(TMP_TOKENIZER_DIR)

    return _onnx_session, _tokenizer


def _predict_distilbert(text: str) -> dict:
    session, tokenizer = _load_distilbert()

    inputs = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="np")
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    logits = session.run(["logits"], ort_inputs)[0][0]  # shape: (2,)
    probs = _softmax(logits)
    pred_label = int(np.argmax(logits))

    return {
        "sentiment": LABEL_MAP[pred_label],
        "confidence": round(float(probs[pred_label]), 4),
        "model": "distilbert-onnx-int8",
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
VALID_MODELS = {"logreg", "distilbert"}


def predict_sentiment(text: str, model: str = "logreg") -> dict:
    """Runs sentiment inference using the requested model.

    Args:
        text: the review text to classify.
        model: "logreg" (default) or "distilbert".

    Returns:
        {"sentiment": "positive"|"negative", "confidence": float, "model": str}

    Raises:
        ValueError: if `model` is not one of VALID_MODELS. Caller (handler.py)
        is responsible for turning this into a 422 validation_error response
        -- this function does not know about HTTP.
    """
    if model not in VALID_MODELS:
        raise ValueError(f"Unknown model '{model}'. Must be one of {sorted(VALID_MODELS)}.")

    if model == "distilbert":
        return _predict_distilbert(text)
    return _predict_logreg(text)