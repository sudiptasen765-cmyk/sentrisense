"""
backend/lambda/analyze/inference.py

Loads the TF-IDF vectorizer + Logistic Regression model from S3 (cached in
/tmp between warm invocations) and runs sentiment inference.
"""

import os
import boto3
import joblib

MODEL_BUCKET = os.environ["MODEL_BUCKET"]
VECTORIZER_KEY = os.environ["VECTORIZER_KEY"]
MODEL_KEY = os.environ["MODEL_KEY"]

TMP_VECTORIZER_PATH = "/tmp/tfidf_vectorizer.joblib"
TMP_MODEL_PATH = "/tmp/logreg_model.joblib"

_vectorizer = None
_model = None

s3_client = boto3.client("s3")


def _download_if_missing(bucket: str, key: str, local_path: str) -> None:
    if not os.path.exists(local_path):
        s3_client.download_file(bucket, key, local_path)


def load_model():
    """Loads vectorizer + model, caching in the Lambda execution environment
    (module-level globals persist across warm invocations)."""
    global _vectorizer, _model

    if _vectorizer is None or _model is None:
        _download_if_missing(MODEL_BUCKET, VECTORIZER_KEY, TMP_VECTORIZER_PATH)
        _download_if_missing(MODEL_BUCKET, MODEL_KEY, TMP_MODEL_PATH)
        _vectorizer = joblib.load(TMP_VECTORIZER_PATH)
        _model = joblib.load(TMP_MODEL_PATH)

    return _vectorizer, _model


def predict_sentiment(text: str) -> dict:
    """Returns {"sentiment": "positive"|"negative", "confidence": float}.

    Confidence is derived from the model's decision function distance,
    passed through a sigmoid, since plain LogisticRegression exposes
    predict_proba directly -- so we use that, which IS a real, calibrated
    probability (not an approximation), matching the roadmap's requirement
    that confidence must come from actual model inference.
    """
    vectorizer, model = load_model()

    vec = vectorizer.transform([text])
    proba = model.predict_proba(vec)[0]  # [P(negative), P(positive)]
    pred_label = int(model.predict(vec)[0])

    sentiment = "positive" if pred_label == 1 else "negative"
    confidence = float(proba[pred_label])

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "model": "tfidf-logreg-v1.0",
    }