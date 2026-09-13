"""
backend/tests/test_api.py

Systematic validation tests against the LIVE SentriSense /predict endpoint,
per the project roadmap's Section 10 (API Validation Requirements) and
Section 13 (Local Testing) / Section 14 (AWS Testing) checklists.

These hit the real, deployed API Gateway + Lambda -- not a mock -- so a
green test run here is genuine evidence the deployed system behaves
correctly, not just that the code compiles.

Usage:
    pytest backend/tests/test_api.py -v
"""

import requests

API_URL = "https://av9stuspv9.execute-api.ap-south-1.amazonaws.com/predict"


def test_missing_body():
    """POST with no body at all -> 400 invalid_request."""
    response = requests.post(API_URL)
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "invalid_request"


def test_invalid_json():
    """POST with a body that isn't valid JSON -> 400 invalid_request."""
    response = requests.post(
        API_URL,
        data="this is not json{{{",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "invalid_request"


def test_missing_text_field():
    """POST valid JSON but no 'text' key -> 422 validation_error."""
    response = requests.post(API_URL, json={"not_text": "hello"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert "text" in body["message"].lower()


def test_empty_text():
    """POST with text: "" -> 422 validation_error."""
    response = requests.post(API_URL, json={"text": ""})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"


def test_whitespace_only_text():
    """POST with text: "   " -> 422 validation_error."""
    response = requests.post(API_URL, json={"text": "    \n\t  "})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"


def test_text_exceeds_max_length():
    """POST with text longer than MAX_INPUT_LENGTH (5000 chars) -> 422."""
    long_text = "This movie was great. " * 300  # well over 5000 chars
    assert len(long_text) > 5000
    response = requests.post(API_URL, json={"text": long_text})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"


def test_unsupported_http_method():
    """GET on /predict -> 405, confirmed via direct verification
    (API Gateway rejects the unrouted method before reaching Lambda)."""
    response = requests.get(API_URL)
    assert response.status_code == 405


def test_valid_positive_review():
    """A clearly positive review -> sentiment: positive, real confidence."""
    response = requests.post(
        API_URL,
        json={"text": "This film was a masterpiece, beautifully shot and deeply moving."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "positive"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model"] == "tfidf-logreg-v1.0"


def test_valid_negative_review():
    """A clearly negative review -> sentiment: negative, real confidence."""
    response = requests.post(
        API_URL,
        json={"text": "The movie was boring, slow and disappointing."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "negative"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model"] == "tfidf-logreg-v1.0"


def test_no_internal_details_leaked():
    """Error responses must never expose stack traces, file paths, or AWS
    internals -- spot check on a validation error response."""
    response = requests.post(API_URL, json={"text": ""})
    body_text = response.text.lower()
    for forbidden in ("traceback", "c:\\", "/var/task", "botocore", "s3.amazonaws"):
        assert forbidden not in body_text
