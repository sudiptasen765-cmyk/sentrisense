"""
backend/lambda/analyze/handler.py

Lambda entry point for POST /predict. Validates input, runs inference,
logs the prediction to DynamoDB, returns a structured JSON response.
"""

import json
import os
import time
import uuid
import logging

import boto3
from inference import predict_sentiment

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PREDICTIONS_TABLE = os.environ["PREDICTIONS_TABLE"]
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", "5000"))

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(PREDICTIONS_TABLE)


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _error(status_code: int, error_code: str, message: str) -> dict:
    return _response(status_code, {"error": error_code, "message": message})


def handler(event, context):
    try:
        http_method = event.get("requestContext", {}).get("http", {}).get("method", "")
        if http_method and http_method != "POST":
            return _error(405, "method_not_allowed", "Only POST is supported.")

        raw_body = event.get("body")
        if raw_body is None:
            return _error(400, "invalid_request", "Request body is required.")

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return _error(400, "invalid_request", "Request body must be valid JSON.")

        text = payload.get("text")
        if text is None:
            return _error(422, "validation_error", "The 'text' field is required.")
        if not isinstance(text, str):
            return _error(422, "validation_error", "The 'text' field must be a string.")
        if text.strip() == "":
            return _error(422, "validation_error", "The 'text' field cannot be empty or whitespace-only.")
        if len(text) > MAX_INPUT_LENGTH:
            return _error(
                422,
                "validation_error",
                f"The 'text' field exceeds the maximum length of {MAX_INPUT_LENGTH} characters.",
            )

        start = time.perf_counter()
        result = predict_sentiment(text)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)

        prediction_id = str(uuid.uuid4())
        try:
            table.put_item(
                Item={
                    "prediction_id": prediction_id,
                    "timestamp": int(time.time()),
                    "sentiment": result["sentiment"],
                    "confidence": str(result["confidence"]),  # DynamoDB Decimal-safe
                    "model": result["model"],
                    "input_length": len(text),
                    "latency_ms": str(latency_ms),
                    "status": "success",
                }
            )
        except Exception:
            logger.exception("Failed to write prediction to DynamoDB (non-fatal)")

        return _response(200, result)

    except Exception:
        logger.exception("Unhandled error in Lambda handler")
        return _error(500, "internal_error", "An internal error occurred.")