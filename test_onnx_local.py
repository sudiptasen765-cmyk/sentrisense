"""
test_onnx_local.py

Standalone local sanity check for the Phase 5 ONNX INT8 DistilBERT model,
BEFORE any Lambda code is written against it. Run this from the project
root (D:\\sentrisense) with the venv activated.

Confirms:
- tokenizer loads from ml/models/distilbert/v1.0/
- ONNX model loads from ml/models/distilbert/v1.0_onnx/model_quantized.onnx
- input tensor names (input_ids, attention_mask) are correct
- output tensor name (logits) is correct
- label mapping (0=negative, 1=positive) is correct
- softmax-derived confidence is a real, sane probability

Usage (from D:\\sentrisense, with .venv active):
    python test_onnx_local.py
"""

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

TOKENIZER_DIR = "ml/models/distilbert/v1.0"
ONNX_MODEL_PATH = "ml/models/distilbert/v1.0_onnx/model_quantized.onnx"
MAX_LENGTH = 512

LABEL_MAP = {0: "negative", 1: "positive"}


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def predict(text: str, tokenizer, session) -> dict:
    inputs = tokenizer(
        text, truncation=True, max_length=MAX_LENGTH, return_tensors="np"
    )
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    logits = session.run(["logits"], ort_inputs)[0][0]  # shape: (2,)
    probs = softmax(logits)
    pred_label = int(np.argmax(logits))

    return {
        "sentiment": LABEL_MAP[pred_label],
        "confidence": round(float(probs[pred_label]), 4),
        "raw_logits": logits.tolist(),
        "raw_probs": probs.tolist(),
    }


def main():
    print(f"[load] tokenizer from {TOKENIZER_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print("[load] tokenizer OK")

    print(f"[load] ONNX session from {ONNX_MODEL_PATH} ...")
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    print("[load] ONNX session OK")
    print(f"[check] input names: {[i.name for i in session.get_inputs()]}")
    print(f"[check] output names: {[o.name for o in session.get_outputs()]}")

    test_cases = [
        ("This movie was fantastic. I loved every minute of it.", "positive"),
        ("This movie was boring, disappointing and painfully slow.", "negative"),
    ]

    print("\n" + "=" * 70)
    all_correct = True
    for text, expected in test_cases:
        result = predict(text, tokenizer, session)
        correct = result["sentiment"] == expected
        all_correct = all_correct and correct
        status = "PASS" if correct else "FAIL"
        print(f"[{status}] expected={expected:<9} got={result['sentiment']:<9} "
              f"confidence={result['confidence']}")
        print(f"       text: {text!r}")
        print(f"       raw_logits: {result['raw_logits']}")

    print("=" * 70)
    if all_correct:
        print("\n[result] All sanity checks PASSED. Tensor contract, label mapping,")
        print("         and confidence calculation are all confirmed correct.")
    else:
        print("\n[result] SOME CHECKS FAILED. Do not proceed to Lambda code until")
        print("         this is resolved -- something in the tensor contract or")
        print("         label mapping is wrong.")


if __name__ == "__main__":
    main()