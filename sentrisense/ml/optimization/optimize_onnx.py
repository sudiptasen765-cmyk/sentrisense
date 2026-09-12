"""
ml/optimization/optimize_onnx.py

Phase 5 — converts the fine-tuned DistilBERT model to ONNX format, applies
onnxruntime's dynamic int8 quantization on top of that, and evaluates the
result against the same test set and latency protocol used elsewhere in
this project, so results are directly comparable to the original PyTorch
model and to ml/optimization/quantize.py's PyTorch-native quantization.

ONNX Runtime is CPU-optimized (uses vendor kernels like oneDNN/MLAS
depending on platform), which is often faster than plain PyTorch CPU
inference even before quantization — this script measures both the ONNX
conversion alone and ONNX + quantization together, so it's clear which
part of the improvement (if any) comes from which step.

Usage:
    python ml/optimization/optimize_onnx.py

Requires: pip install onnx onnxruntime
(not in the original requirements.txt — added specifically for Phase 5)
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metrics import compute_classification_metrics, get_model_size_mb, measure_inference_latency  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
DISTILBERT_DIR = PROJECT_ROOT / "ml" / "models" / "distilbert" / "v1.0"
ONNX_DIR = PROJECT_ROOT / "ml" / "models" / "distilbert" / "v1.0_onnx"
ONNX_FP32_PATH = ONNX_DIR / "model.onnx"
ONNX_INT8_PATH = ONNX_DIR / "model_quantized.onnx"
DISTILBERT_REPORT_PATH = PROJECT_ROOT / "reports" / "results" / "model3_distilbert_metrics.json"
REPORT_OUT = PROJECT_ROOT / "reports" / "results" / "optimization_onnx_metrics.json"

MAX_LENGTH = 512
N_LATENCY_SAMPLES = 100
EVAL_SAMPLE_SIZE = 3000  # full 25k test set x2 variants is impractical for CPU-only DistilBERT
                          # inference here; a fixed-seed stratified sample gives a reliable F1
                          # estimate in a fraction of the time. Documented deviation from
                          # compare_models.py's full-test-set evaluation.
RANDOM_STATE = 42


def export_to_onnx(model, tokenizer):
    import torch

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    dummy_inputs = tokenizer("This is a sample review for tracing the export graph.", return_tensors="pt")

    print(f"[export] tracing and exporting to ONNX -> {ONNX_FP32_PATH.relative_to(PROJECT_ROOT)}...")
    torch.onnx.export(
        model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        str(ONNX_FP32_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        dynamo=False,  # force the legacy TorchScript-based exporter — the newer
                       # dynamo-based exporter (PyTorch 2.x default) requires the
                       # separate 'onnxscript' package, which this project doesn't need
    )
    print(f"[export] done, size={get_model_size_mb(ONNX_FP32_PATH)} MB")


def quantize_onnx():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"[quantize] applying onnxruntime dynamic int8 quantization -> {ONNX_INT8_PATH.relative_to(PROJECT_ROOT)}...")
    quantize_dynamic(str(ONNX_FP32_PATH), str(ONNX_INT8_PATH), weight_type=QuantType.QInt8)
    print(f"[quantize] done, size={get_model_size_mb(ONNX_INT8_PATH)} MB")


def evaluate_onnx_session(session, tokenizer, X_test_text, y_test, label, batch_size=32):
    import numpy as np

    print(f"\n[evaluate:{label}] running inference on test set ({len(X_test_text)} rows, batched)...")
    t0 = time.perf_counter()
    preds = []
    for i in range(0, len(X_test_text), batch_size):
        batch_texts = X_test_text[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, truncation=True, max_length=MAX_LENGTH,
            padding=True, return_tensors="np",
        )
        ort_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        logits = session.run(["logits"], ort_inputs)[0]
        preds.extend(np.argmax(logits, axis=-1).tolist())
        if (i // batch_size) % 20 == 0:
            print(f"  ...{i + len(batch_texts)}/{len(X_test_text)} evaluated", flush=True)
    eval_time_s = time.perf_counter() - t0
    metrics = compute_classification_metrics(y_test, preds)
    print(f"[evaluate:{label}] done in {eval_time_s:.1f}s  acc={metrics['accuracy']}  f1={metrics['f1']}")

    def predict_single(text: str) -> int:
        inputs = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="np")
        ort_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        logits = session.run(["logits"], ort_inputs)[0]
        return int(np.argmax(logits, axis=-1)[0])

    latency = measure_inference_latency(predict_single, X_test_text, n_repeats=N_LATENCY_SAMPLES)
    print(f"[measure:{label}] mean={latency['mean_ms']}ms  p95={latency['p95_ms']}ms")

    return metrics, latency


def main() -> None:
    try:
        import onnxruntime as ort
        import pandas as pd
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        print(
            f"[error] Missing dependency: {exc}. Run: pip install onnx onnxruntime "
            "(also needs transformers/pandas from the main requirements.txt)",
            file=sys.stderr,
        )
        sys.exit(1)

    if not DISTILBERT_DIR.exists():
        print(f"[error] DistilBERT model not found at {DISTILBERT_DIR}. Run Phase 4B first.", file=sys.stderr)
        sys.exit(1)

    test_path = PROCESSED_DIR / "imdb_test.csv"
    if not test_path.exists():
        print("[error] imdb_test.csv not found. Run ml/preprocessing/build_splits.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[load] loading fine-tuned DistilBERT from {DISTILBERT_DIR.relative_to(PROJECT_ROOT)}...")
    tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_DIR))
    model.eval()

    print("[load] reading imdb_test.csv...")
    test_df = pd.read_csv(test_path)
    full_test_size = len(test_df)

    if full_test_size > EVAL_SAMPLE_SIZE:
        print(
            f"[sample] full test set ({full_test_size} rows) x 2 ONNX variants is impractical for "
            f"CPU DistilBERT inference here — using a fixed-seed stratified sample of "
            f"{EVAL_SAMPLE_SIZE} rows instead (documented deviation from compare_models.py's "
            "full-test-set evaluation)."
        )
        test_df = test_df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), EVAL_SAMPLE_SIZE // 2), random_state=RANDOM_STATE)
        ).reset_index(drop=True)

    X_test_text = test_df["text_clean_transformer"].fillna("").tolist()
    y_test = test_df["label"].tolist()
    print(f"[load] evaluating on {len(test_df)} rows (of {full_test_size} total test rows)")

    if not ONNX_FP32_PATH.exists():
        export_to_onnx(model, tokenizer)
    else:
        print(f"[skip] ONNX export already exists at {ONNX_FP32_PATH.relative_to(PROJECT_ROOT)}")

    if not ONNX_INT8_PATH.exists():
        quantize_onnx()
    else:
        print(f"[skip] ONNX quantized model already exists at {ONNX_INT8_PATH.relative_to(PROJECT_ROOT)}")

    fp32_session = ort.InferenceSession(str(ONNX_FP32_PATH), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(ONNX_INT8_PATH), providers=["CPUExecutionProvider"])

    fp32_metrics, fp32_latency = evaluate_onnx_session(fp32_session, tokenizer, X_test_text, y_test, "onnx_fp32")
    int8_metrics, int8_latency = evaluate_onnx_session(int8_session, tokenizer, X_test_text, y_test, "onnx_int8")

    original_report = json.loads(DISTILBERT_REPORT_PATH.read_text(encoding="utf-8"))
    original_test_metrics = original_report["final_test_metrics"]
    original_latency = original_report["inference_latency_cpu_val_samples"]
    original_size_mb = get_model_size_mb(DISTILBERT_DIR)
    onnx_fp32_size_mb = get_model_size_mb(ONNX_FP32_PATH)
    onnx_int8_size_mb = get_model_size_mb(ONNX_INT8_PATH)

    print("\n" + "=" * 84)
    print(f"{'Variant':<22}{'Test F1':>12}{'Size (MB)':>14}{'Mean Lat (ms)':>18}{'p95 Lat (ms)':>18}")
    print("-" * 84)
    print(f"{'original_pytorch':<22}{original_test_metrics['f1']:>12.4f}{original_size_mb:>14.1f}"
          f"{original_latency['mean_ms']:>18.1f}{original_latency['p95_ms']:>18.1f}")
    print(f"{'onnx_fp32':<22}{fp32_metrics['f1']:>12.4f}{onnx_fp32_size_mb:>14.1f}"
          f"{fp32_latency['mean_ms']:>18.1f}{fp32_latency['p95_ms']:>18.1f}")
    print(f"{'onnx_int8':<22}{int8_metrics['f1']:>12.4f}{onnx_int8_size_mb:>14.1f}"
          f"{int8_latency['mean_ms']:>18.1f}{int8_latency['p95_ms']:>18.1f}")
    print("=" * 84)

    f1_delta_int8 = round(int8_metrics["f1"] - original_test_metrics["f1"], 4)
    speedup_int8 = round(original_latency["mean_ms"] / int8_latency["mean_ms"], 2)
    size_reduction_pct_int8 = round((1 - onnx_int8_size_mb / original_size_mb) * 100, 1)

    worthwhile = f1_delta_int8 > -0.01 and (size_reduction_pct_int8 > 5 or speedup_int8 > 1.1)
    verdict = (
        f"ONNX + int8 quantization {'is' if worthwhile else 'is NOT'} a worthwhile optimization: "
        f"F1 changed by {f1_delta_int8:+.4f}, size reduced {size_reduction_pct_int8:.1f}%, "
        f"speedup {speedup_int8:.2f}x vs. the original PyTorch model. "
        + ("Measurable benefit with negligible accuracy cost — recommend keeping."
           if worthwhile else
           "Benefit is too small or accuracy cost too high to justify — recommend discarding.")
    )
    print(f"\n{verdict}\n")

    report = {
        "optimization": "onnx_export_plus_int8_dynamic_quantization",
        "original_pytorch": {
            "test_metrics": original_test_metrics,
            "size_mb": original_size_mb,
            "latency_ms": original_latency,
        },
        "onnx_fp32": {
            "test_metrics": fp32_metrics,
            "size_mb": onnx_fp32_size_mb,
            "latency_ms": fp32_latency,
        },
        "onnx_int8": {
            "test_metrics": int8_metrics,
            "size_mb": onnx_int8_size_mb,
            "latency_ms": int8_latency,
        },
        "comparison_int8_vs_original": {
            "f1_delta": f1_delta_int8,
            "size_reduction_pct": size_reduction_pct_int8,
            "latency_speedup_x": speedup_int8,
        },
        "worthwhile": worthwhile,
        "verdict": verdict,
        "note": "All latency figures in this report were measured on THIS machine in this same "
                "run (unlike the Phase 4 original report, measured on Colab), so the "
                "original_pytorch vs. onnx_int8 latency comparison mixes one cross-machine figure "
                "(original_pytorch, from Colab) with two same-machine figures (onnx_fp32, onnx_int8). "
                "For a strictly same-machine comparison, see ml/optimization/quantize.py's report instead.",
    }

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] Full report -> {REPORT_OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()