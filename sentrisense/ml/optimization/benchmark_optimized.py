"""
ml/optimization/benchmark_optimized.py

Phase 5 — final step: pulls together the two optimization experiments
(ml/optimization/quantize.py and ml/optimization/optimize_onnx.py) into one
summary, and decides which variant (if any) should actually be carried
forward into Phase 6/7 deployment.

Per the roadmap's Phase 5 rule: "Only keep an optimization if it shows a
measurable, worthwhile benefit — optimization isn't applied just to check
a box." This script doesn't re-run any inference; it just reads both prior
reports and picks the best worthwhile candidate (or concludes neither
helps enough, which is also a valid, honest outcome).

Usage:
    python ml/optimization/benchmark_optimized.py

Requires:
    - ml/optimization/quantize.py already run
    - ml/optimization/optimize_onnx.py already run
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

QUANTIZE_REPORT_PATH = RESULTS_DIR / "optimization_quantization_metrics.json"
ONNX_REPORT_PATH = RESULTS_DIR / "optimization_onnx_metrics.json"
ORIGINAL_REPORT_PATH = RESULTS_DIR / "model3_distilbert_metrics.json"

OUT_REPORT = RESULTS_DIR / "optimization_summary.json"
OUT_DOC = PROJECT_ROOT / "docs" / "deployment.md"


def load_json(path, label):
    if not path.exists():
        print(f"[error] {label} not found at {path}. Run its script first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    quant_report = load_json(QUANTIZE_REPORT_PATH, "PyTorch quantization report")
    onnx_report = load_json(ONNX_REPORT_PATH, "ONNX optimization report")
    original_report = load_json(ORIGINAL_REPORT_PATH, "original DistilBERT report")

    original_f1 = original_report["final_test_metrics"]["f1"]
    original_size_mb = round(original_report["disk_size_mb"], 3)
    original_latency_ms = original_report["inference_latency_cpu_val_samples"]["mean_ms"]

    candidates = [
        {
            "name": "pytorch_dynamic_quantization",
            "f1": quant_report["quantized"]["test_metrics"]["f1"],
            "size_mb": quant_report["quantized"]["size_mb"],
            "latency_ms": quant_report["quantized"]["latency_ms"]["mean_ms"],
            "worthwhile": quant_report["worthwhile"],
            "measured_same_machine_as_baseline": True,  # quantize.py measures both on this machine
            "artifact_path": "ml/models/distilbert/v1.0_quantized/model_quantized.pt",
        },
        {
            "name": "onnx_int8",
            "f1": onnx_report["onnx_int8"]["test_metrics"]["f1"],
            "size_mb": onnx_report["onnx_int8"]["size_mb"],
            "latency_ms": onnx_report["onnx_int8"]["latency_ms"]["mean_ms"],
            "worthwhile": onnx_report["worthwhile"],
            "measured_same_machine_as_baseline": False,  # onnx_int8 vs. original mixes machines; see optimize_onnx.py note
            "artifact_path": "ml/models/distilbert/v1.0_onnx/model_quantized.onnx",
        },
        {
            "name": "onnx_fp32",
            "f1": onnx_report["onnx_fp32"]["test_metrics"]["f1"],
            "size_mb": onnx_report["onnx_fp32"]["size_mb"],
            "latency_ms": onnx_report["onnx_fp32"]["latency_ms"]["mean_ms"],
            "worthwhile": None,  # not independently scored — informational middle step
            "measured_same_machine_as_baseline": False,
            "artifact_path": "ml/models/distilbert/v1.0_onnx/model.onnx",
        },
    ]

    for c in candidates:
        c["f1_delta_vs_original"] = round(c["f1"] - original_f1, 4)
        c["size_reduction_pct_vs_original"] = round((1 - c["size_mb"] / original_size_mb) * 100, 1)

    worthwhile_candidates = [c for c in candidates if c["worthwhile"]]

    if worthwhile_candidates:
        # Among worthwhile candidates, prefer the one with lower latency —
        # latency is the deployment-critical constraint per the project's own
        # framing (Lambda has no GPU, cost scales with execution time).
        best = min(worthwhile_candidates, key=lambda c: c["latency_ms"])
        decision = (
            f"'{best['name']}' is the recommended optimization to carry forward: "
            f"F1 delta {best['f1_delta_vs_original']:+.4f}, "
            f"size reduced {best['size_reduction_pct_vs_original']:.1f}%, "
            f"mean latency {best['latency_ms']:.1f}ms (vs. original {original_latency_ms:.1f}ms). "
            "This meets the roadmap's bar for a measurable, worthwhile benefit."
        )
    else:
        best = None
        decision = (
            "None of the tested optimizations (PyTorch dynamic quantization, ONNX+int8) showed "
            "a measurable, worthwhile benefit over the original fine-tuned DistilBERT model. "
            "Per the roadmap's own rule ('optimization isn't applied just to check a box'), "
            "the recommendation is to NOT adopt either optimization and instead carry the "
            "original Phase 4B model forward — or reconsider whether DistilBERT belongs in "
            "the Lambda deployment at all, given its latency was already the dominant concern "
            "in the Phase 4 comparison verdict."
        )

    print("\n" + "=" * 90)
    print(f"{'Variant':<28}{'F1 Δ':>10}{'Size Δ%':>12}{'Latency (ms)':>16}{'Worthwhile':>14}")
    print("-" * 90)
    for c in candidates:
        worthwhile_str = "N/A" if c["worthwhile"] is None else str(c["worthwhile"])
        print(
            f"{c['name']:<28}{c['f1_delta_vs_original']:>+10.4f}"
            f"{c['size_reduction_pct_vs_original']:>11.1f}%{c['latency_ms']:>16.1f}{worthwhile_str:>14}"
        )
    print("=" * 90)
    print(f"\n{decision}\n")

    report = {
        "original_baseline": {
            "f1": original_f1,
            "size_mb": original_size_mb,
            "latency_ms": original_latency_ms,
        },
        "candidates": candidates,
        "recommended_variant": best["name"] if best else None,
        "decision": decision,
    }

    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] Full summary -> {OUT_REPORT.relative_to(PROJECT_ROOT)}")

    doc_lines = [
        "# Phase 5 — Optimization Results\n",
        "Comparing the original fine-tuned DistilBERT against PyTorch dynamic quantization "
        "and ONNX (+int8) conversion.\n",
        "| Variant | F1 Δ vs. original | Size reduction | Mean CPU latency (ms) | Worthwhile? |",
        "|---|---|---|---|---|",
    ]
    for c in candidates:
        worthwhile_str = "N/A" if c["worthwhile"] is None else ("Yes" if c["worthwhile"] else "No")
        doc_lines.append(
            f"| {c['name']} | {c['f1_delta_vs_original']:+.4f} | "
            f"{c['size_reduction_pct_vs_original']:.1f}% | {c['latency_ms']:.1f} | {worthwhile_str} |"
        )
    doc_lines.append(f"\n## Decision\n\n{decision}\n")

    OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOC.write_text("\n".join(doc_lines), encoding="utf-8")
    print(f"[saved] Deployment doc -> {OUT_DOC.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()