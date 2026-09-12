# Phase 5 — Optimization Results

Comparing the original fine-tuned DistilBERT against PyTorch dynamic quantization and ONNX (+int8) conversion.

| Variant | F1 Δ vs. original | Size reduction | Mean CPU latency (ms) | Worthwhile? |
|---|---|---|---|---|
| pytorch_dynamic_quantization | -0.0091 | 48.3% | 259.7 | Yes |
| onnx_int8 | -0.0060 | 74.9% | 222.7 | Yes |
| onnx_fp32 | -0.0047 | 0.2% | 236.5 | N/A |

## Decision

'onnx_int8' is the recommended optimization to carry forward: F1 delta -0.0060, size reduced 74.9%, mean latency 222.7ms (vs. original 307.1ms). This meets the roadmap's bar for a measurable, worthwhile benefit.
