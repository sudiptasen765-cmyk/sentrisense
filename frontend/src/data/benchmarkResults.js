// All figures below are taken directly from the project's locked Phase 3-5
// results (reports/results/, docs/model_comparison.md). Nothing here is
// estimated or fabricated — if a figure isn't in this file, it hasn't been
// measured yet and should say so rather than guess.

export const MODEL_COMPARISON = [
  {
    id: "logreg",
    name: "TF-IDF + Logistic Regression",
    tag: "tfidf-logreg-v1.0",
    role: "Locked baseline (Phase 3)",
    accuracy: 89.77,
    f1: 89.75,
    sizeMb: 2.24,
    latencyMs: 1.83,
    deployed: true,
  },
  {
    id: "svm",
    name: "Linear SVM",
    tag: "svm-v1.0",
    role: "Comparison model (Phase 4) — not deployed",
    accuracy: 89.68,
    f1: 89.65,
    sizeMb: 2.24,
    latencyMs: 1.91,
    deployed: false,
  },
  {
    id: "distilbert",
    name: "DistilBERT (original, fp32)",
    tag: "distilbert-fp32",
    role: "Highest accuracy, before optimization",
    accuracy: 93.12,
    f1: 93.12,
    sizeMb: 256.1,
    latencyMs: 307.1,
    deployed: false,
  },
  {
    id: "distilbert-onnx",
    name: "DistilBERT, ONNX INT8",
    tag: "distilbert-onnx-int8",
    role: "Deployed (Phase 5 optimized)",
    accuracy: null, // F1 delta is documented; absolute accuracy on full test set not separately reported
    f1: null,
    sizeMb: 64.3,
    latencyMs: 222.7,
    deployed: true,
  },
];

export const OPTIMIZATION_RESULTS = [
  {
    name: "PyTorch Dynamic Quantization",
    f1Delta: -0.91,
    sizeReductionPct: 48.3,
    speedup: 1.18,
    verdict: "Worthwhile",
    adopted: false,
  },
  {
    name: "ONNX + INT8 Quantization",
    f1Delta: -0.6,
    sizeReductionPct: 74.9,
    speedup: 1.38,
    verdict: "Recommended",
    adopted: true,
  },
];

export const BASELINE_VALIDATION = {
  accuracy: 91.64,
  f1: 91.67,
  sizeMb: 2.24,
  meanLatencyMs: 0.748,
  p95LatencyMs: 1.318,
};

export const DEPLOYMENT_MODEL = {
  name: "DistilBERT, ONNX INT8",
  tag: "distilbert-onnx-int8",
  file: "model_quantized.onnx",
  sizeMb: 64.3,
  measuredCpuLatencyMs: 222.7,
};

export const RESEARCH_QUESTION =
  "How do lightweight traditional NLP models compare with transformer-based models when accuracy is evaluated alongside inference latency, model size, and serverless deployment constraints?";
