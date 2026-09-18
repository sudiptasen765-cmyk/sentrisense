# Model Weights — Not Tracked in Git

Large trained model files (`.safetensors`, `.onnx`, `.bin`, `.pkl`, `.joblib` under
`ml/models/`) are intentionally excluded from this repository via `.gitignore` —
same convention as `ml/data/raw/` and `ml/data/processed/`. This is standard
practice: large binary artifacts bloat repo size and are regeneratable from the
training scripts, so only the scripts, configs, and small tokenizer files are
version-controlled.

## What's missing locally after a fresh clone

- `ml/models/v1.0/*.joblib` — regenerate by running:
  ```
  python ml/training/train_baseline.py
  python ml/training/train_svm.py
  ```
- `ml/models/distilbert/v1.0/model.safetensors` (~256 MB) — **cannot be
  regenerated quickly** (requires GPU fine-tuning, ~10 min on Colab, 11+ hours
  on CPU). Download instead from:

  **[ADD YOUR GOOGLE DRIVE LINK HERE]**

  Place the downloaded `model.safetensors` into `ml/models/distilbert/v1.0/`
  alongside the existing `config.json` / `tokenizer.json` / `tokenizer_config.json`.

- `ml/models/distilbert/v1.0_onnx/*.onnx` and `v1.0_quantized/*.pt` — the
  Phase 5 optimized artifacts. Regenerate locally (once the weights above are
  in place) by running:
  ```
  python ml/optimization/quantize.py
  python ml/optimization/optimize_onnx.py
  ```
  or download from the same Drive folder if you'd rather skip re-running them.

## Why this matters for the team

If you pull this repo fresh and a script fails with a "model not found" error,
this is why — it's not a bug, the weights are just not meant to live in Git.
Grab them from the Drive link above before running anything in
`ml/evaluation/` or `ml/optimization/`.
