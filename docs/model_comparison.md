# Phase 4 Model Comparison

All three models evaluated on the same held-out `imdb_test.csv`.

| Model | Test Accuracy | Test F1 | Size (MB) | Mean CPU Latency (ms) | Latency measured on |
|---|---|---|---|---|---|
| tfidf_logreg | 0.8977 | 0.8975 | 2.239 | 1.829 | local machine (this run) |
| tfidf_svm | 0.8968 | 0.8965 | 2.239 | 1.905 | local machine (this run) |
| distilbert_finetuned | 0.9312 | 0.9312 | 256.109 | 307.072 | Google Colab CPU during Phase 4B training (NOT this machine) |

## Verdict

Linear SVM does not improve on the Logistic Regression baseline (F1 delta: -0.0010), at effectively the same size and latency. DistilBERT gains +0.0337 F1 over the baseline, but at 114.4x the model size and roughly 167.9x the CPU inference latency (caveat: DistilBERT's latency was measured on a different machine than the classical models' — see latency_measured_on fields). For a cost-sensitive, low-latency AWS Lambda deployment, this is a real accuracy-vs-efficiency tradeoff, not a clear-cut win either way — the right choice depends on whether the accuracy gain matters enough to the product to justify materially slower and more expensive inference.
