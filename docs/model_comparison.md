# Phase 4 Model Comparison

Logistic Regression and SVM evaluated on the same held-out `imdb_test.csv`. DistilBERT (Phase 4B) was deferred -- see verdict below.

| Model | Test Accuracy | Test F1 | Size (MB) | Mean CPU Latency (ms) | Latency measured on |
|---|---|---|---|---|---|
| tfidf_logreg | 0.9005 | 0.9004 | 2.049 | 3.014 | local machine (this run) |
| tfidf_svm | 0.8970 | 0.8967 | 2.049 | 2.813 | local machine (this run) |
| distilbert_finetuned | NOT YET MEASURED | NOT YET MEASURED | -- | -- | deferred -- no local GPU |

## Verdict

Linear SVM does not improve on the Logistic Regression baseline on the held-out test set (F1 delta: -0.0037), at effectively the same size and latency. DistilBERT (Phase 4B) was NOT YET MEASURED in this comparison -- fine-tuning was deferred due to no local GPU being available. This comparison currently reflects only the two classical models.
