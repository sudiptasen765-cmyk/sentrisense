# External Resources

Authoritative sources used across this project. Update this file whenever a new
external dependency, dataset, or reference is introduced in a phase.

## Datasets

| Resource | URL | Purpose | License / Usage Notes |
|---|---|---|---|
| IMDb Large Movie Review Dataset | https://ai.stanford.edu/~amaas/data/sentiment/ | Primary training/eval (binary sentiment) | Free for research use; cite Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, ACL. |
| SST-2 (Stanford Sentiment Treebank) | https://huggingface.co/datasets/stanfordnlp/sst2 | Secondary generalization eval | Research use; see dataset card on Hugging Face for full terms. |
| Amazon/Yelp/Kaggle review sets (optional domain set) | TBD in Phase 1 | Aspect-based sentiment domain testing | License documented per-source when selected — no scraping; only downloadable, licensed datasets. |

## AWS Documentation

| Resource | URL | Purpose |
|---|---|---|
| AWS Lambda Developer Guide | https://docs.aws.amazon.com/lambda/latest/dg/welcome.html | Function packaging, cold start, memory config |
| Amazon API Gateway Developer Guide | https://docs.aws.amazon.com/apigateway/latest/developerguide/welcome.html | REST API design |
| Amazon DynamoDB Developer Guide | https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html | Table design, on-demand capacity, TTL |
| Amazon S3 User Guide | https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html | Storage prefixes, access control |
| Amazon CloudWatch User Guide | https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/WhatIsCloudWatch.html | Logs, metrics, monitoring |
| AWS IAM User Guide | https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html | Least-privilege policy design |
| AWS SAM Developer Guide | https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/what-is-sam.html | Infrastructure-as-code deployment |
| AWS CLI Reference | https://docs.aws.amazon.com/cli/latest/index.html | Local AWS CLI setup/verification |

## ML / NLP

| Resource | URL | Purpose |
|---|---|---|
| scikit-learn documentation | https://scikit-learn.org/stable/ | TF-IDF, Logistic Regression, Linear SVM |
| Hugging Face Transformers docs | https://huggingface.co/docs/transformers/index | DistilBERT comparison model |
| Hugging Face Datasets docs | https://huggingface.co/docs/datasets/index | SST-2 loading |
| NLTK documentation | https://www.nltk.org/ | Tokenization, text utilities |

## Frontend

| Resource | URL | Purpose |
|---|---|---|
| React documentation | https://react.dev/ | Component architecture |
| Vite documentation | https://vitejs.dev/ | Build tooling |
| Recharts documentation | https://recharts.org/ | Dashboard charts |
| Tailwind CSS documentation | https://tailwindcss.com/docs | Styling system |

## Node.js / Python

| Resource | URL | Purpose |
|---|---|---|
| Node.js documentation | https://nodejs.org/en/docs | Runtime reference |
| Python documentation | https://docs.python.org/3/ | Language reference |

---
*Add new rows in the same phase you introduce the dependency — do not batch these updates.*
