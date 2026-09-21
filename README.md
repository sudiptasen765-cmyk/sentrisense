# SentriSense

**Sentiment analysis, measured — a study of accuracy vs. latency vs. size on serverless AWS.**

SentriSense is an end-to-end sentiment analysis system: dataset preparation, classical and transformer models, model optimization, a serverless inference API on AWS, and a React dashboard that lets you try the models and compare their trade-offs.

> **Research question:** How do lightweight traditional NLP models compare with transformer-based models when accuracy is evaluated alongside inference latency, model size, and serverless deployment constraints?

---

## Table of contents

1. [The idea](#1-the-idea)
2. [System overview](#2-system-overview)
3. [Data](#3-data)
4. [Modeling](#4-modeling)
5. [Results](#5-results)
6. [Serverless deployment on AWS](#6-serverless-deployment-on-aws)
7. [API reference](#7-api-reference)
8. [Frontend](#8-frontend)
9. [Running it yourself](#9-running-it-yourself)
10. [Repository structure](#10-repository-structure)
11. [Challenges and lessons learned](#11-challenges-and-lessons-learned)
12. [Limitations](#12-limitations)
13. [Future work](#13-future-work)
14. [Acknowledgements and citations](#14-acknowledgements-and-citations)

---

## 1. The idea

Transformer models such as BERT-family networks routinely beat classical NLP baselines on accuracy. But accuracy is only one axis. In a real deployment, especially on serverless infrastructure, three other things matter just as much:

- **Latency**: how long one prediction takes (and how bad cold starts are).
- **Model size**: what has to be stored, loaded, and packaged.
- **Cost and operational fit**: whether the model fits inside Lambda's constraints at all.

SentriSense was built to answer one practical question with real measurements: **when is the extra accuracy of a transformer worth its cost, and how much of that cost can be optimized away?**

To do that, the project compares a fast classical baseline against a fine-tuned DistilBERT, applies quantization and ONNX optimization to shrink the transformer, and deploys both behind a single serverless endpoint so the trade-off can be observed live.

---

## 2. System overview

```
                 ┌──────────────┐
  Browser  ───▶  │ API Gateway  │  HTTP API, POST /predict
 (React/Vite)    └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │  AWS Lambda  │  container image (Docker, from ECR)
                 │  handler.py  │  validates input, picks model, predicts
                 └──┬────┬───┬──┘
                    │    │   │
        ┌───────────▼┐ ┌─▼───▼──────┐ ┌────────────┐
        │  Amazon S3 │ │  DynamoDB  │ │ CloudWatch │
        │ model files│ │ prediction │ │    logs    │
        │  (private) │ │    log     │ │            │
        └────────────┘ └────────────┘ └────────────┘
```

| Layer         | Technology                                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------- |
| Models        | scikit-learn (TF-IDF + Logistic Regression, Linear SVM), Hugging Face Transformers (DistilBERT), ONNX Runtime |
| Compute       | AWS Lambda, packaged as a Docker container image                                                              |
| API           | Amazon API Gateway (HTTP API)                                                                                 |
| Storage       | Amazon S3 (model artifacts), Amazon DynamoDB (prediction log)                                                 |
| Observability | Amazon CloudWatch                                                                                             |
| Frontend      | React 18, Vite, Tailwind CSS, Recharts, Axios                                                                 |

---

## 3. Data

**Primary dataset: IMDb Large Movie Review Dataset** (Maas et al., 2011) with 25,000 training and 25,000 test reviews, perfectly balanced (12,500 positive and 12,500 negative per split). Labels are binary: `0 = negative`, `1 = positive`.

**Secondary dataset: SST-2** (Socher et al., 2013), used as a generalization check. Models trained on IMDb-derived data are evaluated on SST-2's `validation` split. SST-2's public `test` split has no real labels (`label = -1`, reserved for the GLUE leaderboard), so it is deliberately **not** used for evaluation.

### Data quality inspection

A dedicated script inspects every split for size, duplicates, missing text, class balance, and length distribution. Findings:

- IMDb: 0 missing or empty texts; word count ranges from 4 to 2,470 (mean about 231, median 173), which matters for DistilBERT's 512-token truncation.
- SST-2 train: about 56/44 positive/negative, moderately imbalanced but not severe enough to need resampling.
- **Data leakage found:** 123 IMDb reviews are byte-identical between `train` and `test`.

### Leakage decision

The official test split is left **unmodified**, so results stay comparable with published IMDb benchmarks. The 123 overlapping rows are instead **removed from the training set** during preprocessing. Within-split duplicates are kept because they don't cause leakage.

### Preprocessing decisions

- **Negations are preserved.** Standard stop-word removal deletes words like "not" and "never", which flip sentiment ("not good" ≠ "good").
- HTML line breaks (`<br />`) present in IMDb reviews are explicitly removed.

Full documentation lives in [`ml/data/README.md`](ml/data/README.md). Raw datasets are gitignored and re-downloadable with the scripts below.

---

## 4. Modeling

The work was done in phases, and each phase's results were recorded before moving on.

| Phase | Work                                                                                       |
| ----- | ------------------------------------------------------------------------------------------ |
| 1     | Dataset download scripts, data inspection, leakage analysis                                |
| 2     | Preprocessing pipeline (cleaning, deduplication against test, negation-safe)               |
| 3     | **Baseline:** TF-IDF + Logistic Regression, locked as the reference model                  |
| 4     | **Comparison model:** Linear SVM on the same features                                      |
| 5     | **Transformer:** fine-tuned DistilBERT, then optimized (dynamic quantization, ONNX + INT8) |

### Models

1. **TF-IDF + Logistic Regression** (`tfidf-logreg-v1.0`): the fast, tiny baseline, and the default model on the API.
2. **Linear SVM** (`svm-v1.0`): compared against the baseline but **not deployed**, since it was not meaningfully better.
3. **DistilBERT, fp32** (`distilbert-fp32`): the accuracy ceiling, before optimization.
4. **DistilBERT, ONNX INT8** (`distilbert-onnx-int8`): the optimized transformer that is actually deployed.

### Optimization experiments

Two ways of shrinking DistilBERT were measured against the fp32 model:

| Technique                    | F1 change | Size reduction | Speedup   | Verdict                  |
| ---------------------------- | --------- | -------------- | --------- | ------------------------ |
| PyTorch dynamic quantization | −0.91     | 48.3%          | 1.18×     | Worthwhile, not adopted  |
| **ONNX + INT8 quantization** | **−0.60** | **74.9%**      | **1.38×** | **Recommended, adopted** |

ONNX + INT8 was chosen because it shrinks the model by about three quarters while giving up well under one F1 point.

---

## 5. Results

Measured on CPU, as recorded in the project's locked results (`reports/results/`, `docs/model_comparison.md`):

| Model                        | Accuracy (%)                             | F1 (%) | Size (MB) | Mean latency (ms) | Deployed |
| ---------------------------- | ---------------------------------------- | ------ | --------- | ----------------- | -------- |
| TF-IDF + Logistic Regression | 89.77                                    | 89.75  | 2.24      | 1.83              | Yes      |
| Linear SVM                   | 89.68                                    | 89.65  | 2.24      | 1.91              | No       |
| DistilBERT (fp32)            | 93.12                                    | 93.12  | 256.1     | 307.1             | No       |
| DistilBERT (ONNX INT8)       | not separately reported (F1 delta above) |        | 64.3      | 222.7             | Yes      |

**Generalization check (validation split):** the baseline scored 91.64% accuracy and 91.67% F1, with 0.748 ms mean and 1.318 ms p95 latency.

### What the numbers say

- DistilBERT gains roughly **3.4 accuracy points** over the classical baseline.
- The price is about **100× the model size** and about **170× the latency** (fp32 vs. baseline).
- Quantization to ONNX INT8 cuts the transformer to a quarter of its size and about 27% faster, at a small F1 cost, which makes it practical on Lambda.
- The classical baseline is the right default when speed and cost dominate. The transformer is the right choice when the last few points of accuracy matter.

That is exactly why the API exposes **both** models behind one endpoint.

---

## 6. Serverless deployment on AWS

All resources live in **`ap-south-1` (Mumbai)**.

| Resource                                           | Purpose                                                                                                                                   |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **AWS Lambda** (`sentrisense-inference-container`) | Runs inference. Deployed as a **container image** because the ONNX runtime and model dependencies don't fit comfortably in a zip package. |
| **Amazon ECR**                                     | Stores the Lambda container image.                                                                                                        |
| **Amazon API Gateway** (HTTP API)                  | Public `POST /predict` endpoint, routed to Lambda.                                                                                        |
| **Amazon S3** (private bucket)                     | Holds the model artifacts. Lambda loads them at startup.                                                                                  |
| **Amazon DynamoDB**                                | Logs every prediction (id, timestamp, sentiment, confidence, model, input length, latency).                                               |
| **Amazon CloudWatch**                              | Lambda logs and metrics.                                                                                                                  |

### Security

The Lambda execution role follows least privilege: read-only access to exactly the two model paths in S3, and write access to exactly one DynamoDB table, with nothing broader. The S3 bucket is private.

### Design choices

- **Container image instead of a zip:** dependency size and reproducibility.
- **One function, two models:** the `model` field in the request selects which model runs, so both can be compared through the same endpoint.
- **Non-fatal logging:** a failed DynamoDB write is logged but never breaks a prediction response.
- **CORS handled in the function:** the API's `$default` catch-all route forwards every method, including the browser's `OPTIONS` preflight, to Lambda, so the handler answers `OPTIONS` itself and attaches CORS headers to every response, including errors.

### Building and deploying the container

Replace the placeholders with your own values.

```powershell
# 1. Build the image (Docker Desktop must be running)
docker build -t sentrisense-inference .

# 2. Authenticate Docker to ECR
aws ecr get-login-password --region ap-south-1 |
  docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com

# 3. Tag and push
docker tag sentrisense-inference:latest <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/<ECR_REPO>:latest
docker push <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/<ECR_REPO>:latest

# 4. Point the Lambda at the new image
aws lambda update-function-code `
  --function-name sentrisense-inference-container `
  --image-uri <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/<ECR_REPO>:latest `
  --region ap-south-1
```

Editing `handler.py` locally does **not** change the deployed function. The image has to be rebuilt and pushed, and the function updated, for changes to take effect.

### Lambda environment variables

| Variable            | Meaning                                    | Default  |
| ------------------- | ------------------------------------------ | -------- |
| `PREDICTIONS_TABLE` | DynamoDB table name for the prediction log | required |
| `MAX_INPUT_LENGTH`  | Maximum characters allowed in `text`       | `5000`   |

### Verifying the deployment

Invoke the function directly:

```powershell
[System.IO.File]::WriteAllText("$PWD\payload.json", '{"body": "{\"text\": \"This movie was amazing!\", \"model\": \"logreg\"}"}')
aws lambda invoke --function-name sentrisense-inference-container --region ap-south-1 `
  --cli-binary-format raw-in-base64-out --payload file://payload.json response.json
Get-Content response.json
```

Or call the live API:

```powershell
Invoke-RestMethod -Uri "https://<API_ID>.execute-api.ap-south-1.amazonaws.com/predict" `
  -Method Post -ContentType "application/json" `
  -Body '{"text":"This movie was amazing!","model":"logreg"}'
```

Expected output:

```
sentiment confidence model
--------- ---------- -----
positive      0.9867 tfidf-logreg-v1.0
```

---

## 7. API reference

### `POST /predict`

**Request body**

| Field   | Type   | Required | Notes                                            |
| ------- | ------ | -------- | ------------------------------------------------ |
| `text`  | string | yes      | Non-empty, at most `MAX_INPUT_LENGTH` characters |
| `model` | string | no       | `"logreg"` (default) or `"distilbert"`           |

**Example**

```json
POST /predict
{
  "text": "This movie was fantastic.",
  "model": "distilbert"
}
```

**Success response (200)**

```json
{
  "sentiment": "positive",
  "confidence": 0.994,
  "model": "distilbert-onnx-int8"
}
```

**Errors**

| Status | `error`              | Cause                                                             |
| ------ | -------------------- | ----------------------------------------------------------------- |
| 400    | `invalid_request`    | Missing body or body isn't valid JSON                             |
| 405    | `method_not_allowed` | Any method other than `POST` (and `OPTIONS`)                      |
| 422    | `validation_error`   | `text` missing, not a string, empty, or too long; unknown `model` |
| 500    | `internal_error`     | Unhandled error inside the function                               |

All error bodies look like `{"error": "<code>", "message": "<human-readable text>"}`.

---

## 8. Frontend

A React dashboard with three pages:

- **Predict:** type a review, choose a model, and see the sentiment, confidence, and model tag returned by the live API.
- **Compare:** interactive accuracy vs. latency vs. size chart (log-scale latency, bubble size = model size on disk) and the optimization results, all driven by real measured figures.
- **Deployment:** the architecture diagram, resource list, and API usage.

The frontend reads its API base URL from `VITE_API_BASE_URL` (see [`frontend/.env.example`](frontend/.env.example)).

### Local development and CORS

Browsers send a CORS preflight (`OPTIONS`) request before cross-origin `POST` calls. To avoid depending on preflight behaviour during local development, `vite.config.js` proxies `/api/*` to API Gateway, so the browser only ever talks to `localhost`:

```js
// frontend/vite.config.js
server: {
  port: 5173,
  proxy: {
    "/api": {
      target: "https://<API_ID>.execute-api.ap-south-1.amazonaws.com",
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ""),
    },
  },
},
```

Create `frontend/.env` for local development:

```
VITE_API_BASE_URL=/api
```

`.env` is gitignored. For a production build, point `VITE_API_BASE_URL` at the real API URL instead, since `/api` only exists on the dev server.

---

## 9. Running it yourself

### Prerequisites

Python 3.10+, Node.js 18+, Docker Desktop, and the AWS CLI configured with credentials that can manage Lambda, ECR, API Gateway, S3, and DynamoDB.

### 1. Get the data

```powershell
python ml\scripts\download_imdb.py
python ml\scripts\download_sst2.py
python ml\scripts\inspect_data.py
```

The download scripts are idempotent and safe to re-run.

### 2. Train and evaluate

Run the preprocessing and training scripts under `ml/`, following the phase order in [section 4](#4-modeling). Results are written to `reports/results/`.

### 3. Deploy the backend

Follow [section 6](#6-serverless-deployment-on-aws): create the S3 bucket and DynamoDB table, upload the model artifacts, build and push the container image, create the Lambda function and the API Gateway HTTP API.

### 4. Run the frontend

```powershell
cd frontend
npm install
# create .env with VITE_API_BASE_URL=/api for local dev
npm run dev
```

Open `http://localhost:5173`.

---

## 10. Repository structure

```
sentrisense/
├── backend/
│   └── lambda/
│       └── analyze/
│           ├── handler.py        # Lambda entry point: validation, routing, logging
│           └── inference.py      # Model loading and prediction
├── frontend/
│   ├── src/
│   │   ├── components/           # Layout, ResultPanel
│   │   ├── pages/                # Predict, Compare, Deployment
│   │   ├── services/api.js       # Axios client for POST /predict
│   │   └── data/benchmarkResults.js
│   ├── vite.config.js
│   └── .env.example
├── ml/
│   ├── data/                     # Dataset documentation (raw data is gitignored)
│   ├── scripts/                  # Download and inspection scripts
│   └── preprocessing/            # Cleaning and deduplication
├── reports/results/              # Locked measurement results
└── docs/                         # Model comparison and resources
```

---

## 11. Challenges and lessons learned

- **Data leakage is easy to miss.** Inspecting the raw data turned up 123 train/test duplicates. Handling them without touching the official test split kept results comparable.
- **Stop-word lists can destroy sentiment.** Standard lists remove negations, so preprocessing was made negation-safe on purpose.
- **Accuracy alone is a misleading metric for deployment.** A 3.4-point gain costs roughly 100× the size and 170× the latency until optimization is applied.
- **Quantization is worth doing.** ONNX INT8 recovered most of the size and speed at a cost of 0.6 F1 points.
- **API Gateway HTTP APIs and CORS are subtle.** With a `$default` catch-all route, preflight requests are forwarded to the Lambda, so the handler needs to answer `OPTIONS` itself, and the same headers must appear on error responses too.
- **Container Lambdas need a rebuild-and-push cycle.** Changing handler code locally changes nothing in the cloud until the image is redeployed.
- **A dev-server proxy is a clean way to sidestep browser CORS locally** without weakening any cloud configuration.

---

## 12. Limitations

- **Binary sentiment only.** There is no neutral class, which is a documented limitation of IMDb and SST-2.
- **Domain shift.** Models were trained on movie reviews and are less reliable on other domains such as product reviews, social media, or sarcasm-heavy text.
- **English only.**
- **Cold starts.** The first request after an idle period is slower, especially for the transformer.
- **Latency figures are CPU measurements** taken on the benchmark machine, so real Lambda latency varies with memory configuration and cold-start state.
- **Public API with no authentication** in the current version.

---

## 13. Future work

- Aspect-based sentiment on a domain-specific review dataset (Amazon, Yelp).
- A neutral or three-class formulation.
- Provisioned concurrency or a smaller runtime to reduce cold starts.
- API keys or authorizers, plus rate limiting.
- CI/CD for automatic container builds and deployments.
- Infrastructure as code (Terraform or AWS CDK) so the whole stack is reproducible with one command.
- A production-hosted frontend (S3 + CloudFront) instead of local development only.

---

## 14. Acknowledgements and citations

- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). _Learning Word Vectors for Sentiment Analysis._ ACL 2011.
- Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013). _Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank._ EMNLP 2013.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). _DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter._

Datasets are used for research and educational purposes only, and the raw files are not redistributed in this repository.
