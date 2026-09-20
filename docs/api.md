# SentriSense — Prediction API Documentation

This is the contract the frontend (Phase 8) needs to integrate against. Nothing
about AWS, Lambda, Docker, or the models themselves is required knowledge to
use this API — just the request/response shape below.

## Endpoint

```
POST https://av9stuspv9.execute-api.ap-south-1.amazonaws.com/predict
```

Content-Type: `application/json`

## Request

```json
{
  "text": "The review text goes here",
  "model": "logreg"
}
```

| Field   | Type   | Required | Notes                                                                            |
| ------- | ------ | -------- | -------------------------------------------------------------------------------- |
| `text`  | string | Yes      | The review to classify. Cannot be empty or whitespace-only. Max 5000 characters. |
| `model` | string | No       | Either `"logreg"` or `"distilbert"`. Defaults to `"logreg"` if omitted.          |

**Model choice guidance for the UI:**

- `"logreg"` — fast (~2-5ms typical inference), smaller, ~89.8% test accuracy. Good default.
- `"distilbert"` — much more accurate (~93.1% test accuracy), but noticeably slower
  (hundreds of ms, longer on a cold start) and costs more compute. Consider
  offering this as an opt-in "high accuracy mode" toggle rather than the default.

## Success Response — 200 OK

```json
{
  "sentiment": "positive",
  "confidence": 0.9867,
  "model": "tfidf-logreg-v1.0"
}
```

| Field        | Type   | Notes                                                                                                          |
| ------------ | ------ | -------------------------------------------------------------------------------------------------------------- |
| `sentiment`  | string | `"positive"` or `"negative"`.                                                                                  |
| `confidence` | number | 0.0–1.0. Higher = more confident.                                                                              |
| `model`      | string | Identifies which specific model artifact answered (useful for display, e.g. "Answered by: tfidf-logreg-v1.0"). |

## Error Response — 422 Unprocessable Entity

Returned for invalid input (empty text, oversized text, invalid `model` value).

```json
{
  "error": "validation_error",
  "message": "The 'text' field cannot be empty or whitespace-only"
}
```

The UI should display `message` to the user (or a friendlier paraphrase of it)
rather than a generic failure state, since these are expected, recoverable
input problems — not server failures.

## Known Issue — do not call this endpoint with GET

Visiting the URL directly in a browser (a GET request) currently returns a
raw `500 Internal Server Error` rather than a clean validation message. This
is a known gap being tracked separately — **always call this endpoint with
POST and a JSON body**, never GET, and the frontend should never link
directly to this URL as a clickable link for that reason.

## Example — JavaScript fetch

```javascript
async function analyzeSentiment(text, model = "logreg") {
  const response = await fetch(
    "https://av9stuspv9.execute-api.ap-south-1.amazonaws.com/predict",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model }),
    },
  );

  const data = await response.json();

  if (!response.ok) {
    // data.message holds a user-facing explanation, e.g. empty text
    throw new Error(data.message || "Something went wrong");
  }

  return data; // { sentiment, confidence, model }
}
```

## What the frontend does NOT need to worry about

- No AWS credentials or SDK needed — this is a plain public HTTPS endpoint.
- No knowledge of Lambda, Docker, S3, or DynamoDB required.
- CORS: if browser requests are blocked, this needs to be enabled on the
  API Gateway side (ask the backend/infra owner if this hasn't been
  configured yet — it's a one-time API Gateway setting, not a frontend fix).

## Open items (not blocking frontend work, but worth knowing about)

- DynamoDB prediction logging is implemented but not yet independently
  re-verified for the new container-based deployment.
- The GET-request 500 error above is a known, minor gap.
- Full monitoring (Phase 9 / CloudWatch dashboards) is not yet built.

These don't affect how the frontend should call the API — they're
backend-side follow-ups.
