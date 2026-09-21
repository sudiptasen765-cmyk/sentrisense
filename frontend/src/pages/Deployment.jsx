const RESOURCES = [
  { label: "AWS region", value: "ap-south-1 (Mumbai)" },
  { label: "API endpoint", value: "POST /predict", mono: true },
  { label: "Compute", value: "AWS Lambda (container image)" },
  { label: "Model storage", value: "Amazon S3 (private)" },
  { label: "Prediction log", value: "Amazon DynamoDB" },
  { label: "Observability", value: "Amazon CloudWatch" },
];

const FLOW = ["Client", "API Gateway", "Lambda", "S3 · DynamoDB · CloudWatch"];

export default function Deployment() {
  return (
    <div>
      <p className="font-mono text-xs text-gold-500 mb-3">Deployment</p>
      <h1 className="text-4xl mb-4 leading-tight">How this runs.</h1>
      <p className="text-parchment-300 max-w-column mb-10">
        SentriSense runs entirely on serverless AWS infrastructure. Nothing here is
        always-on — Lambda spins up per request, and you pay only for what gets used.
      </p>

      <h2 className="text-xl mb-4">Architecture</h2>
      <div className="border border-ink-700 rounded-panel bg-ink-900 p-8 mb-12">
        <div className="flex flex-col sm:flex-row items-stretch gap-3">
          {FLOW.map((step, i) => (
            <div key={step} className="flex items-center gap-3">
              <div className="border border-gold-700 bg-ink-950 rounded-panel px-4 py-3 text-sm font-mono text-parchment-100 text-center flex-1">
                {step}
              </div>
              {i < FLOW.length - 1 && (
                <span className="text-gold-500 text-lg hidden sm:inline" aria-hidden="true">
                  →
                </span>
              )}
            </div>
          ))}
        </div>
        <p className="text-xs text-parchment-500 font-mono mt-6">
          IAM grants the Lambda execution role read-only access to its two specific model
          paths in S3, and write access to exactly one DynamoDB table — nothing broader.
        </p>
      </div>

      <h2 className="text-xl mb-4">Two models, one endpoint</h2>
      <p className="text-sm text-parchment-300 max-w-column mb-4">
        The same Lambda function serves both models. Pass{" "}
        <code className="font-mono text-gold-500">"model": "logreg"</code> for the fast
        baseline, or{" "}
        <code className="font-mono text-gold-500">"model": "distilbert"</code> for the
        slower, more accurate transformer. Omitting it defaults to logreg.
      </p>
      <pre className="border border-ink-700 rounded-panel bg-ink-900 p-4 text-xs font-mono text-parchment-300 overflow-x-auto mb-12">
{`POST /predict
{
  "text": "This movie was fantastic.",
  "model": "distilbert"
}

→ {
  "sentiment": "positive",
  "confidence": 0.994,
  "model": "distilbert-onnx-int8"
}`}
      </pre>

      <h2 className="text-xl mb-4">Resources</h2>
      <dl className="border border-ink-700 rounded-panel divide-y divide-ink-700">
        {RESOURCES.map((r) => (
          <div key={r.label} className="flex justify-between px-4 py-3 text-sm">
            <dt className="text-parchment-500">{r.label}</dt>
            <dd className={r.mono ? "font-mono text-parchment-100" : "text-parchment-100"}>
              {r.value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
