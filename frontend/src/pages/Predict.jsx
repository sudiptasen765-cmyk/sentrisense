import { useState } from "react";
import clsx from "clsx";
import { analyzeSentiment } from "../services/api.js";
import ResultPanel from "../components/ResultPanel.jsx";

const EXAMPLES = [
  "This movie was fantastic. I loved every minute of it.",
  "This movie was boring, disappointing and painfully slow.",
];

export default function Predict() {
  const [text, setText] = useState("");
  const [model, setModel] = useState("logreg");
  const [result, setResult] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | error
  const [errorMessage, setErrorMessage] = useState("");

  async function handleAnalyze() {
    const trimmed = text.trim();
    if (!trimmed) {
      setStatus("error");
      setErrorMessage("Enter a review before analyzing.");
      return;
    }

    setStatus("loading");
    setErrorMessage("");
    const start = performance.now();

    try {
      const data = await analyzeSentiment(trimmed, model);
      setLatencyMs(performance.now() - start);
      setResult(data);
      setStatus("idle");
    } catch (err) {
      setStatus("error");
      const apiMessage = err?.response?.data?.message;
      setErrorMessage(
        apiMessage ||
          "Couldn't reach the prediction service. If this is the distilbert model, a cold start can take up to 15-20 seconds — try again."
      );
    }
  }

  return (
    <div className="max-w-column">
      <p className="font-mono text-xs text-gold-500 mb-3">Live demo</p>
      <h1 className="text-4xl mb-3 leading-tight">Sentiment, measured.</h1>
      <p className="text-parchment-300 mb-8">
        Paste a review below. Choose which model reads it — a fast baseline or a slower,
        more accurate transformer — and see the actual confidence behind the call.
      </p>

      <label htmlFor="review-text" className="sr-only">
        Review text
      </label>
      <textarea
        id="review-text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={5}
        maxLength={5000}
        placeholder="Type or paste a movie review…"
        className="w-full bg-ink-900 border border-ink-700 rounded-panel p-4 text-parchment-100 placeholder:text-parchment-500 resize-none mb-3"
      />

      <div className="flex flex-wrap gap-2 mb-6">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => setText(ex)}
            className="text-xs font-mono text-parchment-500 hover:text-gold-500 border border-ink-700 rounded-panel px-2.5 py-1 transition-colors"
          >
            {ex.length > 40 ? ex.slice(0, 40) + "…" : ex}
          </button>
        ))}
      </div>

      <div className="flex items-center justify-between mb-8">
        <div className="flex gap-1 border border-ink-700 rounded-panel p-1">
          {[
            { id: "logreg", label: "Fast (logreg)" },
            { id: "distilbert", label: "Accurate (distilbert)" },
          ].map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setModel(m.id)}
              className={clsx(
                "text-sm px-3 py-1.5 rounded-panel transition-colors",
                model === m.id
                  ? "bg-gold-500 text-ink-950 font-medium"
                  : "text-parchment-300 hover:text-parchment-100"
              )}
            >
              {m.label}
            </button>
          ))}
        </div>

        <button
          type="button"
          onClick={handleAnalyze}
          disabled={status === "loading"}
          className="bg-gold-500 hover:bg-gold-300 disabled:opacity-50 disabled:cursor-not-allowed text-ink-950 font-medium text-sm px-5 py-2 rounded-panel transition-colors"
        >
          {status === "loading" ? "Analyzing…" : "Analyze"}
        </button>
      </div>

      {status === "error" && (
        <p className="text-sm text-parchment-100 bg-ink-900 border border-gold-700 rounded-panel px-4 py-3 mb-6">
          {errorMessage}
        </p>
      )}

      {result && (
        <ResultPanel
          sentiment={result.sentiment}
          confidence={result.confidence}
          model={result.model}
          latencyMs={latencyMs}
        />
      )}
    </div>
  );
}
