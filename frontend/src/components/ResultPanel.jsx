import { ArrowUpRight, ArrowDownRight } from "lucide-react";
import clsx from "clsx";

/**
 * @param {object} props
 * @param {"positive"|"negative"} props.sentiment
 * @param {number} props.confidence - 0..1
 * @param {string} props.model - the model tag string returned by the API
 * @param {number} [props.latencyMs] - optional, measured client-side
 */
export default function ResultPanel({ sentiment, confidence, model, latencyMs }) {
  const isPositive = sentiment === "positive";
  const pct = Math.round(confidence * 1000) / 10; // one decimal place

  return (
    <div className="border border-ink-700 rounded-panel bg-ink-900 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {isPositive ? (
            <ArrowUpRight className="text-gold-500" size={22} strokeWidth={2.5} />
          ) : (
            <ArrowDownRight className="text-parchment-500" size={22} strokeWidth={2.5} />
          )}
          <span
            className={clsx(
              "font-display text-2xl tracking-tight",
              isPositive ? "text-gold-500" : "text-parchment-300"
            )}
          >
            {isPositive ? "Positive" : "Negative"}
          </span>
        </div>
        <span className="font-mono text-2xl text-parchment-100">{pct}%</span>
      </div>

      <div className="h-1.5 bg-ink-700 rounded-full overflow-hidden mb-4" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div
          className={clsx("h-full rounded-full", isPositive ? "bg-gold-500" : "bg-parchment-500")}
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="flex items-center justify-between text-xs font-mono text-parchment-500">
        <span>{model}</span>
        {latencyMs != null && <span>{latencyMs.toFixed(1)}ms (round-trip)</span>}
      </div>
    </div>
  );
}
