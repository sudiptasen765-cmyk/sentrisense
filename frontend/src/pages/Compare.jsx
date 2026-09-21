import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  MODEL_COMPARISON,
  OPTIMIZATION_RESULTS,
  RESEARCH_QUESTION,
} from "../data/benchmarkResults.js";

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-ink-900 border border-ink-700 rounded-panel px-3 py-2 text-xs font-mono">
      <div className="text-parchment-100 mb-1">{d.name}</div>
      <div className="text-parchment-500">accuracy: {d.accuracy ?? "—"}%</div>
      <div className="text-parchment-500">latency: {d.latencyMs}ms</div>
      <div className="text-parchment-500">size: {d.sizeMb}MB</div>
    </div>
  );
}

export default function Compare() {
  const chartable = MODEL_COMPARISON.filter((m) => m.accuracy != null);

  return (
    <div>
      <p className="font-mono text-xs text-gold-500 mb-3">Model comparison</p>
      <h1 className="text-4xl mb-4 leading-tight">The tradeoff, plotted.</h1>
      <p className="text-parchment-300 max-w-column mb-10">{RESEARCH_QUESTION}</p>

      <h2 className="text-xl mb-4">Accuracy vs. latency vs. size</h2>
      <p className="text-sm text-parchment-500 mb-4 font-mono">
        bubble size = model size on disk (MB)
      </p>
      <div className="border border-ink-700 rounded-panel bg-ink-900 p-4 mb-4" style={{ height: 360 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid stroke="#24395C" strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="latencyMs"
              name="latency"
              unit="ms"
              scale="log"
              domain={["auto", "auto"]}
              tick={{ fill: "#C9C2AE", fontSize: 11, fontFamily: "IBM Plex Mono" }}
              stroke="#24395C"
              label={{ value: "Mean CPU latency (ms, log scale)", position: "insideBottom", offset: -5, fill: "#8D8672", fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="accuracy"
              name="accuracy"
              unit="%"
              domain={[85, 95]}
              tick={{ fill: "#C9C2AE", fontSize: 11, fontFamily: "IBM Plex Mono" }}
              stroke="#24395C"
              label={{ value: "Test accuracy (%)", angle: -90, position: "insideLeft", fill: "#8D8672", fontSize: 11 }}
            />
            <ZAxis type="number" dataKey="sizeMb" range={[80, 700]} />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: "#C9A227", strokeWidth: 1 }} />
            <Scatter data={chartable} fill="#C9A227" fillOpacity={0.75} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <h2 className="text-xl mb-4 mt-12">Full comparison</h2>
      <div className="overflow-x-auto border border-ink-700 rounded-panel mb-12">
        <table className="w-full text-sm font-mono">
          <thead>
            <tr className="border-b border-ink-700 text-left text-parchment-500">
              <th className="px-4 py-3 font-normal">Model</th>
              <th className="px-4 py-3 font-normal">Accuracy</th>
              <th className="px-4 py-3 font-normal">F1</th>
              <th className="px-4 py-3 font-normal">Size</th>
              <th className="px-4 py-3 font-normal">Latency</th>
              <th className="px-4 py-3 font-normal">Status</th>
            </tr>
          </thead>
          <tbody>
            {MODEL_COMPARISON.map((m) => (
              <tr key={m.id} className="border-b border-ink-700 last:border-0">
                <td className="px-4 py-3 text-parchment-100">
                  <div className="font-sans">{m.name}</div>
                  <div className="text-xs text-parchment-500">{m.role}</div>
                </td>
                <td className="px-4 py-3">{m.accuracy != null ? `${m.accuracy}%` : "—"}</td>
                <td className="px-4 py-3">{m.f1 != null ? `${m.f1}%` : "—"}</td>
                <td className="px-4 py-3">{m.sizeMb} MB</td>
                <td className="px-4 py-3">{m.latencyMs} ms</td>
                <td className="px-4 py-3">
                  {m.deployed ? (
                    <span className="text-gold-500">deployed</span>
                  ) : (
                    <span className="text-parchment-500">not deployed</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2 className="text-xl mb-4">Phase 5 — optimization results</h2>
      <p className="text-sm text-parchment-300 mb-4 max-w-column">
        Two optimizations were tested against the original fine-tuned DistilBERT model.
        Only one was adopted for deployment.
      </p>
      <div className="overflow-x-auto border border-ink-700 rounded-panel">
        <table className="w-full text-sm font-mono">
          <thead>
            <tr className="border-b border-ink-700 text-left text-parchment-500">
              <th className="px-4 py-3 font-normal">Optimization</th>
              <th className="px-4 py-3 font-normal">F1 Δ</th>
              <th className="px-4 py-3 font-normal">Size reduction</th>
              <th className="px-4 py-3 font-normal">Speedup</th>
              <th className="px-4 py-3 font-normal">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {OPTIMIZATION_RESULTS.map((o) => (
              <tr key={o.name} className="border-b border-ink-700 last:border-0">
                <td className="px-4 py-3 text-parchment-100 font-sans">{o.name}</td>
                <td className="px-4 py-3">{o.f1Delta > 0 ? "+" : ""}{o.f1Delta}%</td>
                <td className="px-4 py-3">{o.sizeReductionPct}%</td>
                <td className="px-4 py-3">{o.speedup}x</td>
                <td className="px-4 py-3">
                  <span className={o.adopted ? "text-gold-500" : "text-parchment-500"}>
                    {o.verdict}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
