import { Fragment, useState } from "react";
import { api, pct, type Metrics } from "../api";
import { Alert, Chip, formatDate, useLoad } from "../components/common";

const LEVELS: { id: string; label: string }[] = [
  { id: "make", label: "Make" },
  { id: "model", label: "Model" },
  { id: "model_year", label: "Model + year range" },
];

function cellColor(v: number | undefined): string {
  if (v === undefined) return "transparent";
  const hue = 0 + 140 * v; // red → green
  return `hsla(${hue}, 70%, 45%, ${0.18 + 0.4 * v})`;
}

function Cell({ m, k }: { m: Metrics | undefined; k: "top1" | "top5" | "top10" | "mrr" }) {
  const v = m?.[k];
  return (
    <td className="heatcell" style={{ background: cellColor(v) }}>
      {v === undefined ? "—" : k === "mrr" ? v.toFixed(2) : pct(v, 0)}
    </td>
  );
}

export function BenchmarksPage() {
  const list = useLoad(() => api.benchmarks(), []);
  const [sel, setSel] = useState<number | null>(null);
  const chosen = sel ?? list.data?.[0]?.id ?? null;
  const report = useLoad(() => (chosen ? api.benchmark(chosen) : Promise.resolve(null)), [chosen]);
  const [level, setLevel] = useState("model");
  const r = report.data;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Algorithm benchmarks</h1>
          <p>
            Hold-out evaluation of each matching mode on clean and degraded copies of reference images (blur, low light, noise,
            compression, occlusion). The best mode is used for identification.
          </p>
        </div>
        <select value={chosen ?? ""} onChange={(e) => setSel(Number(e.target.value))} aria-label="benchmark run">
          {(list.data ?? []).map((b) => (
            <option key={b.id} value={b.id}>
              #{b.id} · {formatDate(b.created_at)} · {b.embedder}
            </option>
          ))}
        </select>
      </div>
      {list.data?.length === 0 && <Alert kind="info">No benchmark yet. Start one from the Jobs page once images are collected.</Alert>}
      {report.error && <Alert kind="bad">{report.error}</Alert>}
      {r && (
        <div className="stack">
          <div className="grid-cards">
            <div className="panel stat">
              <div className="k">Best mode</div>
              <div className="v">{r.best_mode ?? "—"}</div>
              <div className="s">by mean model top-5 across conditions</div>
            </div>
            {Object.entries(r.summary).map(([mode, s]) => (
              <div className="panel stat" key={mode}>
                <div className="k">{mode}</div>
                <div className="v">{pct(s.mean_model_top5)}</div>
                <div className="s">mean model top-5</div>
              </div>
            ))}
            <div className="panel stat">
              <div className="k">Test set</div>
              <div className="v">{r.test_images}</div>
              <div className="s">
                of {r.reference_images} images · {r.classes} classes
              </div>
            </div>
            <div className="panel stat">
              <div className="k">Latency</div>
              <div className="v">{r.mean_latency_ms ?? "—"} ms</div>
              <div className="s">
                {r.embedder} · {r.detector}
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <h2>Accuracy by condition</h2>
              <div className="seg" role="group" aria-label="level">
                {LEVELS.map((l) => (
                  <button key={l.id} className={level === l.id ? "on" : ""} onClick={() => setLevel(l.id)}>
                    {l.label}
                  </button>
                ))}
              </div>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table className="tbl">
                <thead>
                  <tr>
                    <th rowSpan={2}>Condition</th>
                    {r.modes.map((m) => (
                      <th key={m} colSpan={4} style={{ textAlign: "center" }}>
                        {m} {m === r.best_mode && <Chip kind="good">best</Chip>}
                      </th>
                    ))}
                  </tr>
                  <tr>
                    {r.modes.map((m) => ["Top-1", "Top-5", "Top-10", "MRR"].map((h) => <th key={m + h}>{h}</th>))}
                  </tr>
                </thead>
                <tbody>
                  {r.conditions.map((c) => (
                    <tr key={c}>
                      <td>
                        <b>{c.replace("_", " ")}</b>
                      </td>
                      {r.modes.map((m) => {
                        const met = r.results[c]?.[m]?.[level];
                        return (
                          <Fragment key={m}>
                            <Cell m={met} k="top1" />
                            <Cell m={met} k="top5" />
                            <Cell m={met} k="top10" />
                            <Cell m={met} k="mrr" />
                          </Fragment>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
