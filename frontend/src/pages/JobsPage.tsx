import { useState } from "react";
import { api, pct, type Job } from "../api";
import { Alert, Bar, Chip, formatDate, useAnalyst, useInterval, useLoad } from "../components/common";

const KIND_LABEL: Record<string, string> = {
  collect: "Collect images",
  reindex: "Rebuild index",
  benchmark: "Benchmark",
  expand_make: "Expand make (Wikidata)",
  import_bundle: "Import bundle",
  train_adapter: "Train adapter",
};

function describe(j: Job): string {
  const p = j.params;
  if (j.kind === "collect") {
    const parts = [];
    if (Array.isArray(p.make_ids) && p.make_ids.length) parts.push(`${p.make_ids.length} make(s)`);
    else parts.push("all makes");
    if (Array.isArray(p.model_ids) && p.model_ids.length) parts.push(`${p.model_ids.length} model(s)`);
    if (p.per_model_limit) parts.push(`≤${String(p.per_model_limit)}/model`);
    return parts.join(" · ");
  }
  if (j.kind === "expand_make") return String(p.make ?? "");
  return "";
}

export function JobsPage() {
  const jobs = useLoad(() => api.jobs(), []);
  const [sel, setSel] = useState<number | null>(null);
  const detail = useLoad(() => (sel ? api.job(sel) : Promise.resolve(null)), [sel]);
  const [analyst] = useAnalyst();
  const [error, setError] = useState<string | null>(null);
  const active = (jobs.data ?? []).some((j) => j.status === "running" || j.status === "queued");
  useInterval(
    () => {
      void jobs.reload();
      if (sel) void detail.reload();
    },
    2000,
    active,
  );

  const start = async (kind: string, params: Record<string, unknown>, confirmText?: string) => {
    if (confirmText && !window.confirm(confirmText)) return;
    try {
      const { job_id } = await api.createJob(kind, params, analyst || undefined);
      setSel(job_id);
      setError(null);
      await jobs.reload();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const d = detail.data;
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Jobs</h1>
          <p>Collection, indexing and benchmark jobs run in the background; they resume after a restart.</p>
        </div>
        <div className="row">
          <button className="btn" onClick={() => void start("collect", { per_model_limit: 150 }, "Collect images for every make and model? This can take hours.")}>
            Collect all makes
          </button>
          <button className="btn" onClick={() => void start("reindex", {}, "Re-embed every reference image? Needed after changing the embedder.")}>
            Rebuild index
          </button>
          <button
            className="btn"
            title="Learn which interior features separate car models (seconds; re-run after large collections)"
            onClick={() => void start("train_adapter", {})}
          >
            Train adapter
          </button>
          <button className="btn primary" onClick={() => void start("benchmark", { max_test: 200 })}>
            Run benchmark
          </button>
        </div>
      </div>
      {error && <Alert kind="bad">{error}</Alert>}
      <div className="grid-2" style={{ gridTemplateColumns: "minmax(0,1.3fr) minmax(0,1fr)", marginTop: 8 }}>
        <div className="panel">
          <table className="tbl">
            <thead>
              <tr>
                <th>#</th>
                <th>Job</th>
                <th>Status</th>
                <th style={{ width: "28%" }}>Progress</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {(jobs.data ?? []).map((j) => (
                <tr key={j.id} className={`clickable ${sel === j.id ? "open" : ""}`} onClick={() => setSel(j.id)}>
                  <td className="mono">{j.id}</td>
                  <td>
                    <b>{KIND_LABEL[j.kind] ?? j.kind}</b>
                    <div className="small dim">{describe(j)}</div>
                  </td>
                  <td>
                    <Chip kind={j.status}>{j.status}</Chip>
                  </td>
                  <td>
                    <div className="prob">
                      <Bar value={j.progress} tone={j.status === "failed" ? "bad" : j.status === "succeeded" ? "good" : undefined} />
                      <b className="small">{pct(j.progress, 0)}</b>
                    </div>
                    <div className="small dim" title={j.message}>
                      {j.message.slice(0, 60)}
                    </div>
                  </td>
                  <td className="small muted nowrap">{formatDate(j.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {jobs.data?.length === 0 && <div className="empty">No jobs yet.</div>}
        </div>

        <div className="panel" style={{ position: "sticky", top: 12 }}>
          <div className="panel-head">
            <h2>{d ? `Job #${d.id} — ${KIND_LABEL[d.kind] ?? d.kind}` : "Job details"}</h2>
            {d && (d.status === "running" || d.status === "queued") && (
              <button className="btn sm danger" onClick={() => void api.cancelJob(d.id).then(() => jobs.reload())}>
                Cancel
              </button>
            )}
          </div>
          <div className="panel-body stack">
            {!d && <div className="muted">Select a job to see its counters and log.</div>}
            {d && (
              <>
                <dl className="kv">
                  <dt>Status</dt>
                  <dd>
                    <Chip kind={d.status}>{d.status}</Chip> {d.message}
                  </dd>
                  <dt>Requested by</dt>
                  <dd>{d.requested_by ?? "—"}</dd>
                  <dt>Started</dt>
                  <dd>{d.started_at ? formatDate(d.started_at) : "—"}</dd>
                  <dt>Finished</dt>
                  <dd>{d.finished_at ? formatDate(d.finished_at) : "—"}</dd>
                  <dt>Parameters</dt>
                  <dd className="mono small">{JSON.stringify(d.params)}</dd>
                </dl>
                {Object.keys(d.counters).length > 0 && (
                  <div className="grid-cards" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))" }}>
                    {Object.entries(d.counters).map(([k, v]) => (
                      <div className="panel stat" key={k} style={{ padding: 10 }}>
                        <div className="k" style={{ fontSize: 10.5 }}>
                          {k.replace(/_/g, " ")}
                        </div>
                        <div className="v" style={{ fontSize: 19 }}>
                          {String(v)}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                <div className="log" aria-label="job log">
                  {d.log || "No log output yet."}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
