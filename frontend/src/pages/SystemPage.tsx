import { api } from "../api";
import { Alert, Chip, formatDate, useLoad } from "../components/common";

export function SystemPage() {
  const health = useLoad(() => api.health(), []);
  const sources = useLoad(() => api.sources(), []);
  const audit = useLoad(() => api.audit(), []);
  const h = health.data;
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>System</h1>
          <p>Engine configuration, data sources and the audit log.</p>
        </div>
      </div>
      <div className="stack">
        {h?.warnings.map((w) => (
          <Alert key={w} kind="warn">
            {w}
          </Alert>
        ))}
        {health.error && <Alert kind="bad">API unreachable: {health.error}</Alert>}
        <div className="grid-2" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)" }}>
          <div className="panel">
            <div className="panel-head">
              <h2>Engine</h2>
              {h && <Chip kind="good">{h.status}</Chip>}
            </div>
            <div className="panel-body">
              {h && (
                <dl className="kv">
                  <dt>Version</dt>
                  <dd>{h.version}</dd>
                  <dt>Embedder</dt>
                  <dd className="mono">
                    {h.engine.embedder} ({h.engine.embedder_dim}-d)
                  </dd>
                  <dt>Part detector</dt>
                  <dd className="mono">{h.engine.detector}</dd>
                  <dt>Face detector</dt>
                  <dd className="mono">{h.engine.face_detector}</dd>
                  <dt>Matching mode</dt>
                  <dd className="mono">{h.engine.matcher_mode}</dd>
                  <dt>Index</dt>
                  <dd className="mono small">
                    {Object.entries(h.engine.index_sizes)
                      .map(([k, v]) => `${k}: ${v}`)
                      .join(" · ") || "empty"}
                  </dd>
                </dl>
              )}
            </div>
          </div>
          <div className="panel">
            <div className="panel-head">
              <h2>Collection sources</h2>
            </div>
            <table className="tbl">
              <tbody>
                {(sources.data ?? []).map((s) => (
                  <tr key={s.name}>
                    <td>
                      <b className="mono">{s.name}</b>
                      <div className="small dim">{s.description}</div>
                    </td>
                    <td className="nowrap">
                      {s.runnable ? <Chip kind="good">active</Chip> : <Chip kind={s.requires_legal_approval ? "warn" : undefined}>off</Chip>}
                      {s.requires_legal_approval && (
                        <div className="small dim" style={{ marginTop: 4 }}>
                          legal approval: {s.legal_approved ? "recorded" : "required"}
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="panel-body small muted">Sources are configured by the administrator in config/sources.yaml.</div>
          </div>
        </div>
        <div className="panel">
          <div className="panel-head">
            <h2>Audit log</h2>
          </div>
          <table className="tbl">
            <thead>
              <tr>
                <th>When</th>
                <th>Actor</th>
                <th>Action</th>
                <th>Target</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {(audit.data ?? []).map((a) => (
                <tr key={a.id}>
                  <td className="small muted nowrap">{formatDate(a.created_at)}</td>
                  <td>{a.actor ?? "—"}</td>
                  <td className="mono small">{a.action}</td>
                  <td className="mono small">{a.target}</td>
                  <td className="mono small dim">{JSON.stringify(a.details)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {audit.data?.length === 0 && <div className="empty">No audit events yet.</div>}
        </div>
      </div>
    </div>
  );
}
