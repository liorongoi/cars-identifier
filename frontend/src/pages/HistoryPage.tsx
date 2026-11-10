import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, formatYears, pct } from "../api";
import { ResultView } from "../components/ResultView";
import { Alert, Chip, Spinner, formatDate, useAnalyst, useLoad } from "../components/common";

export function HistoryPage() {
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const hist = useLoad(() => api.history(q, offset), [q, offset]);
  const nav = useNavigate();
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Case history</h1>
          <p>Every identification is logged with analyst, case reference, settings and results (audit trail).</p>
        </div>
        <input type="search" placeholder="Search case ref or analyst…" value={q} onChange={(e) => (setOffset(0), setQ(e.target.value))} style={{ width: 280 }} aria-label="search history" />
      </div>
      {hist.error && <Alert kind="bad">{hist.error}</Alert>}
      <div className="panel">
        <table className="tbl">
          <thead>
            <tr>
              <th>Case</th>
              <th>When</th>
              <th>Analyst</th>
              <th>Top result</th>
              <th>Probability</th>
              <th>Image quality</th>
              <th>Feedback</th>
            </tr>
          </thead>
          <tbody>
            {(hist.data?.items ?? []).map((h) => (
              <tr key={h.id} className="clickable" onClick={() => nav(`/cases/${h.id}`)}>
                <td>
                  <b className="mono">#{h.id}</b> {h.case_ref && <span className="muted">{h.case_ref}</span>}
                </td>
                <td className="small muted nowrap">{formatDate(h.created_at)}</td>
                <td>{h.analyst ?? "—"}</td>
                <td>
                  {h.top ? (
                    <>
                      {h.top.make} {h.top.model} <span className="dim small">{formatYears(h.top.year_start, h.top.year_end)}</span>
                    </>
                  ) : (
                    <span className="dim">no match</span>
                  )}
                </td>
                <td>{h.top ? pct(h.top.probability) : "—"}</td>
                <td>{h.quality && <Chip kind={h.quality}>{h.quality}</Chip>}</td>
                <td>{h.feedback_model_id ? <Chip kind="good">confirmed</Chip> : <span className="dim">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {hist.data?.total === 0 && <div className="empty">No identifications yet.</div>}
        {hist.data && hist.data.total > 50 && (
          <div className="row panel-body">
            <button className="btn sm" disabled={offset === 0} onClick={() => setOffset(offset - 50)}>
              ‹ Newer
            </button>
            <span className="small muted">
              {offset + 1}–{Math.min(offset + 50, hist.data.total)} of {hist.data.total}
            </span>
            <button className="btn sm" disabled={offset + 50 >= hist.data.total} onClick={() => setOffset(offset + 50)}>
              Older ›
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export function CasePage() {
  const { id } = useParams();
  const caseId = Number(id);
  const c = useLoad(() => api.getCase(caseId), [caseId]);
  const [analyst] = useAnalyst();
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>
            Case #{caseId} {c.data?.case_ref && <span className="muted">· {c.data.case_ref}</span>}
          </h1>
          {c.data && (
            <p>
              {formatDate(c.data.created_at)} · analyst {c.data.analyst ?? "—"}
              {c.data.feedback_note ? ` · feedback: ${c.data.feedback_note}` : ""}
            </p>
          )}
        </div>
        <Link className="btn" to="/history">
          ‹ Back to history
        </Link>
      </div>
      {c.loading && <Spinner />}
      {c.error && <Alert kind="bad">{c.error}</Alert>}
      {c.data && (
        <ResultView
          result={c.data}
          imageSrc={c.data.has_query_image ? `/api/identifications/${caseId}/query` : null}
          analyst={analyst}
        />
      )}
    </div>
  );
}
