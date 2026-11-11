import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { api } from "./api";
import { useLoad } from "./components/common";
import { AddMakePage } from "./pages/AddMakePage";
import { BenchmarksPage } from "./pages/BenchmarksPage";
import { DatabasePage } from "./pages/DatabasePage";
import { CasePage, HistoryPage } from "./pages/HistoryPage";
import { IdentifyPage } from "./pages/IdentifyPage";
import { JobsPage } from "./pages/JobsPage";
import { SystemPage } from "./pages/SystemPage";

const NAV = [
  { to: "/", label: "Identify", ico: "◎" },
  { to: "/database", label: "Database", ico: "▤" },
  { to: "/add", label: "Add make / model", ico: "＋" },
  { to: "/jobs", label: "Jobs", ico: "⟳" },
  { to: "/history", label: "Case history", ico: "☰" },
  { to: "/benchmarks", label: "Benchmarks", ico: "▥" },
  { to: "/system", label: "System", ico: "⚙" },
];

export function App() {
  const health = useLoad(() => api.health(), []);
  const h = health.data;
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" />
          <div>
            <b>CarID Interior</b>
            <span>Analyst console</span>
          </div>
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}>
              <span className="ico" aria-hidden>
                {n.ico}
              </span>
              <span>{n.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          {health.error ? (
            <span style={{ color: "var(--bad)" }}>● API offline</span>
          ) : h ? (
            <>
              <div>
                <span style={{ color: h.engine.deep_models ? "var(--good)" : "var(--warn)" }}>●</span> v{h.version} ·{" "}
                {h.engine.deep_models ? "deep models" : "fallback models"}
              </div>
              <div className="mono" style={{ marginTop: 4 }}>
                {(h.engine.index_sizes.whole ?? 0).toLocaleString()} images indexed
              </div>
            </>
          ) : (
            "connecting…"
          )}
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<IdentifyPage />} />
          <Route path="/database" element={<DatabasePage />} />
          <Route path="/add" element={<AddMakePage />} />
          <Route path="/jobs" element={<JobsPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/cases/:id" element={<CasePage />} />
          <Route path="/benchmarks" element={<BenchmarksPage />} />
          <Route path="/system" element={<SystemPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
