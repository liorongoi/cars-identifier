import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

export function Chip({ kind, children, title }: { kind?: string; children: ReactNode; title?: string }) {
  return (
    <span className={`chip ${kind ?? ""}`} title={title}>
      {children}
    </span>
  );
}

export function Bar({ value, tone }: { value: number; tone?: "good" | "warn" | "bad" }) {
  const w = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className={`bar ${tone ?? ""}`} role="meter" aria-valuenow={Math.round(w)} aria-valuemin={0} aria-valuemax={100}>
      <i style={{ width: `${w}%` }} />
    </div>
  );
}

export function Spinner() {
  return <span className="spinner" aria-label="loading" />;
}

export function Alert({ kind, children }: { kind: "warn" | "bad" | "info"; children: ReactNode }) {
  const icon = kind === "info" ? "ℹ" : "⚠";
  return (
    <div className={`alert ${kind}`} role={kind === "bad" ? "alert" : "status"}>
      <span aria-hidden>{icon}</span>
      <div>{children}</div>
    </div>
  );
}

export function Stat({ k, v, s }: { k: string; v: ReactNode; s?: ReactNode }) {
  return (
    <div className="panel stat">
      <div className="k">{k}</div>
      <div className="v">{v}</div>
      {s !== undefined && <div className="s">{s}</div>}
    </div>
  );
}

/** Load data, exposing loading/error state and a reload function. */
export function useLoad<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const reload = useCallback(async () => {
    setLoading(true);
    try {
      setData(await fnRef.current());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    void reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, loading, reload, setData };
}

/** Run a callback every `ms` while `active` is true. */
export function useInterval(cb: () => void, ms: number, active = true) {
  const ref = useRef(cb);
  ref.current = cb;
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => ref.current(), ms);
    return () => clearInterval(t);
  }, [ms, active]);
}

const ANALYST_KEY = "carid.analyst";

export function useAnalyst(): [string, (v: string) => void] {
  const [name, setName] = useState<string>(() => {
    try {
      return localStorage.getItem(ANALYST_KEY) ?? "";
    } catch {
      return "";
    }
  });
  const set = (v: string) => {
    setName(v);
    try {
      localStorage.setItem(ANALYST_KEY, v);
    } catch {
      /* storage unavailable */
    }
  };
  return [name, set];
}

export function qualityTone(verdict: string): "good" | "warn" | "bad" {
  return verdict === "good" ? "good" : verdict === "fair" ? "warn" : "bad";
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
