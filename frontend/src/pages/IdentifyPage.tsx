import { useCallback, useEffect, useRef, useState } from "react";
import { api, DRAWABLE_PARTS, type BBox, type IdentifyResult } from "../api";
import { ResultView } from "../components/ResultView";
import { Alert, Chip, Spinner, useAnalyst, useLoad } from "../components/common";

const MODES = [
  { id: "", label: "Auto" },
  { id: "fusion", label: "Fusion" },
  { id: "parts", label: "Parts" },
  { id: "global", label: "Whole" },
];
const DRAWABLE = Object.keys(DRAWABLE_PARTS);
const ACCEPT = "image/jpeg,image/png,image/webp,image/bmp,image/tiff,image/heic,.heic";

export function IdentifyPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [caseRef, setCaseRef] = useState("");
  const [analyst, setAnalyst] = useAnalyst();
  const [topK, setTopK] = useState(10);
  const [enhance, setEnhance] = useState(true);
  const [maskFaces, setMaskFaces] = useState(true);
  const [mode, setMode] = useState("");
  const [makeIds, setMakeIds] = useState<number[]>([]);
  const [manual, setManual] = useState<{ part: string; bbox: BBox }[]>([]);
  const [drawPart, setDrawPart] = useState<string | null>(null);
  const [pickPart, setPickPart] = useState(DRAWABLE[0] ?? "steering_wheel");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<IdentifyResult | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const catalog = useLoad(() => api.catalog(), []);

  const choose = useCallback((f: File | null | undefined) => {
    if (!f) return;
    if (!f.type.startsWith("image/") && !/\.(heic|tif|tiff)$/i.test(f.name)) {
      setError(`"${f.name}" is not an image.`);
      return;
    }
    setError(null);
    setFile(f);
    setResult(null);
    setManual([]);
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(f);
    });
  }, []);

  // Paste from clipboard anywhere on the page.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.type.startsWith("image/"));
      if (item) choose(item.getAsFile());
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [choose]);

  const run = useCallback(async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    setDrawPart(null);
    try {
      const r = await api.identify({
        file,
        topK,
        enhance,
        maskFaces,
        mode: mode || undefined,
        makeIds,
        manualParts: manual,
        caseRef: caseRef || undefined,
        analyst: analyst || undefined,
      });
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [file, topK, enhance, maskFaces, mode, makeIds, manual, caseRef, analyst]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") void run();
      if (e.key === "Escape") setDrawPart(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [run]);

  const makes = catalog.data?.makes ?? [];

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Identify vehicle</h1>
          <p>Drop an interior photo. The system masks people, enhances poor images, finds parts and ranks matching vehicles.</p>
        </div>
        <div className="row small muted">
          <kbd className="mono">Ctrl</kbd>+<kbd className="mono">Enter</kbd> run · <kbd className="mono">Ctrl</kbd>+
          <kbd className="mono">V</kbd> paste
        </div>
      </div>

      <div className="grid-2">
        <div className="stack">
          <div className="panel">
            <div className="panel-body stack">
              <div
                className={`drop ${over ? "over" : ""}`}
                role="button"
                tabIndex={0}
                onClick={() => input.current?.click()}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setOver(true);
                }}
                onDragLeave={() => setOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setOver(false);
                  choose(e.dataTransfer.files[0]);
                }}
              >
                {preview ? (
                  <img src={preview} alt="selected" style={{ maxWidth: "100%", maxHeight: 160, borderRadius: 6 }} />
                ) : (
                  <>
                    <div style={{ fontSize: 28, marginBottom: 6 }}>⤓</div>
                    <b>Drop image here</b> or click to browse
                    <div className="small dim">JPEG · PNG · WebP · TIFF · HEIC, up to 40 MB</div>
                  </>
                )}
              </div>
              <input
                ref={input}
                type="file"
                accept={ACCEPT}
                hidden
                data-testid="file-input"
                onChange={(e) => choose(e.target.files?.[0])}
              />
              {file && (
                <div className="small muted mono" title={file.name}>
                  {file.name.length > 40 ? `${file.name.slice(0, 37)}…` : file.name} · {(file.size / 1024 / 1024).toFixed(2)} MB
                </div>
              )}
              <div className="row" style={{ gap: 8 }}>
                <label className="field" style={{ flex: 1 }}>
                  Case reference
                  <input type="text" value={caseRef} onChange={(e) => setCaseRef(e.target.value)} placeholder="e.g. 2026-0457" />
                </label>
                <label className="field" style={{ flex: 1 }}>
                  Analyst
                  <input type="text" value={analyst} onChange={(e) => setAnalyst(e.target.value)} placeholder="Your name" />
                </label>
              </div>
              <button className="btn primary" disabled={!file || busy} onClick={() => void run()}>
                {busy ? <Spinner /> : "◎"} {busy ? "Analysing…" : result ? "Re-run identification" : "Identify"}
              </button>
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <h3>Options</h3>
            </div>
            <div className="panel-body stack">
              <label className="field">
                <span>
                  Candidates to show: <b style={{ color: "var(--text)" }}>{topK}</b>
                </span>
                <input type="range" min={3} max={20} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
              </label>
              <label className="check">
                <input type="checkbox" checked={enhance} onChange={(e) => setEnhance(e.target.checked)} /> Enhance poor images
                (low light, noise, blur)
              </label>
              <label className="check">
                <input type="checkbox" checked={maskFaces} onChange={(e) => setMaskFaces(e.target.checked)} /> Mask faces and people
              </label>
              <div className="field">
                Matching mode
                <div className="seg" role="group" aria-label="matching mode">
                  {MODES.map((m) => (
                    <button key={m.id} className={mode === m.id ? "on" : ""} onClick={() => setMode(m.id)}>
                      {m.label}
                    </button>
                  ))}
                </div>
              </div>
              <label className="field">
                Restrict to makes (optional)
                <select
                  multiple
                  size={Math.min(6, Math.max(3, makes.length))}
                  value={makeIds.map(String)}
                  onChange={(e) => setMakeIds(Array.from(e.target.selectedOptions).map((o) => Number(o.value)))}
                >
                  {makes.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} ({m.images})
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <h3>Manual part boxes</h3>
            </div>
            <div className="panel-body stack">
              <div className="small muted">
                Draw a box around a small item (gear knob, vent, switch block, badge) or a missed part, then re-run.
                Small items are matched against detail regions of every reference photo.
              </div>
              <div className="row" style={{ gap: 8 }}>
                <select value={pickPart} onChange={(e) => setPickPart(e.target.value)} aria-label="part to draw" style={{ flex: 1 }}>
                  {DRAWABLE.map((p) => (
                    <option key={p} value={p}>
                      {DRAWABLE_PARTS[p]}
                    </option>
                  ))}
                </select>
                <button
                  className={`btn sm ${drawPart ? "primary" : ""}`}
                  disabled={!result}
                  onClick={() => setDrawPart(drawPart ? null : pickPart)}
                  title={result ? "" : "Run identification first"}
                >
                  {drawPart ? "Drawing… (Esc)" : "Draw box"}
                </button>
              </div>
              {manual.map((m) => (
                <div key={m.part} className="row small">
                  <Chip kind="good">{DRAWABLE_PARTS[m.part] ?? m.part}</Chip>
                  <span className="mono dim">{m.bbox.join(", ")}</span>
                  <span className="spacer" />
                  <button className="btn sm ghost" onClick={() => setManual(manual.filter((x) => x.part !== m.part))}>
                    remove
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="stack">
          {error && <Alert kind="bad">{error}</Alert>}
          {busy && !result && (
            <div className="panel scan" style={{ minHeight: 320 }}>
              <div className="empty">
                <Spinner /> Analysing image: quality → enhancement → face masking → part detection → matching…
              </div>
            </div>
          )}
          {!busy && !result && !error && (
            <div className="panel">
              <div className="empty">
                <div style={{ fontSize: 40, opacity: 0.6 }}>◎</div>
                Results will appear here: ranked make and model candidates with probability, reference-image counts and visual evidence.
              </div>
            </div>
          )}
          {result && (
            <div className={busy ? "scan" : ""}>
              <ResultView
                result={result}
                imageSrc={preview}
                manual={manual}
                drawPart={drawPart}
                analyst={analyst}
                onDraw={(part, bbox) => {
                  setManual((m) => [...m.filter((x) => x.part !== part), { part, bbox }]);
                  setDrawPart(null);
                }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
