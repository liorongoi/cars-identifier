import { useState } from "react";
import { api, PART_LABELS, pct, type BBox, type IdentifyResult } from "../api";
import { ImageViewer } from "./ImageViewer";
import { ResultsTable } from "./ResultsTable";
import { Alert, Bar, Chip } from "./common";

interface Props {
  result: IdentifyResult;
  imageSrc: string | null;
  manual?: { part: string; bbox: BBox }[];
  drawPart?: string | null;
  onDraw?: (part: string, bbox: BBox) => void;
  analyst?: string;
}

export function ResultView({ result, imageSrc, manual = [], drawPart = null, onDraw, analyst = "" }: Props) {
  const [imgEl, setImgEl] = useState<HTMLImageElement | null>(null);
  const [showParts, setShowParts] = useState(true);
  const [showFaces, setShowFaces] = useState(true);
  const [hl, setHl] = useState<string | null>(null);
  const q = result.quality;
  const top = result.candidates[0];

  return (
    <div className="stack">
      {result.warnings.map((w) => (
        <Alert key={w} kind={w.startsWith("No matches") ? "bad" : "warn"}>
          {w}
        </Alert>
      ))}

      <div className="grid-2" style={{ gridTemplateColumns: "minmax(0, 1.15fr) minmax(0, 1fr)" }}>
        <div className="panel">
          <div className="panel-head">
            <h2>Query image</h2>
            <div className="row">
              <label className="check small">
                <input type="checkbox" checked={showParts} onChange={(e) => setShowParts(e.target.checked)} /> Parts
              </label>
              <label className="check small">
                <input type="checkbox" checked={showFaces} onChange={(e) => setShowFaces(e.target.checked)} /> Masked faces
              </label>
            </div>
          </div>
          <div className="panel-body">
            {imageSrc ? (
              <ImageViewer
                src={imageSrc}
                onImageLoad={setImgEl}
                overlay={{ parts: result.parts, faces: result.faces, imageSize: result.image_size }}
                showParts={showParts}
                showFaces={showFaces}
                highlight={hl}
                manual={manual}
                drawPart={drawPart}
                onDraw={onDraw}
              />
            ) : (
              <div className="empty">Query image not retained (privacy retention period passed).</div>
            )}
            <div className="row small muted" style={{ marginTop: 10 }}>
              <span>Detected:</span>
              {result.parts
                .filter((p) => p.part !== "whole")
                .map((p) => (
                  <span key={p.part} onMouseEnter={() => setHl(p.part)} onMouseLeave={() => setHl(null)}>
                    <Chip kind="part" title={p.method === "layout-prior" ? "position estimated from typical layout" : `detected (${p.method}), score ${p.score}`}>
                      {PART_LABELS[p.part] ?? p.part}
                      {p.method === "layout-prior" ? " ·est" : ""}
                    </Chip>
                  </span>
                ))}
              {result.faces.length > 0 && <Chip kind="warn">{result.faces.length} face(s) masked</Chip>}
            </div>
          </div>
        </div>

        <div className="stack">
          <div className="panel">
            <div className="panel-head">
              <h2>Best match</h2>
              <span className="dim small mono">
                case #{result.id} · {result.duration_ms} ms
              </span>
            </div>
            <div className="panel-body">
              {top ? (
                <div className="stack" style={{ gap: 8 }}>
                  <div style={{ fontSize: 22, fontWeight: 650 }}>
                    {top.make} {top.model}
                  </div>
                  <div className="row">
                    <Chip kind={top.confidence}>{top.confidence} confidence</Chip>
                    {top.generation && <Chip>{top.generation}</Chip>}
                    <span className="muted">
                      {top.year_start ?? "?"}–{top.year_end ?? "present"}
                    </span>
                  </div>
                  <div className="prob">
                    <Bar value={top.probability} />
                    <b>{pct(top.probability)}</b>
                  </div>
                </div>
              ) : (
                <div className="muted">No candidate.</div>
              )}
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <h2>Make probability</h2>
            </div>
            <div className="panel-body stack" style={{ gap: 7 }}>
              {result.makes.slice(0, 5).map((m) => (
                <div className="metric" key={m.make_id}>
                  <span>{m.make}</span>
                  <Bar value={m.probability} />
                  <span className="v">{pct(m.probability, 0)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <h2>Image quality</h2>
              <Chip kind={q.verdict}>
                {q.verdict} · {q.score.toFixed(0)}/100
              </Chip>
            </div>
            <div className="panel-body stack" style={{ gap: 7 }}>
              <div className="metric">
                <span>Sharpness</span>
                <Bar value={Math.min(1, q.sharpness / 300)} tone={q.sharpness < 30 ? "bad" : q.sharpness < 80 ? "warn" : "good"} />
                <span className="v">{q.sharpness.toFixed(0)}</span>
              </div>
              <div className="metric">
                <span>Brightness</span>
                <Bar value={q.brightness / 255} tone={q.brightness < 50 ? "bad" : q.brightness < 80 ? "warn" : "good"} />
                <span className="v">{q.brightness.toFixed(0)}</span>
              </div>
              <div className="metric">
                <span>Contrast</span>
                <Bar value={Math.min(1, q.contrast / 80)} tone={q.contrast < 25 ? "warn" : "good"} />
                <span className="v">{q.contrast.toFixed(0)}</span>
              </div>
              <div className="metric">
                <span>Noise</span>
                <Bar value={Math.min(1, q.noise / 20)} tone={q.noise > 12 ? "bad" : q.noise > 7 ? "warn" : "good"} />
                <span className="v">{q.noise.toFixed(1)}</span>
              </div>
              <div className="small muted">
                {q.width}×{q.height}px
                {q.issues.length ? ` · ${q.issues.join(", ")}` : " · no issues"}
                {result.enhancement.length ? ` · applied: ${result.enhancement.join(", ")}` : ""}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <h2>Ranked candidates ({result.candidates.length})</h2>
          <div className="row">
            <a className="btn sm" href={`/api/identifications/${result.id}/export?format=csv`}>
              Export CSV
            </a>
            <a className="btn sm" href={`/api/identifications/${result.id}/export?format=json`}>
              Export JSON
            </a>
          </div>
        </div>
        <ResultsTable
          candidates={result.candidates}
          queryParts={result.parts}
          imageSize={result.image_size}
          queryImg={imgEl}
          onHoverPart={setHl}
        />
      </div>

      <Feedback result={result} analyst={analyst} />
      <div className="dim small mono">
        engine: {result.engine.embedder} · detector {result.engine.detector} · faces {result.engine.face_detector} · mode{" "}
        {result.engine.matcher_mode}
      </div>
    </div>
  );
}

function Feedback({ result, analyst }: { result: IdentifyResult; analyst: string }) {
  const [modelId, setModelId] = useState<string>("");
  const [note, setNote] = useState("");
  const [state, setState] = useState<"idle" | "saving" | "saved" | string>("idle");
  const save = async () => {
    setState("saving");
    try {
      await api.feedback(result.id, modelId ? Number(modelId) : null, note, analyst);
      setState("saved");
    } catch (e) {
      setState(e instanceof Error ? e.message : "error");
    }
  };
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Analyst feedback</h2>
        <span className="dim small">Ground truth improves future accuracy</span>
      </div>
      <div className="panel-body row">
        <select value={modelId} onChange={(e) => setModelId(e.target.value)} aria-label="confirmed vehicle">
          <option value="">Correct vehicle (if known)…</option>
          {result.candidates.map((c) => (
            <option key={c.rank} value={c.model_id}>
              {c.make} {c.model} {c.generation ?? ""}
            </option>
          ))}
        </select>
        <input type="text" placeholder="Note (optional)" value={note} onChange={(e) => setNote(e.target.value)} style={{ flex: 1, minWidth: 200 }} />
        <button className="btn" onClick={save} disabled={state === "saving"}>
          Save feedback
        </button>
        {state === "saved" && <Chip kind="good">saved</Chip>}
        {state !== "idle" && state !== "saving" && state !== "saved" && <Chip kind="bad">{state}</Chip>}
      </div>
    </div>
  );
}
