import { useRef, useState, type MouseEvent } from "react";
import { PART_LABELS, type BBox, type DetectedPart } from "../api";

export interface Overlay {
  parts: DetectedPart[];
  faces: BBox[];
  imageSize: [number, number]; // coordinate space of the boxes
}

interface Props {
  src: string;
  overlay?: Overlay | null;
  showParts: boolean;
  showFaces: boolean;
  highlight?: string | null;
  manual: { part: string; bbox: BBox }[];
  drawPart: string | null; // when set, dragging draws a box for this part
  onDraw?: (part: string, bbox: BBox) => void;
  onImageLoad?: (img: HTMLImageElement) => void;
}

/** Image with part/face overlays; boxes are positioned in % so they scale with the image. */
export function ImageViewer({ src, overlay, showParts, showFaces, highlight, manual, drawPart, onDraw, onImageLoad }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [natural, setNatural] = useState<[number, number] | null>(null);
  const space: [number, number] | null = overlay?.imageSize ?? natural;

  const rel = (e: MouseEvent) => {
    const r = wrap.current!.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)),
      y: Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)),
    };
  };

  const boxStyle = (b: BBox) => {
    if (!space) return {};
    const [w, h] = space;
    return {
      left: `${(b[0] / w) * 100}%`,
      top: `${(b[1] / h) * 100}%`,
      width: `${((b[2] - b[0]) / w) * 100}%`,
      height: `${((b[3] - b[1]) / h) * 100}%`,
    };
  };

  const finish = () => {
    if (drag && drawPart && space && onDraw) {
      const [w, h] = space;
      const x0 = Math.min(drag.x0, drag.x1) * w;
      const x1 = Math.max(drag.x0, drag.x1) * w;
      const y0 = Math.min(drag.y0, drag.y1) * h;
      const y1 = Math.max(drag.y0, drag.y1) * h;
      if (x1 - x0 > 12 && y1 - y0 > 12) onDraw(drawPart, [Math.round(x0), Math.round(y0), Math.round(x1), Math.round(y1)]);
    }
    setDrag(null);
  };

  const parts = (overlay?.parts ?? []).filter((p) => p.part !== "whole" && p.method !== "manual");
  return (
    <div
      ref={wrap}
      className={`viewer ${drawPart ? "drawing" : ""}`}
      onMouseDown={(e) => {
        if (!drawPart) return;
        const p = rel(e);
        setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
      }}
      onMouseMove={(e) => {
        if (!drag) return;
        const p = rel(e);
        setDrag({ ...drag, x1: p.x, y1: p.y });
      }}
      onMouseUp={finish}
      onMouseLeave={() => drag && finish()}
    >
      <img
        src={src}
        alt="Query"
        draggable={false}
        onLoad={(e) => {
          const el = e.currentTarget;
          setNatural([el.naturalWidth, el.naturalHeight]);
          onImageLoad?.(el);
        }}
      />
      {showParts &&
        parts.map((p) => (
          <div
            key={p.part}
            className={`box ${highlight === p.part ? "hl" : ""}`}
            style={boxStyle(p.bbox)}
            title={`${PART_LABELS[p.part] ?? p.part} · ${p.method} · ${(p.score * 100).toFixed(0)}%`}
          >
            <span>{PART_LABELS[p.part] ?? p.part}</span>
          </div>
        ))}
      {showFaces &&
        (overlay?.faces ?? []).map((f, i) => (
          <div key={`f${i}`} className="box face" style={boxStyle(f)}>
            <span>masked</span>
          </div>
        ))}
      {manual.map((m) => (
        <div key={`m-${m.part}`} className="box manual" style={boxStyle(m.bbox)}>
          <span>{PART_LABELS[m.part] ?? m.part} (manual)</span>
        </div>
      ))}
      {drag && (
        <div
          className="box manual"
          style={{
            left: `${Math.min(drag.x0, drag.x1) * 100}%`,
            top: `${Math.min(drag.y0, drag.y1) * 100}%`,
            width: `${Math.abs(drag.x1 - drag.x0) * 100}%`,
            height: `${Math.abs(drag.y1 - drag.y0) * 100}%`,
          }}
        />
      )}
    </div>
  );
}
