import { Fragment, useEffect, useRef, useState } from "react";
import { formatYears, imageUrl, PART_LABELS, pct, type BBox, type Candidate, type DetectedPart } from "../api";
import { Bar, Chip } from "./common";

interface Props {
  candidates: Candidate[];
  queryParts: DetectedPart[];
  imageSize: [number, number];
  queryImg: HTMLImageElement | null;
  onHoverPart?: (part: string | null) => void;
}

/** Crops the query image client-side so evidence shows query part vs reference part. */
function QueryCrop({ img, bbox, size }: { img: HTMLImageElement | null; bbox: BBox | undefined; size: [number, number] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !img || !bbox || !img.naturalWidth) return;
    const sx = img.naturalWidth / size[0];
    const sy = img.naturalHeight / size[1];
    const [x0, y0, x1, y1] = bbox;
    const w = Math.max(1, (x1 - x0) * sx);
    const h = Math.max(1, (y1 - y0) * sy);
    c.width = Math.min(400, w);
    c.height = (c.width / w) * h;
    c.getContext("2d")?.drawImage(img, x0 * sx, y0 * sy, w, h, 0, 0, c.width, c.height);
  }, [img, bbox, size]);
  return <canvas ref={ref} aria-label="query part" />;
}

export function ResultsTable({ candidates, queryParts, imageSize, queryImg, onHoverPart }: Props) {
  const [open, setOpen] = useState<number | null>(candidates[0]?.rank ?? null);
  const partBox = (part: string) => queryParts.find((p) => p.part === part)?.bbox;

  if (!candidates.length) return <div className="empty">No candidates.</div>;
  return (
    <table className="tbl" aria-label="Identification results">
      <thead>
        <tr>
          <th style={{ width: 52 }}>#</th>
          <th>Vehicle</th>
          <th>Years</th>
          <th style={{ width: "26%" }}>Probability</th>
          <th>Confidence</th>
          <th className="right" title="Reference images in the database for this candidate">Ref. images</th>
          <th>Matched parts</th>
        </tr>
      </thead>
      <tbody>
        {candidates.map((c) => {
          const isOpen = open === c.rank;
          const parts = Object.entries(c.part_scores).sort((a, b) => b[1] - a[1]);
          return (
            <Fragment key={`${c.model_id}-${c.generation_id ?? "x"}`}>
              <tr
                className={`clickable ${c.rank === 1 ? "top1" : ""} ${isOpen ? "open" : ""}`}
                onClick={() => setOpen(isOpen ? null : c.rank)}
                aria-expanded={isOpen}
              >
                <td>
                  <div className="rank">{c.rank}</div>
                </td>
                <td className="car">
                  <b>
                    {c.make} {c.model}
                  </b>
                  <div className="gen">{c.generation ? `Generation ${c.generation}` : "Generation not resolved"}</div>
                </td>
                <td className="nowrap">{formatYears(c.year_start, c.year_end)}</td>
                <td>
                  <div className="prob">
                    <Bar value={c.probability} />
                    <b>{pct(c.probability)}</b>
                  </div>
                </td>
                <td>
                  <Chip kind={c.confidence}>{c.confidence}</Chip>
                </td>
                <td className="right">
                  <span className={c.reference_images < 5 ? "muted" : ""} title={c.reference_images < 5 ? "Few reference images: less reliable" : undefined}>
                    {c.reference_images}
                  </span>
                </td>
                <td>
                  <div className="row" style={{ gap: 4 }}>
                    {parts.slice(0, 4).map(([p]) => (
                      <Chip key={p} kind="part">
                        {PART_LABELS[p] ?? p}
                      </Chip>
                    ))}
                    {parts.length > 4 && <span className="dim small">+{parts.length - 4}</span>}
                  </div>
                </td>
              </tr>
              {isOpen && (
                <tr>
                  <td colSpan={7} style={{ background: "#0d131d" }}>
                    <div className="small muted" style={{ marginBottom: 8 }}>
                      Evidence: query part (left) against the closest reference image of this candidate (right).
                    </div>
                    <div className="evidence">
                      {c.evidence.map((e) => (
                        <div
                          className="ev"
                          key={`${e.part}-${e.image_id}`}
                          onMouseEnter={() => onHoverPart?.(e.part)}
                          onMouseLeave={() => onHoverPart?.(null)}
                        >
                          <div className="row" style={{ marginBottom: 6 }}>
                            <b className="small">{PART_LABELS[e.part] ?? e.part}</b>
                            <span className="spacer" />
                            <span className="mono small" title="cosine similarity">
                              sim {e.similarity.toFixed(3)}
                            </span>
                          </div>
                          <div className="pair">
                            <figure>
                              <QueryCrop img={queryImg} bbox={partBox(e.part)} size={imageSize} />
                              <figcaption>Query</figcaption>
                            </figure>
                            <figure>
                              <a href={imageUrl(e.image_id)} target="_blank" rel="noreferrer">
                                <img src={imageUrl(e.image_id, { size: 400, bbox: e.bbox })} alt={`reference ${e.part}`} loading="lazy" />
                              </a>
                              <figcaption>
                                Ref #{e.image_id} · {e.source}
                              </figcaption>
                            </figure>
                          </div>
                          {(e.license || e.credit) && (
                            <div className="dim small" style={{ marginTop: 6 }}>
                              {e.license}
                              {e.credit ? ` · ${e.credit}` : ""}
                              {e.page_url && (
                                <>
                                  {" · "}
                                  <a href={e.page_url} target="_blank" rel="noreferrer">
                                    source
                                  </a>
                                </>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </td>
                </tr>
              )}
            </Fragment>
          );
        })}
      </tbody>
    </table>
  );
}
