import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, formatYears, imageUrl, pct, type CatalogMake, type CatalogModel, type RefImage } from "../api";
import { Alert, Chip, Spinner, Stat, useAnalyst, useLoad } from "../components/common";

const FIRST = 1996;
const LAST = new Date().getFullYear();
const YEARS = Array.from({ length: LAST - FIRST + 1 }, (_, i) => FIRST + i);

function heatColor(n: number, max: number): string {
  if (n <= 0) return "var(--bg-2)";
  const t = Math.min(1, Math.log1p(n) / Math.log1p(Math.max(max, 10)));
  return `rgba(56, 189, 248, ${0.18 + 0.82 * t})`;
}

function Heat({ model, max }: { model: CatalogModel; max: number }) {
  const start = model.year_start ?? FIRST;
  const end = model.year_end ?? LAST;
  return (
    <div className="heat" style={{ gridTemplateColumns: `repeat(${YEARS.length}, 1fr)` }} aria-label="images per year">
      {YEARS.map((y) => {
        const n = model.years[String(y)] ?? 0;
        const inRange = y >= start && y <= end;
        return (
          <i
            key={y}
            className={inRange ? "" : "out"}
            style={inRange ? { background: heatColor(n, max) } : undefined}
            title={`${y}: ${n} image${n === 1 ? "" : "s"}${inRange ? "" : " (outside production years)"}`}
          />
        );
      })}
    </div>
  );
}

export function DatabasePage() {
  const catalog = useLoad(() => api.catalog(), []);
  const stats = useLoad(() => api.stats(), []);
  const [q, setQ] = useState("");
  const [gapsOnly, setGapsOnly] = useState(false);
  const [openMakes, setOpenMakes] = useState<Set<number>>(new Set());
  const [openModels, setOpenModels] = useState<Set<number>>(new Set());
  const [browse, setBrowse] = useState<{ make: CatalogMake; model: CatalogModel } | null>(null);
  const [analyst] = useAnalyst();
  const [msg, setMsg] = useState<string | null>(null);

  const makes = useMemo(() => {
    const term = q.trim().toLowerCase();
    return (catalog.data?.makes ?? [])
      .map((m) => {
        const makeHit = !term || m.name.toLowerCase().includes(term) || m.aliases.some((a) => a.toLowerCase().includes(term));
        const models = m.models.filter(
          (md) =>
            (makeHit || md.name.toLowerCase().includes(term) || md.aliases.some((a) => a.toLowerCase().includes(term))) &&
            (!gapsOnly || md.images < 10),
        );
        return { ...m, models };
      })
      .filter((m) => m.models.length > 0 || (!gapsOnly && (!term || m.name.toLowerCase().includes(term))));
  }, [catalog.data, q, gapsOnly]);

  const maxPerYear = useMemo(() => {
    let mx = 1;
    for (const m of catalog.data?.makes ?? []) for (const md of m.models) for (const v of Object.values(md.years)) mx = Math.max(mx, v);
    return mx;
  }, [catalog.data]);

  const toggle = (set: Set<number>, id: number, fn: (s: Set<number>) => void) => {
    const n = new Set(set);
    if (n.has(id)) n.delete(id);
    else n.add(id);
    fn(n);
  };

  const collect = async (params: Record<string, unknown>, label: string) => {
    try {
      const { job_id } = await api.createJob("collect", { per_model_limit: 150, ...params }, analyst || undefined);
      setMsg(`Collection job #${job_id} queued for ${label}. Track it under Jobs.`);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    }
  };

  const s = stats.data;
  const totalModels = s?.models ?? 0;
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Reference database</h1>
          <p>Make › Model › Year, with the number of reference images collected for each.</p>
        </div>
        <Link className="btn primary" to="/add">
          ＋ Add make / model
        </Link>
      </div>

      {s && (
        <div className="grid-cards" style={{ marginBottom: 16 }}>
          <Stat k="Makes" v={s.makes} />
          <Stat k="Models" v={s.models} s={`${s.models_with_images} with images (${totalModels ? pct(s.models_with_images / totalModels, 0) : "0%"})`} />
          <Stat k="Reference images" v={s.images.toLocaleString()} s={`${s.images_by_status.approved ?? 0} approved · ${s.images_by_status.pending ?? 0} pending`} />
          <Stat k="Indexed part vectors" v={Object.values(s.index).reduce((a, b) => a + b, 0).toLocaleString()} s={`${Object.keys(s.index).length} part types`} />
          <Stat
            k="Sources"
            v={Object.keys(s.images_by_source).length}
            s={Object.entries(s.images_by_source)
              .map(([k, v]) => `${k} ${v}`)
              .join(" · ") || "none yet"}
          />
        </div>
      )}
      {msg && (
        <div style={{ marginBottom: 12 }}>
          <Alert kind="info">{msg}</Alert>
        </div>
      )}

      <div className="grid-2" style={{ gridTemplateColumns: browse ? "minmax(0,1fr) minmax(0,1fr)" : "1fr" }}>
        <div className="panel">
          <div className="panel-head">
            <input type="search" placeholder="Search make or model…" value={q} onChange={(e) => setQ(e.target.value)} style={{ flex: 1 }} aria-label="search catalog" />
            <label className="check small">
              <input type="checkbox" checked={gapsOnly} onChange={(e) => setGapsOnly(e.target.checked)} /> Gaps only (&lt;10 images)
            </label>
            <button className="btn sm" onClick={() => setOpenMakes(new Set(makes.map((m) => m.id)))}>
              Expand all
            </button>
          </div>
          {catalog.loading && !catalog.data && (
            <div className="empty">
              <Spinner />
            </div>
          )}
          {catalog.error && <Alert kind="bad">{catalog.error}</Alert>}
          <div className="heat-axis" style={{ maxWidth: "none", padding: "6px 14px 0", justifyContent: "flex-end", gap: 12 }}>
            <span>heat strip: {FIRST} → {LAST}, darker = more images; dashed = outside production</span>
          </div>
          {makes.map((m) => {
            const open = openMakes.has(m.id) || !!q;
            const empty = m.models.filter((md) => md.images === 0).length;
            return (
              <div className="tree-make" key={m.id}>
                <div className="head" onClick={() => toggle(openMakes, m.id, setOpenMakes)}>
                  <span className={`caret ${open ? "open" : ""}`}>▸</span>
                  <b>{m.name}</b>
                  {m.country && <span className="dim small">{m.country}</span>}
                  {m.added_by && m.added_by !== "seed" && <Chip kind="part">{m.added_by}</Chip>}
                  <span className="spacer" />
                  <span className="small muted">
                    {m.models.length} models{empty ? ` · ${empty} without images` : ""}
                  </span>
                  <span className={`count ${m.images === 0 ? "zero" : ""}`}>{m.images.toLocaleString()}</span>
                  <button
                    className="btn sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      void collect({ make_ids: [m.id] }, m.name);
                    }}
                  >
                    Collect
                  </button>
                </div>
                {open &&
                  m.models.map((md) => {
                    const mOpen = openModels.has(md.id);
                    return (
                      <div className="tree-model" key={md.id}>
                        <div className="head" onClick={() => toggle(openModels, md.id, setOpenModels)}>
                          <span className={`caret ${mOpen ? "open" : ""}`}>▸</span>
                          <span style={{ minWidth: 150 }}>{md.name}</span>
                          <span className="dim small nowrap" style={{ minWidth: 92 }}>
                            {formatYears(md.year_start, md.year_end)}
                          </span>
                          <Heat model={md} max={maxPerYear} />
                          <span className="spacer" />
                          <span className={`count ${md.images === 0 ? "zero" : ""}`} title={`${md.approved} approved · ${md.unknown_year} without year`}>
                            {md.images}
                          </span>
                        </div>
                        {mOpen && (
                          <div className="gens">
                            {md.generations.map((g) => (
                              <Chip key={g.id} kind={g.images === 0 ? "bad" : g.images < 10 ? "warn" : "good"} title={`${g.images} images`}>
                                {g.name} · {formatYears(g.year_start, g.year_end)} · {g.images}
                              </Chip>
                            ))}
                            {md.unknown_year > 0 && <Chip>year unknown · {md.unknown_year}</Chip>}
                            <span className="spacer" />
                            <button className="btn sm" onClick={() => setBrowse({ make: m, model: md })}>
                              Browse images
                            </button>
                            <button className="btn sm" onClick={() => void collect({ make_ids: [m.id], model_ids: [md.id] }, `${m.name} ${md.name}`)}>
                              Collect more
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })}
              </div>
            );
          })}
          {!catalog.loading && makes.length === 0 && <div className="empty">Nothing matches.</div>}
        </div>

        {browse && (
          <Gallery
            key={browse.model.id}
            make={browse.make}
            model={browse.model}
            actor={analyst}
            onClose={() => setBrowse(null)}
            onChanged={() => {
              void catalog.reload();
              void stats.reload();
            }}
          />
        )}
      </div>
    </div>
  );
}

function Gallery({
  make,
  model,
  actor,
  onClose,
  onChanged,
}: {
  make: CatalogMake;
  model: CatalogModel;
  actor: string;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [status, setStatus] = useState("");
  const [year, setYear] = useState("");
  const [offset, setOffset] = useState(0);
  const imgs = useLoad(
    () => api.images({ model_id: model.id, status: status || undefined, year: year || undefined, limit: 48, offset }),
    [model.id, status, year, offset],
  );
  const [error, setError] = useState<string | null>(null);

  const act = async (im: RefImage, body: Record<string, unknown>) => {
    try {
      await api.patchImage(im.id, { ...body, actor: actor || undefined });
      setError(null);
      await imgs.reload();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="panel" style={{ position: "sticky", top: 12 }}>
      <div className="panel-head">
        <h2>
          {make.name} {model.name} <span className="muted small">· {imgs.data?.total ?? "…"} images</span>
        </h2>
        <button className="btn sm ghost" onClick={onClose} aria-label="close gallery">
          ✕
        </button>
      </div>
      <div className="panel-body stack">
        <div className="row">
          <select value={status} onChange={(e) => (setOffset(0), setStatus(e.target.value))} aria-label="review status">
            <option value="">All statuses</option>
            <option value="pending">Pending review</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
          </select>
          <input type="number" placeholder="Year" value={year} onChange={(e) => (setOffset(0), setYear(e.target.value))} style={{ width: 100 }} aria-label="year" />
        </div>
        {error && <Alert kind="bad">{error}</Alert>}
        <div className="gallery">
          {(imgs.data?.items ?? []).map((im) => (
            <div className={`thumb ${im.review_status}`} key={im.id}>
              <a href={imageUrl(im.id)} target="_blank" rel="noreferrer">
                <img src={imageUrl(im.id, { size: 360 })} alt={im.title ?? `image ${im.id}`} loading="lazy" />
              </a>
              <div className="meta">
                <div className="row" style={{ gap: 4 }}>
                  <Chip kind={im.review_status}>{im.review_status}</Chip>
                  <span className="dim">{formatYears(im.year_start, im.year_end)}</span>
                </div>
                <div className="dim" title={im.title ?? ""}>
                  {im.source} · label {pct(im.label_confidence, 0)}
                  {im.faces_blurred ? ` · ${im.faces_blurred} face(s) blurred` : ""}
                </div>
                <div className="row" style={{ gap: 4 }}>
                  {im.review_status !== "approved" && (
                    <button className="btn sm" onClick={() => void act(im, { review_status: "approved" })}>
                      ✓
                    </button>
                  )}
                  {im.review_status !== "rejected" && (
                    <button className="btn sm danger" onClick={() => void act(im, { review_status: "rejected" })}>
                      ✕
                    </button>
                  )}
                  <button
                    className="btn sm ghost"
                    onClick={() => {
                      const y = window.prompt("Correct model year (e.g. 2014):", String(im.year_start ?? ""));
                      if (y && /^\d{4}$/.test(y)) void act(im, { year_start: Number(y), year_end: Number(y) });
                    }}
                  >
                    year
                  </button>
                  <select
                    className="small"
                    value=""
                    onChange={(e) => e.target.value && void act(im, { model_id: Number(e.target.value) })}
                    aria-label="move to model"
                    style={{ padding: "3px 4px", maxWidth: 80 }}
                  >
                    <option value="">model…</option>
                    {make.models.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          ))}
        </div>
        {imgs.data && imgs.data.total === 0 && <div className="empty">No images yet. Use “Collect more” or upload brochures into the import folder.</div>}
        {imgs.data && imgs.data.total > 48 && (
          <div className="row">
            <button className="btn sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 48))}>
              ‹ Prev
            </button>
            <span className="small muted">
              {offset + 1}–{Math.min(offset + 48, imgs.data.total)} of {imgs.data.total}
            </span>
            <button className="btn sm" disabled={offset + 48 >= imgs.data.total} onClick={() => setOffset(offset + 48)}>
              Next ›
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
