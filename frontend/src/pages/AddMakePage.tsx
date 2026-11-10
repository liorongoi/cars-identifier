import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Alert, useAnalyst, useLoad } from "../components/common";

interface ModelRow {
  name: string;
  from: string;
  to: string;
  generations: string; // "E210:2018-; E170:2013-2019"
}

const emptyRow = (): ModelRow => ({ name: "", from: "", to: "", generations: "" });

/** Parse "E170:2013-2019; E210:2018-" into generation objects. */
export function parseGenerations(text: string): { name: string; year_start: number; year_end: number | null }[] {
  return text
    .split(/[;\n]/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => {
      const m = /^(.+?)\s*:\s*(\d{4})\s*[-–]\s*(\d{4})?$/.exec(s);
      if (!m) throw new Error(`Cannot read generation "${s}". Use NAME:START-END, e.g. E210:2018-2024 or E210:2018-`);
      return { name: m[1]!.trim(), year_start: Number(m[2]), year_end: m[3] ? Number(m[3]) : null };
    });
}

export function AddMakePage() {
  const [analyst, setAnalyst] = useAnalyst();
  const [name, setName] = useState("");
  const [country, setCountry] = useState("");
  const [aliases, setAliases] = useState("");
  const [rows, setRows] = useState<ModelRow[]>([emptyRow()]);
  const [useWikidata, setUseWikidata] = useState(false);
  const [collect, setCollect] = useState(true);
  const [limit, setLimit] = useState(150);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const catalog = useLoad(() => api.catalog(), []);
  const [modelMake, setModelMake] = useState("");
  const [modelRow, setModelRow] = useState<ModelRow>(emptyRow());

  const update = (i: number, patch: Partial<ModelRow>) => setRows(rows.map((r, j) => (i === j ? { ...r, ...patch } : r)));

  const submitMake = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setDone(null);
    try {
      const models = rows
        .filter((r) => r.name.trim())
        .map((r) => ({
          name: r.name.trim(),
          year_start: r.from ? Number(r.from) : null,
          year_end: r.to ? Number(r.to) : null,
          generations: parseGenerations(r.generations),
        }));
      setBusy(true);
      const res = await api.addMake({
        name,
        country: country || null,
        aliases: aliases.split(",").map((a) => a.trim()).filter(Boolean),
        models,
        use_wikidata: useWikidata,
        collect,
        per_model_limit: limit,
        requested_by: analyst || null,
      });
      setDone(
        `${res.existed ? "Updated" : "Added"} ${name}: +${res.models_added} models, +${res.generations_added} generations.` +
          (res.jobs.length ? ` Queued job(s) #${res.jobs.join(", #")}.` : " No collection queued."),
      );
      setName("");
      setCountry("");
      setAliases("");
      setRows([emptyRow()]);
      void catalog.reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const submitModel = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setDone(null);
    try {
      setBusy(true);
      const res = await api.addModel({
        make_id: Number(modelMake),
        name: modelRow.name.trim(),
        year_start: modelRow.from ? Number(modelRow.from) : null,
        year_end: modelRow.to ? Number(modelRow.to) : null,
        generations: parseGenerations(modelRow.generations),
        collect,
        per_model_limit: limit,
        requested_by: analyst || null,
      });
      setDone(`Model saved (#${res.model_id}).${res.jobs.length ? ` Collection job #${res.jobs[0]} queued.` : ""}`);
      setModelRow(emptyRow());
      void catalog.reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page" style={{ maxWidth: 1100 }}>
      <div className="page-head">
        <div>
          <h1>Add a make or model</h1>
          <p>Missing a manufacturer? Add it here and the collector will gather interior images for it.</p>
        </div>
      </div>
      {error && <Alert kind="bad">{error}</Alert>}
      {done && (
        <Alert kind="info">
          {done} <Link to="/jobs">Open Jobs →</Link>
        </Alert>
      )}

      <div className="stack" style={{ marginTop: 12 }}>
        <form className="panel" onSubmit={(e) => void submitMake(e)}>
          <div className="panel-head">
            <h2>New make</h2>
          </div>
          <div className="panel-body stack">
            <div className="row" style={{ alignItems: "flex-end" }}>
              <label className="field" style={{ flex: 2 }}>
                Make name *
                <input type="text" required value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Chery" />
              </label>
              <label className="field" style={{ flex: 1 }}>
                Country
                <input type="text" value={country} onChange={(e) => setCountry(e.target.value)} placeholder="China" />
              </label>
              <label className="field" style={{ flex: 2 }}>
                Other names (comma separated)
                <input type="text" value={aliases} onChange={(e) => setAliases(e.target.value)} placeholder="Chery Automobile" />
              </label>
            </div>

            <label className="check">
              <input type="checkbox" checked={useWikidata} onChange={(e) => setUseWikidata(e.target.checked)} />
              Fill in models and generations automatically from Wikidata (needs internet on the server)
            </label>

            <div>
              <h3 style={{ marginBottom: 8 }}>Models {useWikidata && <span className="dim">(optional with Wikidata)</span>}</h3>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th style={{ width: 90 }}>From</th>
                    <th style={{ width: 90 }}>To</th>
                    <th>Generations (NAME:START-END; …)</th>
                    <th style={{ width: 40 }} />
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i}>
                      <td>
                        <input type="text" value={r.name} onChange={(e) => update(i, { name: e.target.value })} placeholder="Tiggo 7" style={{ width: "100%" }} aria-label={`model ${i + 1} name`} />
                      </td>
                      <td>
                        <input type="number" value={r.from} onChange={(e) => update(i, { from: e.target.value })} style={{ width: "100%" }} aria-label={`model ${i + 1} from`} />
                      </td>
                      <td>
                        <input type="number" value={r.to} onChange={(e) => update(i, { to: e.target.value })} placeholder="now" style={{ width: "100%" }} aria-label={`model ${i + 1} to`} />
                      </td>
                      <td>
                        <input type="text" value={r.generations} onChange={(e) => update(i, { generations: e.target.value })} placeholder="T1E:2016-2020; Pro:2020-" style={{ width: "100%" }} aria-label={`model ${i + 1} generations`} />
                      </td>
                      <td>
                        <button type="button" className="btn sm ghost" onClick={() => setRows(rows.length > 1 ? rows.filter((_, j) => j !== i) : [emptyRow()])} aria-label="remove row">
                          ✕
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button type="button" className="btn sm" style={{ marginTop: 8 }} onClick={() => setRows([...rows, emptyRow()])}>
                ＋ Add model row
              </button>
            </div>
            <CollectOptions collect={collect} setCollect={setCollect} limit={limit} setLimit={setLimit} analyst={analyst} setAnalyst={setAnalyst} />
            <div>
              <button className="btn primary" disabled={busy || !name.trim()}>
                Add make{collect ? " and start collecting" : ""}
              </button>
            </div>
          </div>
        </form>

        <form className="panel" onSubmit={(e) => void submitModel(e)}>
          <div className="panel-head">
            <h2>Add a model to an existing make</h2>
          </div>
          <div className="panel-body row" style={{ alignItems: "flex-end" }}>
            <label className="field">
              Make *
              <select required value={modelMake} onChange={(e) => setModelMake(e.target.value)}>
                <option value="">Choose…</option>
                {(catalog.data?.makes ?? []).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              Model *
              <input type="text" required value={modelRow.name} onChange={(e) => setModelRow({ ...modelRow, name: e.target.value })} />
            </label>
            <label className="field" style={{ width: 90 }}>
              From
              <input type="number" value={modelRow.from} onChange={(e) => setModelRow({ ...modelRow, from: e.target.value })} />
            </label>
            <label className="field" style={{ width: 90 }}>
              To
              <input type="number" value={modelRow.to} onChange={(e) => setModelRow({ ...modelRow, to: e.target.value })} />
            </label>
            <label className="field" style={{ flex: 1 }}>
              Generations
              <input type="text" value={modelRow.generations} onChange={(e) => setModelRow({ ...modelRow, generations: e.target.value })} placeholder="NAME:START-END; …" />
            </label>
            <button className="btn" disabled={busy || !modelMake || !modelRow.name.trim()}>
              Add model
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function CollectOptions(p: {
  collect: boolean;
  setCollect: (v: boolean) => void;
  limit: number;
  setLimit: (v: number) => void;
  analyst: string;
  setAnalyst: (v: string) => void;
}) {
  return (
    <div className="row" style={{ alignItems: "flex-end" }}>
      <label className="check">
        <input type="checkbox" checked={p.collect} onChange={(e) => p.setCollect(e.target.checked)} /> Start image collection now
      </label>
      <label className="field" style={{ width: 170 }}>
        Max new images per model
        <input type="number" min={1} max={5000} value={p.limit} onChange={(e) => p.setLimit(Number(e.target.value))} />
      </label>
      <label className="field" style={{ width: 200 }}>
        Requested by
        <input type="text" value={p.analyst} onChange={(e) => p.setAnalyst(e.target.value)} />
      </label>
    </div>
  );
}
