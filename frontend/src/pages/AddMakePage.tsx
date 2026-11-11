import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Alert, useAnalyst, useLoad } from "../components/common";

/** "Tiggo 7, Arrizo 5\nOmoda 5" -> ["Tiggo 7", "Arrizo 5", "Omoda 5"] (duplicates removed). */
export function parseModelNames(text: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of text.split(/[,;\n]/)) {
    const name = raw.trim();
    if (name && !seen.has(name.toLowerCase())) {
      seen.add(name.toLowerCase());
      out.push(name);
    }
  }
  return out;
}

export function AddMakePage() {
  const [analyst, setAnalyst] = useAnalyst();
  const [name, setName] = useState("");
  const [country, setCountry] = useState("");
  const [aliases, setAliases] = useState("");
  const [models, setModels] = useState("");
  const [useWikidata, setUseWikidata] = useState(false);
  const [collect, setCollect] = useState(true);
  const [limit, setLimit] = useState(150);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const catalog = useLoad(() => api.catalog(), []);
  const [modelMake, setModelMake] = useState("");
  const [modelName, setModelName] = useState("");

  const run = async (fn: () => Promise<string>) => {
    setError(null);
    setDone(null);
    setBusy(true);
    try {
      setDone(await fn());
      void catalog.reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const submitMake = (e: FormEvent) => {
    e.preventDefault();
    void run(async () => {
      const res = await api.addMake({
        name,
        country: country || null,
        aliases: parseModelNames(aliases),
        models: parseModelNames(models).map((m) => ({ name: m })),
        use_wikidata: useWikidata,
        collect,
        per_model_limit: limit,
        requested_by: analyst || null,
      });
      const msg =
        `${res.existed ? "Updated" : "Added"} ${name}: +${res.models_added} models.` +
        (res.jobs.length ? ` Queued job(s) #${res.jobs.join(", #")}.` : " No collection queued.");
      setName("");
      setCountry("");
      setAliases("");
      setModels("");
      return msg;
    });
  };

  const submitModel = (e: FormEvent) => {
    e.preventDefault();
    void run(async () => {
      const res = await api.addModel({
        make_id: Number(modelMake),
        name: modelName.trim(),
        collect,
        per_model_limit: limit,
        requested_by: analyst || null,
      });
      setModelName("");
      return `Model saved (#${res.model_id}).${res.jobs.length ? ` Collection job #${res.jobs[0]} queued.` : ""}`;
    });
  };

  return (
    <div className="page" style={{ maxWidth: 1100 }}>
      <div className="page-head">
        <div>
          <h1>Add a make or model</h1>
          <p>Missing a manufacturer or model? Add it here and the collector will gather interior images for it.</p>
        </div>
      </div>
      {error && <Alert kind="bad">{error}</Alert>}
      {done && (
        <Alert kind="info">
          {done} <Link to="/jobs">Open Jobs →</Link>
        </Alert>
      )}

      <div className="stack" style={{ marginTop: 12 }}>
        <form className="panel" onSubmit={submitMake}>
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
            <label className="field">
              Models (comma or one per line){useWikidata && <span className="dim"> — optional with Wikidata</span>}
              <textarea
                rows={4}
                value={models}
                onChange={(e) => setModels(e.target.value)}
                placeholder={"Tiggo 7\nTiggo 8\nArrizo 5"}
                aria-label="models"
              />
            </label>
            <label className="check">
              <input type="checkbox" checked={useWikidata} onChange={(e) => setUseWikidata(e.target.checked)} />
              Fill in the model list automatically from Wikidata (needs internet on the server)
            </label>
            <CollectOptions collect={collect} setCollect={setCollect} limit={limit} setLimit={setLimit} analyst={analyst} setAnalyst={setAnalyst} />
            <div>
              <button className="btn primary" disabled={busy || !name.trim()}>
                Add make{collect ? " and start collecting" : ""}
              </button>
            </div>
          </div>
        </form>

        <form className="panel" onSubmit={submitModel}>
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
            <label className="field" style={{ flex: 1 }}>
              Model *
              <input type="text" required value={modelName} onChange={(e) => setModelName(e.target.value)} />
            </label>
            <button className="btn" disabled={busy || !modelMake || !modelName.trim()}>
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
