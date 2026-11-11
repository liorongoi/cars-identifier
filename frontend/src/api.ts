// Typed client for the CarID REST API (see /api/docs on the server).

export type BBox = [number, number, number, number];

export interface Evidence {
  part: string;
  similarity: number;
  image_id: number;
  bbox: BBox | null;
  source: string | null;
  page_url: string | null;
  credit: string | null;
  license: string | null;
}

export interface Candidate {
  rank: number;
  make_id: number;
  make: string;
  model_id: number;
  model: string;
  probability: number;
  confidence: "high" | "medium" | "low";
  score: number;
  part_scores: Record<string, number>;
  reference_images: number;
  evidence: Evidence[];
}

export interface Quality {
  width: number;
  height: number;
  sharpness: number;
  brightness: number;
  contrast: number;
  noise: number;
  dark_fraction: number;
  score: number;
  verdict: "good" | "fair" | "poor";
  issues: string[];
}

export interface DetectedPart {
  part: string;
  bbox: BBox;
  score: number;
  method: string;
}

export interface IdentifyResult {
  id: number;
  candidates: Candidate[];
  makes: { make_id: number; make: string; probability: number }[];
  quality: Quality;
  enhancement: string[];
  faces: BBox[];
  parts: DetectedPart[];
  image_size: [number, number];
  warnings: string[];
  duration_ms: number;
  engine: EngineInfo;
  created_at: string;
}

export interface CaseRecord extends IdentifyResult {
  case_ref: string | null;
  analyst: string | null;
  options: Record<string, unknown>;
  has_query_image: boolean;
  feedback_model_id: number | null;
  feedback_note: string | null;
}

export interface EngineInfo {
  embedder: string;
  embedder_dim: number;
  detector: string;
  face_detector: string;
  deep_models: boolean;
  index_sizes: Record<string, number>;
  matcher_mode: string;
  adapter: string[];
}

export interface Health {
  status: string;
  version: string;
  engine: EngineInfo;
  warnings: string[];
}

export interface Stats {
  makes: number;
  models: number;
  models_with_images: number;
  images: number;
  images_by_status: Record<string, number>;
  images_by_source: Record<string, number>;
  identifications: number;
  index: Record<string, number>;
}

export interface CatalogModel {
  id: number;
  name: string;
  aliases: string[];
  images: number;
  approved: number;
}

export interface CatalogMake {
  id: number;
  name: string;
  country: string | null;
  aliases: string[];
  added_by: string | null;
  images: number;
  models: CatalogModel[];
}

export interface Catalog {
  total_images: number;
  makes: CatalogMake[];
}

export interface RefImage {
  id: number;
  make_id: number;
  model_id: number;
  make: string;
  model: string;
  label_confidence: number;
  source: string;
  page_url: string | null;
  license: string | null;
  credit: string | null;
  title: string | null;
  width: number;
  height: number;
  interior_score: number | null;
  faces_blurred: number;
  review_status: "pending" | "approved" | "rejected";
  created_at: string;
}

export interface Job {
  id: number;
  kind: string;
  params: Record<string, unknown>;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  progress: number;
  message: string;
  counters: Record<string, number | string>;
  requested_by: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  log?: string;
}

export interface Source {
  name: string;
  description: string;
  enabled: boolean;
  requires_legal_approval: boolean;
  legal_approved: boolean;
  runnable: boolean;
  reason: string;
}

export interface HistoryItem {
  id: number;
  case_ref: string | null;
  analyst: string | null;
  created_at: string;
  duration_ms: number;
  has_query_image: boolean;
  top: { make: string; model: string; probability: number } | null;
  quality: string | null;
  feedback_model_id: number | null;
}

export interface Metrics {
  n: number;
  top1?: number;
  top5?: number;
  top10?: number;
  mrr?: number;
}

export interface BenchmarkSummary {
  id: number;
  embedder: string;
  created_at: string;
  best_mode: string | null;
  summary: Record<string, { mean_model_top5: number }>;
  test_images: number;
}

export interface BenchmarkReport {
  id: number;
  created_at: string;
  embedder: string;
  detector: string;
  reference_images: number;
  test_images: number;
  classes: number;
  conditions: string[];
  modes: string[];
  results: Record<string, Record<string, Record<string, Metrics>>>;
  summary: Record<string, { mean_model_top5: number }>;
  best_mode: string | null;
  mean_latency_ms: number | null;
}

export interface AuditEvent {
  id: number;
  actor: string | null;
  action: string;
  target: string;
  details: Record<string, unknown>;
  created_at: string;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) {
        detail = body.detail
          .map((d: { msg?: string; loc?: unknown[] }) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg ?? ""}`)
          .join("; ");
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export interface IdentifyParams {
  file: File;
  topK: number;
  enhance: boolean;
  maskFaces: boolean;
  mode?: string;
  makeIds?: number[];
  manualParts?: { part: string; bbox: BBox }[];
  caseRef?: string;
  analyst?: string;
}

export function buildIdentifyForm(p: IdentifyParams): FormData {
  const fd = new FormData();
  fd.append("file", p.file);
  fd.append("top_k", String(p.topK));
  fd.append("enhance", String(p.enhance));
  fd.append("mask_faces", String(p.maskFaces));
  if (p.mode) fd.append("mode", p.mode);
  if (p.makeIds?.length) fd.append("make_ids", p.makeIds.join(","));
  if (p.manualParts?.length) fd.append("manual_parts", JSON.stringify(p.manualParts));
  if (p.caseRef) fd.append("case_ref", p.caseRef);
  if (p.analyst) fd.append("analyst", p.analyst);
  return fd;
}

export const api = {
  health: () => request<Health>("/api/health"),
  stats: () => request<Stats>("/api/stats"),
  sources: () => request<Source[]>("/api/sources"),
  audit: () => request<AuditEvent[]>("/api/audit?limit=100"),
  identify: (p: IdentifyParams) =>
    request<IdentifyResult>("/api/identify", { method: "POST", body: buildIdentifyForm(p) }),
  history: (q = "", offset = 0) =>
    request<{ total: number; items: HistoryItem[] }>(
      `/api/identifications?limit=50&offset=${offset}&q=${encodeURIComponent(q)}`,
    ),
  getCase: (id: number) => request<CaseRecord>(`/api/identifications/${id}`),
  feedback: (id: number, modelId: number | null, note: string, analyst: string) =>
    request<{ ok: boolean }>(`/api/identifications/${id}/feedback`, {
      method: "POST",
      ...json({ model_id: modelId, note, analyst }),
    }),
  catalog: () => request<Catalog>("/api/catalog"),
  addMake: (body: unknown) =>
    request<{ make_id: number; existed: boolean; models_added: number; jobs: number[] }>(
      "/api/catalog/makes",
      { method: "POST", ...json(body) },
    ),
  addModel: (body: unknown) =>
    request<{ model_id: number; models_added: number; jobs: number[] }>("/api/catalog/models", {
      method: "POST",
      ...json(body),
    }),
  images: (params: Record<string, string | number | undefined>) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== "") q.set(k, String(v));
    return request<{ total: number; items: RefImage[] }>(`/api/images?${q.toString()}`);
  },
  patchImage: (id: number, body: unknown) =>
    request<RefImage>(`/api/images/${id}`, { method: "PATCH", ...json(body) }),
  jobs: () => request<Job[]>("/api/jobs?limit=100"),
  job: (id: number) => request<Job>(`/api/jobs/${id}`),
  createJob: (kind: string, params: Record<string, unknown>, requestedBy?: string) =>
    request<{ job_id: number }>("/api/jobs", { method: "POST", ...json({ kind, params, requested_by: requestedBy }) }),
  cancelJob: (id: number) => request<{ cancelled: boolean }>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  benchmarks: () => request<BenchmarkSummary[]>("/api/benchmarks"),
  benchmark: (id: number) => request<BenchmarkReport>(`/api/benchmarks/${id}`),
};

export const imageUrl = (id: number, opts: { size?: number; bbox?: BBox | null } = {}): string => {
  const q = new URLSearchParams();
  if (opts.size) q.set("size", String(opts.size));
  if (opts.bbox) q.set("bbox", opts.bbox.join(","));
  const qs = q.toString();
  return `/api/images/${id}/file${qs ? `?${qs}` : ""}`;
};

export const pct = (p: number, digits = 1): string => `${(p * 100).toFixed(digits)}%`;

export const PART_LABELS: Record<string, string> = {
  whole: "Whole image",
  dashboard: "Dashboard",
  steering_wheel: "Steering wheel",
  instrument_cluster: "Instrument cluster",
  infotainment: "Infotainment screen",
  air_vent: "Air vents",
  center_console: "Center console",
  gear_selector: "Gear selector",
  climate_control: "Climate controls",
  door_panel: "Door panel",
};
