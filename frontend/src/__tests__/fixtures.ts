import type { IdentifyResult } from "../api";

export const RESULT: IdentifyResult = {
  id: 7,
  candidates: [
    {
      rank: 1, make_id: 1, make: "Toyota", model_id: 10, model: "Corolla", generation_id: 3, generation: "E210",
      year_start: 2018, year_end: null, probability: 0.72, confidence: "high", score: 3.2,
      part_scores: { steering_wheel: 1.2, whole: 1.0, infotainment: 0.6 }, reference_images: 48,
      evidence: [{ part: "steering_wheel", similarity: 0.91, image_id: 55, bbox: [1, 2, 30, 40], source: "wikimedia",
        page_url: "https://commons.wikimedia.org/x", credit: "Jane", license: "CC BY-SA 4.0" }],
    },
    {
      rank: 2, make_id: 2, make: "Honda", model_id: 20, model: "Civic", generation_id: null, generation: null,
      year_start: 2016, year_end: 2021, probability: 0.18, confidence: "low", score: 0.8,
      part_scores: { whole: 0.8 }, reference_images: 3, evidence: [],
    },
  ],
  makes: [{ make_id: 1, make: "Toyota", probability: 0.75 }, { make_id: 2, make: "Honda", probability: 0.25 }],
  quality: { width: 800, height: 600, sharpness: 140, brightness: 60, contrast: 40, noise: 3, dark_fraction: 0.3,
    score: 73, verdict: "fair", issues: ["low light"] },
  enhancement: ["low-light"],
  faces: [[10, 10, 50, 60]],
  parts: [{ part: "whole", bbox: [0, 0, 800, 600], score: 1, method: "full-frame" },
    { part: "steering_wheel", bbox: [80, 200, 400, 560], score: 0.45, method: "circle" }],
  image_size: [800, 600],
  warnings: ["Poor image quality (low light); results are less reliable."],
  duration_ms: 812,
  engine: { embedder: "handcrafted-v1", embedder_dim: 512, detector: "layout", face_detector: "haar", deep_models: false,
    index_sizes: { whole: 100 }, matcher_mode: "fusion" },
  created_at: "2026-11-10T10:00:00Z",
};
