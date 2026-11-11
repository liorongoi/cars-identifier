"""Deep-learning backends (optional ``ml`` extra: torch, open_clip, transformers).

Imported lazily so the rest of the system works without them.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from carid.vision.embed import Vectors, l2_normalize
from carid.vision.imageio import RGBImage
from carid.vision.parts import LayoutPartDetector, PartBox, whole_box

log = logging.getLogger(__name__)

PART_PROMPTS: dict[str, list[str]] = {
    "dashboard": ["a car dashboard"],
    "steering_wheel": ["a car steering wheel"],
    "instrument_cluster": ["a car instrument cluster with speedometer gauges"],
    "infotainment": ["a car infotainment touchscreen", "a car dashboard display screen"],
    "air_vent": ["a car air conditioning vent"],
    "center_console": ["a car center console"],
    "gear_selector": ["a car gear shifter", "a car gear selector knob"],
    "climate_control": ["car climate control knobs and buttons"],
    "door_panel": ["a car interior door panel with door handle"],
}

INTERIOR_PROMPTS = [
    "a photo of the interior of a car",
    "a photo of a car dashboard",
    "a photo of a car steering wheel and instrument cluster",
    "a photo of car seats inside a vehicle",
]
NON_INTERIOR_PROMPTS = [
    "a photo of the exterior of a car",
    "a photo of a car engine",
    "a photo of a car wheel and tyre",
    "a photo of a person",
    "a document or brochure page with text",
    "a photo of a building or a landscape",
    "a logo or a drawing",
]


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class ClipEmbedder:
    """OpenCLIP image tower. Weights are cached under ``models_cache`` for offline use."""

    def __init__(self, model: str, pretrained: str, models_cache: str,
                 allow_download: bool) -> None:
        import open_clip
        import torch

        cache = Path(models_cache) / "open_clip"
        cache.mkdir(parents=True, exist_ok=True)
        if not allow_download:
            import os

            os.environ["HF_HUB_OFFLINE"] = "1"
        self._torch = torch
        self.device = _device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, cache_dir=str(cache), device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self.name = f"clip-{model}-{pretrained}"
        with torch.no_grad():
            dummy = self.preprocess(Image.new("RGB", (64, 64))).unsqueeze(0).to(self.device)
            self.dim = int(self.model.encode_image(dummy).shape[-1])
        self._lock = threading.Lock()
        self._prompt_vecs: Vectors | None = None

    def embed(self, images: list[RGBImage]) -> Vectors:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        out: list[np.ndarray] = []
        with self._lock, torch.no_grad():
            for i in range(0, len(images), 32):
                batch = torch.stack(
                    [self.preprocess(Image.fromarray(im)) for im in images[i : i + 32]]
                ).to(self.device)
                out.append(self.model.encode_image(batch).float().cpu().numpy())
        return l2_normalize(np.concatenate(out))

    def embed_text(self, texts: list[str]) -> Vectors:
        torch = self._torch
        with self._lock, torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            return l2_normalize(self.model.encode_text(tokens).float().cpu().numpy())

    def interior_score(self, img: RGBImage) -> float:
        """Probability-like score that the image shows a car interior (zero-shot)."""
        img_vec = self.embed([img])[0]
        if self._prompt_vecs is None:  # constant prompts: encode once
            self._prompt_vecs = self.embed_text(INTERIOR_PROMPTS + NON_INTERIOR_PROMPTS)
        txt = self._prompt_vecs
        logits = 100.0 * (txt @ img_vec)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return float(p[: len(INTERIOR_PROMPTS)].sum())


class OwlPartDetector:
    name = "owlv2"

    def __init__(self, model: str, models_cache: str, allow_download: bool,
                 threshold: float = 0.12) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        kwargs: dict[str, Any] = {"cache_dir": str(Path(models_cache) / "hf")}
        if not allow_download:
            kwargs["local_files_only"] = True
        self._torch = torch
        self.device = _device()
        self.processor = Owlv2Processor.from_pretrained(model, **kwargs)
        self.model = cast(Any, Owlv2ForObjectDetection.from_pretrained(model, **kwargs)).to(
            self.device)
        self.model.eval()
        self.threshold = threshold
        self._labels: list[tuple[str, str]] = [
            (part, prompt) for part, prompts in PART_PROMPTS.items() for prompt in prompts
        ]
        self._lock = threading.Lock()
        self._layout = LayoutPartDetector()

    def detect(self, img: RGBImage) -> list[PartBox]:
        torch = self._torch
        h, w = img.shape[:2]
        pil = Image.fromarray(img)
        texts = [[p for _, p in self._labels]]
        with self._lock, torch.no_grad():
            inputs = self.processor(text=texts, images=pil, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # OWLv2 pads to a square; target size is the padded square side.
            side = max(h, w)
            # image_processor's method exists in transformers 4.x and 5.x alike.
            results = self.processor.image_processor.post_process_object_detection(
                outputs, threshold=self.threshold,
                target_sizes=torch.tensor([[side, side]], device=self.device),
            )[0]
        best: dict[str, PartBox] = {}
        for score, label, box in zip(
            results["scores"].tolist(), results["labels"].tolist(), results["boxes"].tolist(),
            strict=True,
        ):
            part = self._labels[int(label)][0]
            x0, y0, x1, y1 = (int(max(0, v)) for v in box)
            x1, y1 = min(x1, w), min(y1, h)
            if x1 - x0 < 12 or y1 - y0 < 12:
                continue
            if part not in best or score > best[part].score:
                best[part] = PartBox(part, (x0, y0, x1, y1), float(score), "owlv2")
        # Parts the detector missed fall back to the layout prior, so every image
        # (reference or query) yields the same set of regions to compare.
        for prior in self._layout.detect(img):
            if prior.part != "whole" and prior.part not in best:
                best[prior.part] = prior
        return [whole_box(img), *best.values()]
