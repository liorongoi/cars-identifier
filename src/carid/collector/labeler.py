"""Infer a reference image's model-year label from its text metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass

from carid.collector.base import Target
from carid.taxonomy import normalize_name

INTERIOR_WORDS = (
    "interior", "dashboard", "dash board", "cockpit", "steering wheel", "instrument",
    "console", "cabin", "infotainment", "innenraum", "armaturenbrett", "interieur",
    "intérieur", "tableau de bord", "interni", "cruscotto", "interior del", "salpicadero",
    "habitáculo", "inside", "front seats", "gauges",
)
EXTERIOR_WORDS = ("engine bay", "engine compartment", "exterior", "rear view", "front view",
                  "badge", "emblem", "wheel rim", "taillight", "headlight")

_GENERIC_GEN_WORDS = {"original", "facelift", "gen", "generation", "refresh", "1st", "2nd",
                      "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th",
                      "13th", "14th"}
_YEAR_RE = re.compile(r"(?<!\d)(19[89]\d|20[0-4]\d)(?!\d)")
_RANGE_RE = re.compile(r"(?<!\d)(19[89]\d|20[0-4]\d)\s*[-–—/]\s*(19[89]\d|20[0-4]\d|\d{2})(?!\d)")


@dataclass
class LabelGuess:
    year_start: int | None
    year_end: int | None
    generation_id: int | None
    confidence: float
    interior_hint: bool | None  # True: text says interior; False: says exterior; None: unknown
    reason: str


def _contains(text_norm_words: str, phrase: str) -> bool:
    p = normalize_name(phrase)
    if not p:
        return False
    # match on word boundaries of the space-separated normalised text
    return re.search(rf"(?:^| ){re.escape(p)}(?: |$)", text_norm_words) is not None


def _norm_words(text: str) -> str:
    words = re.split(r"[\s_/,;:()\[\]|\"'.]+", text)
    return " ".join(w for w in (normalize_name(w) for w in words) if w)


def _phrase_in(text: str, phrase: str) -> bool:
    """Phrase match tolerant to punctuation and spacing ("CR-V" ~ "CRV" ~ "CR V")."""
    words = _norm_words(text)
    joined = words.replace(" ", "")
    p = normalize_name(phrase)
    if not p:
        return False
    if len(p) <= 3:  # short names (e.g. "A3", "X5", "206") need word boundaries
        return _contains(words, phrase)
    return p in joined


def infer_label(text: str, target: Target, preset_year: int | None = None) -> LabelGuess | None:
    """Return a label guess, or None if the text does not clearly refer to the target model."""
    low = text.lower()
    model_names = [target.model, *target.model_aliases]
    if not any(_phrase_in(text, n) for n in model_names):
        return None
    make_named = any(_phrase_in(text, n) for n in [target.make, *target.make_aliases])

    interior_hint: bool | None = None
    if any(w in low for w in INTERIOR_WORDS):
        interior_hint = True
    elif any(w in low for w in EXTERIOR_WORDS):
        interior_hint = False

    lo = target.year_start or 1990
    hi = target.year_end or 2100
    year_start = year_end = None
    if preset_year is not None:
        year_start = year_end = preset_year
    else:
        m = _RANGE_RE.search(text)
        if m:
            a = int(m.group(1))
            b_raw = m.group(2)
            b = int(b_raw) if len(b_raw) == 4 else (a // 100) * 100 + int(b_raw)
            if lo - 1 <= a <= b <= hi + 1 and b - a <= 12:
                year_start, year_end = a, b
        if year_start is None:
            years = sorted({int(y) for y in _YEAR_RE.findall(text) if lo - 1 <= int(y) <= hi + 1})
            if len(years) == 1:
                year_start = year_end = years[0]
            elif 1 < len(years) <= 3 and years[-1] - years[0] <= 8:
                year_start, year_end = years[0], years[-1]

    generation_id = None
    words = _norm_words(text)
    model_tokens = {normalize_name(w) for n in model_names for w in n.split()}
    for g in target.generations:
        tokens = [t for t in re.split(r"[/\s]+", g.name)
                  if len(t) >= 2 and normalize_name(t) not in _GENERIC_GEN_WORDS
                  and normalize_name(t) not in model_tokens]
        if tokens and any(_contains(words, t) for t in tokens):
            generation_id = g.id
            if year_start is None:
                year_start, year_end = g.year_start, g.year_end or g.year_start
            break

    conflicts = [m for m in target.sibling_models
                 if normalize_name(m) not in {normalize_name(n) for n in model_names}
                 and len(normalize_name(m)) > 3 and _phrase_in(text, m)]

    confidence = 0.55
    reasons = ["model named"]
    if make_named:
        confidence += 0.1
        reasons.append("make named")
    if year_start is not None:
        confidence += 0.2
        reasons.append("year found")
    if generation_id is not None:
        confidence += 0.1
        reasons.append("generation found")
    if conflicts:
        confidence -= 0.35
        reasons.append("also mentions " + ", ".join(conflicts[:3]))
    return LabelGuess(
        year_start=year_start, year_end=year_end, generation_id=generation_id,
        confidence=round(max(0.05, min(confidence, 0.95)), 2),
        interior_hint=interior_hint, reason="; ".join(reasons),
    )
