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
    """Whole-word phrase match tolerant to punctuation and spacing ("CR-V" ~ "CRV" ~ "CR V").

    The phrase must equal a run of consecutive whole words: "Note" does not match
    "footnote" and "A3" does not match "A35".
    """
    p = normalize_name(phrase)
    if not p:
        return False
    words = re.split(r"[\s_/,;:()\[\]|\"'.\-]+", text)
    norm = [w for w in (normalize_name(w) for w in words) if w]
    for size in range(1, 7):
        for i in range(len(norm) - size + 1):
            if "".join(norm[i : i + size]) == p:
                return True
    return False


def _mentions_model(text: str, make_names: list[str], model_name: str) -> bool:
    """Code-like names (A3, CR-V, 3008, Model 3) may stand alone. Plain words
    (Focus, Note, Leaf, Fit, Ranger) are ordinary words in captions, so they count
    only right after the make ("Ford Focus"), or capitalised in a text that names
    the make ("Focus ST (Ford)"), never as "shallow focus"."""
    if any(ch.isdigit() for ch in model_name) or not model_name.replace(" ", "").isalpha():
        return _phrase_in(text, model_name)
    if any(_phrase_in(text, f"{mk} {model_name}") for mk in make_names):
        return True
    capitalised = re.search(rf"(?<![\w-]){re.escape(model_name)}(?![\w-])", text) is not None
    return capitalised and any(_phrase_in(text, mk) for mk in make_names)


_ANY_YEAR_RE = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")


def _year_bounds(target: Target) -> tuple[int, int]:
    starts = [g.year_start for g in target.generations]
    if target.year_start:
        starts.append(target.year_start)
    ends = [g.year_end for g in target.generations] + [target.year_end]
    lo = min(starts) if starts else 1990
    hi = 2100 if (not ends or any(e is None for e in ends)) else max(e for e in ends if e)
    return lo, hi


def _without_model_numbers(text: str, target: Target) -> str:
    """Blank out numbers that are model names ("Peugeot 2008", "3008") before year parsing."""
    numbers = {w for n in [target.model, *target.model_aliases, *target.sibling_models]
               for w in re.findall(r"\d{4}", n)}
    for num in numbers:
        text = re.sub(rf"(?<!\d){num}(?!\d)", " ", text)
    return text


def _out_of_scope(ytext: str, lo: int, hi: int) -> bool:
    """True when the text states years and none falls in the model's covered range."""
    years = [int(y) for y in _ANY_YEAR_RE.findall(ytext)]
    return bool(years) and not any(lo - 1 <= y <= hi + 1 for y in years)


def title_out_of_scope(title: str, target: Target) -> bool:
    lo, hi = _year_bounds(target)
    return _out_of_scope(_without_model_numbers(title, target), lo, hi)


def title_verdict(title: str, target: Target) -> str:
    """'match' if the title names the model, 'other' if it names the make but not the
    model (e.g. "Audi R8 Spyder" for an A1 search), 'silent' if it names neither."""
    model_names = [target.model, *target.model_aliases]
    make_names = [target.make, *target.make_aliases]
    if any(_mentions_model(title, make_names, n) for n in model_names):
        return "match"
    if any(_phrase_in(title, n) for n in make_names):
        return "other"
    return "silent"


def infer_label(text: str, target: Target, preset_year: int | None = None,
                title: str | None = None) -> LabelGuess | None:
    """Return a label guess, or None if the text does not clearly refer to the target model.

    ``title`` (when given, also part of ``text``) outranks tags and descriptions: a title
    that names the make but another model rejects the image, and a model found only in
    tags/description lowers the confidence.
    """
    low = text.lower()
    model_names = [target.model, *target.model_aliases]
    make_names = [target.make, *target.make_aliases]
    if not any(_mentions_model(text, make_names, n) for n in model_names):
        return None
    verdict = title_verdict(title, target) if title else "match"
    if verdict == "other":
        return None
    make_named = any(_phrase_in(text, n) for n in make_names)

    interior_hint: bool | None = None
    if any(w in low for w in INTERIOR_WORDS):
        interior_hint = True
    elif any(w in low for w in EXTERIOR_WORDS):
        interior_hint = False

    lo, hi = _year_bounds(target)
    ytext = _without_model_numbers(text, target)
    if preset_year is None and _out_of_scope(ytext, lo, hi):
        return None  # e.g. "1967 Chevelle Malibu": outside the model's covered years
    year_start = year_end = None
    if preset_year is not None:
        year_start = year_end = preset_year
    else:
        m = _RANGE_RE.search(ytext)
        if m:
            a = int(m.group(1))
            b_raw = m.group(2)
            b = int(b_raw) if len(b_raw) == 4 else (a // 100) * 100 + int(b_raw)
            if lo - 1 <= a <= b <= hi + 1 and b - a <= 12:
                year_start, year_end = a, b
        if year_start is None:
            years = sorted({int(y) for y in _YEAR_RE.findall(ytext) if lo - 1 <= int(y) <= hi + 1})
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
                 and len(normalize_name(m)) > 3 and _mentions_model(text, make_names, m)]

    confidence = 0.55 if verdict == "match" else 0.4
    reasons = ["model named" if verdict == "match" else "model only in tags/description"]
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
