"""Infer a reference image's make/model label from its text metadata (years are not used)."""

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

@dataclass
class LabelGuess:
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


def infer_label(text: str, target: Target, title: str | None = None) -> LabelGuess | None:
    """Return a label guess, or None if the text does not clearly refer to the target model.

    ``title`` (when given, also part of ``text``) outranks tags and descriptions: a title
    that names the make but another model rejects the image, and a model found only in
    tags/description lowers the confidence. Model years are deliberately ignored: the
    system identifies make and model only, across all years.
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

    conflicts = [m for m in target.sibling_models
                 if normalize_name(m) not in {normalize_name(n) for n in model_names}
                 and len(normalize_name(m)) > 3 and _mentions_model(text, make_names, m)]

    confidence = 0.65 if verdict == "match" else 0.45
    reasons = ["model named" if verdict == "match" else "model only in tags/description"]
    if make_named:
        confidence += 0.15
        reasons.append("make named")
    if conflicts:
        confidence -= 0.35
        reasons.append("also mentions " + ", ".join(conflicts[:3]))
    return LabelGuess(confidence=round(max(0.05, min(confidence, 0.95)), 2),
                      interior_hint=interior_hint, reason="; ".join(reasons))
