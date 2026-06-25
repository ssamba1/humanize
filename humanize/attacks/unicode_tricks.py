"""Unicode-level tricks: homoglyph substitution (attack) + hidden-watermark scrubbing (defense).

Two sub-semantic operations several competitor repos have and we didn't:

- ``scrub_hidden`` (defense, recommended): strip invisible LLM watermarks / steganography — zero-width
  chars, variation selectors, Unicode tag chars, control chars — and NFKC-normalize confusable
  homoglyphs back to ASCII. Genuinely useful: cleans text that had a hidden watermark embedded.
- ``homoglyph_substitute`` (attack, OPT-IN, caveated): replace a fraction of ASCII letters with
  visually identical Cyrillic/Greek homoglyphs to disrupt detector tokenization (silverspeak,
  arXiv 2406.11239). **Caveats:** invisible to humans but breaks copy-paste/search, is removed by any
  detector that NFKC-normalizes first, and detectors like Winston flag unusual Unicode as an attack.
  Last resort only — ``scrub_hidden`` is the opposite of robust evasion, use deliberately.
"""

from __future__ import annotations

import re
import unicodedata

# ASCII -> visually-identical homoglyph (Cyrillic/Greek). Conservative set that renders identically.
_HOMOGLYPH = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х",
    "y": "у", "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х",
}
# Reverse map (+ a few extra confusables) for scrubbing back to ASCII.
_UNHOMOGLYPH = {v: k for k, v in _HOMOGLYPH.items()}

_INVISIBLE = re.compile(
    "[​‌‍⁠﻿]"  # zero-width space/non-joiner/joiner/word-joiner/BOM
    "|[︀-️]"  # variation selectors
    "|[\U000e0000-\U000e007f]"  # Unicode tag chars (used for invisible-tag watermarks)
)


def scrub_hidden(text: str) -> str:
    """Remove invisible watermark/steganography characters and normalize confusables to ASCII."""
    text = _INVISIBLE.sub("", text)
    # strip other control chars except tab/newline
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    text = "".join(_UNHOMOGLYPH.get(ch, ch) for ch in text)
    return unicodedata.normalize("NFKC", text)


def homoglyph_substitute(text: str, rate: float = 0.15) -> str:
    """Replace a fraction (``rate``) of eligible ASCII letters with homoglyphs. OPT-IN attack.

    Deterministic (every Nth eligible letter) so it is reproducible and testable. See module caveats.
    """
    if rate <= 0:
        return text
    period = max(1, round(1 / rate))
    out = []
    n = 0
    for ch in text:
        if ch in _HOMOGLYPH:
            n += 1
            out.append(_HOMOGLYPH[ch] if n % period == 0 else ch)
        else:
            out.append(ch)
    return "".join(out)


def count_hidden(text: str) -> int:
    """How many invisible/homoglyph chars are present — a quick 'is this watermarked?' check."""
    invisible = len(_INVISIBLE.findall(text))
    homoglyphs = sum(1 for ch in text if ch in _UNHOMOGLYPH)
    return invisible + homoglyphs
