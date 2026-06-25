"""Text-perturbation attacks/augmentations from the detection-evasion literature.

These are *mechanical* transforms (no LLM) that can pre- or post-process text in the loop or stand
alone for research comparisons. Currently: back-translation (round-trip MT). Each degrades to a
safe no-op when its optional models/deps are unavailable.
"""

from __future__ import annotations

from .back_translation import BackTranslator, back_translate

__all__ = ["BackTranslator", "back_translate"]
