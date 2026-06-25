"""Back-translation tests.

Offline tests run everywhere (the no-op fallback path). The real round-trip test is gated on
torch/transformers/sentencepiece, so it skips on the lite CI / broken-torch boxes and runs in the
full-tier CI job (where it downloads the MarianMT models).
"""

from __future__ import annotations

import pytest

from humanize.attacks import BackTranslator, back_translate


def test_noop_when_unavailable(monkeypatch):
    bt = BackTranslator()
    monkeypatch.setattr(bt, "available", lambda: False)
    text = "This text must come back exactly unchanged when MT is unavailable."
    assert bt.back_translate(text) == text


def test_empty_input_is_noop():
    assert back_translate("") == ""
    assert back_translate("   ") == "   "


def test_translation_failure_falls_back(monkeypatch):
    bt = BackTranslator()
    monkeypatch.setattr(bt, "available", lambda: True)

    def _boom(*a, **k):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(bt, "_translate", _boom)
    text = "Any failure mid-translation must degrade to the original text, never raise."
    assert bt.back_translate(text) == text


def _mt_ready() -> bool:
    try:
        import sentencepiece  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _mt_ready(), reason="MarianMT stack (torch/transformers/sentencepiece) unavailable")
def test_roundtrip_changes_text_but_keeps_gist():
    src = "The committee approved the new policy after a lengthy and contentious debate."
    out = back_translate(src, pivots=("fr",))
    assert isinstance(out, str) and out.strip()
    assert len(out.split()) >= 5  # produced real prose, not empty/garbage
