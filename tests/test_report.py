"""Benchmark report tests (lite, builtin)."""

from __future__ import annotations

from eval.benchmark import run
from eval.report import _bypass_rate, render, summarize


def _by():
    return run("builtin", 4, "lite", 0.30, ["noop", "single_pass", "full_loop"])


def test_summarize_shape():
    s = summarize(_by(), 0.30)
    assert set(s["strategies"]) == {"noop", "single_pass", "full_loop"}
    for st in s["strategies"].values():
        assert st["n"] == 4
        assert 0.0 <= st["bypass_rate"] <= 1.0
        assert 0.0 <= st["mean_similarity"] <= 1.0
        assert "perplexity_burstiness" in st["per_detector"]
        pd = st["per_detector"]["perplexity_burstiness"]
        assert 0.0 <= pd["pre"] <= 1.0 and 0.0 <= pd["post"] <= 1.0
    assert "thesis_pass" in s and isinstance(s["thesis_pass"], bool)


def test_render_is_ascii_safe_and_complete():
    md = render(_by(), 0.30)
    md.encode("ascii")  # no emoji -> never crashes a Windows cp1252 console
    assert "# humanize benchmark" in md
    assert "Per-detector" in md
    assert "Thesis" in md


def test_bypass_rate_empty_is_zero():
    assert _bypass_rate([], 0.30) == 0.0


def test_noop_never_bypasses_builtin():
    # builtin samples are deliberately AI-flagged, so the identity strategy bypasses 0%.
    s = summarize(run("builtin", 3, "lite", 0.30, ["noop"]), 0.30)
    assert s["strategies"]["noop"]["bypass_rate"] == 0.0
