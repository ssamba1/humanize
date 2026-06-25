"""Training-stack tests: multi-objective reward + loop distillation (offline, lite tier)."""

from __future__ import annotations

from training.distill import distill
from training.reward import fluency, humanness_reward


def test_fluency_penalizes_repetition():
    assert fluency("the quick brown fox jumps over the lazy dog") > fluency("spam spam spam spam spam spam")
    assert fluency("hi") == 1.0  # too short -> neutral


def test_reward_penalizes_degenerate_and_meaning_drift():
    src = "The committee approved the budget after a brief discussion on Tuesday afternoon."
    good = src  # sim 1.0, fluent -> reward >= 0
    degenerate = "budget budget budget budget budget budget budget budget budget budget budget"
    assert humanness_reward(src, good, tier="lite") > humanness_reward(src, degenerate, tier="lite")
    assert humanness_reward(src, "", tier="lite") == -1.0


def test_distill_keeps_passing_samples(monkeypatch):
    import humanize.scripts.run as run_mod

    monkeypatch.setattr(
        run_mod, "humanize_text", lambda text, **k: {"final": "a human rewrite", "flagged": False, "similarity": 0.9}
    )
    out = distill("builtin", n=3, tier="lite")
    assert out["kept"] == 3
    assert len(out["rows"]) == 3
    assert all("source" in r and "humanized" in r and "prompt" in r for r in out["rows"])


def test_distill_drops_flagged_or_low_similarity(monkeypatch):
    import humanize.scripts.run as run_mod

    monkeypatch.setattr(run_mod, "humanize_text", lambda text, **k: {"final": "x", "flagged": True, "similarity": 0.9})
    assert distill("builtin", n=3, tier="lite")["kept"] == 0

    monkeypatch.setattr(run_mod, "humanize_text", lambda text, **k: {"final": "x", "flagged": False, "similarity": 0.2})
    assert distill("builtin", n=3, tier="lite")["kept"] == 0
