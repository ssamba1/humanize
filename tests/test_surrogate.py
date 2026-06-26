"""Surrogate + pluggable-reward tests (no torch needed — the loaders and reward routing only)."""

from __future__ import annotations

from training import reward as reward_mod
from training.surrogate import _builtin_labeled, load_labeled


def test_builtin_labeled_balanced():
    data = _builtin_labeled(6)
    assert len(data) == 6
    assert any(lbl >= 0.5 for _, lbl in data)  # has AI
    assert any(lbl < 0.5 for _, lbl in data)   # has human
    for text, label in data:
        assert isinstance(text, str) and text
        assert label in (0.0, 1.0)


def test_load_labeled_csv(tmp_path):
    csv = tmp_path / "labels.csv"
    csv.write_text(
        'text,score\n"some ai-sounding text that is plenty long",0.9\n"a person typed this one by hand",0.1\n',
        encoding="utf-8",
    )
    rows = load_labeled(str(csv))
    assert len(rows) == 2
    assert all(0.0 <= s <= 1.0 for _, s in rows)


def test_reward_uses_surrogate_when_env_set(monkeypatch):
    """With UNTELL_SURROGATE_DIR set, the reward target is the surrogate, not the local ensemble —
    this is the whole point: train against a GPTZero-mimicking model, not the non-transferring proxies."""

    class FakeSurrogate:
        def __init__(self, _dir):
            pass

        def score(self, _text):
            return 0.02  # surrogate says "human"

    import training.surrogate as surr

    monkeypatch.setattr(surr, "SurrogateDetector", FakeSurrogate)
    reward_mod._SURROGATE = None
    monkeypatch.setenv("UNTELL_SURROGATE_DIR", "out/whatever")
    try:
        assert reward_mod.target_ai_score("anything") == 0.02
    finally:
        reward_mod._SURROGATE = None


def test_reward_default_is_local_ensemble(monkeypatch):
    monkeypatch.delenv("UNTELL_SURROGATE_DIR", raising=False)
    reward_mod._SURROGATE = None
    s = reward_mod.target_ai_score("Furthermore, the system operates predictably and uniformly.", tier="lite")
    assert 0.0 <= s <= 1.0


def test_humanness_reward_still_works_lite(monkeypatch):
    monkeypatch.delenv("UNTELL_SURROGATE_DIR", raising=False)
    reward_mod._SURROGATE = None
    r = reward_mod.humanness_reward("The cat sat on the mat.", "A cat was sitting on the mat.", tier="lite")
    assert -1.0 <= r <= 1.0
