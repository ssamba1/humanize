"""Headless humanize-loop tests — offline (rewriter mocked; no network, no keys)."""

from __future__ import annotations

import json

from humanize.scripts.run import humanize_text, main

AI = (
    "Furthermore, artificial intelligence has fundamentally transformed numerous industries. "
    "Moreover, organizations utilize it to significantly improve operational efficiency. Overall, "
    "the impact continues to grow across various sectors according to Smith (2020), rising 47%."
)


class _GoodRW:
    """A rewriter that returns bursty, human-ish text while preserving the sentinels it is given."""

    name = "fake"

    def available(self):
        return True

    def rewrite(self, text, score_result, threshold=0.30):
        # Keep any sentinels present in the input so restore still works.
        import re

        sentinels = re.findall(r"⟦HZ\d{4}⟧", text)
        tail = (" " + " ".join(sentinels)) if sentinels else ""
        return "It shifted. Fast. Nobody saw it coming, and then everything was different." + tail


def test_humanize_text_runs_loop_and_restores(monkeypatch):
    import humanize.scripts.run as run_mod

    monkeypatch.setattr(run_mod, "get_rewriter", lambda prefer=None: _GoodRW())
    res = humanize_text(AI, tier="lite", max_iters=3)
    assert "error" not in res
    assert res["iterations"] >= 1
    assert res["post"]["max"] <= res["pre"]["max"] + 1e-9
    # Locked facts must survive into the final output.
    assert "Smith (2020)" in res["final"]
    assert "47%" in res["final"]


def test_humanize_text_no_rewriter_returns_error(monkeypatch):
    import humanize.scripts.run as run_mod

    monkeypatch.setattr(run_mod, "get_rewriter", lambda prefer=None: None)
    res = humanize_text(AI, tier="lite")
    assert "error" in res
    assert res["final"] == AI  # unchanged


def test_humanize_text_survives_rewriter_exception(monkeypatch):
    import humanize.scripts.run as run_mod

    class _Boom:
        name = "boom"

        def available(self):
            return True

        def rewrite(self, text, score_result, threshold=0.30):
            raise RuntimeError("api down")

    monkeypatch.setattr(run_mod, "get_rewriter", lambda prefer=None: _Boom())
    res = humanize_text(AI, tier="lite")
    assert "error" in res and "rewriter failed" in res["error"]


def test_cli_json_output(monkeypatch, capsys):
    import humanize.scripts.run as run_mod

    monkeypatch.setattr(run_mod, "get_rewriter", lambda prefer=None: _GoodRW())
    rc = main(["--tier", "lite", "--json", AI])
    assert rc == 0
    out = capsys.readouterr().out
    out.encode("ascii")  # ensure_ascii -> portable
    parsed = json.loads(out)
    assert "final" in parsed and parsed["iterations"] >= 1


def test_cli_no_rewriter_exits_nonzero(monkeypatch, capsys):
    import humanize.scripts.run as run_mod

    monkeypatch.setattr(run_mod, "get_rewriter", lambda prefer=None: None)
    rc = main(["--tier", "lite", "some text to humanize here please"])
    assert rc == 1
    assert "ERROR" in capsys.readouterr().out


def test_cli_empty_input_returns_2(capsys):
    rc = main(["--tier", "lite", "   "])
    assert rc == 2
