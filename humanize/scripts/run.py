"""Headless humanize loop — run the full lock -> score -> rewrite -> restore loop as a CLI.

Inside Claude Code the SKILL.md procedure drives the loop with Claude as the rewriter. This module
is the *standalone* path: a `humanize-loop` console command (and `humanize_text` API) that runs the
same loop programmatically using a hosted-LLM rewriter (``humanize.rewriter``). It reuses the exact
same scripts the skill calls — preserve-lock, the detector ensemble, and the quality gate — so the
two paths stay behaviourally identical.

A rewriter must be configured (``pip install -e ".[api]"`` + ``ANTHROPIC_API_KEY``/``OPENAI_API_KEY``);
without one this returns a clear error rather than silently no-op'ing (use the Claude skill instead).
"""

from __future__ import annotations

import argparse
import json
import sys

from humanize.rewriter import get_rewriter
from humanize.scripts.preserve import lock, restore
from humanize.scripts.quality import method, recommended_bar, similarity
from humanize.scripts.score import DEFAULT_THRESHOLD, score_text


def _browser_scorer(site: str, mapping: dict, threshold: float):
    """Return a scorer(masked_text)->score-dict that drives a free web detector (real, no key).

    Scores the *restored* text (what a real detector actually sees), so the loop optimizes directly
    against the live checker. Returns None if the browser checker isn't available.
    """
    from humanize.browser_check import get_browser_checker
    from humanize.scripts.preserve import restore

    chk = get_browser_checker(site)
    if chk is None or not chk.available():
        return None

    def _score(masked_text: str) -> dict:
        ai = float(chk.check(restore(masked_text, mapping)))
        return {
            "tier": f"browser:{site}",
            "detectors": {site: round(ai, 4)},
            "max": round(ai, 4),
            "mean": round(ai, 4),
            "threshold": threshold,
            "flagged": ai >= threshold,
        }

    return _score


def humanize_text(
    text: str,
    tier: str = "full",
    threshold: float = DEFAULT_THRESHOLD,
    max_iters: int = 5,
    sim_bar: float | None = None,
    rewriter=None,
    browser: str | None = None,
) -> dict:
    """Run the closed loop on ``text``; return a structured result dict.

    Keys: ``final`` (humanized text, spans restored), ``iterations``, ``pre``/``post`` score dicts,
    ``similarity``, ``tier``, ``sim_bar``, ``flagged`` (final), and ``stopped`` (why it stopped).
    If no rewriter is available, returns ``{"error": ...}`` without modifying the text.

    ``browser`` (e.g. ``"zerogpt"``) scores each iteration against a free web detector instead of the
    local proxies — the loop then optimizes against a *real* checker, no API key (but slow: ~10s/iter).
    """
    if sim_bar is None:
        sim_bar = recommended_bar()
    rw = rewriter if rewriter is not None else get_rewriter()
    if rw is None:
        return {
            "error": "no rewriter configured — install .[api] and set ANTHROPIC_API_KEY or "
            "OPENAI_API_KEY, or use the /humanize Claude skill (Claude is the rewriter).",
            "final": text,
        }

    masked, mapping = lock(text)

    browser_score = _browser_scorer(browser, mapping, threshold) if browser else None
    if browser and browser_score is None:
        return {
            "error": f"browser checker '{browser}' unavailable — pip install .[browser] && playwright install chromium",
            "final": text,
        }

    def score(masked_text: str) -> dict:
        if browser_score is not None:
            return browser_score(masked_text)
        return score_text(masked_text, tier=tier, threshold=threshold)

    pre = score(masked)
    best_masked, best_score = masked, pre
    iters = 0
    stopped = "max_iters"
    for i in range(1, max_iters + 1):
        iters = i
        if not best_score["flagged"] and similarity(masked, best_masked) >= sim_bar:
            stopped = "passed"
            break
        try:
            candidate = rw.rewrite(best_masked, best_score, threshold)
        except Exception as exc:  # surface the failure rather than silently looping
            return {"error": f"rewriter failed: {type(exc).__name__}: {str(exc)[:160]}", "final": restore(best_masked, mapping)}
        cand_score = score(candidate)
        if similarity(masked, candidate) >= sim_bar and cand_score["max"] <= best_score["max"]:
            best_masked, best_score = candidate, cand_score
        if not best_score["flagged"]:
            stopped = "passed"
            break

    final = restore(best_masked, mapping)
    return {
        "final": final,
        "iterations": iters,
        "pre": pre,
        "post": best_score,
        "similarity": similarity(masked, best_masked),
        "tier": best_score.get("tier", tier),
        "sim_bar": sim_bar,
        "quality_metric": method(),
        "flagged": best_score["flagged"],
        "stopped": stopped,
    }


def _render(result: dict) -> str:
    if "error" in result:
        return f"ERROR: {result['error']}"
    pre, post = result["pre"], result["post"]
    lines = ["# humanize result", ""]
    lines.append(f"tier={result['tier']}  iterations={result['iterations']}  stopped={result['stopped']}")
    lines.append(f"max P(AI): {pre['max']:.3f} -> {post['max']:.3f}  (threshold {post['threshold']})")
    lines.append(f"similarity: {result['similarity']:.3f} (bar {result['sim_bar']}, {result['quality_metric']})")
    lines.append("\nper-detector (pre -> post):")
    for name in pre.get("detectors", {}):
        if "__error" in name:
            continue
        p = pre["detectors"].get(name)
        q = post["detectors"].get(name)
        if isinstance(p, (int, float)) and isinstance(q, (int, float)):
            lines.append(f"  {name}: {p:.3f} -> {q:.3f}")
    lines.append("\n--- humanized text ---\n" + result["final"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from humanize._env import load_env

    load_env()  # pick up ANTHROPIC_API_KEY / commercial keys from a .env file if present
    parser = argparse.ArgumentParser(prog="humanize-loop", description="Run the headless humanize loop.")
    parser.add_argument("text", nargs="?", help="text to humanize (or --file / stdin)")
    parser.add_argument("--file", "-f", help="read text from this file")
    parser.add_argument("--tier", default="full", choices=["lite", "full", "heavy", "commercial"])
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument(
        "--browser",
        help="score each iteration against a free web detector (e.g. 'zerogpt') instead of local "
        "proxies — real checker, no key, but slow (~10s/iter). Needs .[browser] + playwright.",
    )
    parser.add_argument("--json", action="store_true", help="emit the full result as JSON")
    args = parser.parse_args(argv)

    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()
    if not text.strip():
        print(json.dumps({"error": "empty input"}))
        return 2

    result = humanize_text(
        text, tier=args.tier, threshold=args.threshold, max_iters=args.max_iters, browser=args.browser
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=True, indent=2))
    else:
        print(_render(result))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
