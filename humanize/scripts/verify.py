"""Pass/fail verification against the commercial AI checkers.

This is the literal "does it pass all major AI detectors" tool. It scores text with every
*configured* commercial detector (those whose API keys are set) and reports, per checker, the AI
probability and whether it is under the pass threshold — plus an overall ``passes_all`` verdict.

    humanize-verify "text to check" --threshold 0.30
    humanize-verify --file out.txt --json

With no commercial keys set it reports that no checkers are configured (and exits non-zero), because
"passes all major checkers" cannot be asserted without running against them.
"""

from __future__ import annotations

import argparse
import json
import sys

from humanize.detectors.base import clamp01
from humanize.detectors.commercial import commercial_detectors
from humanize.scripts.score import DEFAULT_THRESHOLD


def verify(text: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Score ``text`` against every configured commercial checker; return a verdict dict."""
    configured = [d for d in commercial_detectors() if d.available()]
    results: dict[str, dict] = {}
    for d in configured:
        try:
            ai = clamp01(float(d.score(text)))
            results[d.name] = {"ai": round(ai, 4), "passes": ai < threshold}
        except Exception as exc:  # surface per-checker failure rather than crashing the verdict
            results[d.name] = {"ai": None, "passes": False, "error": str(exc)[:160]}

    passing = [n for n, r in results.items() if r.get("passes")]
    return {
        "configured": [d.name for d in configured],
        "threshold": threshold,
        "results": results,
        "passes_all": bool(configured) and all(r.get("passes") for r in results.values()),
        "n_configured": len(configured),
        "n_passing": len(passing),
    }


def _render(v: dict) -> str:
    if not v["configured"]:
        return (
            "No commercial checkers configured. Set API keys (ORIGINALITY_API_KEY, GPTZERO_API_KEY, "
            "WINSTON_API_KEY, SAPLING_API_KEY, ZEROGPT_API_KEY, COPYLEAKS_EMAIL+COPYLEAKS_API_KEY) "
            "and install .[commercial]. Cannot verify 'passes all checkers' without them."
        )
    lines = [f"AI-checker verification (threshold {v['threshold']}: AI prob must be below it)", ""]
    for name, r in v["results"].items():
        if r.get("error"):
            lines.append(f"  {name:12} ERROR: {r['error']}")
        else:
            mark = "PASS" if r["passes"] else "FAIL"
            lines.append(f"  {name:12} AI={r['ai']:.3f}  [{mark}]")
    lines.append("")
    lines.append(
        f"PASSES ALL {v['n_configured']} CHECKERS"
        if v["passes_all"]
        else f"FAILS — {v['n_passing']}/{v['n_configured']} checkers passed"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(prog="humanize-verify", description="Verify text against commercial AI checkers.")
    parser.add_argument("text", nargs="?", help="text to verify (or --file / stdin)")
    parser.add_argument("--file", "-f", help="read text from this file")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--json", action="store_true")
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

    v = verify(text, threshold=args.threshold)
    print(json.dumps(v, ensure_ascii=True, indent=2) if args.json else _render(v))
    # exit 0 only when there is at least one checker AND all pass
    return 0 if v["passes_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
