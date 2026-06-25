"""Browser-driven AI-detection via free web UIs (no API key).

Some detectors have no affordable API but a free web checker. This drives a real browser
(Playwright) to paste text into one and read the score — a $0 way to get a real-checker signal.

Selectors for ZeroGPT were confirmed live (2026-06): input ``#textArea``, button
``button.scoreButton`` ("Detect Text"), result ``.percentage-div`` (e.g. "100%AI GPT*"). A JS
``.click()`` is used because ad overlays intercept normal pointer clicks.

CAVEATS — read these:
  * **Slow + fragile.** Page layouts/selectors change; ads/Cloudflare/captchas can block automation.
    This is for occasional *verification*, NOT a step inside the rewrite loop.
  * **Respect the site's terms.** Automating a free web UI may violate ToS. Use low volume on your
    own content; do not hammer the service. You are responsible for how you use it.
  * Optional: needs ``pip install -e ".[browser]"`` then ``playwright install chromium``.
"""

from __future__ import annotations

import re

from humanize.detectors.base import clamp01

_PCT = re.compile(r"([\d.]+)\s*%")


def parse_ai_percent(text: str) -> float | None:
    """Pull the first percentage out of a result string and return it as P(AI) in [0, 1].

    e.g. "100%AI GPT*" -> 1.0, "55% AI Generated" -> 0.55. Returns None if no percentage found.
    """
    m = _PCT.search(text or "")
    if not m:
        return None
    try:
        return clamp01(float(m.group(1)) / 100.0)
    except ValueError:
        return None


class ZeroGPTChecker:
    name = "zerogpt"
    url = "https://www.zerogpt.com/"

    def available(self) -> bool:
        try:
            import playwright  # noqa: F401
        except Exception:
            return False
        return True

    def check(self, text: str, headless: bool = True, timeout_s: float = 45.0) -> float:
        """Drive a browser through the ZeroGPT web UI and return P(AI) in [0, 1]."""
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page()
            try:
                page.goto(self.url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
                page.fill("#textArea", text, timeout=timeout_s * 1000)
                # JS click — ad overlays steal normal pointer events on this site.
                page.evaluate(
                    "() => { const b=[...document.querySelectorAll('button.scoreButton')]"
                    ".find(x=>/detect/i.test(x.textContent||'')); if (b) b.click(); }"
                )
                page.wait_for_selector(".percentage-div", timeout=timeout_s * 1000)
                pct = parse_ai_percent(page.inner_text(".percentage-div"))
                return pct if pct is not None else 0.5
            finally:
                browser.close()


_CHECKERS = {"zerogpt": ZeroGPTChecker}


def get_browser_checker(name: str):
    """Return a browser checker instance for ``name``, or None if unknown."""
    cls = _CHECKERS.get(name.lower())
    return cls() if cls else None


def available_browser_checkers() -> list[str]:
    return sorted(_CHECKERS)
