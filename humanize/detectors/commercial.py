"""Commercial AI-detector adapters (the *real* checkers the product must beat).

Each wraps a paid HTTP API and is **key-gated**: ``available()`` is true only when ``requests`` is
installed (the ``[commercial]`` extra) and the service's API key env var(s) are set. With no keys
the detectors are simply absent from the ensemble, so nothing here runs — or costs money — unless
you configure it. Endpoints/field paths below are from each provider's current public docs.

Tier ``commercial`` sits above ``heavy``: ``load_detectors("commercial")`` returns every available
lite/full/heavy detector *plus* every configured commercial one, and the loop must drive the
``max`` across all of them under threshold — i.e. pass **every** checker you've wired up.

Env vars:
  ORIGINALITY_API_KEY · WINSTON_API_KEY · GPTZERO_API_KEY · SAPLING_API_KEY · ZEROGPT_API_KEY
  COPYLEAKS_EMAIL + COPYLEAKS_API_KEY
"""

from __future__ import annotations

import hashlib
import os
import time

from .base import clamp01


def _post_json(url: str, headers: dict, body: dict, timeout: float = 45.0) -> dict:
    """POST JSON and return parsed JSON. Isolated so tests can monkeypatch it (no network)."""
    import requests

    resp = requests.post(url, headers={"Content-Type": "application/json", **headers}, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _has(*env_vars: str) -> bool:
    try:
        import requests  # noqa: F401
    except Exception:
        return False
    return all(os.environ.get(v) for v in env_vars)


class OriginalityDetector:
    name = "originality"
    tier = "commercial"

    def available(self) -> bool:
        return _has("ORIGINALITY_API_KEY")

    def score(self, text: str) -> float:
        if not self.available() or not text.strip():
            return 0.5
        data = _post_json(
            "https://api.originality.ai/api/v1/scan/ai",
            {"X-OAI-API-KEY": os.environ["ORIGINALITY_API_KEY"], "Accept": "application/json"},
            {"content": text, "aiModelVersion": "1"},
        )
        return clamp01(float(data["score"]["ai"]))  # 0-1, higher = more AI


class WinstonDetector:
    name = "winston"
    tier = "commercial"

    def available(self) -> bool:
        return _has("WINSTON_API_KEY")

    def score(self, text: str) -> float:
        if not self.available() or not text.strip():
            return 0.5
        data = _post_json(
            "https://api.gowinston.ai/v2/ai-content-detection",
            {"Authorization": f"Bearer {os.environ['WINSTON_API_KEY']}"},
            {"text": text, "sentences": False, "language": "auto"},
        )
        # `score` is a 0-100 *human* likelihood; AI probability is the complement.
        return clamp01((100.0 - float(data["score"])) / 100.0)


class GPTZeroDetector:
    name = "gptzero"
    tier = "commercial"

    def available(self) -> bool:
        return _has("GPTZERO_API_KEY")

    def score(self, text: str) -> float:
        if not self.available() or not text.strip():
            return 0.5
        data = _post_json(
            "https://api.gptzero.me/v2/predict/text",
            {"x-api-key": os.environ["GPTZERO_API_KEY"], "Accept": "application/json"},
            {"document": text},
        )
        doc = data["documents"][0]
        ai = doc.get("class_probabilities", {}).get("ai")
        if ai is None:
            ai = doc.get("completely_generated_prob", 0.5)
        return clamp01(float(ai))


class SaplingDetector:
    name = "sapling"
    tier = "commercial"

    def available(self) -> bool:
        return _has("SAPLING_API_KEY")

    def score(self, text: str) -> float:
        if not self.available() or not text.strip():
            return 0.5
        data = _post_json(
            "https://api.sapling.ai/api/v1/aidetect",
            {},
            {"key": os.environ["SAPLING_API_KEY"], "text": text, "sent_scores": False},
        )
        return clamp01(float(data["score"]))  # 0-1, overall AI probability


class ZeroGPTDetector:
    name = "zerogpt"
    tier = "commercial"

    def available(self) -> bool:
        return _has("ZEROGPT_API_KEY")

    def score(self, text: str) -> float:
        if not self.available() or not text.strip():
            return 0.5
        data = _post_json(
            "https://api.zerogpt.com/api/v1/detectText",
            {"Authorization": f"Bearer {os.environ['ZEROGPT_API_KEY']}"},
            {"input_text": text},
        )
        return clamp01(float(data["data"]["is_gpt_generated"]) / 100.0)  # 0-100 -> 0-1


# Copyleaks needs a 2-step auth: login (email+key) -> 48h Bearer token -> detect.
_CL_TOKEN: dict = {"token": None, "exp": 0.0}


def _copyleaks_token() -> str:
    if _CL_TOKEN["token"] and time.time() < _CL_TOKEN["exp"]:
        return _CL_TOKEN["token"]
    data = _post_json(
        "https://id.copyleaks.com/v3/account/login/api",
        {},
        {"email": os.environ["COPYLEAKS_EMAIL"], "key": os.environ["COPYLEAKS_API_KEY"]},
    )
    _CL_TOKEN["token"] = data["access_token"]
    _CL_TOKEN["exp"] = time.time() + 40 * 3600  # token lives 48h; refresh well inside that
    return _CL_TOKEN["token"]


class CopyleaksDetector:
    name = "copyleaks"
    tier = "commercial"

    def available(self) -> bool:
        return _has("COPYLEAKS_EMAIL", "COPYLEAKS_API_KEY")

    def score(self, text: str) -> float:
        if not self.available() or not text.strip():
            return 0.5
        token = _copyleaks_token()
        scan_id = "hz" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]
        data = _post_json(
            f"https://api.copyleaks.com/v2/writer-detector/{scan_id}/check",
            {"Authorization": f"Bearer {token}"},
            {"text": text, "sandbox": False},
        )
        return clamp01(float(data["summary"]["ai"]))  # 0-1


def commercial_detectors() -> list:
    """Every commercial adapter (cheap to instantiate; no network until score())."""
    return [
        OriginalityDetector(),
        GPTZeroDetector(),
        WinstonDetector(),
        SaplingDetector(),
        ZeroGPTDetector(),
        CopyleaksDetector(),
    ]
