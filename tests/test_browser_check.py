"""Browser-checker tests — offline (the percentage parser + registry; no real browser)."""

from __future__ import annotations

import builtins

import pytest

from humanize.browser_check import (
    ZeroGPTChecker,
    available_browser_checkers,
    get_browser_checker,
    parse_ai_percent,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("100%AI GPT*", 1.0),  # confirmed ZeroGPT result string
        ("55% AI Generated", 0.55),
        ("Your text is 0% AI", 0.0),
        ("12.5%", 0.125),
        ("AI: 150%", 1.0),  # clamped
    ],
)
def test_parse_ai_percent(text, expected):
    assert abs(parse_ai_percent(text) - expected) < 1e-6


def test_parse_ai_percent_none_when_no_number():
    assert parse_ai_percent("no percentage here") is None
    assert parse_ai_percent("") is None
    assert parse_ai_percent(None) is None


def test_registry():
    assert "zerogpt" in available_browser_checkers()
    assert isinstance(get_browser_checker("ZeroGPT"), ZeroGPTChecker)  # case-insensitive
    assert get_browser_checker("nonexistent-site") is None


def test_available_false_without_playwright(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "playwright" or name.startswith("playwright."):
            raise ImportError("playwright not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert ZeroGPTChecker().available() is False
