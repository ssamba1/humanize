"""MCP server — expose the humanizer as tools to Claude Desktop and other MCP clients.

Run: ``untell-mcp`` (after ``pip install -e ".[mcp]"``). Register it in your MCP client's config.
Tools: ``score`` (AI-likelihood + per-detector), ``sentences`` (per-sentence flags), ``untell``
(run the loop), ``verify`` (pass/fail vs commercial checkers), ``scrub`` (strip hidden watermarks).

The ``mcp`` package is imported lazily so this module imports fine without it (the server build needs
it). Keeping logic thin: each tool delegates to the same functions the CLIs use.
"""

from __future__ import annotations


def _server():
    from mcp.server.fastmcp import FastMCP

    from untell.attacks import count_hidden, scrub_hidden
    from untell.scripts.run import untell_text
    from untell.scripts.score import score_text
    from untell.scripts.sentences import score_sentences
    from untell.scripts.verify import verify

    server = FastMCP("untell")

    @server.tool()
    def score(text: str, tier: str = "lite") -> dict:
        """Score text for AI-likelihood (max + ai_percent 0-100 + per-detector)."""
        return score_text(text, tier=tier)

    @server.tool()
    def sentences(text: str, tier: str = "lite") -> dict:
        """Per-sentence AI scores and the list of sentences flagged as AI."""
        return score_sentences(text, tier=tier)

    @server.tool()
    def untell(text: str, tier: str = "lite", style: str | None = None, max_iters: int = 5) -> dict:
        """Run the closed untell loop (needs an LLM rewriter key, or use the /untell skill)."""
        return untell_text(text, tier=tier, style=style, max_iters=max_iters)

    @server.tool()
    def verify_commercial(text: str, threshold: float = 0.30) -> dict:
        """Pass/fail vs every configured commercial checker (needs API keys)."""
        return verify(text, threshold=threshold)

    @server.tool()
    def scrub(text: str) -> dict:
        """Strip hidden watermark / zero-width / homoglyph characters from text."""
        return {"clean": scrub_hidden(text), "hidden_chars_removed": count_hidden(text)}

    return server


def main(argv: list[str] | None = None) -> int:
    _server().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
