"""The MCP server builds and registers all its tools.

Skips when the optional ``[mcp]`` extra is absent; CI installs it on the lite job so this actually
runs. Building the server exercises every ``@server.tool()`` registration and the CLI-delegate
imports it depends on — catching a renamed delegate or a FastMCP API change before a user's MCP
client fails to start the server. The plain ``callable(main)`` smoke test cannot catch that.
"""

import pytest

pytest.importorskip("mcp")

EXPECTED_TOOLS = {"score", "sentences", "untell", "verify_commercial", "scrub"}


def test_server_builds():
    from untell.mcp_server import _server

    server = _server()  # raises if any tool delegate import or registration is broken
    assert server is not None


def test_all_tools_registered():
    from untell.mcp_server import _server

    server = _server()
    mgr = getattr(server, "_tool_manager", None)
    tools = getattr(mgr, "_tools", None) if mgr is not None else None
    if not isinstance(tools, dict):
        pytest.skip("FastMCP tool registry layout not introspectable in this version")
    assert EXPECTED_TOOLS <= set(tools)
