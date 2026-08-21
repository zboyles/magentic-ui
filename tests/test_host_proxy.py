"""Tests for the Quicksand host-app origin-rewrite proxy."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest
from typer.testing import CliRunner

from magentic_ui.backend.cli import app
from magentic_ui.backend.host_proxy import (
    HostProxyServer,
    guest_browser_url,
    incoming_origin,
    localhost_origin,
    origin_from_url,
    rewrite_referer,
    rewrite_url_origin,
)


class TestOriginHelpers:
    def test_localhost_origin_maps_loopback(self) -> None:
        assert localhost_origin("http", "127.0.0.1", 3000) == "http://localhost:3000"
        assert localhost_origin("http", "localhost", 3000) == "http://localhost:3000"

    def test_origin_from_url(self) -> None:
        assert origin_from_url("http://127.0.0.1:3000/") == "http://localhost:3000"

    def test_incoming_origin_prefers_origin_header(self) -> None:
        assert (
            incoming_origin("http://10.0.2.2:3100", "127.0.0.1:3100")
            == "http://10.0.2.2:3100"
        )

    def test_incoming_origin_falls_back_to_host(self) -> None:
        assert incoming_origin(None, "10.0.2.2:3100") == "http://10.0.2.2:3100"

    def test_rewrite_referer_from_gateway(self) -> None:
        assert (
            rewrite_referer(
                "http://10.0.2.2:3100/wallet",
                "http://10.0.2.2:3100",
                "http://localhost:3000",
            )
            == "http://localhost:3000/wallet"
        )

    def test_rewrite_referer_from_unknown_host(self) -> None:
        # Dynamic: any guest-facing host is rewritten, not just 10.0.2.2.
        assert (
            rewrite_referer(
                "http://onyx.local:3100/wallet",
                "http://onyx.local:3100",
                "http://localhost:3000",
            )
            == "http://localhost:3000/wallet"
        )

    def test_rewrite_location_back_to_guest(self) -> None:
        assert (
            rewrite_url_origin(
                "http://localhost:3000/login",
                "http://localhost:3000",
                "http://10.0.2.2:3100",
            )
            == "http://10.0.2.2:3100/login"
        )

    def test_guest_browser_url(self) -> None:
        assert guest_browser_url(3100) == "http://10.0.2.2:3100"


class _CaptureHandler(BaseHTTPRequestHandler):
    last_origin = ""
    last_referer = ""
    last_host = ""

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: ARG002
        return

    def do_GET(self) -> None:  # noqa: N802
        type(self).last_origin = self.headers.get("Origin") or ""
        type(self).last_referer = self.headers.get("Referer") or ""
        type(self).last_host = self.headers.get("Host") or ""
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def upstream_server() -> tuple[str, HTTPServer]:
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}"
    try:
        yield url, server
    finally:
        server.shutdown()
        server.server_close()


def test_proxy_rewrites_gateway_origin(
    upstream_server: tuple[str, HTTPServer],
) -> None:
    _CaptureHandler.last_origin = ""
    upstream_url, _upstream = upstream_server
    proxy = HostProxyServer(("127.0.0.1", 0), upstream_url)
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = proxy.server_address[:2]
        req = Request(
            f"http://{host}:{port}/api/proxy/example",
            headers={
                "Origin": "http://10.0.2.2:3100",
                "Referer": "http://10.0.2.2:3100/wallet",
            },
        )
        with urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            assert json.loads(resp.read().decode())["ok"] is True
        expected = origin_from_url(upstream_url)
        assert _CaptureHandler.last_origin == expected
        assert _CaptureHandler.last_referer == expected + "/wallet"
        assert _CaptureHandler.last_host == urlsplit(expected).netloc
    finally:
        proxy.shutdown()
        proxy.server_close()


class TestCli:
    def test_host_proxy_help(self) -> None:
        result = CliRunner().invoke(app, ["host-proxy", "--help"])
        assert result.exit_code == 0
        assert "--upstream-port" in result.output
        assert "--listen-port" in result.output

    def test_main_help_lists_host_proxy_flag(self) -> None:
        result = CliRunner().invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "--host-proxy" in result.output
