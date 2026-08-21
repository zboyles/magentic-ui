"""Origin-rewrite reverse proxy so Quicksand can reach host web apps.

The agent browser runs in a QEMU VM. ``localhost`` inside the VM is the
guest, not the Mac. The host is reachable as ``http://10.0.2.2:<port>``,
but many local apps allowlist ``Origin: http://localhost:<port>`` and
reject the gateway origin with 403.

This proxy binds on all interfaces (so the guest can hit 10.0.2.2) and
forwards to a localhost upstream while rewriting ``Origin`` / ``Referer``
/ ``Host`` to the upstream's localhost origin. Incoming origin is taken
from the request (dynamic) — not a hardcoded allowlist of IPs.

Start with the UI:

    magentic-ui --port 8081 --host-proxy 3000

Or standalone:

    magentic-ui host-proxy --upstream-port 3000

Then open ``http://10.0.2.2:3100`` in the agent browser (not ``:3000``).
"""

from __future__ import annotations

import http.client
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import typer

logger = logging.getLogger(__name__)

# Prevent daemon-thread servers from being garbage-collected.
_active_servers: list[HostProxyServer] = []

QEMU_SLIRP_GATEWAY = "10.0.2.2"
DEFAULT_LISTEN_HOST = "0.0.0.0"
DEFAULT_LISTEN_PORT = 3100

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "proxy-connection",
}


def localhost_origin(scheme: str, hostname: str, port: int) -> str:
    """Origin string apps typically allowlist (localhost, not 127.0.0.1)."""
    host = hostname
    if host in {"127.0.0.1", "::1"}:
        host = "localhost"
    default_port = 443 if scheme == "https" else 80
    netloc = host if port == default_port else f"{host}:{port}"
    return f"{scheme}://{netloc}"


def origin_from_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"Invalid upstream URL: {url}")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return localhost_origin(parts.scheme, parts.hostname, port)


def incoming_origin(origin_header: str | None, host_header: str | None) -> str | None:
    """Best-effort origin of the browser request (dynamic, not allowlisted)."""
    if origin_header:
        return origin_header.rstrip("/")
    if host_header:
        return f"http://{host_header}"
    return None


def rewrite_url_origin(url: str, from_origin: str, to_origin: str) -> str:
    if not url or not from_origin:
        return url
    if url.startswith(from_origin):
        return to_origin + url[len(from_origin) :]
    return url


def rewrite_referer(referer: str, incoming: str | None, upstream_origin: str) -> str:
    """Map a guest-facing referer onto the localhost upstream origin."""
    if incoming:
        rewritten = rewrite_url_origin(referer, incoming, upstream_origin)
        if rewritten != referer:
            return rewritten
    parts = urlsplit(referer)
    up = urlsplit(upstream_origin)
    if not parts.scheme or not parts.hostname:
        return referer
    if parts.hostname in {up.hostname, "localhost", "127.0.0.1"}:
        return referer
    return urlunsplit((up.scheme, up.netloc, parts.path, parts.query, parts.fragment))


class HostProxyServer(ThreadingHTTPServer):
    """Threading HTTP server that carries upstream rewrite config."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(
        self,
        server_address: tuple[str, int],
        upstream_url: str,
    ) -> None:
        self.upstream = urlsplit(upstream_url)
        self.upstream_origin = origin_from_url(upstream_url)
        super().__init__(server_address, _HostProxyHandler)

    def server_bind(self) -> None:
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        super().server_bind()


class _HostProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: HostProxyServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _upstream_conn(self) -> http.client.HTTPConnection:
        host = self.server.upstream.hostname or "127.0.0.1"
        port = self.server.upstream.port or (
            443 if self.server.upstream.scheme == "https" else 80
        )
        if self.server.upstream.scheme == "https":
            return http.client.HTTPSConnection(host, port, timeout=120)
        return http.client.HTTPConnection(host, port, timeout=120)

    def _request_headers(self) -> dict[str, str]:
        incoming = incoming_origin(self.headers.get("Origin"), self.headers.get("Host"))
        upstream_origin = self.server.upstream_origin
        out: dict[str, str] = {}
        for key, value in self.headers.items():
            lk = key.lower()
            if lk in HOP_BY_HOP or lk in {"host", "origin", "referer"}:
                continue
            out[key] = value
        out["Host"] = urlsplit(upstream_origin).netloc
        out["Origin"] = upstream_origin
        referer = self.headers.get("Referer")
        if referer:
            out["Referer"] = rewrite_referer(referer, incoming, upstream_origin)
        else:
            out["Referer"] = upstream_origin + "/"
        return out

    def _proxy(self) -> None:
        if self.headers.get("Upgrade", "").lower() == "websocket":
            self.send_error(501, "WebSocket upgrade is not supported by host-proxy")
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else None
        incoming = incoming_origin(self.headers.get("Origin"), self.headers.get("Host"))
        conn = self._upstream_conn()
        try:
            conn.request(
                self.command,
                self.path,
                body=body,
                headers=self._request_headers(),
            )
            resp = conn.getresponse()
            payload = resp.read()
            self.send_response(resp.status, resp.reason)
            for key, value in resp.getheaders():
                lk = key.lower()
                if lk in HOP_BY_HOP:
                    continue
                if lk in {"location", "content-location"} and incoming:
                    value = rewrite_url_origin(
                        value, self.server.upstream_origin, incoming
                    )
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)
        finally:
            conn.close()

    def do_GET(self) -> None:  # noqa: N802
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def do_PUT(self) -> None:  # noqa: N802
        self._proxy()

    def do_PATCH(self) -> None:  # noqa: N802
        self._proxy()

    def do_DELETE(self) -> None:  # noqa: N802
        self._proxy()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._proxy()

    def do_HEAD(self) -> None:  # noqa: N802
        self._proxy()


def guest_browser_url(listen_port: int) -> str:
    """URL to open inside the Quicksand Chromium (host via slirp gateway)."""
    return f"http://{QEMU_SLIRP_GATEWAY}:{listen_port}"


def start_background(
    *,
    listen_host: str,
    listen_port: int,
    upstream_url: str,
) -> HostProxyServer:
    """Start the proxy on a daemon thread. Raises OSError if the port is taken."""
    server = HostProxyServer((listen_host, listen_port), upstream_url)
    thread = threading.Thread(
        target=server.serve_forever,
        name="magentic-ui-host-proxy",
        daemon=True,
    )
    thread.start()
    _active_servers.append(server)
    return server


def run_foreground(
    *,
    listen_host: str,
    listen_port: int,
    upstream_url: str,
) -> None:
    server = HostProxyServer((listen_host, listen_port), upstream_url)
    typer.echo(f"host-proxy listening on http://{listen_host}:{listen_port}")
    typer.echo(f"  upstream: {upstream_url}")
    typer.echo(f"  rewrites Origin/Referer/Host → {server.upstream_origin}")
    typer.echo(f"  Quicksand browser URL: {guest_browser_url(listen_port)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nhost-proxy stopped")
    finally:
        server.server_close()


def log_sidecar_banner(
    listen_port: int, upstream_url: str, upstream_origin: str
) -> None:
    typer.echo(typer.style("Host-app proxy", fg=typer.colors.GREEN, bold=True))
    typer.echo(f"  upstream {upstream_url}  (Origin/Referer → {upstream_origin})")
    typer.echo(
        f"  In the agent browser open {guest_browser_url(listen_port)} "
        "(not localhost, and not the app's raw host port)."
    )
