"""Bounded dev-proxy failures without starting models or the real backend."""

import http.client
import io
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler

import pytest

import run_dev


@contextmanager
def serving(handler):
    server = run_dev.DevHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("send_headers", [False, True])
def test_proxy_bounds_stalled_upstream(monkeypatch, send_headers):
    release = threading.Event()

    class StalledBackend(BaseHTTPRequestHandler):
        def do_GET(self):
            if send_headers:
                self.send_response(200)
                self.send_header("Content-Length", "20")
                self.end_headers()
                self.wfile.flush()
            release.wait(timeout=5)

    # Exercise a real socket timeout, not a mocked TimeoutError.
    monkeypatch.setattr(run_dev, "BACKEND_TIMEOUT_SECONDS", 0.1)
    with serving(StalledBackend) as upstream_port:
        monkeypatch.setattr(run_dev, "BACKEND_PORT", upstream_port)
        with serving(run_dev.DevRequestHandler) as proxy_port:
            connection = http.client.HTTPConnection("127.0.0.1", proxy_port, timeout=2)
            try:
                connection.request("GET", "/api/search")
                response = connection.getresponse()
                if send_headers:
                    assert response.status == 200
                    with pytest.raises(http.client.IncompleteRead) as error:
                        response.read()
                    # Never append a second 502/504 HTTP response to the body.
                    assert error.value.partial == b""
                else:
                    assert response.status == 504
                    assert b"Backend proxy timed out" in response.read()
            finally:
                release.set()
                connection.close()


def test_proxy_network_error_closes_upstream(monkeypatch):
    connections = []

    class RefusedConnection:
        def __init__(self, host, port, *, timeout):
            assert 0 < timeout == run_dev.BACKEND_TIMEOUT_SECONDS < float("inf")
            self.closed = False
            connections.append(self)

        def request(self, *args, **kwargs):
            raise ConnectionRefusedError("connection refused")

        def close(self):
            self.closed = True

    monkeypatch.setattr(run_dev.http.client, "HTTPConnection", RefusedConnection)
    handler = object.__new__(run_dev.DevRequestHandler)
    handler.command = "GET"
    handler.path = "/api/search"
    handler.headers = {}
    handler.rfile = io.BytesIO()
    errors = []
    handler.send_error = lambda code, message: errors.append((code, message))
    handler._proxy_api()
    assert errors == [(502, "Backend proxy error: connection refused")]
    assert handler.close_connection is True
    assert connections[0].closed is True
