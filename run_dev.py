#!/usr/bin/env python3
"""
Development runner for the Chi Lăng frontend.

Run from the project root:
    python run_dev.py

What it does:
- starts the existing FastAPI backend with:
  python projectctl.py dev --host 127.0.0.1 --port 8000
- serves frontend/src/ at http://127.0.0.1:3000
- proxies every /api/* request from port 3000 to the backend on port 8000
- Ctrl+C shuts down the frontend server and the backend process group

Only the Python standard library is used.
"""

from __future__ import annotations

import http.client
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend" / "src"
PROJECTCTL = ROOT / "projectctl.py"

FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = 3000
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
# Per-socket-operation timeout; allow time for model inference, but never hang.
BACKEND_TIMEOUT_SECONDS = 120.0

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class DevHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class DevRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def _is_api_request(self) -> bool:
        return self.path.startswith("/api/")

    def _request_body(self) -> bytes:
        transfer_encoding = self.headers.get("Transfer-Encoding", "").lower()
        if transfer_encoding and transfer_encoding != "identity":
            self.send_error(501, "Chunked request bodies are not supported by this dev proxy.")
            raise RuntimeError("unsupported transfer encoding")

        content_length = self.headers.get("Content-Length")
        if not content_length:
            return b""

        try:
            size = int(content_length)
        except ValueError as exc:
            self.send_error(400, "Invalid Content-Length header.")
            raise RuntimeError("invalid content length") from exc

        if size < 0:
            self.send_error(400, "Invalid Content-Length header.")
            raise RuntimeError("invalid content length")

        return self.rfile.read(size)

    def _proxy_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        for name, value in self.headers.items():
            lower = name.lower()
            if lower in HOP_BY_HOP_HEADERS or lower in {"host", "content-length"}:
                continue
            headers[name] = value
        return headers

    def _proxy_api(self) -> None:
        try:
            body = self._request_body()
        except RuntimeError:
            return

        connection = http.client.HTTPConnection(
            BACKEND_HOST, BACKEND_PORT, timeout=BACKEND_TIMEOUT_SECONDS
        )
        response_started = False

        try:
            connection.request(
                self.command,
                self.path,
                body=body if body else None,
                headers=self._proxy_headers(),
            )
            upstream = connection.getresponse()

            self.send_response(upstream.status, upstream.reason)
            for name, value in upstream.getheaders():
                lower = name.lower()
                if lower in HOP_BY_HOP_HEADERS or lower in {"server", "date"}:
                    continue
                self.send_header(name, value)
            self.send_header("Connection", "close")
            self.close_connection = True
            response_started = True
            self.end_headers()

            if self.command != "HEAD":
                shutil.copyfileobj(upstream, self.wfile)
        except (OSError, http.client.HTTPException) as exc:
            self.close_connection = True
            if not response_started:
                if isinstance(exc, TimeoutError):
                    self.send_error(504, "Backend proxy timed out.")
                else:
                    self.send_error(502, f"Backend proxy error: {exc}")
            # After headers have been sent, close the truncated response rather
            # than appending a second HTTP response to its body.
        finally:
            connection.close()

    def do_GET(self) -> None:
        if self._is_api_request():
            self._proxy_api()
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self._is_api_request():
            self._proxy_api()
            return
        super().do_HEAD()

    def do_POST(self) -> None:
        if self._is_api_request():
            self._proxy_api()
            return
        self.send_error(405, "POST is only supported for /api/* requests.")

    def do_PUT(self) -> None:
        if self._is_api_request():
            self._proxy_api()
            return
        self.send_error(405, "PUT is only supported for /api/* requests.")

    def do_PATCH(self) -> None:
        if self._is_api_request():
            self._proxy_api()
            return
        self.send_error(405, "PATCH is only supported for /api/* requests.")

    def do_DELETE(self) -> None:
        if self._is_api_request():
            self._proxy_api()
            return
        self.send_error(405, "DELETE is only supported for /api/* requests.")

    def do_OPTIONS(self) -> None:
        if self._is_api_request():
            self._proxy_api()
            return
        self.send_error(405, "OPTIONS is only supported for /api/* requests.")


def _backend_process() -> subprocess.Popen:
    command = [
        sys.executable,
        str(PROJECTCTL),
        "dev",
        "--host",
        BACKEND_HOST,
        "--port",
        str(BACKEND_PORT),
    ]
    kwargs: dict[str, object] = {
        "cwd": str(ROOT),
    }

    if os.name == "posix":
        kwargs["start_new_session"] = True
    elif os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    return subprocess.Popen(command, **kwargs)


def _stop_backend(process: subprocess.Popen, timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    elif os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, ValueError):
            process.terminate()
    else:
        process.terminate()

    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()

    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def _validate_layout() -> None:
    missing = [
        path
        for path in (
            PROJECTCTL,
            FRONTEND_DIR / "index.html",
            FRONTEND_DIR / "styles" / "main.css",
            FRONTEND_DIR / "scripts" / "api.js",
            FRONTEND_DIR / "scripts" / "shortcuts.js",
            FRONTEND_DIR / "scripts" / "app.js",
        )
        if not path.is_file()
    ]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required project files are missing:\n{joined}")


def main() -> int:
    _validate_layout()

    backend = _backend_process()
    server: DevHTTPServer | None = None
    server_thread: threading.Thread | None = None
    exit_code = 0

    try:
        server = DevHTTPServer((FRONTEND_HOST, FRONTEND_PORT), DevRequestHandler)
        server_thread = threading.Thread(
            target=server.serve_forever,
            name="chi-lang-frontend",
            daemon=True,
        )
        server_thread.start()

        print(f"Frontend: http://{FRONTEND_HOST}:{FRONTEND_PORT}")
        print(f"Backend:  http://{BACKEND_HOST}:{BACKEND_PORT}")
        print("Press Ctrl+C to stop both servers.")

        while True:
            backend_code = backend.poll()
            if backend_code is not None:
                if backend_code != 0:
                    print(
                        f"Backend exited unexpectedly with code {backend_code}.",
                        file=sys.stderr,
                    )
                    exit_code = backend_code
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2.0)
        _stop_backend(backend)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
