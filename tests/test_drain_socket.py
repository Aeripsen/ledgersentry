"""The drain works on a real TCP connection, not just in the ASGI test client.

tests/test_drain.py checks the `Connection: close` header through Starlette's
TestClient, which has no socket and no uvicorn protocol layer. That shows the
header is set, not that uvicorn then closes the connection. Here a real
uvicorn.Server listens on an ephemeral port, once with each of its HTTP parsers
(h11 and httptools), and a raw socket speaks HTTP/1.1 to it:

- drain file present: the first response carries `Connection: close` and the
  server then closes the TCP connection on its own (recv returns b''), so the
  second request on that socket gets nothing back;
- drain file absent (control): the same socket carries a second request and
  gets a second 200, so the test can tell keep-alive from close.
"""
from __future__ import annotations

import importlib.util
import re
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI

from ledgersentry.drain import DrainMiddleware

REQUEST = b"GET /ping HTTP/1.1\r\nHost: test\r\n\r\n"


def _app(drain_file: Path) -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"ok": "yes"}

    app.add_middleware(DrainMiddleware, drain_file=drain_file)
    return app


@pytest.fixture
def serve() -> Iterator[Callable[[FastAPI, str], int]]:
    """Start uvicorn in a thread on 127.0.0.1:<ephemeral>; return the port."""
    running: list[tuple[uvicorn.Server, threading.Thread]] = []

    def start(app: FastAPI, http: str) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        config = uvicorn.Config(app, http=http, lifespan="off", log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise RuntimeError(f"uvicorn ({http}) did not start")
            time.sleep(0.02)
        running.append((server, thread))
        return int(sock.getsockname()[1])

    yield start
    for server, thread in running:
        server.should_exit = True
        thread.join(timeout=10)


def _read_response(s: socket.socket) -> bytes:
    """Read one HTTP/1.1 response (headers + Content-Length body); return the head."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            raise ConnectionError(f"connection closed mid-response after {buf!r}")
        buf += chunk
    head, body = buf.split(b"\r\n\r\n", 1)
    m = re.search(rb"(?im)^content-length:\s*(\d+)\s*$", head)
    assert m, head
    while len(body) < int(m.group(1)):
        chunk = s.recv(4096)
        if not chunk:
            raise ConnectionError("connection closed mid-body")
        body += chunk
    return head


def _connection_header(head: bytes) -> list[bytes]:
    return [v.strip().lower() for v in re.findall(rb"(?im)^connection:\s*(.*?)\s*$", head)]


PARSERS = [
    pytest.param("h11", id="h11"),
    pytest.param(
        "httptools", id="httptools",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("httptools") is None,
            reason="httptools not installed (uvicorn[standard] installs it)",
        ),
    ),
]


@pytest.mark.parametrize("http", PARSERS)
def test_drain_makes_uvicorn_close_the_tcp_connection(
    tmp_path: Path, serve: Callable[[FastAPI, str], int], http: str
) -> None:
    flag = tmp_path / "draining"
    flag.touch()
    port = serve(_app(flag), http)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(REQUEST)
        head = _read_response(s)
        assert head.startswith(b"HTTP/1.1 200")
        assert _connection_header(head) == [b"close"]
        # The server closes the connection by itself: an orderly FIN, before any
        # second request is sent.
        assert s.recv(4096) == b""
        # A client that tries to reuse the socket anyway gets no response.
        got = b""
        try:
            s.sendall(REQUEST)
            got = s.recv(4096)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        assert got == b""


@pytest.mark.parametrize("http", PARSERS)
def test_without_drain_file_the_connection_is_reused(
    tmp_path: Path, serve: Callable[[FastAPI, str], int], http: str
) -> None:
    port = serve(_app(tmp_path / "draining"), http)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        for _ in range(2):
            s.sendall(REQUEST)
            head = _read_response(s)
            assert head.startswith(b"HTTP/1.1 200")
            assert b"close" not in _connection_header(head)
