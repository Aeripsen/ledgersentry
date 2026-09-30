"""DrainMiddleware adds `Connection: close` only once the drain file exists, and
leaves every response untouched before that (see src/ledgersentry/drain.py)."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ledgersentry.drain import DrainMiddleware
from ledgersentry.service import app as service_app


def _client(drain_file: Path) -> TestClient:
    app = FastAPI()

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"ok": "yes"}

    app.add_middleware(DrainMiddleware, drain_file=drain_file)
    return TestClient(app)


def test_no_drain_file_means_keep_alive_is_untouched(tmp_path: Path) -> None:
    r = _client(tmp_path / "draining").get("/ping")
    assert r.status_code == 200
    assert r.headers.get("connection") != "close"


def test_drain_file_turns_on_connection_close(tmp_path: Path) -> None:
    flag = tmp_path / "draining"
    client = _client(flag)
    flag.touch()
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.json() == {"ok": "yes"}
    assert r.headers["connection"] == "close"
    # exactly one connection header, never a keep-alive left beside it
    assert r.headers.get_list("connection") == ["close"]


def test_service_app_has_the_middleware() -> None:
    assert any(m.cls is DrainMiddleware for m in service_app.user_middleware)
