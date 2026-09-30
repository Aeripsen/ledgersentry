"""Connection draining for rolling restarts.

Measured on a kind cluster in CI (DEPLOY.md): with only a preStop sleep, a
rolling restart under load still failed a few requests with "connection reset
by peer" / EOF. The cause is HTTP keep-alive. Removing a
terminating pod from the Service stops NEW connections reaching it, but a
client's existing persistent connection stays pinned to the old pod. When
uvicorn gets SIGTERM it closes idle keep-alive connections, and a client that
sends its next request on one at that moment gets a reset.

The fix: the Kubernetes preStop hook touches DRAIN_FILE before its sleep. From
then on every response carries `Connection: close`, so each client finishes
its current request, closes the connection cleanly, and reconnects through the
Service to a pod that is not terminating. By the time SIGTERM arrives no client
is holding a connection it expects to reuse.

A file, not an endpoint, so nothing reachable over the network can put a pod
into drain mode. The per-request cost is one stat() call.
"""
from __future__ import annotations

from pathlib import Path

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Must match the preStop command in deploy/k8s and deploy/terraform/kubernetes.
DRAIN_FILE = Path("/tmp/draining")


class DrainMiddleware:
    """Adds `Connection: close` to every HTTP response once DRAIN_FILE exists."""

    def __init__(self, app: ASGIApp, drain_file: Path | None = None) -> None:
        self.app = app
        self.drain_file = drain_file

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        drain_file = self.drain_file or DRAIN_FILE
        if scope["type"] != "http" or not drain_file.exists():
            await self.app(scope, receive, send)
            return

        async def send_with_close(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k != b"connection"]
                message["headers"] = [*headers, (b"connection", b"close")]
            await send(message)

        await self.app(scope, receive, send_with_close)
