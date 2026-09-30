"""
HTTP load test for the running service: throughput and latency under
concurrent load, stepped until it saturates.

Why this exists: artifacts/benchmark.json times the scorer in-process, one row
at a time. That answers "how fast is one score" but not the question a backend
reviewer asks next: how many requests per second does the service sustain, what
happens to p99 as concurrent clients pile on, and where does it knee. Those are
properties of the whole stack (HTTP parsing, pydantic validation, FastAPI's
threadpool, the GIL, the model's own threading), so they have to be measured
over HTTP against a real server process.

Method, stated so the numbers can be read correctly:
  * The server is started exactly the way the Dockerfile starts it
    (uvicorn ledgersentry.service:app, access log on), as a separate process,
    unless --url points at one that is already running.
  * The client is a minimal keep-alive HTTP/1.1 client on asyncio streams,
    stdlib only, spread over several processes so the load generator is not
    the thing that saturates first.
  * CLOSED loop: at concurrency c there are c connections, each sending its
    next request as soon as the previous response is fully read. Latency is
    client-side, send to last byte. A closed loop slows down when the server
    slows down, so its tail understates what an open-loop (fixed arrival rate)
    client would see past saturation. Below the knee the two agree.
  * Each level runs a warmup (discarded) and then a fixed measurement window.
    Throughput = successful responses completed inside the window / window.
  * Payloads are real held-out test rows (the same split bench.py uses), not a
    single repeated body, so any value-dependent cost shows up.
  * Client and server share the one machine. Everything is recorded next to
    the numbers: CPU model, logical cores, RAM, OS, Python and library
    versions, uvicorn worker count and HTTP parser.

Knee rule (computed, not eyeballed): the knee is the first concurrency level c
where doubling to the next level buys less than 10% more throughput. Past it,
extra concurrency only adds queueing, which shows up as latency.

Run: python scripts/loadtest.py            (see --help)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import REPO_ROOT, get_settings

DEFAULT_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
REQUEST_TIMEOUT_S = 30.0  # a response slower than this is recorded as a failed request
KNEE_GAIN = 1.10  # doubling concurrency must buy >= 10% more req/s to be "before the knee"
LATENCY_BUDGET_MS = 10.0  # the repo's self-imposed single-row budget (bench.py)


# --------------------------------------------------------------------------
# client: one keep-alive HTTP/1.1 connection per concurrent user
# --------------------------------------------------------------------------

def _request_bytes(host: str, port: int, path: str, body: bytes) -> bytes:
    head = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: keep-alive\r\n\r\n"
    )
    return head.encode("ascii") + body


async def _read_response(reader: asyncio.StreamReader) -> int:
    """Read one response fully (status line, headers, Content-Length body).
    Returns the status code. uvicorn always sends Content-Length for these
    JSON responses; a chunked response would be a harness error, raised."""
    status_line = await reader.readline()
    if not status_line:
        raise ConnectionError("server closed the connection")
    status = int(status_line.split(b" ", 2)[1])
    length = None
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            length = int(value.strip())
    if length is None:
        raise ValueError("response without Content-Length")
    await reader.readexactly(length)
    return status


async def _user(
    host: str, port: int, requests: list[bytes], offset: int,
    t_stop: float, out: list[tuple[float, float, int]],
) -> None:
    """One closed-loop user: send, read, record, repeat until t_stop.
    Records (start_time, latency_s, status); status 0 = connection error or
    no response within REQUEST_TIMEOUT_S (a stalled server shows up as errors
    in the result, never as a harness that waits forever)."""
    i = offset
    n = len(requests)
    writer: asyncio.StreamWriter | None = None
    while True:
        t0 = time.perf_counter()
        if t0 >= t_stop:
            break
        try:
            if writer is None:
                reader, writer = await asyncio.open_connection(host, port)
            writer.write(requests[i % n])
            await writer.drain()
            status = await asyncio.wait_for(_read_response(reader), REQUEST_TIMEOUT_S)
        except (OSError, ValueError, asyncio.IncompleteReadError, TimeoutError):
            # A refused, reset or truncated exchange is a failed request, not a
            # harness crash: record it, drop the connection, reconnect next turn.
            out.append((t0, time.perf_counter() - t0, 0))
            if writer is not None:
                writer.close()
                writer = None
            await asyncio.sleep(0.01)
            i += 1
            continue
        out.append((t0, time.perf_counter() - t0, status))
        i += 1
    if writer is not None:
        writer.close()


def _client_proc(args: tuple) -> list[tuple[float, float, int]]:
    """One client process running `users` concurrent connections."""
    host, port, requests, users, first_offset, t_stop_wall = args
    # perf_counter is per-process; translate the shared wall-clock deadline.
    t_stop = time.perf_counter() + (t_stop_wall - time.time())
    out: list[tuple[float, float, int]] = []
    # Record wall-clock starts so the parent can cut one shared window.
    shift = time.time() - time.perf_counter()

    async def main() -> None:
        await asyncio.gather(*[
            _user(host, port, requests, first_offset + k * 7919, t_stop, out)
            for k in range(users)
        ])

    asyncio.run(main())
    return [(t0 + shift, lat, st) for t0, lat, st in out]


def _split(total: int, parts: int) -> list[int]:
    base, extra = divmod(total, parts)
    return [base + (1 if k < extra else 0) for k in range(parts) if base + (1 if k < extra else 0)]


def run_level(
    pool: Any, host: str, port: int, path: str, bodies: list[bytes],
    concurrency: int, client_procs: int, warmup_s: float, measure_s: float,
    server_pids: list[int] | None = None,
) -> dict[str, Any]:
    requests = [_request_bytes(host, port, path, b) for b in bodies]
    shares = _split(concurrency, min(client_procs, concurrency))
    t_begin = time.time() + 0.5  # let every process start before the clock matters
    t_window0 = t_begin + warmup_s
    t_stop = t_window0 + measure_s
    jobs = [
        (host, port, requests, users, sum(shares[:k]), t_stop)
        for k, users in enumerate(shares)
    ]
    cpu = _CpuSampler(server_pids)
    async_result = pool.map_async(_client_proc, jobs)
    time.sleep(max(0.0, t_window0 - time.time()))
    cpu.start()
    time.sleep(max(0.0, t_stop - time.time()))
    server_cpu = cpu.stop()
    records = [
        r for chunk in async_result.get(timeout=REQUEST_TIMEOUT_S + 60.0) for r in chunk
    ]

    window = [(lat, st) for t0, lat, st in records if t_window0 <= t0 and t0 + lat <= t_stop]
    ok = np.array([lat for lat, st in window if st == 200], dtype=np.float64) * 1000.0
    n_err = sum(1 for _, st in window if st != 200)
    n_total = len(window)
    res: dict[str, Any] = {
        "concurrency": concurrency,
        "client_processes": len(shares),
        "requests": n_total,
        "ok": int(ok.size),
        "errors": n_err,
        "error_rate": round(n_err / n_total, 6) if n_total else None,
        "req_per_s": round(ok.size / measure_s, 1),
    }
    if ok.size:
        res.update({
            "p50_ms": round(float(np.percentile(ok, 50)), 3),
            "p95_ms": round(float(np.percentile(ok, 95)), 3),
            "p99_ms": round(float(np.percentile(ok, 99)), 3),
            "max_ms": round(float(ok.max()), 3),
            "mean_ms": round(float(ok.mean()), 3),
        })
    if server_cpu is not None:
        res["server_cpu_percent"] = server_cpu[0]  # 100 = one logical core fully busy
        res["machine_cpu_percent"] = server_cpu[1]  # 100 = all logical cores busy
    return res


class _CpuSampler:
    """Server-process CPU over the measurement window, summed over the uvicorn
    parent and its worker children. psutil is optional: without it the field
    is simply absent, never guessed."""

    def __init__(self, pids: list[int] | None) -> None:
        self.procs: list[Any] = []
        try:
            import psutil
        except ImportError:
            return
        for pid in pids or []:
            try:
                p = psutil.Process(pid)
                self.procs.append(p)
                self.procs.extend(p.children(recursive=True))
            except psutil.Error:
                pass
        self.t0 = 0.0
        self.c0: list[float] = []

    def _cpu(self) -> list[float]:
        vals = []
        for p in self.procs:
            try:
                t = p.cpu_times()
                vals.append(t.user + t.system)
            except Exception:
                vals.append(float("nan"))
        return vals

    def start(self) -> None:
        if self.procs:
            import psutil

            psutil.cpu_percent(interval=None)  # arm the machine-wide counter
            self.t0 = time.perf_counter()
            self.c0 = self._cpu()

    def stop(self) -> tuple[float, float] | None:
        """(server CPU %, whole-machine CPU %). Server: 100 = one logical core
        fully busy. Machine: 100 = every logical core busy, client included."""
        if not self.procs:
            return None
        import psutil

        dt = time.perf_counter() - self.t0
        used = sum(b - a for a, b in zip(self.c0, self._cpu(), strict=True) if b == b)
        return round(100.0 * used / dt, 1), psutil.cpu_percent(interval=None)


def find_knee(levels: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First level where the next doubling buys < 10% more throughput."""
    for cur, nxt in zip(levels, levels[1:], strict=False):
        if cur["req_per_s"] <= 0:
            continue
        if nxt["req_per_s"] / cur["req_per_s"] < KNEE_GAIN:
            return {
                "concurrency": cur["concurrency"],
                "req_per_s": cur["req_per_s"],
                "p99_ms": cur.get("p99_ms"),
                "next_level_gain": round(nxt["req_per_s"] / cur["req_per_s"], 3),
            }
    return None


def summarize(levels: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(levels, key=lambda r: r["req_per_s"])
    in_budget = [
        r for r in levels if r.get("p99_ms") is not None and r["p99_ms"] <= LATENCY_BUDGET_MS
        and r["errors"] == 0
    ]
    top_in_budget = max(in_budget, key=lambda r: r["req_per_s"]) if in_budget else None
    return {
        "peak": {k: best.get(k) for k in ("concurrency", "req_per_s", "p50_ms", "p99_ms")},
        "knee": find_knee(levels),
        "best_with_p99_under_budget": (
            {k: top_in_budget.get(k) for k in ("concurrency", "req_per_s", "p50_ms", "p99_ms")}
            if top_in_budget else None
        ),
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "total_errors": sum(r["errors"] for r in levels),
    }


# --------------------------------------------------------------------------
# server under test
# --------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_ready(
    base: str, proc: subprocess.Popen | None = None, log: Path | None = None,
    timeout_s: float = 120.0,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; see {log}")
        try:
            with urllib.request.urlopen(base + "/ready", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"server at {base} never became ready")


def start_server(
    port: int, workers: int, http: str, log_path: Path, extra_env: dict[str, str]
) -> subprocess.Popen:
    """uvicorn with the Dockerfile's flags, plus --workers/--http when asked."""
    cmd = [
        sys.executable, "-m", "uvicorn", "ledgersentry.service:app",
        "--app-dir", str(REPO_ROOT / "src"),
        "--host", "127.0.0.1", "--port", str(port),
    ]
    if workers > 1:
        cmd += ["--workers", str(workers)]
    if http != "auto":
        cmd += ["--http", http]
    # Thread-count variables change the result (see the OpenMP finding in the
    # README), so an inherited value never leaks in: only --env sets them.
    inherited = {
        k: v for k, v in os.environ.items()
        if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    env = {**inherited, **extra_env}
    log = open(log_path, "wb")  # noqa: SIM115 - handed to the child, closed with it
    return subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def stop_server(proc: subprocess.Popen) -> None:
    try:
        import psutil

        for child in psutil.Process(proc.pid).children(recursive=True):
            child.kill()
    except Exception:
        pass
    proc.kill()
    proc.wait(timeout=30)


# --------------------------------------------------------------------------
# payloads and environment
# --------------------------------------------------------------------------

def predict_bodies(n: int) -> tuple[list[bytes], list[dict[str, Any]], str]:
    """Real held-out rows as /predict JSON bodies, in the shape bench.py uses."""
    from .bench import _feature_dicts
    from .scoring import build_scorer
    from .stream import load_bundle, load_stream

    bundle = load_bundle()
    df, source = load_stream(n=0)
    scorer = build_scorer(bundle)
    rows = _feature_dicts(df.iloc[:n], scorer)
    clean = [{k: v for k, v in r.items() if v == v} for r in rows]  # drop NaN: a client omits it
    bodies = [json.dumps({"features": r}).encode() for r in clean]
    return bodies, clean, source


def batch_bodies(rows: list[dict[str, Any]], batch_size: int) -> list[bytes]:
    out = []
    for start in range(0, len(rows) - batch_size + 1, batch_size):
        out.append(json.dumps({"transactions": rows[start:start + batch_size]}).encode())
    return out


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        return None


def _physical_cores() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except ImportError:
        return None


def _version(mod: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(mod)
    except Exception:
        return None


def _idle_load() -> float | None:
    """Whole-machine CPU over 1 s before the server starts: other processes on
    a shared dev machine are noise the numbers carry, so it is recorded."""
    try:
        import psutil

        return float(psutil.cpu_percent(interval=1.0))
    except ImportError:
        return None


def environment(workers: int, http: str) -> dict[str, Any]:
    resolved_http = http
    if http == "auto":
        resolved_http = "httptools" if _version("httptools") else "h11"
    return {
        "cpu": _cpu_name(),
        "physical_cores": _physical_cores(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "fastapi": _version("fastapi"),
        "uvicorn": _version("uvicorn"),
        "http_parser": resolved_http,
        "event_loop": "uvloop" if (_version("uvloop") and sys.platform != "win32") else "asyncio",
        "scikit_learn": _version("scikit-learn"),
        "numpy": _version("numpy"),
        "uvicorn_workers": workers,
        "machine_cpu_percent_idle_before_run": _idle_load(),
        "client": "stdlib asyncio keep-alive HTTP/1.1, same machine as the server",
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(
    endpoint: str, levels: list[int], workers: int, http: str, warmup_s: float,
    measure_s: float, client_procs: int, batch_size: int, n_payloads: int,
    url: str | None, label: str, extra_env: dict[str, str],
) -> dict[str, Any]:
    bodies, rows, source = predict_bodies(n_payloads)
    if endpoint == "/predict/batch":
        bodies = batch_bodies(rows, batch_size)
    print(f"[load] source={source} endpoint={endpoint} bodies={len(bodies)} "
          f"workers={workers} http={http} levels={levels}")

    env = environment(workers, http)  # before the server starts: idle load is honest
    proc = None
    pids: list[int] = []
    tmp = Path(tempfile.mkdtemp(prefix="ledgersentry_load_"))
    if url is None:
        port = _free_port()
        base = f"http://127.0.0.1:{port}"
        proc = start_server(port, workers, http, tmp / "server.log", extra_env)
        pids = [proc.pid]
    else:
        base = url.rstrip("/")
    host, port_s = base.removeprefix("http://").split(":")
    try:
        _wait_ready(base, proc, tmp / "server.log")
        # server-side warmup: the first requests in each worker pay one-time costs
        with mp.get_context("spawn").Pool(client_procs) as pool:
            run_level(pool, host, int(port_s), endpoint, bodies, max(workers, 4),
                      client_procs, 0.0, 2.0)
            results = []
            for c in levels:
                r = run_level(pool, host, int(port_s), endpoint, bodies, c,
                              client_procs, warmup_s, measure_s, pids)
                if endpoint == "/predict/batch":
                    r["rows_per_s"] = round(r["req_per_s"] * batch_size, 1)
                results.append(r)
                print(
                    f"[c={c:>4}] {r['req_per_s']:>9,.1f} req/s  "
                    f"p50={r.get('p50_ms', float('nan')):8.2f}  "
                    f"p95={r.get('p95_ms', float('nan')):8.2f}  "
                    f"p99={r.get('p99_ms', float('nan')):8.2f} ms  "
                    f"err={r['errors']}  server_cpu={r.get('server_cpu_percent', '-')}%"
                )
    finally:
        if proc is not None:
            stop_server(proc)

    out: dict[str, Any] = {
        "what": (
            "closed-loop HTTP load test against the service process; client-side "
            "latency, send to last byte; real held-out rows as payloads"
        ),
        "label": label,
        "endpoint": endpoint,
        "batch_size": batch_size if endpoint == "/predict/batch" else 1,
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "warmup_s": warmup_s,
        "measure_s": measure_s,
        "server_env_overrides": extra_env,
        "environment": env,
        "levels": results,
        "summary": summarize(results),
    }
    s = out["summary"]
    print(f"[peak] {s['peak']}\n[knee] {s['knee']}\n[<{LATENCY_BUDGET_MS:.0f}ms p99] "
          f"{s['best_with_p99_under_budget']}")
    return out


def table(paths: list[str]) -> str:
    """Markdown table straight from result files, so a README can never quote
    a number the JSON does not contain."""
    import glob

    files = sorted({f for pat in paths for f in glob.glob(pat)})
    lines = [
        "| run | workers | server env | c | req/s | p50 ms | p95 ms | p99 ms | errors "
        "| server CPU % |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in files:
        d = json.loads(Path(f).read_text())
        env = " ".join(f"{k}={v}" for k, v in d.get("server_env_overrides", {}).items()) or "-"
        for r in d["levels"]:
            lines.append(
                f"| {Path(f).stem} | {d['environment']['uvicorn_workers']} | {env} "
                f"| {r['concurrency']} | {r['req_per_s']:,.1f} | {r.get('p50_ms', '-')} "
                f"| {r.get('p95_ms', '-')} | {r.get('p99_ms', '-')} | {r['errors']} "
                f"| {r.get('server_cpu_percent', '-')} |"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="HTTP load test for the scoring service.")
    ap.add_argument("--endpoint", default="predict", choices=["predict", "batch"],
                    help="predict = POST /predict, batch = POST /predict/batch")
    ap.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)),
                    help="comma-separated concurrency levels")
    ap.add_argument("--workers", type=int, default=1, help="uvicorn worker processes")
    ap.add_argument("--http", default="auto", choices=["auto", "h11", "httptools"])
    ap.add_argument("--warmup", type=float, default=2.0, help="seconds discarded per level")
    ap.add_argument("--duration", type=float, default=10.0, help="measured seconds per level")
    ap.add_argument("--client-procs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--payloads", type=int, default=2000, help="distinct held-out rows used")
    ap.add_argument("--url", default=None, help="target an already-running server instead")
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VALUE set in the server's environment (repeatable)")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--out", default=None,
                    help="JSON path (default artifacts/loadtest_<label>.json)")
    ap.add_argument("--table", nargs="+", default=None, metavar="JSON",
                    help="print a markdown table from result files (globs ok) and exit")
    args = ap.parse_args()
    if args.table:
        print(table(args.table))
        return

    extra_env = dict(kv.split("=", 1) for kv in args.env)
    result = run(
        endpoint={"predict": "/predict", "batch": "/predict/batch"}[args.endpoint],
        levels=[int(x) for x in args.levels.split(",")],
        workers=args.workers, http=args.http, warmup_s=args.warmup,
        measure_s=args.duration, client_procs=args.client_procs,
        batch_size=args.batch_size, n_payloads=args.payloads, url=args.url,
        label=args.label, extra_env=extra_env,
    )
    out_path = Path(args.out) if args.out else (
        get_settings().artifact_dir / f"loadtest_{args.label}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
