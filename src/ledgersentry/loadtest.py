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
  * The server is the same app the Dockerfile serves (uvicorn
    ledgersentry.service:app, access log on) with the Dockerfile's thread-count
    ENV lines applied, started as a separate process. It is NOT the image: host
    127.0.0.1, the host's Python and uvicorn (both recorded), and the worker
    count comes from WEB_CONCURRENCY, which the image's uvicorn also reads but
    which neither the Dockerfile nor the k8s manifests set. --url points the
    client at a server that is already running (the Linux-image workflow does
    that with the real container).
  * An "arm" names the one thing an A/B changes (ARMS below). The shipped
    configuration is the first arm; the other is the "before".
  * The client is a minimal keep-alive HTTP/1.1 client on asyncio streams,
    stdlib only, spread over several processes. It shares the machine with the
    server, so at high concurrency it competes with the server for CPU; the
    whole-machine CPU column shows when that happens.
  * CLOSED loop: at concurrency c there are c connections, each sending its
    next request as soon as the previous response is fully read. Latency is
    client-side, send to last byte. A closed loop slows down when the server
    slows down, so its tail understates what an open-loop (fixed arrival rate)
    client would see past saturation. Below the knee the two agree.
  * Each level runs a warmup (discarded) and then a fixed measurement window.
    Throughput = successful responses completed inside the window / window.
  * Payloads are real held-out test rows (the same split bench.py uses), not a
    single repeated body, so any value-dependent cost shows up.
  * Every result file records what produced it: git SHA and whether the tree
    had uncommitted changes outside artifacts/, the arm, the server's resolved
    thread environment, the OpenMP team size scikit-learn picks under that
    environment, CPU model, cores, RAM, OS, Python and library versions, and
    the machine's CPU load in the second before the run.

psutil is required (requirements-loadtest.txt): it measures server CPU, the
machine, and cleans up uvicorn worker children. Without it the harness stops
instead of writing a file with those fields missing.

Knee rule (computed, not eyeballed): the knee is the first concurrency level c
where doubling to the next level buys less than 10% more throughput. Past it,
extra concurrency only adds queueing, which shows up as latency.

Run: python scripts/loadtest.py            (see --help)
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import multiprocessing as mp
import os
import platform
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import REPO_ROOT, get_settings

PKG = "ledgersentry"
APP = f"{PKG}.service:app"
DEFAULT_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
REQUEST_TIMEOUT_S = 30.0  # a response slower than this is recorded as a failed request
KNEE_GAIN = 1.10  # doubling concurrency must buy >= 10% more req/s to be "before the knee"
LATENCY_BUDGET_MS = 10.0  # the repo's self-imposed single-row budget (bench.py)
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

# The A/B: the only thing that differs between the two arms. The first arm is
# what ships (the Dockerfile's ENV OMP_NUM_THREADS=1); the second is the
# configuration before the fix, where scikit-learn sizes the OpenMP team itself.
ARMS: dict[str, dict[str, Any]] = {
    "omp1": {
        "what": "thread-count ENV from the Dockerfile (OMP_NUM_THREADS=1), as shipped",
        "unset_thread_env": False,
        "env": {},
    },
    "ompdefault": {
        "what": "thread-count variables unset: scikit-learn sizes the OpenMP team (before the fix)",
        "unset_thread_env": True,
        "env": {},
    },
}

# py-spy buckets: share of GIL-holding samples whose stack contains a match.
# Inclusive, so they can overlap; the regexes are saved next to the result.
PROFILE_BUCKETS: dict[str, str] = {
    "scoring (ledgersentry/scoring.py)": r"ledgersentry[\\/]scoring\.py",
    "model predict (sklearn)": r"sklearn[\\/]",
    "json decode": r"json[\\/]decoder\.py",
    "pydantic validation": r"pydantic",
    "pandas": r"pandas[\\/]",
    "logging": r"logging[\\/]",
}


def arm_record(server_env: dict[str, str]) -> dict[str, Any]:
    """What the arm resolved to inside the server's environment."""
    return {k: server_env.get(k) for k in THREAD_VARS}


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
    server_pids: list[int] | None = None, pyspy: dict[str, Any] | None = None,
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
    spy = subprocess.Popen(pyspy["cmd"]) if pyspy else None  # noqa: S603 - our own argv
    cpu.start()
    time.sleep(max(0.0, t_stop - time.time()))
    server_cpu = cpu.stop()
    records = [
        r for chunk in async_result.get(timeout=REQUEST_TIMEOUT_S + 60.0) for r in chunk
    ]
    if spy is not None:
        spy.wait(timeout=120)

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
    if server_cpu[0] is not None:
        res["server_cpu_percent"] = server_cpu[0]  # 100 = one logical core fully busy
    res["machine_cpu_percent"] = server_cpu[1]  # 100 = all logical cores busy, client included
    return res


def _psutil() -> Any:
    try:
        import psutil
    except ImportError as exc:  # pragma: no cover - exercised by hand, not in CI
        raise SystemExit(
            "the load test needs psutil (server CPU, machine load, worker cleanup): "
            "pip install -r requirements-loadtest.txt"
        ) from exc
    return psutil


class _CpuSampler:
    """Server-process CPU over the measurement window, summed over the uvicorn
    parent and its worker children, plus whole-machine CPU."""

    def __init__(self, pids: list[int] | None) -> None:
        psutil = _psutil()
        self.procs: list[Any] = []
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
        _psutil().cpu_percent(interval=None)  # arm the machine-wide counter
        self.t0 = time.perf_counter()
        self.c0 = self._cpu()

    def stop(self) -> tuple[float | None, float]:
        """(server CPU %, whole-machine CPU %). Server: 100 = one logical core
        fully busy; None when no server PID is known (--url without
        --server-pid). Machine: 100 = every logical core busy."""
        machine = float(_psutil().cpu_percent(interval=None))
        if not self.procs:
            return None, machine
        dt = time.perf_counter() - self.t0
        used = sum(b - a for a, b in zip(self.c0, self._cpu(), strict=True) if b == b)
        return round(100.0 * used / dt, 1), machine


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

def dockerfile_env(path: Path | None = None) -> dict[str, str]:
    """The ENV values the image ends up with, parsed the way `docker build`
    applies them: comment lines ignored, backslash continuations joined, both
    `ENV K=V [K2=V2 ...]` and legacy `ENV K V`, and a later ENV overrides an
    earlier one. A commented-out `# ENV OMP_NUM_THREADS=1` sets nothing."""
    text = (path or REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    logical: list[str] = []
    buf = ""
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("#"):
            continue  # Docker drops comment lines, also inside a continuation
        if s.endswith("\\"):
            buf += s[:-1] + " "
            continue
        logical.append(buf + s)
        buf = ""
    if buf:
        logical.append(buf)
    env: dict[str, str] = {}
    for line in logical:
        parts = line.split(None, 1)
        if len(parts) < 2 or parts[0].upper() != "ENV":
            continue
        tokens = shlex.split(parts[1], posix=True)
        if tokens and "=" not in tokens[0]:
            env[tokens[0]] = " ".join(tokens[1:])  # legacy form: ENV KEY value words
            continue
        for tok in tokens:
            key, sep, value = tok.partition("=")
            if sep:
                env[key] = value
    return env


def dockerfile_thread_env(path: Path | None = None) -> dict[str, str]:
    return {k: v for k, v in dockerfile_env(path).items() if k in THREAD_VARS}


def server_env(workers: int, thread_env: dict[str, str], extra_env: dict[str, str],
               base: dict[str, str] | None = None) -> dict[str, str]:
    """The server's environment. Inherited thread-count and worker-count
    variables never leak in (they change the result); only the arm sets them."""
    inherited = {
        k: v for k, v in (os.environ if base is None else base).items()
        if k not in THREAD_VARS and k != "WEB_CONCURRENCY"
    }
    return {**inherited, **thread_env, "WEB_CONCURRENCY": str(workers), **extra_env}


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


def server_cmd(port: int, http: str) -> list[str]:
    """The Dockerfile's uvicorn command, on 127.0.0.1. Workers come from
    WEB_CONCURRENCY in the environment, the variable the image's uvicorn reads."""
    cmd = [
        sys.executable, "-m", "uvicorn", APP,
        "--app-dir", str(REPO_ROOT / "src"),
        "--host", "127.0.0.1", "--port", str(port),
    ]
    if http != "auto":
        cmd += ["--http", http]
    return cmd


def start_server(cmd: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen:
    log = open(log_path, "wb")  # noqa: SIM115 - handed to the child, closed with it
    return subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def stop_server(proc: subprocess.Popen) -> None:
    psutil = _psutil()
    try:
        for child in psutil.Process(proc.pid).children(recursive=True):
            child.kill()
    except psutil.Error:
        pass
    proc.kill()
    proc.wait(timeout=30)


def probe_openmp() -> dict[str, Any]:
    """The OpenMP team scikit-learn's HistGradientBoosting uses per predict
    call in THIS process's environment. The number is
    sklearn.utils._openmp_helpers._openmp_effective_n_threads(), the value
    HistGradientBoosting passes to its Cython prange: with OMP_NUM_THREADS set
    it is omp_get_max_threads(); unset, it is that capped at the physical cores
    (and any cgroup CPU quota). threadpoolctl reports the loaded OpenMP
    runtime and its pool size after a real predict call."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    from threadpoolctl import threadpool_info

    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 4))
    y = (X[:, 0] > 0).astype(int)
    HistGradientBoostingClassifier(max_iter=5).fit(X, y).predict(X[:1])
    omp = [p for p in threadpool_info() if p.get("user_api") == "openmp"]
    return {
        # the versions the SERVER runs, which for an external server (a container)
        # are not the client host's versions in "environment"
        "python": platform.python_version(),
        **{mod.replace("-", "_"): _version(mod)
           for mod in ("uvicorn", "fastapi", "scikit-learn", "numpy", "httptools")},
        "hgb_predict_openmp_threads": int(_openmp_effective_n_threads()),
        "openmp_runtime": omp[0].get("prefix") if omp else None,
        "openmp_pool_num_threads": omp[0].get("num_threads") if omp else None,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }


def run_probe(env: dict[str, str]) -> dict[str, Any]:
    """probe_openmp() in a fresh process with the server's environment."""
    penv = {**env, "PYTHONPATH": str(REPO_ROOT / "src")}
    out = subprocess.run(
        [sys.executable, "-m", f"{PKG}.loadtest", "--probe-openmp"],
        cwd=REPO_ROOT, env=penv, capture_output=True, text=True, timeout=300, check=False,
    )
    if out.returncode != 0:
        return {"error": out.stderr.strip()[-500:]}
    return dict(json.loads(out.stdout.strip().splitlines()[-1]))


# --------------------------------------------------------------------------
# py-spy: the profile is a committed file, the percentages are computed from it
# --------------------------------------------------------------------------

def pyspy_cmd(pid: int, seconds: float, out_path: Path, subprocesses: bool) -> list[str]:
    exe = shutil.which("py-spy")
    if exe is None:
        raise SystemExit("py-spy is not installed: pip install -r requirements-loadtest.txt")
    cmd = [
        exe, "record", "--pid", str(pid), "--duration", str(max(1, int(seconds))),
        "--rate", "100", "--gil", "--format", "raw", "--output", str(out_path),
    ]
    if subprocesses:
        cmd.append("--subprocesses")
    return cmd


def _shown_cmd(cmd: list[str]) -> str:
    """The py-spy command as it ran, without the machine-specific exe path,
    PID and output directory."""
    shown = [Path(cmd[0]).stem]
    for prev, arg in zip(cmd, cmd[1:], strict=False):
        if prev == "--pid":
            arg = "<server pid>"
        elif prev == "--output":
            arg = Path(arg).name
        shown.append(arg)
    return " ".join(shown)


def profile_shares(path: Path, buckets: dict[str, str]) -> dict[str, Any]:
    """Share of samples (py-spy raw/collapsed format: 'frame;frame;... count')
    whose stack matches each bucket regex, plus the top leaf frames. With
    --gil every sample is a moment when a thread held the GIL."""
    stacks: list[tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stack, _, count = line.rpartition(" ")
        if stack and count.isdigit():
            stacks.append((stack, int(count)))
    total = sum(c for _, c in stacks)
    shares = {
        name: round(100.0 * sum(c for s, c in stacks if re.search(rx, s)) / total, 1)
        if total else None
        for name, rx in buckets.items()
    }
    leaves: dict[str, int] = defaultdict(int)
    for s, c in stacks:
        leaves[s.rsplit(";", 1)[-1]] += c
    top = sorted(leaves.items(), key=lambda kv: -kv[1])[:10]
    return {
        "file": path.name,
        "samples": total,
        "bucket_regex": buckets,
        "share_percent": shares,
        "top_leaf_frames_percent": (
            {k: round(100.0 * v / total, 1) for k, v in top} if total else {}
        ),
    }


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


def _version(mod: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(mod)
    except Exception:
        return None


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
            check=False,
        )
    except OSError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def provenance() -> dict[str, Any]:
    """Which code produced the file. Changes under artifacts/ do not count as
    dirty: the harness writes there while it runs."""
    import hashlib

    sha = _git("rev-parse", "HEAD") or None
    dirty = _git("status", "--porcelain", "--", ".", ":(exclude)artifacts") if sha else ""
    model = get_settings().artifact_dir / f"{PKG}.joblib"  # gitignored, so hashed here
    return {
        "git_sha": sha,
        "git_dirty_outside_artifacts": bool(dirty) if sha else None,
        "dirty_paths": [ln.split(maxsplit=1)[-1] for ln in dirty.splitlines()][:20],
        "model_artifact_path": str(model),
        "model_artifact_sha256": (
            hashlib.sha256(model.read_bytes()).hexdigest() if model.exists() else None
        ),
        "harness_argv": sys.argv[1:],
    }


def environment(workers: int, http: str) -> dict[str, Any]:
    psutil = _psutil()
    resolved_http = http
    if http == "auto":
        resolved_http = "httptools" if _version("httptools") else "h11"
    return {
        "cpu": _cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "fastapi": _version("fastapi"),
        "uvicorn": _version("uvicorn"),
        "http_parser": resolved_http,
        "event_loop": "uvloop" if (_version("uvloop") and sys.platform != "win32") else "asyncio",
        "scikit_learn": _version("scikit-learn"),
        "numpy": _version("numpy"),
        "uvicorn_workers": workers,
        # BUSY percent, not idle: other processes on a shared machine are noise
        # the numbers carry, so the load in the second before the run is kept.
        "machine_cpu_busy_percent_before_run": float(psutil.cpu_percent(interval=1.0)),
        "client": "stdlib asyncio keep-alive HTTP/1.1, same machine as the server",
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def run(
    endpoint: str, levels: list[int], workers: int, http: str, warmup_s: float,
    measure_s: float, client_procs: int, batch_size: int, n_payloads: int,
    url: str | None, label: str, arm: str, extra_env: dict[str, str],
    server_pid: int | None = None, meta: dict[str, Any] | None = None,
    pyspy_at: int | None = None, profile_dir: Path | None = None,
) -> dict[str, Any]:
    bodies, rows, source = predict_bodies(n_payloads)
    if endpoint == "/predict/batch":
        bodies = batch_bodies(rows, batch_size)
    print(f"[load] source={source} endpoint={endpoint} bodies={len(bodies)} arm={arm} "
          f"workers={workers} http={http} levels={levels}")

    if url is not None:
        # An external server may still be starting (a fresh container importing
        # the model): wait for it first, or its own startup reads as "load before".
        _wait_ready(url.rstrip("/"))
    env = environment(workers, http)  # before a harness-started server starts
    prov = provenance()  # at the start: the code the server is about to load
    proc = None
    pids: list[int] = []
    tmp = Path(tempfile.mkdtemp(prefix=f"{PKG}_load_"))
    server: dict[str, Any]
    if url is None:
        spec = ARMS[arm]
        thread_env = {} if spec["unset_thread_env"] else dockerfile_thread_env()
        senv = server_env(workers, thread_env, {**spec["env"], **extra_env})
        port = _free_port()
        base = f"http://127.0.0.1:{port}"
        cmd = server_cmd(port, http)
        server = {
            "mode": "started by the harness",
            # repo-relative, so a result file carries no machine-specific path
            "cmd": [Path(cmd[0]).name, *("src" if a == str(REPO_ROOT / "src") else a
                                         for a in cmd[1:])],
            "workers_via": "WEB_CONCURRENCY",
            "thread_env": {k: senv[k] for k in THREAD_VARS if k in senv},
            "thread_env_source": "unset" if spec["unset_thread_env"] else "Dockerfile ENV",
            "env_overrides": {**spec["env"], **extra_env},
            "arm_resolved": arm_record(senv),
            "openmp_probe": run_probe(senv),
        }
        proc = start_server(cmd, senv, tmp / "server.log")
        pids = [proc.pid]
    else:
        base = url.rstrip("/")
        pids = [server_pid] if server_pid else []
        env["describes"] = "the client host only; the server's versions come from --meta"
        server = {
            "mode": "external (--url)",
            "url": base,
            "server_pid": server_pid,
            "note": "configuration of an external server is what --meta records",
        }
    host, port_s = base.removeprefix("http://").split(":")
    profile: dict[str, Any] | None = None
    try:
        _wait_ready(base, proc, tmp / "server.log")
        # server-side warmup: the first requests in each worker pay one-time costs
        with mp.get_context("spawn").Pool(client_procs) as pool:
            run_level(pool, host, int(port_s), endpoint, bodies, max(workers, 4),
                      client_procs, 0.0, 2.0)
            results = []
            for c in levels:
                spy: dict[str, Any] | None = None
                if pyspy_at == c and pids:
                    pdir = profile_dir or get_settings().artifact_dir / "profiles"
                    pdir.mkdir(parents=True, exist_ok=True)
                    ppath = pdir / f"{label}_c{c}.txt"
                    spy = {"cmd": pyspy_cmd(pids[0], measure_s, ppath, workers > 1),
                           "path": ppath}
                r = run_level(pool, host, int(port_s), endpoint, bodies, c,
                              client_procs, warmup_s, measure_s, pids, spy)
                if endpoint == "/predict/batch":
                    r["rows_per_s"] = round(r["req_per_s"] * batch_size, 1)
                results.append(r)
                if spy is not None:
                    profile = {
                        "concurrency": c,
                        "command": _shown_cmd(spy["cmd"]),
                        "pid_target": "the uvicorn process the harness started"
                        + (" and its worker children" if workers > 1 else ""),
                        **profile_shares(spy["path"], PROFILE_BUCKETS),
                    }
                print(
                    f"[c={c:>4}] {r['req_per_s']:>9,.1f} req/s  "
                    f"p50={r.get('p50_ms', float('nan')):8.2f}  "
                    f"p95={r.get('p95_ms', float('nan')):8.2f}  "
                    f"p99={r.get('p99_ms', float('nan')):8.2f} ms  "
                    f"err={r['errors']}  server_cpu={r.get('server_cpu_percent', '-')}%  "
                    f"machine_cpu={r['machine_cpu_percent']}%"
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
        "arm": arm,
        "arm_what": ARMS[arm]["what"] if arm in ARMS else None,
        "endpoint": endpoint,
        "batch_size": batch_size if endpoint == "/predict/batch" else 1,
        "data_source": source,
        "is_synthetic": source == "synthetic",
        "warmup_s": warmup_s,
        "measure_s": measure_s,
        "provenance": {
            **prov,
            "git_sha_changed_during_run": provenance()["git_sha"] != prov["git_sha"],
        },
        "server": server,
        "meta": meta or {},
        "environment": env,
        "levels": results,
        "summary": summarize(results),
    }
    if profile is not None:
        out["profile"] = profile
    s = out["summary"]
    print(f"[peak] {s['peak']}\n[knee] {s['knee']}\n[<{LATENCY_BUDGET_MS:.0f}ms p99] "
          f"{s['best_with_p99_under_budget']}")
    return out


def check_result(d: dict[str, Any]) -> list[str]:
    """What a result file must carry for its numbers to be re-checkable."""
    problems = []
    prov = d.get("provenance") or {}
    if not prov.get("git_sha"):
        problems.append("no git SHA")
    if d.get("arm") not in ARMS:
        problems.append(f"unknown arm {d.get('arm')!r}")
    env = d.get("environment") or {}
    for k in ("physical_cores", "ram_gb", "machine_cpu_busy_percent_before_run"):
        if env.get(k) is None:
            problems.append(f"environment.{k} missing")
    started = (d.get("server") or {}).get("mode") == "started by the harness"
    pid_known = started or bool((d.get("server") or {}).get("server_pid"))
    for r in d.get("levels", []):
        if pid_known and "server_cpu_percent" not in r:
            problems.append(f"c={r['concurrency']}: no server CPU")
        if "machine_cpu_percent" not in r:
            problems.append(f"c={r['concurrency']}: no machine CPU")
    if started and "error" in (d["server"].get("openmp_probe") or {}):
        problems.append("OpenMP probe failed")
    return problems


def table(paths: list[str]) -> str:
    """Markdown table straight from result files, so a README can never quote
    a number the JSON does not contain."""
    files = sorted({f for pat in paths for f in glob.glob(pat)})
    lines = [
        "| run | arm | workers | git | c | req/s | p50 ms | p95 ms | p99 ms | errors "
        "| server CPU % | machine CPU % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in files:
        d = json.loads(Path(f).read_text())
        sha = (d.get("provenance") or {}).get("git_sha") or "-"
        dirty = (d.get("provenance") or {}).get("git_dirty_outside_artifacts")
        git = sha[:7] + ("+dirty" if dirty else "")
        for r in d["levels"]:
            lines.append(
                f"| {Path(f).stem} | {d.get('arm', '-')} "
                f"| {d['environment']['uvicorn_workers']} | {git} "
                f"| {r['concurrency']} | {r['req_per_s']:,.1f} | {r.get('p50_ms', '-')} "
                f"| {r.get('p95_ms', '-')} | {r.get('p99_ms', '-')} | {r['errors']} "
                f"| {r.get('server_cpu_percent', '-')} | {r.get('machine_cpu_percent', '-')} |"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# A/B: both arms, alternating, repeated; one command regenerates both halves
# --------------------------------------------------------------------------

def ab_order(arms: list[str], round_no: int) -> list[str]:
    """ABBA: odd rounds run the arms in order, even rounds reversed, so a slow
    drift in background load does not always land on the same arm."""
    return list(arms) if round_no % 2 else list(reversed(arms))


def ab_summary(files: list[Path]) -> dict[str, Any]:
    """Per (endpoint, workers, concurrency, arm): min / median / max req/s and
    p99 across rounds, and the two ratios a claim can use: median over median,
    and the worst case (slowest shipped run over fastest before run)."""
    runs: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    shas, busy = set(), []
    for f in files:
        d = json.loads(f.read_text())
        shas.add((d.get("provenance") or {}).get("git_sha"))
        busy.append(d["environment"].get("machine_cpu_busy_percent_before_run"))
        w = int(d["environment"]["uvicorn_workers"])
        for r in d["levels"]:
            runs[(d["endpoint"], w, r["concurrency"], d["arm"])].append(r)
    arms = list(ARMS)
    shipped, before = arms[0], arms[1]
    rows = []
    for ep, w, c in sorted({k[:3] for k in runs}):
        row: dict[str, Any] = {"endpoint": ep, "workers": w, "concurrency": c}
        for a in arms:
            rs = runs.get((ep, w, c, a), [])
            tput = [r["req_per_s"] for r in rs]
            p99 = [r["p99_ms"] for r in rs if r.get("p99_ms") is not None]
            row[a] = {
                "runs": len(rs),
                "req_per_s": [min(tput), statistics.median(tput), max(tput)] if tput else None,
                "p99_ms": [min(p99), statistics.median(p99), max(p99)] if p99 else None,
                "errors": sum(r["errors"] for r in rs),
                "server_cpu_percent": [r.get("server_cpu_percent") for r in rs],
                "machine_cpu_percent": [r.get("machine_cpu_percent") for r in rs],
            }
        a_t, b_t = row[shipped]["req_per_s"], row[before]["req_per_s"]
        if a_t and b_t and b_t[1] > 0 and b_t[2] > 0:
            row["median_ratio"] = round(a_t[1] / b_t[1], 2)
            row["worst_case_ratio"] = round(a_t[0] / b_t[2], 2)
        rows.append(row)
    return {
        "arms": {a: ARMS[a]["what"] for a in arms},
        "files": sorted(f.name for f in files),
        "git_shas": sorted(s for s in shas if s),
        "machine_cpu_busy_percent_before_runs": [min(busy), max(busy)] if busy else None,
        "rows": rows,
    }


def ab_table(summary: dict[str, Any]) -> str:
    shipped, before = list(ARMS)
    lines = [
        f"| endpoint | workers | c | {shipped} req/s min / median / max | {shipped} p99 ms median "
        f"| {before} req/s min / median / max | {before} p99 ms median | median ratio "
        "| worst-case ratio |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def trio(v: list[float] | None) -> str:
        return " / ".join(f"{x:,.1f}" for x in v) if v else "-"

    for r in summary["rows"]:
        a, b = r[shipped], r[before]
        lines.append(
            f"| {r['endpoint']} | {r['workers']} | {r['concurrency']} | {trio(a['req_per_s'])} "
            f"| {a['p99_ms'][1] if a['p99_ms'] else '-'} | {trio(b['req_per_s'])} "
            f"| {b['p99_ms'][1] if b['p99_ms'] else '-'} | {r.get('median_ratio', '-')} "
            f"| {r.get('worst_case_ratio', '-')} |"
        )
    return "\n".join(lines)


def validate_args(
    ab_rounds: int, url: str | None, extra_env: dict[str, str], ab_dir: Path | None,
) -> str | None:
    """Combinations that would write a result whose labels are wrong."""
    owned = [k for k in extra_env if k in THREAD_VARS or k == "WEB_CONCURRENCY"]
    if owned:
        return (f"--env may not set {', '.join(owned)}: the arm sets the thread variables "
                "and --workers sets WEB_CONCURRENCY, and the result labels follow them")
    if ab_rounds and url:
        return ("--ab-rounds starts its own server per arm; against --url every arm would be "
                "the same server (scripts/loadtest_linux.sh runs an external A/B)")
    if ab_rounds and ab_dir is not None and any(
        p.name != "summary.json" for p in ab_dir.glob("*.json")
    ):
        return (f"{ab_dir} already holds results; the summary would mix them with this run. "
                "Use an empty --ab-dir")
    return None


def _maybe_json(value: str) -> Any:
    """--meta values that are JSON (a probe's output) are stored as JSON."""
    try:
        return json.loads(value)
    except ValueError:
        return value


def main() -> None:
    ap = argparse.ArgumentParser(description="HTTP load test for the scoring service.")
    ap.add_argument("--endpoint", default="predict", choices=["predict", "batch"],
                    help="predict = POST /predict, batch = POST /predict/batch")
    ap.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)),
                    help="comma-separated concurrency levels")
    ap.add_argument("--workers", type=int, default=1,
                    help="uvicorn worker processes (set through WEB_CONCURRENCY)")
    ap.add_argument("--arm", default=next(iter(ARMS)), choices=list(ARMS),
                    help="which side of the A/B to serve; the default is what ships")
    ap.add_argument("--http", default="auto", choices=["auto", "h11", "httptools"])
    ap.add_argument("--warmup", type=float, default=2.0, help="seconds discarded per level")
    ap.add_argument("--duration", type=float, default=10.0, help="measured seconds per level")
    ap.add_argument("--client-procs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--payloads", type=int, default=2000, help="distinct held-out rows used")
    ap.add_argument("--url", default=None, help="target an already-running server instead")
    ap.add_argument("--server-pid", type=int, default=None,
                    help="with --url: PID of that server, so its CPU is measured")
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VALUE added to the server's environment (repeatable)")
    ap.add_argument("--meta", action="append", default=[],
                    help="KEY=VALUE recorded in the result as-is (repeatable)")
    ap.add_argument("--pyspy-at", type=int, default=None, metavar="C",
                    help="record a py-spy --gil profile of the server at concurrency C")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None,
                    help="JSON path (default artifacts/loadtest_<label>.json)")
    ap.add_argument("--ab-rounds", type=int, default=0,
                    help="run every arm this many times, alternating (ABBA), into --ab-dir")
    ap.add_argument("--ab-workers", default="1,4", help="worker counts for --ab-rounds")
    ap.add_argument("--ab-dir", default=None,
                    help="A/B output dir (default artifacts/loadtest_ab/<endpoint>)")
    ap.add_argument("--ab-summary", default=None, metavar="DIR",
                    help="summarize an A/B dir into DIR/summary.json + a table, and exit")
    ap.add_argument("--table", nargs="+", default=None, metavar="JSON",
                    help="print a markdown table from result files (globs ok) and exit")
    ap.add_argument("--check", nargs="+", default=None, metavar="JSON",
                    help="fail if a result file is missing provenance or CPU fields")
    ap.add_argument("--profile-summary", default=None, metavar="RAW",
                    help="recompute bucket shares from a committed py-spy raw file")
    ap.add_argument("--probe-openmp", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.probe_openmp:
        print(json.dumps(probe_openmp()))
        return
    if args.table:
        print(table(args.table))
        return
    if args.profile_summary:
        print(json.dumps(profile_shares(Path(args.profile_summary), PROFILE_BUCKETS), indent=2))
        return
    if args.check:
        bad = 0
        for f in sorted({f for pat in args.check for f in glob.glob(pat)}):
            probs = check_result(json.loads(Path(f).read_text()))
            print(f"[check] {f}: {'ok' if not probs else '; '.join(probs)}")
            bad += bool(probs)
        if bad:
            raise SystemExit(1)
        return
    if args.ab_summary:
        d = Path(args.ab_summary)
        files = sorted(p for p in d.glob("*.json") if p.name != "summary.json")
        summary = ab_summary(files)
        (d / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(ab_table(summary))
        return

    extra_env = dict(kv.split("=", 1) for kv in args.env)
    meta = {k: _maybe_json(v) for k, v in (kv.split("=", 1) for kv in args.meta)}
    endpoint = {"predict": "/predict", "batch": "/predict/batch"}[args.endpoint]
    common: dict[str, Any] = {
        "endpoint": endpoint, "levels": [int(x) for x in args.levels.split(",")],
        "http": args.http, "warmup_s": args.warmup, "measure_s": args.duration,
        "client_procs": args.client_procs, "batch_size": args.batch_size,
        "n_payloads": args.payloads, "url": args.url, "extra_env": extra_env,
        "server_pid": args.server_pid, "meta": meta,
    }
    ab_dir = Path(args.ab_dir) if args.ab_dir else (
        get_settings().artifact_dir / "loadtest_ab" / args.endpoint
    )
    problem = validate_args(args.ab_rounds, args.url, extra_env, ab_dir)
    if problem:
        raise SystemExit(problem)
    if args.ab_rounds:
        ab_dir.mkdir(parents=True, exist_ok=True)
        for rnd in range(1, args.ab_rounds + 1):
            for w in [int(x) for x in args.ab_workers.split(",")]:
                for arm in ab_order(list(ARMS), rnd):
                    label = f"{arm}_w{w}_r{rnd}"
                    res = run(workers=w, label=label, arm=arm, **common)
                    res["ab_round"] = rnd
                    (ab_dir / f"{label}.json").write_text(json.dumps(res, indent=2) + "\n")
                    print(f"[save] {ab_dir / (label + '.json')}")
        files = sorted(p for p in ab_dir.glob("*.json") if p.name != "summary.json")
        summary = ab_summary(files)
        (ab_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(ab_table(summary))
        return

    label = args.label or f"{args.endpoint}_w{args.workers}_{args.arm}"
    result = run(workers=args.workers, label=label, arm=args.arm,
                 pyspy_at=args.pyspy_at, **common)
    out_path = Path(args.out) if args.out else (
        get_settings().artifact_dir / f"loadtest_{label}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
