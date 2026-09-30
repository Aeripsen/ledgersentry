"""Summarize a k8s-drain-ab workflow run: failed requests per rolling restart,
with the connection drain and without it.

    python scripts/k8s_drain_ab.py ab --app ledgersentry \
        --out artifacts/k8s_drain_ab_ledgersentry.json

`ab/` holds one directory per matrix job (ab-on-1, ab-off-3, ...), each with the
report.json that scripts/k8s_report.py wrote in --no-gate mode. A run counts only
if the test itself was valid: every pod replaced, the rollout finished while the
load was still running, and the other deploy checks held. Failed requests are
the outcome being compared, so they never make a run invalid.

Each run is one rolling restart, so the unit is the run: the summary reports how
many runs had any failed request, and the failures in each, not a per-request
rate pooled across runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _run(path: Path) -> dict[str, Any]:
    r = json.loads(path.read_text())
    restart = r["load_during_rolling_restart"]
    problems = [p for p in r.get("gate", {}).get("problems", [])
                if "failed requests" not in p]
    return {
        "job": path.parent.name,
        "drain": r["environment"].get("drain"),
        "run_url": r["environment"].get("run_url"),
        "git_sha": r["environment"].get("git_sha"),
        "restart_requests": restart["requests"],
        "restart_failed": restart["failed_total"],
        "restart_failed_connection": restart["failed_connection"],
        "steady_failed": r["load_steady"]["failed_total"],
        "valid": not problems,
        "invalid_because": problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ab_dir")
    ap.add_argument("--app", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = sorted((_run(p) for p in Path(args.ab_dir).glob("*/report.json")),
                  key=lambda x: (x["drain"] or "", x["job"]))
    if not runs:
        print("no report.json found", file=sys.stderr)
        return 1
    by: dict[str, dict[str, Any]] = {}
    for drain in ("on", "off"):
        rs = [x for x in runs if x["drain"] == drain and x["valid"]]
        by[drain] = {
            "valid_runs": len(rs),
            "runs_with_any_failed_request": sum(1 for x in rs if x["restart_failed"]),
            "failed_requests_per_run": [x["restart_failed"] for x in rs],
            "requests_per_run": [x["restart_requests"] for x in rs],
            "failed_requests_total": sum(x["restart_failed"] for x in rs),
            "requests_total": sum(x["restart_requests"] for x in rs),
        }
    out = {
        "what": f"{args.app}: scripts/k8s_e2e.sh repeated on fresh runners with the "
                "connection drain (preStop touches the drain file, responses get "
                "`Connection: close`) and without it (DRAIN=off, preStop is a plain "
                "sleep 5). One rolling restart of 5 pods under 16-VU closed-loop "
                "load per run.",
        "unit": "one run = one rolling restart; failures are counted per run",
        "by_drain": by,
        "runs": runs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(by, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
