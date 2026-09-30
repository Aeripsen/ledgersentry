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
import math
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
        "report": str(path.parent),
        "drain": r["environment"].get("drain"),
        "run_url": r["environment"].get("run_url"),
        "git_sha": r["environment"].get("git_sha"),
        "restart_requests": restart["requests"],
        "restart_failed": restart["failed_total"],
        "restart_failed_connection": restart["failed_connection"],
        "steady_failed": r["load_steady"]["failed_total"],
        "valid": not problems,
        "invalid_because": problems,
        # the same run's throughput windows and per-pod spread, so the
        # connection-balancing findings in DEPLOY.md rest on every run
        "steady_req_per_s": _rates(r["load_steady"]),
        "restart_req_per_s": _rates(restart),
        # only windows whose access-log counts add up (see k8s_report.py)
        "pods_serving": _pods_serving(r.get("requests_by_pod") or {}),
        "hpa_desired_max_first_seen_s": r["hpa"].get("desired_max_first_seen_s"),
        "hpa_current_max_first_seen_s": r["hpa"].get("current_max_first_seen_s"),
        "pdb_held": bool(r.get("pod_disruption_budget")
                         and r["pod_disruption_budget"]["first_eviction_allowed"]
                         and r["pod_disruption_budget"]["second_eviction_refused"]),
    }


def fisher_one_sided(fail_off: int, n_off: int, fail_on: int, n_on: int) -> float:
    """P(at least fail_off of the failing runs land in the drain-off group) if
    the drain made no difference: a one-sided Fisher exact test on the 2x2 table
    of runs (failed / clean) by drain setting, from the hypergeometric law."""
    k, n = fail_off + fail_on, n_off + n_on
    total = math.comb(n, k)
    return sum(math.comb(n_off, i) * math.comb(n_on, k - i)
               for i in range(fail_off, min(k, n_off) + 1)) / total


def _pods_serving(by_pod: dict[str, Any]) -> dict[str, int]:
    complete = by_pod.get("complete", {})
    return {k: v["pods_serving"] for k, v in by_pod.items()
            if isinstance(v, dict) and "pods_serving" in v and complete.get(k, False)}


def _rates(phase: dict[str, Any]) -> dict[str, Any]:
    return {k: v["req_per_s"] for k, v in phase.get("by_window", {}).items()}


def _describe(values: list[float]) -> dict[str, float] | None:
    return {"min": min(values), "median": sorted(values)[len(values) // 2],
            "max": max(values)} if values else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ab_dir")
    ap.add_argument("--app", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = sorted((_run(p) for p in Path(args.ab_dir).glob("**/report.json")),
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
    valid = [x for x in runs if x["valid"]]
    comparison = {
        "test": "one-sided Fisher exact on runs with any failed request, off vs on",
        "p_value": round(fisher_one_sided(
            by["off"]["runs_with_any_failed_request"], by["off"]["valid_runs"],
            by["on"]["runs_with_any_failed_request"], by["on"]["valid_runs"]), 4)
        if by["off"]["valid_runs"] and by["on"]["valid_runs"] else None,
    }

    def ratio(key: str, a: str, b: str) -> list[float]:
        return sorted(round(x[key][b] / x[key][a], 2) for x in valid
                      if x[key].get(a) and x[key].get(b))

    across = {
        "note": "every valid run, drain on and off alike; min / median / max",
        "steady_all_pods_running_over_before_scale_out": _describe(
            ratio("steady_req_per_s", "before_scale_out", "all_pods_running")),
        "restart_after_over_before": _describe(
            ratio("restart_req_per_s", "before_restart", "after_restart")),
        "pods_serving_steady_phase": sorted(
            x["pods_serving"]["steady_phase"] for x in valid
            if "steady_phase" in x["pods_serving"]),
        "pods_serving_fresh_client_before_restart": sorted(
            x["pods_serving"]["restart_phase_before_restart"] for x in valid
            if "restart_phase_before_restart" in x["pods_serving"]),
        "hpa_desired_max_first_seen_s": sorted(
            x["hpa_desired_max_first_seen_s"] for x in valid
            if x["hpa_desired_max_first_seen_s"] is not None),
        "hpa_current_max_first_seen_s": sorted(
            x["hpa_current_max_first_seen_s"] for x in valid
            if x["hpa_current_max_first_seen_s"] is not None),
        "pdb_held": sum(1 for x in valid if x["pdb_held"]),
    }
    out = {
        "what": f"{args.app}: scripts/k8s_e2e.sh repeated on fresh runners with the "
                "connection drain (preStop touches the drain file, responses get "
                "`Connection: close`) and without it (DRAIN=off, preStop is a plain "
                "sleep 5). One rolling restart of 5 pods under 16-VU closed-loop "
                "load per run.",
        "unit": "one run = one rolling restart; failures are counted per run",
        "by_drain": by,
        "runs_compared": comparison,
        "across_runs": across,
        "runs": runs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"by_drain": by, "runs_compared": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
