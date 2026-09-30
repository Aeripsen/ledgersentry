"""Turn the raw output of scripts/k8s_e2e.sh and scripts/tf_kind_e2e.sh into one
JSON artifact each, and fail the CI job when the run shows a real problem.

    python scripts/k8s_report.py kind --app ledgersentry --out-dir k8s-run \\
        --artifact artifacts/k8s_kind_ledgersentry.json
    python scripts/k8s_report.py terraform --out-dir tf-run \\
        --artifact artifacts/terraform_kind_ledgersentry.json

Every number in the artifact is read from files the run wrote (k6's own summary,
kubectl output, timestamps); nothing is estimated here. Stdlib only, so it runs
on a bare CI runner.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

CAVEATS = [
    "Single-node kind cluster on one GitHub Actions runner. The k6 load generator "
    "runs in the same cluster and competes for the same CPUs as the pods.",
    "Synthetic load: one fixed request body sent in a closed loop. Not production "
    "traffic, not users.",
    "Throughput and latency vary between runners; the gate is on failed requests, "
    "not on a throughput number.",
]


def _kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def _k6(path: Path) -> dict[str, Any]:
    for line in path.read_text().splitlines():
        if line.startswith("K6_SUMMARY "):
            s: dict[str, Any] = json.loads(line[len("K6_SUMMARY "):])
            s["failed_total"] = (
                s["failed_5xx"] + s["failed_other_status"] + s["failed_connection"]
            )
            s["req_per_s"] = round(s["req_per_s"], 1)
            s["latency_ms"] = {k: round(v, 2) for k, v in s["latency_ms"].items()}
            return s
    raise SystemExit(f"no K6_SUMMARY line in {path}")


def _mem_mi(v: str) -> float:
    m = re.fullmatch(r"(\d+)(Ki|Mi|Gi)", v)
    if not m:
        return float("nan")
    n, unit = int(m.group(1)), m.group(2)
    return {"Ki": n / 1024, "Mi": float(n), "Gi": n * 1024.0}[unit]


def _top(path: Path) -> list[dict[str, Any]]:
    pods = []
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3:
            pods.append({
                "pod": parts[0],
                "cpu_millicores": int(parts[1].rstrip("m")),
                "memory_mi": round(_mem_mi(parts[2]), 1),
            })
    return pods


def _hpa(path: Path, t0: int) -> dict[str, Any]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1].isdigit():
            util = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
            rows.append([int(parts[0]) - t0, int(parts[1]), int(parts[2]), util])
    utils = [r[3] for r in rows if r[3] is not None]
    top = max((r[2] for r in rows), default=None)
    return {
        "columns": ["seconds_since_steady_load_start", "current_replicas",
                    "desired_replicas", "cpu_percent_of_request"],
        "replicas_min_seen": min((r[1] for r in rows), default=None),
        "replicas_max_seen": max((r[1] for r in rows), default=None),
        # When the HPA decided on its largest size, and when that many pods were
        # running. The log is sampled every 5 s, so each is the first sample
        # that showed it, up to 5 s after it happened.
        "desired_max_first_seen_s": next((r[0] for r in rows if r[2] == top), None),
        "current_max_first_seen_s": next((r[0] for r in rows if r[1] == top), None),
        # averageUtilization is usage / request. With a 1000m limit on a 250m
        # request it cannot pass 400: a value near 400 means the pods sat at
        # their CPU limit (throttled), not that they did 4x the work asked.
        "cpu_percent_max_seen": max(utils, default=None),
        "timeline": rows,
    }


def _hpa_epochs(path: Path) -> list[tuple[int, int, int]]:
    """(epoch, current, desired) for every sample of hpa.log."""
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1].isdigit():
            out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return out


def _seconds(duration: str) -> int:
    m = re.fullmatch(r"(?:(\d+)m)?(?:(\d+)s)?", duration)
    if not m or not duration:
        raise SystemExit(f"cannot read k6 duration {duration!r}")
    return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)


def _windows(k6: dict[str, Any],
             split: list[tuple[str, float | None, float | None]]) -> dict[str, Any]:
    """Requests per 10 s from k6, grouped into windows of wall-clock time.

    split: (name, not_before_epoch, not_after_epoch). A bucket belongs to a
    window only if it lies entirely inside it, and only full buckets (inside
    the phase duration) count, so no rate is diluted by a partial bucket."""
    t0 = k6["t0_epoch_ms"] / 1000
    width = k6["bucket_seconds"]
    dur = _seconds(k6["duration"])
    out: dict[str, Any] = {}
    for name, lo, hi in split:
        rows = [b for b in k6["buckets"] if b[0] + width <= dur
                and (lo is None or t0 + b[0] >= lo)
                and (hi is None or t0 + b[0] + width <= hi)]
        n = sum(b[1] for b in rows)
        out[name] = {
            "full_buckets": len(rows),
            "seconds_into_phase": [rows[0][0], rows[-1][0] + width] if rows else None,
            "requests": n,
            "failed": sum(b[2] for b in rows),
            "req_per_s": round(n / (len(rows) * width), 1) if rows else None,
        }
    return out


def _pod_counts(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                out[parts[0]] = int(parts[1])
    return out


def _spread(counts: dict[str, int]) -> dict[str, Any]:
    total = sum(counts.values())
    return {
        "requests_by_pod": dict(sorted(counts.items())),
        "total": total,
        # pods that served at least 5% of the window's requests
        "pods_serving": sum(1 for v in counts.values() if total and v >= 0.05 * total),
        "pods_running": len(counts),
    }


def _max_batch_in_openapi(openapi: dict[str, Any]) -> int | None:
    """maxItems of POST /predict/batch's list field, read from the live schema."""
    body = openapi["paths"]["/predict/batch"]["post"]["requestBody"]
    ref = body["content"]["application/json"]["schema"]["$ref"]
    schema = openapi["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    for prop in schema.get("properties", {}).values():
        if prop.get("type") == "array" and "maxItems" in prop:
            return int(prop["maxItems"])
    return None


def _pid1_is_uvicorn(cmdline: str) -> bool:
    """/proc/1/cmdline of a console-script launch reads `python .../uvicorn ...`.
    A shell as PID 1 (`sh -c uvicorn ...`) is the case this guards against."""
    argv = cmdline.split()
    return bool(argv) and not argv[0].endswith("sh") and any(
        Path(a).name == "uvicorn" for a in argv[:2]
    )


def kind_report(args: argparse.Namespace) -> int:
    d = Path(args.out_dir)
    env = _kv(d / "env.txt")
    t = {k: int(v) for k, v in _kv(d / "times.env").items()}
    steady, restart = _k6(d / "k6-steady.log"), _k6(d / "k6-restart.log")
    cm = json.loads((d / "configmap.json").read_text())["data"]
    cm_batch = next((int(v) for k, v in cm.items() if k.endswith("MAX_BATCH")
                     or k.endswith("MAX_BATCH_ROWS")), None)
    seen_batch = _max_batch_in_openapi(json.loads((d / "openapi.json").read_text()))
    before = set((d / "pods_before.txt").read_text().split())
    after = set((d / "pods_after.txt").read_text().split())
    pid1 = (d / "pid1.txt").read_text().strip()

    hpa_rows = _hpa_epochs(d / "hpa.log")
    top = max((r[2] for r in hpa_rows), default=0)
    low = min((r[1] for r in hpa_rows), default=0)
    # last sample still at the starting size, first sample at the largest size
    last_low = max((r[0] for r in hpa_rows if r[1] == low), default=None)
    first_top = next((r[0] for r in hpa_rows if r[1] == top), None)
    if steady.get("buckets"):
        steady["by_window"] = _windows(steady, [
            ("before_scale_out", None, last_low),
            ("all_pods_running", first_top, None),
        ])
    if restart.get("buckets"):
        restart["by_window"] = _windows(restart, [
            ("before_restart", None, t["restart_start"]),
            ("during_restart", t["restart_start"], t["restart_end"] + 1),
            ("after_restart", t["restart_end"] + 1, None),
        ])

    # Requests each pod served, from its uvicorn access log. steady_end counts
    # the whole steady phase (plus the one smoke-test /predict); the pre-restart
    # window is the difference between the two counts.
    at_steady_end = _pod_counts(d / "pod_requests_steady_end.txt")
    at_pre = _pod_counts(d / "pod_requests_pre_restart.txt")
    pods_served: dict[str, Any] | None = None
    if at_steady_end and at_pre:
        window = {p: n - at_steady_end.get(p, 0) for p, n in at_pre.items()}
        pods_served = {
            "steady_phase": _spread(at_steady_end),
            "restart_phase_before_restart": _spread(window),
            "counted_seconds_into_restart_phase":
                t["pre_restart_count_at"] - t["restart_load_start"],
            # Log counts are only usable if nothing was lost: kubectl logs
            # returns the current log file only, so a rotation drops lines. The
            # steady total must match k6's count (the smoke-test /predict can
            # add one), and no pod's count may go down between the two samples.
            "complete": {
                "steady_phase": abs(sum(at_steady_end.values()) - steady["requests"]) <= 2,
                "restart_phase_before_restart": all(v >= 0 for v in window.values()),
            },
        }
        # Replacement pods only ever served the restart phase, so their counts
        # show where the clients' connections ended up after the rollout.
        at_end = _pod_counts(d / "pod_requests_restart_end.txt")
        if at_end:
            pods_served["replacement_pods_whole_restart_phase"] = _spread(
                {p: n for p, n in at_end.items() if p in after})
            # Not checkable: the old pods' share of this phase is gone with
            # them, so there is no total to compare with. The 200Mi log limit in
            # deploy/k8s/kind-cluster.yaml is what keeps these counts whole.
            pods_served["complete"]["replacement_pods_whole_restart_phase"] = None

    pdb_kv = _kv(d / "pdb.txt")
    pdb = None
    if pdb_kv:
        pdb = {
            "first_eviction_allowed": pdb_kv.get("first_rc") == "0",
            "second_eviction_refused": pdb_kv.get("second_rc") != "0"
            and "disruption budget" in pdb_kv.get("second_out", ""),
            "second_eviction_message": pdb_kv.get("second_out", "").strip(),
        }

    restart["rolling_restart"] = {
        "started_seconds_into_load": t["restart_start"] - t["restart_load_start"],
        "rollout_seconds": t["restart_end"] - t["restart_start"],
        "finished_before_load_ended": t["restart_end"] <= t["restart_load_end"],
        "pods_before": sorted(before),
        "pods_after": sorted(after),
        "every_pod_replaced": bool(after) and not (before & after),
    }

    report: dict[str, Any] = {
        "what": f"{args.app} deployed to a kind cluster from deploy/k8s/overlays/kind, "
                "smoke-tested, load-tested through its Service, and rolled while under load.",
        "environment": {
            "runner_vcpu": int(env.get("vcpu", "0")),
            "runner_memory_mb": int(env.get("mem_mb", "0")),
            "kind": env.get("kind_version"),
            "kubernetes": env.get("kubernetes_version"),
            "git_sha": env.get("git_sha"),
            "run_url": env.get("run_url"),
            "drain": env.get("drain", "on"),
        },
        "deploy": {
            "initial_rollout_seconds": t.get("rollout_seconds"),
            "ready": json.loads((d / "ready.json").read_text()),
            "predict_example": json.loads((d / "predict.json").read_text()),
            "pid1": pid1,
            "runs_as": (d / "id.txt").read_text().strip(),
            "configmap_max_batch": cm_batch,
            "max_batch_seen_in_openapi": seen_batch,
            "configmap_reaches_process": cm_batch is not None and cm_batch == seen_batch,
        },
        "load_steady": steady,
        "load_during_rolling_restart": restart,
        "hpa": _hpa(d / "hpa.log", t["steady_load_start"]),
        "pods_after_steady_load": _top(d / "top.txt"),
        "requests_by_pod": pods_served,
        "pod_disruption_budget": pdb,
        "caveats": CAVEATS,
    }

    problems = []
    if steady["failed_total"]:
        problems.append(f"steady load: {steady['failed_total']} failed requests")
    if restart["failed_total"]:
        problems.append(f"rolling restart: {restart['failed_total']} failed requests")
    if not restart["rolling_restart"]["finished_before_load_ended"]:
        problems.append("rollout finished after the load ended, so it was not tested under load")
    if not restart["rolling_restart"]["every_pod_replaced"]:
        problems.append("rolling restart did not replace every pod")
    if not report["deploy"]["configmap_reaches_process"]:
        problems.append(f"ConfigMap max batch {cm_batch} != served schema {seen_batch}")
    if not _pid1_is_uvicorn(pid1):
        problems.append(f"PID 1 is not uvicorn: {pid1!r}")
    if pdb is not None and not (pdb["first_eviction_allowed"] and pdb["second_eviction_refused"]):
        problems.append(f"PodDisruptionBudget check did not hold: {pdb}")
    report["gate"] = {"enforced": not args.no_gate, "problems": problems}
    Path(args.artifact).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("deploy", "load_steady",
                                             "load_during_rolling_restart",
                                             "requests_by_pod", "pod_disruption_budget",
                                             "gate")}, indent=2))
    for p in problems:
        print("RECORDED (no gate):" if args.no_gate else "FAIL:", p, file=sys.stderr)
    return 1 if problems and not args.no_gate else 0


def terraform_report(args: argparse.Namespace) -> int:
    d = Path(args.out_dir)
    kv = _kv(d / "tf.env")
    rc = {k: int(v) for k, v in kv.items() if k.endswith("_rc")}
    report = {
        "what": "deploy/terraform/kubernetes applied to a kind cluster with the "
                "hashicorp/kubernetes provider, checked, then destroyed.",
        "environment": {k: kv.get(k) for k in ("terraform_version", "provider_version",
                                                 "kubernetes_version", "git_sha", "run_url")},
        "apply_exit_code": rc.get("apply_rc"),
        "resources_created": int(kv.get("resources_created", "0")),
        "apply_seconds": int(kv.get("apply_seconds", "0")),
        # terraform plan -detailed-exitcode: 0 = no changes, 2 = changes pending
        "second_plan_exit_code": rc.get("plan_rc"),
        "second_plan_is_empty": rc.get("plan_rc") == 0,
        "ready": json.loads((d / "ready.json").read_text()),
        "predict_example": json.loads((d / "predict.json").read_text()),
        "parity_with_yaml_exit_code": rc.get("parity_rc"),
        "destroy_exit_code": rc.get("destroy_rc"),
        "not_done": "No cloud apply. deploy/terraform/cloudrun is fmt-checked and "
                    "validated in CI only; applying it needs a GCP project with billing.",
    }
    Path(args.artifact).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    bad = [k for k in ("apply_rc", "plan_rc", "parity_rc", "destroy_rc") if rc.get(k) != 0]
    for k in bad:
        print(f"FAIL: {k}={rc.get(k)}", file=sys.stderr)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kind")
    k.add_argument("--app", required=True)
    k.add_argument("--out-dir", required=True)
    k.add_argument("--artifact", required=True)
    k.add_argument("--no-gate", action="store_true",
                   help="record problems in the artifact but exit 0 (drain A/B runs)")
    tf = sub.add_parser("terraform")
    tf.add_argument("--out-dir", required=True)
    tf.add_argument("--artifact", required=True)
    args = ap.parse_args()
    return kind_report(args) if args.cmd == "kind" else terraform_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
