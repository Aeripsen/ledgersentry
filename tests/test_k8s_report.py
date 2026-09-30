"""scripts/k8s_report.py turns k6's per-10 s counts into before / during / after
windows and the access-log counts into requests per pod. DEPLOY.md quotes those
numbers, so the arithmetic is pinned here on hand-made inputs."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("k8s_report", REPO / "scripts" / "k8s_report.py")
assert _spec and _spec.loader
k8s_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(k8s_report)


def _k6(buckets: list[list[int]], duration: str = "60s") -> dict:
    return {"t0_epoch_ms": 1_000_000, "bucket_seconds": 10, "duration": duration,
            "buckets": buckets}


def test_windows_keep_only_full_buckets_inside_each_window() -> None:
    # t0 = 1000 s. Buckets start at 1000, 1010, ..., 1060 (the last one is the
    # graceful-stop tail past the 60 s duration and must be ignored).
    k6 = _k6([[0, 100, 0], [10, 100, 0], [20, 300, 1], [30, 500, 0],
              [40, 500, 0], [50, 500, 0], [60, 7, 0]])
    w = k8s_report._windows(k6, [
        ("before", None, 1020),       # buckets 0 and 10
        ("during", 1020, 1036),       # only bucket 20 lies fully inside
        ("after", 1036, None),        # buckets 40 and 50; 30 straddles 1036
    ])
    assert w["before"] == {"full_buckets": 2, "seconds_into_phase": [0, 20],
                           "requests": 200, "failed": 0, "req_per_s": 10.0}
    assert w["during"]["requests"] == 300 and w["during"]["failed"] == 1
    assert w["after"] == {"full_buckets": 2, "seconds_into_phase": [40, 60],
                          "requests": 1000, "failed": 0, "req_per_s": 50.0}


def test_empty_window_reports_none_rate() -> None:
    w = k8s_report._windows(_k6([[0, 5, 0]]), [("x", 2000, None)])
    assert w["x"]["full_buckets"] == 0 and w["x"]["req_per_s"] is None


def test_spread_counts_pods_with_at_least_5_percent() -> None:
    s = k8s_report._spread({"a": 1000, "b": 900, "c": 30, "d": 0})
    assert s["total"] == 1930
    assert s["pods_serving"] == 2
    assert s["pods_running"] == 4


def test_durations() -> None:
    assert k8s_report._seconds("120s") == 120
    assert k8s_report._seconds("2m30s") == 150
