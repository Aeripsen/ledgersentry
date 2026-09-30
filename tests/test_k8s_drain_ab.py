"""scripts/k8s_drain_ab.py compares runs with and without the connection drain.
The unit is the run (one rolling restart), and the test on it is a one-sided
Fisher exact test; pinned here against values worked out by hand."""
import importlib.util
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("k8s_drain_ab", REPO / "scripts" / "k8s_drain_ab.py")
assert _spec and _spec.loader
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)


def test_fisher_matches_the_hypergeometric_by_hand() -> None:
    # 4 of 6 drain-off runs failed, 0 of 6 drain-on: all 4 failing runs in the
    # off group has probability C(6,4) C(6,0) / C(12,4) = 15 / 495.
    assert abs(ab.fisher_one_sided(4, 6, 0, 6) - 15 / 495) < 1e-12
    assert ab.fisher_one_sided(6, 6, 0, 6) == 1 / comb(12, 6)


def test_fisher_is_one_when_nothing_differs() -> None:
    assert ab.fisher_one_sided(0, 6, 0, 6) == 1.0
    assert ab.fisher_one_sided(2, 6, 2, 6) > 0.5
