"""The load-test harness's own logic: the knee rule, the summary, the response
reader, the Dockerfile ENV parser, the server environment, the py-spy share
computation, the A/B summary and the README table. The end-to-end run is the
CI smoke step (ci.yml), which starts a real server."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ledgersentry import loadtest as lt


def _lvl(c: int, rps: float, p99: float = 5.0, errors: int = 0) -> dict:
    return {"concurrency": c, "req_per_s": rps, "p50_ms": p99 / 2, "p99_ms": p99,
            "errors": errors}


def test_knee_is_first_level_whose_doubling_buys_under_ten_percent() -> None:
    levels = [_lvl(1, 100), _lvl(2, 190), _lvl(4, 205), _lvl(8, 400)]
    knee = lt.find_knee(levels)
    assert knee is not None
    assert knee["concurrency"] == 2  # 205 / 190 = 1.079 < 1.10
    assert knee["next_level_gain"] == 1.079


def test_knee_is_none_while_every_doubling_still_pays() -> None:
    assert lt.find_knee([_lvl(1, 100), _lvl(2, 200), _lvl(4, 400)]) is None


def test_knee_skips_a_level_with_zero_throughput() -> None:
    knee = lt.find_knee([_lvl(1, 0), _lvl(2, 100), _lvl(4, 105)])
    assert knee is not None and knee["concurrency"] == 2


def test_summary_budget_row_excludes_levels_with_errors_or_slow_p99() -> None:
    levels = [_lvl(1, 100, 4.0), _lvl(4, 300, 9.0, errors=1), _lvl(8, 350, 20.0)]
    s = lt.summarize(levels)
    assert s["peak"]["concurrency"] == 8
    assert s["best_with_p99_under_budget"]["concurrency"] == 1
    assert s["total_errors"] == 1


def _read(raw: bytes) -> int:
    async def go() -> int:
        reader = asyncio.StreamReader()
        reader.feed_data(raw)
        reader.feed_eof()
        return await lt._read_response(reader)

    return asyncio.run(go())


def test_read_response_consumes_exactly_one_response() -> None:
    body = b'{"ok":true}'
    raw = (b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
           b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    assert _read(raw) == 200


def test_read_response_rejects_a_response_without_content_length() -> None:
    with pytest.raises(ValueError):
        _read(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")


def test_read_response_raises_on_a_closed_connection() -> None:
    with pytest.raises(ConnectionError):
        _read(b"")


def test_dockerfile_env_ignores_comments_and_takes_the_last_value(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM python:3.12-slim\n"
        "# ENV OMP_NUM_THREADS=1\n"
        "ENV A=1 B=\"two words\"\n"
        "ENV LEGACY some value\n"
        "ENV C=1 \\\n"
        "    # a comment inside a continuation\n"
        "    D=4\n"
        "ENV A=3\n"
    )
    env = lt.dockerfile_env(df)
    assert "OMP_NUM_THREADS" not in env
    assert env == {"A": "3", "B": "two words", "LEGACY": "some value", "C": "1", "D": "4"}


def test_dockerfile_env_sees_an_override_after_the_pin(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("ENV OMP_NUM_THREADS=1\nENV OMP_NUM_THREADS=6\n")
    assert lt.dockerfile_thread_env(df) == {"OMP_NUM_THREADS": "6"}


def test_server_env_never_inherits_thread_or_worker_counts() -> None:
    base = {"PATH": "/bin", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
            "WEB_CONCURRENCY": "9"}
    env = lt.server_env(4, {}, {}, base=base)
    assert env == {"PATH": "/bin", "WEB_CONCURRENCY": "4"}
    env = lt.server_env(1, {"OMP_NUM_THREADS": "1"}, {"X": "y"}, base=base)
    assert env["OMP_NUM_THREADS"] == "1" and env["WEB_CONCURRENCY"] == "1" and env["X"] == "y"


def test_arms_the_first_is_what_ships() -> None:
    first, second = list(lt.ARMS)
    assert lt.ARMS[first]["unset_thread_env"] is False
    assert lt.ARMS[second]["unset_thread_env"] is True


def test_ab_order_alternates() -> None:
    assert lt.ab_order(["a", "b"], 1) == ["a", "b"]
    assert lt.ab_order(["a", "b"], 2) == ["b", "a"]


def test_profile_shares_from_raw_py_spy_output(tmp_path: Path) -> None:
    raw = tmp_path / "p.txt"
    raw.write_text(
        "run (uvicorn/main.py:1);predict (ledgersentry/scoring.py:9);"
        "predict (sklearn/ensemble/x.py:3) 30\n"
        "run (uvicorn/main.py:1);decode (json/decoder.py:5) 10\n"
        "run (uvicorn/main.py:1);emit (logging/__init__.py:2) 60\n"
    )
    s = lt.profile_shares(raw, {"scoring": r"ledgersentry[\\/]scoring\.py",
                                "json": r"json[\\/]decoder\.py"})
    assert s["samples"] == 100
    assert s["share_percent"] == {"scoring": 30.0, "json": 10.0}
    assert next(iter(s["top_leaf_frames_percent"])) == "emit (logging/__init__.py:2)"


def _result(arm: str, workers: int, rps: dict[int, float], rnd: int) -> dict:
    return {
        "arm": arm, "endpoint": "/predict", "ab_round": rnd,
        "provenance": {"git_sha": "abc1234" + "0" * 33, "git_dirty_outside_artifacts": False},
        "environment": {"uvicorn_workers": workers, "machine_cpu_busy_percent_before_run": 10.0},
        "levels": [dict(_lvl(c, v), server_cpu_percent=100.0, machine_cpu_percent=50.0)
                   for c, v in rps.items()],
    }


def test_ab_summary_reports_median_and_worst_case_ratios(tmp_path: Path) -> None:
    shipped, before = list(lt.ARMS)
    files = []
    for i, (a, v) in enumerate([(shipped, 300), (shipped, 280), (shipped, 320),
                                (before, 100), (before, 150), (before, 140)]):
        f = tmp_path / f"{i}.json"
        f.write_text(json.dumps(_result(a, 4, {4: v}, i)))
        files.append(f)
    s = lt.ab_summary(files)
    (row,) = s["rows"]
    assert row[shipped]["req_per_s"] == [280, 300, 320]
    assert row[before]["req_per_s"] == [100, 140, 150]
    assert row["median_ratio"] == round(300 / 140, 2)
    assert row["worst_case_ratio"] == round(280 / 150, 2)
    assert "| /predict | 4 | 4 | 280.0 / 300.0 / 320.0 |" in lt.ab_table(s)


def test_table_prints_every_level_with_arm_and_commit(tmp_path: Path) -> None:
    f = tmp_path / "r.json"
    f.write_text(json.dumps(_result(next(iter(lt.ARMS)), 1, {1: 250.0, 4: 260.5}, 1)))
    out = lt.table([str(f)])
    assert out.count("\n") == 3  # header, rule, 2 levels
    assert "abc1234" in out and "260.5" in out


def test_check_result_flags_missing_provenance_and_cpu() -> None:
    d = _result(next(iter(lt.ARMS)), 1, {1: 250.0}, 1)
    d["server"] = {"mode": "started by the harness", "openmp_probe": {}}
    d["environment"].update(physical_cores=6, ram_gb=16.0)
    assert lt.check_result(d) == []
    del d["levels"][0]["server_cpu_percent"]
    d["provenance"]["git_sha"] = None
    assert set(lt.check_result(d)) == {"no git SHA", "c=1: no server CPU"}


def test_shown_cmd_hides_pid_and_paths() -> None:
    cmd = ["C:/x/py-spy.exe", "record", "--pid", "123", "--output", "/tmp/a/b.txt", "--gil"]
    assert lt._shown_cmd(cmd) == "py-spy record --pid <server pid> --output b.txt --gil"


def test_validate_args_refuses_combinations_that_mislabel_results(tmp_path: Path) -> None:
    assert lt.validate_args(0, None, {}, tmp_path) is None
    assert lt.validate_args(3, None, {"X": "1"}, tmp_path) is None
    assert "OMP_NUM_THREADS" in lt.validate_args(3, None, {"OMP_NUM_THREADS": "1"}, tmp_path)
    assert "WEB_CONCURRENCY" in lt.validate_args(0, None, {"WEB_CONCURRENCY": "4"}, tmp_path)
    assert "--url" in lt.validate_args(1, "http://127.0.0.1:8000", {}, tmp_path)
    (tmp_path / "summary.json").write_text("{}")
    assert lt.validate_args(1, None, {}, tmp_path) is None  # a summary alone is not a result
    (tmp_path / "omp1_w1_r1.json").write_text("{}")
    assert "already holds results" in lt.validate_args(1, None, {}, tmp_path)
    assert lt.validate_args(0, None, {}, tmp_path) is None  # single runs do not summarize
