#!/usr/bin/env python
"""Require a regenerated JSON file to equal its committed version, field by field,
except for named wall-clock fields.
Run:  python scripts/same_json.py artifacts/comparison_ulb_creditcard.json

compare.py and compare_boosters.py stamp measured_at and time every fit, so their
reports can never be byte-identical to the committed ones and `git diff
--exit-code` cannot check them. This compares the working-tree file with
`git show HEAD:<file>` and ignores only those fields (and the Python patch
version, which setup-python does not pin). Every metric, interval, selection and
count must match exactly. Exit 1 on any difference, listing each one.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

IGNORED_KEYS = {"measured_at", "fit_seconds"}
IGNORED_PATHS = {"/environment/python"}


def differences(old: Any, new: Any, path: str = "") -> list[str]:
    if path in IGNORED_PATHS:
        return []
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new)):
            if k in IGNORED_KEYS:
                continue
            if k not in old or k not in new:
                out.append(f"{path}/{k}: only in {'new' if k in new else 'committed'}")
                continue
            out.extend(differences(old[k], new[k], f"{path}/{k}"))
        return out
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}: {len(old)} items committed, {len(new)} regenerated"]
        return [d for i, (a, b) in enumerate(zip(old, new, strict=True))
                for d in differences(a, b, f"{path}[{i}]")]
    return [] if old == new else [f"{path}: committed {old!r}, regenerated {new!r}"]


def main(files: list[str]) -> int:
    failed = False
    for f in files:
        committed = subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True,
                                   text=True, check=True).stdout
        diffs = differences(json.loads(committed), json.loads(Path(f).read_text()))
        for d in diffs:
            print(f"[diff] {f}{d}")
        print(f"[check] {f}: {'MISMATCH' if diffs else 'reproduced'} "
              f"(ignoring {sorted(IGNORED_KEYS | IGNORED_PATHS)})")
        failed |= bool(diffs)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
