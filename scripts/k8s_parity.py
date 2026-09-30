"""Check that the Terraform module and the YAML manifest describe the same
Kubernetes objects, so the two deploy paths cannot drift apart.

    kubectl kustomize deploy/k8s/base \\
      | kubectl apply -n parity --dry-run=server -o json -f - > yaml.json
    kubectl -n ledgersentry get deploy,svc,hpa,pdb,cm -o json > tf.json
    python scripts/k8s_parity.py yaml.json tf.json

Both sides come from the API server (a server-side dry run of the YAML, and the
live objects Terraform created), so both carry the same server defaults and the
comparison is like for like. Every field the YAML sets must have the same value
on the Terraform object. Extra fields on the Terraform side are allowed only if
they are explicit zero values (false, 0, "", empty), which the provider sends
for unset booleans. The image is skipped on purpose: CI deploys a local build.
"""
from __future__ import annotations

import json
import sys
from typing import Any

# Which parts of each object are the deploy contract.
DEPLOYMENT_POD = ("securityContext", "enableServiceLinks", "terminationGracePeriodSeconds")
CONTAINER = ("name", "imagePullPolicy", "ports", "env", "envFrom", "readinessProbe",
             "livenessProbe", "lifecycle", "resources", "securityContext")


def _items(path: str) -> dict[tuple[str, str], dict[str, Any]]:
    doc = json.load(open(path))
    items = doc.get("items", [doc]) if doc.get("kind") == "List" else [doc]
    return {(i["kind"], i["metadata"]["name"]): i for i in items}


def _contract(obj: dict[str, Any]) -> Any:
    kind, spec = obj["kind"], obj.get("spec", {})
    if kind == "Deployment":
        pod = spec["template"]["spec"]
        return {
            "selector": spec["selector"],
            "strategy": spec["strategy"],
            "pod_labels": spec["template"]["metadata"]["labels"],
            "pod": {k: pod.get(k) for k in DEPLOYMENT_POD},
            "containers": [{k: c.get(k) for k in CONTAINER} for c in pod["containers"]],
        }
    if kind == "Service":
        return {"type": spec["type"], "selector": spec["selector"], "ports": spec["ports"]}
    if kind in ("HorizontalPodAutoscaler", "PodDisruptionBudget"):
        return spec
    if kind == "ConfigMap":
        return obj.get("data", {})
    raise ValueError(kind)


def _diff(want: Any, got: Any, path: str, out: list[str]) -> None:
    if isinstance(want, dict) and isinstance(got, dict):
        for k, v in want.items():
            _diff(v, got.get(k), f"{path}.{k}", out)
        for k, v in got.items():
            if k not in want and v not in (False, 0, "", None, [], {}):
                out.append(f"{path}.{k}: only in Terraform ({v!r})")
    elif isinstance(want, list) and isinstance(got, list) and len(want) == len(got):
        for i, (a, b) in enumerate(zip(want, got, strict=True)):
            _diff(a, b, f"{path}[{i}]", out)
    elif want != got and not (want is None and got in (False, 0, "", [], {})):
        out.append(f"{path}: yaml={want!r} terraform={got!r}")


def main() -> int:
    yaml_objs, tf_objs = _items(sys.argv[1]), _items(sys.argv[2])
    problems: list[str] = []
    checked = 0
    for key, obj in sorted(yaml_objs.items()):
        if key not in tf_objs:
            problems.append(f"{key[0]}/{key[1]}: in YAML, not created by Terraform")
            continue
        _diff(_contract(obj), _contract(tf_objs[key]), f"{key[0]}/{key[1]}", problems)
        checked += 1
    for p in problems:
        print("MISMATCH", p)
    print(f"parity: {checked} objects compared, {len(problems)} mismatches")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
