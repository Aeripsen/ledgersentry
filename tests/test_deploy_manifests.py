"""The deploy manifests are wired correctly - checked in CI, not just asserted
in a doc. These pin the render.yaml / Kubernetes / Dockerfile contract so an edit
that breaks a selector, a probe path, or the non-root uid turns CI red instead of
failing silently at deploy time (the tools themselves - docker/kubectl/terraform -
are not in CI, so this is the offline guard for their inputs)."""
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _k8s_docs() -> dict[str, dict]:
    text = (REPO / "deploy" / "k8s" / "base" / "ledgersentry.yaml").read_text()
    return {d["kind"]: d for d in yaml.safe_load_all(text) if d}


def test_render_blueprint_is_a_docker_web_service() -> None:
    services = yaml.safe_load((REPO / "render.yaml").read_text())["services"]
    assert len(services) == 1
    svc = services[0]
    assert svc["type"] == "web"
    assert svc["runtime"] == "docker"
    assert svc["dockerfilePath"] == "./Dockerfile"
    assert svc["healthCheckPath"] == "/health"
    port_vars = [e for e in svc.get("envVars", []) if e["key"] == "PORT"]
    assert len(port_vars) == 1
    assert str(port_vars[0]["value"]) == "8000"


def test_k8s_service_selector_matches_deployment_pods() -> None:
    docs = _k8s_docs()
    pod_labels = docs["Deployment"]["spec"]["template"]["metadata"]["labels"]
    # Deployment must select its own pods, and the Service must too, or traffic
    # never reaches a running pod.
    assert docs["Deployment"]["spec"]["selector"]["matchLabels"] == pod_labels
    assert docs["Service"]["spec"]["selector"] == pod_labels


def test_k8s_probes_hit_the_right_endpoints() -> None:
    docs = _k8s_docs()
    container = docs["Deployment"]["spec"]["template"]["spec"]["containers"][0]
    port = container["ports"][0]
    assert port["containerPort"] == 8000
    # Liveness on /health (stays 200 while the model loads, so a warming pod is
    # not killed); readiness on /ready (gates traffic until the scorer is built).
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    # Probes and the Service all target the one named container port.
    assert container["livenessProbe"]["httpGet"]["port"] == port["name"]
    assert container["readinessProbe"]["httpGet"]["port"] == port["name"]
    assert docs["Service"]["spec"]["ports"][0]["targetPort"] == port["name"]


def test_k8s_nonroot_uid_matches_the_dockerfile() -> None:
    docs = _k8s_docs()
    sec = docs["Deployment"]["spec"]["template"]["spec"]["securityContext"]
    assert sec["runAsNonRoot"] is True
    # Must equal the Dockerfile's `useradd --uid 10001`, or runAsNonRoot makes the
    # pod fail to start.
    assert sec["runAsUser"] == 10001
    assert "10001" in (REPO / "Dockerfile").read_text()


def test_dockerfile_exposes_8000_and_healthchecks() -> None:
    df = (REPO / "Dockerfile").read_text()
    assert "EXPOSE 8000" in df
    assert "HEALTHCHECK" in df
    assert "/health" in df


def test_k8s_rollout_never_drops_capacity() -> None:
    dep = _k8s_docs()["Deployment"]
    rolling = dep["spec"]["strategy"]["rollingUpdate"]
    # New pod must pass /ready before an old one goes: the rolling-restart
    # load test in scripts/k8s_e2e.sh depends on this.
    assert rolling["maxUnavailable"] == 0
    assert rolling["maxSurge"] >= 1
    pod = dep["spec"]["template"]["spec"]
    container = pod["containers"][0]
    shell, flag, script = container["lifecycle"]["preStop"]["exec"]["command"]
    assert (shell, flag) == ("sh", "-c")
    # preStop must create the file DrainMiddleware watches, then outwait kube-proxy
    from ledgersentry.drain import DRAIN_FILE

    assert f"touch {DRAIN_FILE.as_posix()}" in script
    sleep_s = int(re.search(r"sleep (\d+)", script).group(1))
    assert sleep_s < pod["terminationGracePeriodSeconds"]
    assert "exec uvicorn" in (REPO / "Dockerfile").read_text()


def test_k8s_configmap_is_wired_and_matches_code_defaults() -> None:
    from ledgersentry.config import Settings

    docs = _k8s_docs()
    cm = docs["ConfigMap"]
    container = docs["Deployment"]["spec"]["template"]["spec"]["containers"][0]
    assert {"configMapRef": {"name": cm["metadata"]["name"]}} in container["envFrom"]
    # Base values are the code defaults, so a deploy serves exactly the model
    # behind the committed metrics. The kind overlay changes one on purpose.
    defaults = Settings()
    for key, value in cm["data"].items():
        field = key.removeprefix("LEDGERSENTRY_").lower()
        assert float(value) == float(getattr(defaults, field)), key


def test_k8s_hpa_and_pdb_target_the_deployment() -> None:
    docs = _k8s_docs()
    dep = docs["Deployment"]
    hpa = docs["HorizontalPodAutoscaler"]["spec"]
    assert hpa["scaleTargetRef"] == {
        "apiVersion": "apps/v1", "kind": "Deployment", "name": dep["metadata"]["name"],
    }
    assert hpa["minReplicas"] <= dep["spec"]["replicas"] <= hpa["maxReplicas"]
    # CPU utilization is a percent of the request, so a request must exist.
    assert dep["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]["cpu"]
    pdb = docs["PodDisruptionBudget"]["spec"]
    assert pdb["selector"]["matchLabels"] == dep["spec"]["template"]["metadata"]["labels"]
    assert pdb["minAvailable"] < hpa["minReplicas"]


def test_terraform_kubernetes_settings_match_the_yaml_configmap() -> None:
    tf = (REPO / "deploy" / "terraform" / "kubernetes" / "variables.tf").read_text()
    tf_settings = dict(re.findall(r'(LEDGERSENTRY_\w+)\s*=\s*"([^"]*)"', tf))
    assert tf_settings == _k8s_docs()["ConfigMap"]["data"]


def test_kind_overlay_changes_max_batch_so_ci_can_prove_the_wiring() -> None:
    text = (REPO / "deploy" / "k8s" / "overlays" / "kind" / "kustomization.yaml").read_text()
    overlay = yaml.safe_load(text)
    value = yaml.safe_load(overlay["patches"][0]["patch"])[0]["value"]
    assert value != _k8s_docs()["ConfigMap"]["data"]["LEDGERSENTRY_MAX_BATCH"]
