#!/usr/bin/env bash
# End-to-end Kubernetes check on a throwaway kind cluster. This is what the
# kind-e2e job in .github/workflows/k8s.yml runs, and it runs the same on any
# machine with docker, kind and kubectl:
#
#   bash scripts/k8s_e2e.sh
#
# 1. build the serving image and load it into a kind node
# 2. install metrics-server (the HPA needs it) and apply deploy/k8s/overlays/kind
# 3. wait for the rollout (every pod passed /ready), smoke-test /health, /ready,
#    /predict through the Service, check the ConfigMap value reached the process,
#    record PID 1 and the uid the container runs as
# 4. steady load: k6 in-cluster against the Service, while logging HPA replicas
# 5. rolling restart DURING load: count every failed request
# 6. scripts/k8s_report.py writes artifacts/k8s_kind_<app>.json and exits 1 if
#    any request failed
#
# Everything here is a kind cluster on one machine. The numbers describe that
# machine under synthetic load, nothing more.
set -euo pipefail

APP=${APP:-ledgersentry}
NS=${NS:-$APP}
CLUSTER=${CLUSTER:-$APP}
OUT=${OUT:-k8s-run}
VUS=${VUS:-16}
STEADY=${STEADY:-90s}
RESTART=${RESTART:-120s}
RESTART_AFTER=${RESTART_AFTER:-15}
METRICS_SERVER=${METRICS_SERVER:-v0.9.0}
K6_IMAGE=grafana/k6:2.3.0

mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
now() { date +%s; }
kn() { kubectl -n "$NS" "$@"; }

{
  echo "vcpu=$(nproc)"
  echo "mem_mb=$(free -m | awk '/^Mem:/{print $2}')"
  echo "kind_version=$(kind version | awk '{print $2}')"
  echo "run_url=${GITHUB_SERVER_URL:-}/${GITHUB_REPOSITORY:-}/actions/runs/${GITHUB_RUN_ID:-}"
  echo "git_sha=$(git rev-parse HEAD)"
} > "$OUT/env.txt"

log "build image $APP:ci"
docker build -q -t "$APP:ci" .
docker pull -q "$K6_IMAGE"

if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  log "create kind cluster $CLUSTER"
  kind create cluster --name "$CLUSTER" --wait 120s
fi
kubectl config use-context "kind-$CLUSTER"
kind load docker-image "$APP:ci" "$K6_IMAGE" --name "$CLUSTER"
echo "kubernetes_version=$(kubectl version -o json | python3 -c 'import json,sys; print(json.load(sys.stdin)["serverVersion"]["gitVersion"])')" >> "$OUT/env.txt"

log "install metrics-server $METRICS_SERVER"
kubectl apply -f "https://github.com/kubernetes-sigs/metrics-server/releases/download/$METRICS_SERVER/components.yaml"
# kind's kubelets serve self-signed certs; this flag is for the throwaway cluster only
kubectl -n kube-system patch deploy metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl -n kube-system rollout status deploy/metrics-server --timeout=180s

log "apply deploy/k8s/overlays/kind"
t0=$(now)
kubectl apply -k deploy/k8s/overlays/kind
kn rollout status "deploy/$APP" --timeout=300s
echo "rollout_seconds=$(( $(now) - t0 ))" >> "$OUT/times.env"
kn get deploy,rs,pod,svc,endpoints,cm,hpa,pdb -o wide | tee "$OUT/objects.txt"

log "smoke test through the Service"
kn port-forward "svc/$APP" 18000:80 > /dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
for _ in $(seq 30); do curl -sf localhost:18000/health > /dev/null && break; sleep 1; done
curl -sf localhost:18000/health | tee "$OUT/health.json"; echo
curl -sf localhost:18000/ready | tee "$OUT/ready.json"; echo
curl -sf -X POST localhost:18000/predict -H 'Content-Type: application/json' \
  -d @deploy/k8s/loadtest/payload.json | tee "$OUT/predict.json"; echo
curl -sf localhost:18000/openapi.json > "$OUT/openapi.json"
kn get cm "$APP-config" -o json > "$OUT/configmap.json"
kill $PF 2>/dev/null || true
POD=$(kn get pod -l "app=$APP" -o jsonpath='{.items[0].metadata.name}')
kn exec "$POD" -- cat /proc/1/cmdline | tr '\0' ' ' > "$OUT/pid1.txt"
kn exec "$POD" -- id > "$OUT/id.txt"
log "PID 1: $(cat "$OUT/pid1.txt")   $(cat "$OUT/id.txt")"

log "wait for metrics-server to report pod CPU"
for _ in $(seq 60); do kn top pod > /dev/null 2>&1 && break; sleep 5; done
kn top pod

# HPA timeline in the background: epoch, current replicas, desired, CPU % of request
(
  while true; do
    echo "$(now) $(kn get hpa "$APP" -o jsonpath='{.status.currentReplicas} {.status.desiredReplicas} {.status.currentMetrics[0].resource.current.averageUtilization}' 2>/dev/null)"
    sleep 5
  done
) > "$OUT/hpa.log" &
HPA_WATCH=$!
trap 'kill $PF $HPA_WATCH 2>/dev/null || true' EXIT

kn create configmap k6-script \
  --from-file=deploy/k8s/loadtest/load.js --from-file=deploy/k8s/loadtest/payload.json \
  --dry-run=client -o yaml | kn apply -f -

start_phase() {
  sed "s/__PHASE__/$1/g; s/__DURATION__/$2/g; s/__VUS__/$VUS/g" deploy/k8s/loadtest/job.yaml | kn apply -f -
  kn wait --for=jsonpath='{.status.phase}'=Running pod -l "job-name=k6-$1" --timeout=120s
  echo "$1_load_start=$(now)" >> "$OUT/times.env"
}
finish_phase() {
  kn wait --for=condition=complete "job/k6-$1" --timeout=900s
  echo "$1_load_end=$(now)" >> "$OUT/times.env"
  kn logs "job/k6-$1" > "$OUT/k6-$1.log"
  grep K6_SUMMARY "$OUT/k6-$1.log"
}

log "phase 1: steady load, $VUS VUs for $STEADY"
start_phase steady "$STEADY"
finish_phase steady
kn top pod | tee "$OUT/top.txt"

log "phase 2: rolling restart $RESTART_AFTER s into $RESTART of load"
kn get pod -l "app=$APP" -o jsonpath='{.items[*].metadata.name}' > "$OUT/pods_before.txt"
start_phase restart "$RESTART"
sleep "$RESTART_AFTER"
echo "restart_start=$(now)" >> "$OUT/times.env"
kn rollout restart "deploy/$APP"
kn rollout status "deploy/$APP" --timeout=600s
echo "restart_end=$(now)" >> "$OUT/times.env"
finish_phase restart
for p in $(cat "$OUT/pods_before.txt"); do kn wait --for=delete "pod/$p" --timeout=120s || true; done
kn get pod -l "app=$APP" --field-selector=status.phase=Running \
  -o jsonpath='{.items[*].metadata.name}' > "$OUT/pods_after.txt"

kill $HPA_WATCH 2>/dev/null || true
log "write report"
python3 scripts/k8s_report.py kind --app "$APP" --out-dir "$OUT" \
  --artifact "artifacts/k8s_kind_$APP.json" "$@"
