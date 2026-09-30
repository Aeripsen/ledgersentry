#!/usr/bin/env bash
# Terraform, actually applied: deploy/terraform/kubernetes against a throwaway
# kind cluster with the hashicorp/kubernetes provider. This is what the
# terraform-kind job in .github/workflows/k8s.yml runs, and it runs the same on
# any machine with docker, kind, kubectl and terraform:
#
#   bash scripts/tf_kind_e2e.sh
#
# init -> validate -> apply (returns only after every pod passed /ready)
# -> smoke /ready and /predict through the Service
# -> a second plan must be empty (the module is idempotent, no drift)
# -> the live objects must match deploy/k8s/base (scripts/k8s_parity.py)
# -> destroy
# scripts/k8s_report.py writes artifacts/terraform_kind_<app>.json and exits 1
# if any step failed.
set -uo pipefail

APP=${APP:-ledgersentry}
CLUSTER=${CLUSTER:-$APP-tf}
OUT=${OUT:-tf-run}
TF_DIR=deploy/terraform/kubernetes
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
rec() { echo "$1=$2" >> "$OUT/tf.env"; }
: > "$OUT/tf.env"
rec git_sha "$(git rev-parse HEAD)"
rec run_url "${GITHUB_SERVER_URL:-}/${GITHUB_REPOSITORY:-}/actions/runs/${GITHUB_RUN_ID:-}"

set -e
log "build image $APP:ci"
docker build -q -t "$APP:ci" .
if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER" --wait 120s
fi
kind load docker-image "$APP:ci" --name "$CLUSTER"
rec kubernetes_version "$(kubectl --context "kind-$CLUSTER" version -o json | python3 -c 'import json,sys; print(json.load(sys.stdin)["serverVersion"]["gitVersion"])')"
set +e

TFVARS=(-var "image=$APP:ci" -var "kube_context=kind-$CLUSTER")
cd "$TF_DIR"
terraform init -input=false -no-color | tee "$OUT/init.log"
rec terraform_version "$(terraform version -json | python3 -c 'import json,sys; print(json.load(sys.stdin)["terraform_version"])')"
rec provider_version "$(grep -A1 'hashicorp/kubernetes' .terraform.lock.hcl | awk -F'"' '/version/{print $2}')"
terraform validate -no-color | tee "$OUT/validate.log"

log "terraform apply"
t0=$(date +%s)
terraform apply -auto-approve -input=false -no-color "${TFVARS[@]}" | tee "$OUT/apply.log"
rec apply_rc "${PIPESTATUS[0]}"
rec apply_seconds "$(( $(date +%s) - t0 ))"
rec resources_created "$(grep -oE '[0-9]+ added' "$OUT/apply.log" | grep -oE '[0-9]+' | tail -1)"
NS=$(terraform output -raw namespace)
cd - > /dev/null

log "smoke test through the Service"
kubectl --context "kind-$CLUSTER" -n "$NS" get deploy,pod,svc,cm,hpa,pdb -o wide | tee "$OUT/objects.txt"
kubectl --context "kind-$CLUSTER" -n "$NS" port-forward "svc/$APP" 18001:80 > /dev/null 2>&1 &
PF=$!
for _ in $(seq 30); do curl -sf localhost:18001/health > /dev/null && break; sleep 1; done
curl -sf localhost:18001/ready | tee "$OUT/ready.json"; echo
curl -sf -X POST localhost:18001/predict -H 'Content-Type: application/json' \
  -d @deploy/k8s/loadtest/payload.json | tee "$OUT/predict.json"; echo
kill $PF 2>/dev/null

log "second plan must be empty"
(cd "$TF_DIR" && terraform plan -detailed-exitcode -input=false -no-color "${TFVARS[@]}") \
  | tee "$OUT/plan2.log"
rec plan_rc "${PIPESTATUS[0]}"

log "parity: Terraform objects vs deploy/k8s/base"
kubectl --context "kind-$CLUSTER" create namespace parity > /dev/null 2>&1
kubectl kustomize deploy/k8s/base \
  | kubectl --context "kind-$CLUSTER" apply -n parity --dry-run=server -o json -f - > "$OUT/parity_yaml.json"
kubectl --context "kind-$CLUSTER" -n "$NS" get deploy,svc,hpa,pdb,cm -o json > "$OUT/parity_tf.json"
python3 scripts/k8s_parity.py "$OUT/parity_yaml.json" "$OUT/parity_tf.json" | tee "$OUT/parity.log"
rec parity_rc "${PIPESTATUS[0]}"

log "terraform destroy"
(cd "$TF_DIR" && terraform destroy -auto-approve -input=false -no-color "${TFVARS[@]}") \
  | tee "$OUT/destroy.log"
rec destroy_rc "${PIPESTATUS[0]}"

python3 scripts/k8s_report.py terraform --out-dir "$OUT" \
  --artifact "artifacts/terraform_kind_$APP.json"
