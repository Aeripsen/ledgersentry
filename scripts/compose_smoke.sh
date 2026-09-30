#!/usr/bin/env bash
# Brings up docker-compose.yml (API on 8000, Streamlit dashboard on 8501), waits
# for the API's healthcheck, hits /health, /ready and /predict and the
# dashboard's health endpoint, then tears the stack down. The k8s workflow runs
# this on every deploy change; locally it needs docker with the compose plugin.
#
#   bash scripts/compose_smoke.sh
set -euo pipefail

OUT=${OUT:-compose-run}
mkdir -p "$OUT"
trap 'docker compose down -v > /dev/null 2>&1 || true' EXIT

docker compose up -d --build --wait --wait-timeout 300
docker compose ps | tee "$OUT/ps.txt"
curl -sf localhost:8000/health | tee "$OUT/health.json"; echo
curl -sf localhost:8000/ready | tee "$OUT/ready.json"; echo
curl -sf -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d @deploy/k8s/loadtest/payload.json | tee "$OUT/predict.json"; echo
for _ in $(seq 60); do curl -sf localhost:8501/_stcore/health > /dev/null && break; sleep 2; done
curl -sf localhost:8501/_stcore/health | tee "$OUT/dashboard_health.txt"; echo
echo "compose smoke test passed"
