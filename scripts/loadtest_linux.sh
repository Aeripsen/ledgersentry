#!/usr/bin/env bash
# The load test against the real serving image on Linux: the image's Python
# 3.12, the pinned requirements.txt (uvicorn 0.51.0), scikit-learn's Linux wheel
# with libgomp. Run by .github/workflows/loadtest-linux.yml on a GitHub-hosted
# runner; runs anywhere with docker and `pip install -r requirements.txt -r
# requirements-loadtest.txt`.
#
# Arms, alternating ABBA per round, at 1 and 4 workers (WEB_CONCURRENCY, which
# the image's uvicorn reads; nothing else changes):
#   omp1        the image exactly as built: its CMD, its ENV OMP_NUM_THREADS=1
#   ompdefault  the same image and command with OMP_NUM_THREADS unset first
# No --cpus limit, --network host (docker-proxy would add its own CPU cost to
# every request). The client runs on the runner, outside the container, and
# the container's PID is handed to it so server CPU is measured.
set -euo pipefail
IMAGE=${IMAGE:-ledgersentry:loadtest}
ROUNDS=${ROUNDS:-3}
WORKERS=${WORKERS:-"1 4"}
OUT=${OUT:-artifacts/loadtest_linux}
LEVELS=${LEVELS:-1,2,4,8,16,32,64,128}
mkdir -p "$OUT"
CMD='exec uvicorn ledgersentry.service:app --host 0.0.0.0 --port ${PORT:-8000}'
IMAGE_ID=$(docker image inspect -f '{{.Id}}' "$IMAGE")

run_arm() {
  local arm=$1 workers=$2 round=$3 pre=""
  [ "$arm" = ompdefault ] && pre='unset OMP_NUM_THREADS; '
  local probe cid pid
  probe=$(docker run --rm "$IMAGE" sh -c "${pre}exec python -m ledgersentry.loadtest --probe-openmp")
  cid=$(docker run -d --network host -e WEB_CONCURRENCY="$workers" "$IMAGE" sh -c "${pre}${CMD}")
  pid=$(docker inspect -f '{{.State.Pid}}' "$cid")
  python scripts/loadtest.py --url http://127.0.0.1:8000 --server-pid "$pid" \
    --arm "$arm" --workers "$workers" --levels "$LEVELS" \
    --label "${arm}_w${workers}_r${round}" --out "$OUT/${arm}_w${workers}_r${round}.json" \
    --meta "where=docker container on a GitHub-hosted ubuntu runner" \
    --meta "image_id=$IMAGE_ID" \
    --meta "container_cmd=sh -c '${pre}${CMD}'" \
    --meta "docker_run=--network host -e WEB_CONCURRENCY=$workers (no --cpus)" \
    --meta "openmp_probe_in_image=$probe" \
    --meta "ab_round=$round"
  docker logs "$cid" 2>&1 | tail -n 3
  docker rm -f "$cid" >/dev/null
}

for round in $(seq 1 "$ROUNDS"); do
  for workers in $WORKERS; do
    if [ $((round % 2)) -eq 1 ]; then arms="omp1 ompdefault"; else arms="ompdefault omp1"; fi
    for arm in $arms; do run_arm "$arm" "$workers" "$round"; done
  done
done
python scripts/loadtest.py --ab-summary "$OUT"
