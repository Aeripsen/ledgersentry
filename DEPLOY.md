# Deploying LedgerSentry

The serving image (`Dockerfile`) is self-contained. If no trained model is baked
in, which is the case for any build from a clean git checkout because the
`.joblib` is gitignored, the build generates the deterministic synthetic-fixture
model the same way `scripts/train.py` does with no data file present. So the API
comes up green on first boot with zero data download. Drop a real dataset in
`data/` and rebuild (or train first) and that real model is served instead.

**Read this before quoting any deploy result:** every deployment below was built
from a clean checkout, so it served the **synthetic-fixture model** (5 input
features, `/ready` reports `expected_features: 5`), not the ULB model behind the
README's metrics. The deploy results are about the serving path (image, probes,
rollout, autoscaling, draining), not about model quality. Never put a deploy
number next to a ULB fraud metric as if one model produced both.

The service listens on `$PORT` (default `8000`) and exposes:

- `GET /health` - liveness, 200 while the process is up (even while the model loads)
- `GET /ready` - readiness, 200 only once the compiled scorer can actually score
- `POST /predict`, `POST /predict/batch`, `POST /drift`, `GET /curve`

## What has actually run

| Target | Status | Evidence |
|---|---|---|
| Kubernetes (kind) | Deployed, smoke-tested, load-tested, PodDisruptionBudget exercised and rolled under load in CI on every deploy change | [`.github/workflows/k8s.yml`](.github/workflows/k8s.yml), [`artifacts/k8s_kind_ledgersentry.json`](artifacts/k8s_kind_ledgersentry.json) |
| Connection drain, with vs without | 24 runs on fresh runners, 12 each way | [`.github/workflows/k8s-drain-ab.yml`](.github/workflows/k8s-drain-ab.yml), [`artifacts/k8s_drain_ab_ledgersentry.json`](artifacts/k8s_drain_ab_ledgersentry.json) |
| Terraform, `hashicorp/kubernetes` | Applied to a throwaway kind cluster, re-planned, diffed against the YAML, destroyed, in CI | [`artifacts/terraform_kind_ledgersentry.json`](artifacts/terraform_kind_ledgersentry.json) |
| docker compose | Brought up and smoke-tested in CI on every deploy change (API `/health`, `/ready`, `/predict`, dashboard health) | the `compose` job in `k8s.yml`, [`scripts/compose_smoke.sh`](scripts/compose_smoke.sh) |
| Manifest schemas | `kubeconform -strict` in CI on every deploy change | the `kubeconform` job, [`scripts/k8s_schema.sh`](scripts/k8s_schema.sh) |
| Terraform, GCP Cloud Run | `fmt -check` and `validate` in CI only. **Never applied** | needs a GCP project with billing |
| Render | **Not deployed** | needs a Render account connected to the repo |

Everything in CI runs on a free GitHub-hosted runner (4 vCPU, 16 GB). No cloud
account is used anywhere. "Applied" for Terraform means applied to a kind cluster
that exists for the length of one CI job, not to cloud infrastructure.

---

## 1. Kubernetes

Layout:

- [`deploy/k8s/base/ledgersentry.yaml`](deploy/k8s/base/ledgersentry.yaml): ConfigMap
  (serving settings, injected with `envFrom`), Deployment (2 replicas, non-root uid
  10001, liveness `/health`, readiness `/ready`, `maxUnavailable: 0` rolling update,
  preStop drain, requests 250m CPU / 256Mi, limits 1 CPU / 512Mi), ClusterIP Service,
  HorizontalPodAutoscaler (CPU 70% of request, 2 to 5 replicas) and a
  PodDisruptionBudget (`minAvailable: 1`).
- [`deploy/k8s/overlays/kind`](deploy/k8s/overlays/kind): the base in its own
  namespace, with the locally built `ledgersentry:ci` image and one ConfigMap value
  changed (`LEDGERSENTRY_MAX_BATCH=2000`) so the run can prove the ConfigMap reaches
  the process.
- [`deploy/k8s/loadtest`](deploy/k8s/loadtest): a k6 Job that runs inside the
  cluster and posts to `/predict` through the Service.

### Reproduce it

Needs docker, [kind](https://kind.sigs.k8s.io/) and kubectl. No account.

```bash
make k8s-e2e          # or: bash scripts/k8s_e2e.sh
make k8s-schema       # kubeconform -strict on everything below
make compose-smoke    # docker compose up, smoke test, down
```

`scripts/k8s_e2e.sh` is exactly what CI runs:

1. `docker build -t ledgersentry:ci .`, `kind create cluster`, `kind load docker-image`
2. install metrics-server (the HPA needs it), `kubectl apply -k deploy/k8s/overlays/kind`,
   `kubectl rollout status` (returns once every pod passed `/ready`)
3. through the Service: `/health`, `/ready`, `/predict`, and `/openapi.json`, whose
   `/predict/batch` `maxItems` must equal the ConfigMap value; record PID 1 and the uid
4. PodDisruptionBudget: with 2 pods up, evict one through the Eviction API (must be
   allowed), then the other straight away (must be refused)
5. steady load: 16 k6 virtual users for 90 s, closed loop, while logging the HPA;
   then count the requests each pod served from its uvicorn access log
6. rolling restart under load: a new 120 s k6 run, `kubectl rollout restart` 30 s
   into it; per-pod counts again just before the restart and, for the replacement
   pods, at the end. k6 counts requests and failures per 10 s of wall clock, so the
   report can compare the 30 s before the restart with the windows during and after
7. [`scripts/k8s_report.py`](scripts/k8s_report.py) writes the artifact and **fails
   the job on any failed request**, an unreplaced pod, a rollout that outlasted the
   load, a ConfigMap value that did not reach the process, a shell as PID 1, or a
   PodDisruptionBudget that did not refuse the second eviction

To deploy the base to a real cluster instead, push the image somewhere the cluster
can pull, set `image:` and run `kubectl apply -k deploy/k8s/base`.

### Captured output

From CI run [36777977754](https://github.com/Aeripsen/ledgersentry/actions/runs/36777977754)
on master (commit `45b8754`), kind v0.33.0, Kubernetes v1.37.0. The full JSON is
[`artifacts/k8s_kind_ledgersentry.json`](artifacts/k8s_kind_ledgersentry.json).

```text
rollout: 2 pods ready in 6 s
PID 1:   /usr/local/bin/python3.12 /usr/local/bin/uvicorn ledgersentry.service:app ...
runs as: uid=10001(app)
ConfigMap LEDGERSENTRY_MAX_BATCH=2000, served /predict/batch maxItems=2000
/predict: {"decision": "legit", "p_fraud": 0.0002, "confidence": 0.9998, ...}
PDB:     first eviction allowed; second refused: "Cannot evict pod as it would
         violate the pod's disruption budget."

steady (16 VUs, 90 s): 20,870 requests, 0 failed
  before the scale-out 231.9 req/s, with all 5 pods running 230.4 req/s
  requests per pod: 10,508 / 10,362 / 0 / 0 / 0
rolling restart (120 s): 62,532 requests, 0 failed
  restart began 32 s into the load, took 36 s, all 5 pods replaced
  before the restart 640.2 req/s, during 484.6, after 475.0
  per pod in the 30 s before the restart: 5,023 / 4,898 / 4,816 / 4,020 / 1,047
  per replacement pod over the phase:    10,375 / 10,247 / 8,829 / 966 / 0
HPA: chose 5 replicas 34 s into the load, 5 pods running at 50 s;
  the 2 busy pods sat at their CPU limit (999m of 1000m)
```

### What the runs found

**The first run failed its own gate.** With only a preStop `sleep 5`, a rolling
restart under load failed 4 of 114,308 requests
([run 36769111760](https://github.com/Aeripsen/ledgersentry/actions/runs/36769111760),
[`artifacts/k8s_kind_ledgersentry_before_drain.json`](artifacts/k8s_kind_ledgersentry_before_drain.json)).
k6's error text was `connection reset by peer` three times and `EOF` once, all on
reused keep-alive connections. The steady phase, with no restart, failed none.

Why: the sleep only covers *new* connections. When a pod starts terminating it
leaves the Service's endpoints and kube-proxy stops sending new connections to it,
but a client's existing keep-alive connection stays pinned to that pod. When
uvicorn gets SIGTERM it closes idle keep-alive connections, and a client that sends
its next request on one at that instant gets a reset.

The fix ([`src/ledgersentry/drain.py`](src/ledgersentry/drain.py)): preStop runs
`touch /tmp/draining && sleep 5`. While that file exists, a small ASGI middleware
adds `Connection: close` to every response, so each client finishes its current
request, closes the connection cleanly and reconnects to a pod that is staying.
It is a file rather than an endpoint so nothing reachable over the network can put
a pod into drain mode.
[`tests/test_drain_socket.py`](tests/test_drain_socket.py) checks the part a test
client cannot: it starts a real uvicorn server on a local port, once with each of
uvicorn's HTTP parsers (h11 and httptools), and over a raw socket asserts that the
server closes the TCP connection after the drain response, and that without the
drain file the same socket carries a second request. It runs in the normal CI test
job.

**Whether the drain works, measured.** Each rolling restart is one event, so the
unit is the run, not the request.
[`k8s-drain-ab.yml`](.github/workflows/k8s-drain-ab.yml) runs the whole of
`k8s_e2e.sh` 12 times on fresh runners: 6 with the drain and 6 with `DRAIN=off`,
which puts the old plain `sleep 5` back. It ran twice, on commits `578ad1a`
([run 36776368018](https://github.com/Aeripsen/ledgersentry/actions/runs/36776368018))
and `45b8754` ([run 36777997492](https://github.com/Aeripsen/ledgersentry/actions/runs/36777997492));
the drain code is the same in both. All 24 runs were valid (every pod replaced,
rollout finished under load, the other checks held).

| | runs | runs with a failed request | failed requests per run | requests in the restart phases |
|---|---|---|---|---|
| drain off | 12 | 9 | 0, 0, 5, 0, 2, 5, 4, 5, 5, 2, 4, 1 | 1,049,755 |
| drain on | 12 | 0 | all 0 | 1,178,138 |

A one-sided Fisher exact test on those runs gives p = 0.0002: if the drain made no
difference, all 9 failing runs landing in the drain-off group would happen about 2
times in 10,000 ([`artifacts/k8s_drain_ab_ledgersentry.json`](artifacts/k8s_drain_ab_ledgersentry.json)). Before the A/B, the record was 1 of 1 runs failed without the drain and 0 of
3 with it (74,490, 71,629 and 108,825 requests: attempts 1 and 2 of
[run 36770455800](https://github.com/Aeripsen/ledgersentry/actions/runs/36770455800),
which are two attempts of one commit, and the master run
[36771694619](https://github.com/Aeripsen/ledgersentry/actions/runs/36771694619);
[`artifacts/k8s_kind_ledgersentry_postfix_attempt1.json`](artifacts/k8s_kind_ledgersentry_postfix_attempt1.json),
[`_postfix_attempt2.json`](artifacts/k8s_kind_ledgersentry_postfix_attempt2.json),
[`_postfix_master.json`](artifacts/k8s_kind_ledgersentry_postfix_master.json); attempt 1's
artifact was replaced on GitHub when attempt 2 uploaded, so its report is copied from
the committed job log in [`artifacts/ci_logs/`](artifacts/ci_logs)).

**Keep-alive also defeats the autoscaler's new pods.** Across the 24 A/B runs,
the HPA chose 5 replicas 19 to 40 s into the load and 5 pods were running 34 to 55
s in (the log is sampled every 5 s, so each time is up to 5 s late). But the steady
phase's throughput did not move: after the scale-out it was 0.92 to 2.43 times the
rate before it, median 1.00. The access logs say why. In 21 of the 24 runs, and in
the master run, 2 of the 5 pods served every steady-phase request and the other 3
served none (the per-pod log totals match k6's request count in every run). The
16 k6 connections were opened while 2 pods existed and stay on those pods, because
kube-proxy balances connections, not requests. In the other 3 runs a third pod
picked up connections during the steady phase, which is where the ratios above 1
come from; the first, pre-drain run also had 3 busy pods (by `kubectl top`).
Nothing recorded says which connection moved or why, so that stays unexplained.

**The rolling restart did not spread the connections.** An earlier version of this
file said it did and credited it with a 2.6x throughput rise. That compared the
steady phase with a whole restart phase whose k6 Job is a *new* client: its
connections open against 5 ready pods before the restart begins. Measured
separately: in the 30 s before the restart, that fresh client was already served by
4 or 5 of the 5 pods (at least 5% of its requests each) in every run, and after the
restart throughput was 0.70 to 1.18 times the pre-restart rate, median 1.00, across
the 24 A/B runs. So a fresh client spreads over the pods that exist when it
connects; the restart itself adds nothing. It can take away: in the master run
above, throughput fell from 640.2 to 475.0 req/s after the restart, and the
replacement pods served 10,375, 10,247, 8,829, 966 and 0 requests, because clients
reconnect while only some of the new pods are up and stay where they land. A real deployment would fix this with an
L7 proxy or service mesh that balances per request. Not done here.

**What the fresh client's higher rate means.** 640.2 req/s before the restart
against 231.9 in the steady phase, in the master run, is what the node gives 4 or 5
busy pods against 2 pods held at their 1-CPU limits. The node has 4 vCPU shared with k6 and
the control plane, so 5 pods with 1-CPU limits cannot all get their limit. It shows
the per-pod limit was the steady phase's bottleneck, not a capacity figure.

**CPU and memory.** The HPA's `averageUtilization` peaked at 368% of the 250m
request in the master run. It cannot pass 400% here, because the limit is 1000m: it means the busy
pods sat at their CPU limit and were throttled. Memory was about 140 Mi per pod
under load, against a 256Mi request and a 512Mi limit.

### What these numbers are not

- One single-node kind cluster on one 4-vCPU runner, with the k6 load generator in
  the same cluster, competing for the same CPUs as the pods. The HPA going from 2
  to 5 pods on one node shows the control loop working, not added capacity.
- Synthetic load: one fixed request body (a transaction the model scores `legit`)
  in a closed loop. Every request takes the same code path. Not production
  traffic, not users.
- The served model is the synthetic fixture (see the top of this file).
- Throughput varies a lot between runners: steady-phase rates of 231.8 to 630.6
  req/s across the committed runs, and one A/B runner ran about twice as fast as the
  rest. The CI gate is therefore on failed requests, never on a throughput number,
  and no req/s or latency figure here is a benchmark.
- The PodDisruptionBudget check proves the eviction API refuses to take the last
  pod. It is not a node drain under load.

---

## 2. Terraform

### 2a. Kubernetes provider (applied to kind in CI)

[`deploy/terraform/kubernetes`](deploy/terraform/kubernetes) creates the same six
objects as the YAML (namespace, ConfigMap, Deployment, Service, HPA, PDB) as typed
`hashicorp/kubernetes` resources. `wait_for_rollout = true` makes `apply` return only
after every pod passed `/ready`, and `ignore_changes` on `replicas` stops Terraform
from fighting the HPA.

```bash
make tf-kind          # or: bash scripts/tf_kind_e2e.sh
```

does `terraform init`, `validate`, `apply` to a kind cluster, a `/ready` and
`/predict` smoke test through the Service, a second `terraform plan
-detailed-exitcode` that must be empty, [`scripts/k8s_parity.py`](scripts/k8s_parity.py)
(the live Terraform objects against a server-side dry run of `deploy/k8s/base`),
then `terraform destroy`.

From the same master run, Terraform 1.16.4, provider 3.2.1
([`artifacts/terraform_kind_ledgersentry.json`](artifacts/terraform_kind_ledgersentry.json)):

```text
Apply complete! Resources: 6 added, 0 changed, 0 destroyed.      (8 s)
second plan: No changes. Your infrastructure matches the configuration.
parity: 5 objects compared, 0 mismatches
Destroy complete! Resources: 6 destroyed.
```

The parity check compares a chosen contract per object, not every field: for the
Deployment the selector, rollout strategy, pod labels, pod security context,
service links and grace period, and per container the ports, env, probes,
lifecycle, resources and security context; the Service's type, selector and
ports; the full HPA and PDB specs; the ConfigMap data. It skips the image, the
replica count, object metadata labels and the namespace object, and a field set
only on the Terraform side passes if its value is a zero value. "0 mismatches"
means 0 on those fields, not full equivalence. It
earned its place on the first run anyway: the provider always writes an HPA
`behavior` block (the Kubernetes default scaling policies) while the YAML left it
unset. Both now state the policy explicitly.

To use it against another cluster: `terraform -chdir=deploy/terraform/kubernetes
apply -var kube_context=<context> -var image=<registry>/ledgersentry:<tag>`.

### 2b. GCP Cloud Run (validated, never applied)

[`deploy/terraform/cloudrun`](deploy/terraform/cloudrun) provisions a scale-to-zero
Cloud Run v2 service (startup probe on `/ready`, liveness on `/health`) and,
optionally, public access. CI runs `terraform fmt -check` and `terraform validate`
on it. **It has never been applied**: that needs a GCP project with a billing
account, and nothing in this repo creates one. It is not evidence of Cloud Run
experience.

If a project with billing exists:

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud auth configure-docker us-central1-docker.pkg.dev
gcloud artifacts repositories create ledgersentry --repository-format=docker --location=us-central1
docker build -t us-central1-docker.pkg.dev/PROJECT_ID/ledgersentry/ledgersentry:latest .
docker push  us-central1-docker.pkg.dev/PROJECT_ID/ledgersentry/ledgersentry:latest

cd deploy/terraform/cloudrun
terraform init
terraform apply -var project_id=PROJECT_ID \
  -var image=us-central1-docker.pkg.dev/PROJECT_ID/ledgersentry/ledgersentry:latest
curl "$(terraform output -raw service_url)/health"
terraform destroy -var project_id=PROJECT_ID
```

Variables: `project_id` (required), `region` (default `us-central1`),
`service_name`, `image`, `allow_unauthenticated` (default `true`).

---

## 3. Render (not deployed)

[`render.yaml`](render.yaml) is a Blueprint for a Docker web service with a
`/health` check. Deploying it needs a Render account connected to the GitHub repo,
so it has not been done. A build from git serves the synthetic model (top of this
file), and the free plan sleeps after about 15 minutes idle.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Aeripsen/ledgersentry)

---

## 4. Local: docker compose

Runs the API on 8000 and the Streamlit dashboard on 8501 from the one image:

```bash
docker compose up --build
curl localhost:8000/health
# dashboard at http://localhost:8501
```

CI brings this stack up with `docker compose up --wait` on every deploy change
([`scripts/compose_smoke.sh`](scripts/compose_smoke.sh)) and checks the API's
`/health`, `/ready` and `/predict` and the dashboard's `/_stcore/health`. Its first
run found a real bug: the dashboard container inherited the image's `HEALTHCHECK`,
which probes the API on port 8000, so the dashboard was always "unhealthy". It now
checks Streamlit's own health endpoint.

---

## Validation status

- CI, every push that touches a deploy file (`k8s.yml`): the kind deployment, PDB
  check, load test and rolling restart (section 1), the Terraform
  apply/re-plan/parity/destroy (2a), `fmt -check` plus `validate` for both Terraform
  modules, `kubeconform -strict` (v0.8.0, its newest published schemas) on the base,
  the kind overlay and the k6 Job, and the docker compose smoke test (section 4).
- CI, every push: `tests/test_deploy_manifests.py` pins the contracts offline:
  selectors, probe paths, named ports, the non-root uid against the Dockerfile,
  `maxUnavailable: 0`, the preStop drain path against `drain.py`, the ConfigMap
  keys against the code defaults, the HPA and PDB targets, and the Terraform
  settings against the YAML ConfigMap. `tests/test_drain_socket.py` runs the drain
  against real uvicorn sockets; `tests/test_k8s_report.py` and
  `tests/test_k8s_drain_ab.py` pin the report arithmetic.
- Manual (`k8s-drain-ab.yml`): the drain A/B above. The committed summary was
  built from both runs' uploaded files, each report regenerated with the current
  `k8s_report.py` so one version of the arithmetic produced every number:

  ```bash
  for run in 36776368018 36777997492; do gh run download $run -p 'ab-*' -D ab/$run; done
  for d in ab/*/ab-*; do
    python scripts/k8s_report.py kind --app ledgersentry --out-dir $d --artifact $d/report.json --no-gate
  done
  python scripts/k8s_drain_ab.py ab --app ledgersentry --out artifacts/k8s_drain_ab_ledgersentry.json
  ```

  GitHub keeps those run files for 90 days (to about 2026-12-29); the summary JSON
  is committed so the numbers outlive them.
- Not done: any cloud deploy (Cloud Run, Render). Each needs an account.
