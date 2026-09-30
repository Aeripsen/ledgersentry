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
rollout, autoscaling, draining), not about model quality.

The service listens on `$PORT` (default `8000`) and exposes:

- `GET /health` - liveness, 200 while the process is up (even while the model loads)
- `GET /ready` - readiness, 200 only once the compiled scorer can actually score
- `POST /predict`, `POST /predict/batch`, `POST /drift`, `GET /curve`

## What has actually run

| Target | Status | Evidence |
|---|---|---|
| Kubernetes (kind) | Deployed, smoke-tested, load-tested and rolled under load in CI on every deploy change | [`.github/workflows/k8s.yml`](.github/workflows/k8s.yml), [`artifacts/k8s_kind_ledgersentry.json`](artifacts/k8s_kind_ledgersentry.json) |
| Terraform, `hashicorp/kubernetes` | Applied to kind, re-planned, diffed against the YAML, destroyed, in CI | [`artifacts/terraform_kind_ledgersentry.json`](artifacts/terraform_kind_ledgersentry.json) |
| Terraform, GCP Cloud Run | `fmt -check` and `validate` in CI only. **Never applied** | needs a GCP project with billing |
| Render | **Not deployed** | needs a Render account connected to the repo |
| docker compose | Runs locally | section 4 |

Everything in CI runs on a free GitHub-hosted runner (4 vCPU, 16 GB). No cloud
account is used anywhere.

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
```

That script is exactly what CI runs:

1. `docker build -t ledgersentry:ci .`, `kind create cluster`, `kind load docker-image`
2. install metrics-server (the HPA needs it), `kubectl apply -k deploy/k8s/overlays/kind`,
   `kubectl rollout status` (returns once every pod passed `/ready`)
3. through the Service: `/health`, `/ready`, `/predict`, and `/openapi.json`, whose
   `/predict/batch` `maxItems` must equal the ConfigMap value; record PID 1 and the uid
4. steady load: 16 k6 virtual users for 90 s, closed loop, while logging the HPA
5. rolling restart under load: `kubectl rollout restart` 15 s into a 120 s k6 run
6. [`scripts/k8s_report.py`](scripts/k8s_report.py) writes the artifact and **fails
   the job on any failed request**, an unreplaced pod, a rollout that outlasted the
   load, a ConfigMap value that did not reach the process, or a shell as PID 1

To deploy the base to a real cluster instead, push the image somewhere the cluster
can pull, set `image:` and run `kubectl apply -k deploy/k8s/base`.

### Captured output

From CI run [36771694619](https://github.com/Aeripsen/ledgersentry/actions/runs/36771694619)
on master (commit `05cc9ff`), kind v0.33.0, Kubernetes v1.37.0. The full JSON is
[`artifacts/k8s_kind_ledgersentry.json`](artifacts/k8s_kind_ledgersentry.json).

```text
rollout: 2 pods ready in 7 s
PID 1:   /usr/local/bin/python3.12 /usr/local/bin/uvicorn ledgersentry.service:app ...
runs as: uid=10001(app)
ConfigMap LEDGERSENTRY_MAX_BATCH=2000, served /predict/batch maxItems=2000
/predict: {"decision": "legit", "p_fraud": 0.0002, "confidence": 0.9998, ...}

steady (16 VUs, 90 s):  31,730 requests, 352.3 req/s, p50 42.5 ms, p99 97.6 ms, 0 failed
rolling restart (120 s): 108,825 requests, 906.8 req/s, p50 14.7 ms, p99 56.0 ms, 0 failed
  restart began 15 s into the load, took 36 s, finished before the load ended,
  all 5 pods replaced
HPA: 2 -> 5 replicas, CPU peaked at 384% of the 250m request
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
a pod into drain mode. Checked on real sockets that uvicorn closes the TCP
connection after such a response with both of its HTTP parsers (httptools, h11).

After the fix, three runs of the same test all passed with 0 failed requests:
74,490 and 71,629 requests
([run 36770455800](https://github.com/Aeripsen/ledgersentry/actions/runs/36770455800),
attempts 1 and 2) and 108,825 (the master run above), 254,944 in total. This is
a rare race, so zero failures is evidence, not proof: if the fix did nothing and
the failure rate stayed at the first run's 4 per 114,308, zero failures in 254,944
requests would have a probability of about e^-8.9, roughly 0.0001.

**Keep-alive also defeats the autoscaler's new pods.** In all four runs the HPA
reached 5 replicas 50 to 51 s after the load started. But at the end of the steady
phase `kubectl top` showed only 2 of the 5 pods working in three runs, and 3 of 5
in the first, each at about 1000m (the CPU limit), with the rest idle at 2m. The
16 k6 connections were opened before the scale-up and stayed on the pods that
existed then, because kube-proxy balances connections, not requests. The rolling
restart forced every client to reconnect, which spread them over all 5 pods, and
throughput rose 2.6x (352.3 to 906.8 req/s) in the master run. A real deployment
would fix this with an L7 proxy or service mesh that balances per request, or by
closing client connections periodically. Not done here.

**Memory** was about 140 Mi per pod under load, against a 256Mi request and a
512Mi limit.

### What these numbers are not

- One single-node kind cluster on one 4-vCPU runner, with the k6 load generator in
  the same cluster, competing for the same CPUs as the pods.
- Synthetic load: one fixed request body in a closed loop. Not production traffic,
  not users.
- Throughput varies a lot between runners. Across the four runs above, steady-phase
  throughput was 630.6, 238.8, 239.2 and 352.3 req/s. The CI gate is therefore on
  failed requests, never on a throughput number.
- The served model is the synthetic fixture (see the top of this file).

---

## 2. Terraform

### 2a. Kubernetes provider (applied in CI)

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
(the live Terraform objects against a server-side dry run of `deploy/k8s/base`, so
the two definitions cannot drift), then `terraform destroy`.

From the same master run, Terraform 1.16.4, provider 3.2.1
([`artifacts/terraform_kind_ledgersentry.json`](artifacts/terraform_kind_ledgersentry.json)):

```text
Apply complete! Resources: 6 added, 0 changed, 0 destroyed.      (8 s)
second plan: No changes. Your infrastructure matches the configuration.
parity: 5 objects compared, 0 mismatches
Destroy complete! Resources: 6 destroyed.
```

The parity check earned its place on the first run: the provider always writes an
HPA `behavior` block (the Kubernetes default scaling policies) while the YAML left
it unset. Both now state the policy explicitly.

To use it against another cluster: `terraform -chdir=deploy/terraform/kubernetes
apply -var kube_context=<context> -var image=<registry>/ledgersentry:<tag>`.

### 2b. GCP Cloud Run (validated, never applied)

[`deploy/terraform/cloudrun`](deploy/terraform/cloudrun) provisions a scale-to-zero
Cloud Run v2 service (startup probe on `/ready`, liveness on `/health`) and,
optionally, public access. CI runs `terraform fmt -check` and `terraform validate`
on it. **It has never been applied**: that needs a GCP project with a billing
account, and nothing in this repo creates one.

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

---

## Validation status

- CI, every push that touches a deploy file: the kind deployment, load test and
  rolling restart (section 1), the Terraform apply/re-plan/parity/destroy (2a), and
  `fmt -check` plus `validate` for both Terraform modules.
- CI, every push: `tests/test_deploy_manifests.py` pins the contracts offline:
  selectors, probe paths, named ports, the non-root uid against the Dockerfile,
  `maxUnavailable: 0`, the preStop drain path against `drain.py`, the ConfigMap
  keys against the code defaults, the HPA and PDB targets, and the Terraform
  settings against the YAML ConfigMap.
- Locally, before the first push: `kubeconform -strict` against the Kubernetes 1.33
  schemas on the base, the kind overlay and the k6 Job.
- Not done: any cloud deploy (Cloud Run, Render). Each needs an account.
