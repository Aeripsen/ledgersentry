# Deploying LedgerSentry

The serving image (`Dockerfile`) is self-contained: if no trained model is baked
in - which is the case for any build from a clean git checkout, because the
`.joblib` is gitignored - the build generates the deterministic synthetic-fixture
model the same way `scripts/train.py` does with no data file present. So the API
comes up green on first boot with zero data download. Drop a real dataset in
`data/` and rebuild (or train first) and that real model is served instead; the
build keeps a model it finds rather than overwriting it.

The service listens on `$PORT` (default `8000`) and exposes:

- `GET /health` - liveness, 200 while the process is up (even while the model loads)
- `GET /ready` - readiness, 200 only once the compiled scorer can actually score
- `POST /predict`, `POST /predict/batch`, `POST /drift`, `GET /curve`

Three deploy targets are wired and validated in this repo. Pick one.

---

## 1. Render - one click (the fastest path)

Render reads [`render.yaml`](render.yaml) and builds the Dockerfile. Nothing to
install locally.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Aeripsen/ledgersentry)

Or do it from the dashboard (the reliable, explicit path):

1. Go to <https://dashboard.render.com/blueprints> and click **New Blueprint Instance**.
2. **Connect** your GitHub account (first time only) and pick the
   `Aeripsen/ledgersentry` repo.
3. Render detects `render.yaml` and shows the `ledgersentry` web service. Name the
   blueprint and click **Apply**.
4. Render builds the image (the build bakes the model in) and deploys it. When the
   build finishes it shows a live `https://ledgersentry-*.onrender.com` URL and
   starts health-checking `/health`.

That is the whole thing: connect the account, click Apply. Then:

```bash
curl https://<your-service>.onrender.com/health
curl -X POST https://<your-service>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"amount": 812.50, "category": "electronics", "timestamp": "2026-01-15T02:14:00"}, "review_threshold": 0.9}'
```

Note: `plan: free` in `render.yaml` spins the service down after ~15 min idle and
cold-starts it on the next request (fine for a demo). Switch to a paid plan in
`render.yaml` for always-on.

---

## 2. Kubernetes

Manifest: [`deploy/k8s/ledgersentry.yaml`](deploy/k8s/ledgersentry.yaml) - a
2-replica Deployment (non-root, liveness on `/health`, readiness on `/ready`) plus
a ClusterIP Service. Validate it offline without a cluster:

```bash
kubectl apply --dry-run=client -f deploy/k8s/ledgersentry.yaml
```

Deploy it for real. First build and push the image to a registry your cluster can
pull (GitHub Container Registry shown; Artifact Registry works the same way):

```bash
docker build -t ghcr.io/aeripsen/ledgersentry:latest .
docker push ghcr.io/aeripsen/ledgersentry:latest
# if you push somewhere else, update the `image:` in deploy/k8s/ledgersentry.yaml

kubectl apply -f deploy/k8s/ledgersentry.yaml
kubectl rollout status deploy/ledgersentry
kubectl port-forward svc/ledgersentry 8000:80
curl localhost:8000/health          # in another shell
```

---

## 3. Terraform - GCP Cloud Run

Module: [`deploy/terraform/`](deploy/terraform) - provisions a serverless Cloud Run
service (scale-to-zero, startup probe on `/ready`, liveness on `/health`) and,
optionally, public access. Cloud Run is the minimal single-container host: it pulls
the image, scales it, and terminates TLS.

One-time GCP setup (you already have `gcloud` authed):

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# build + push to Artifact Registry (Cloud Run pulls from here)
gcloud auth configure-docker us-central1-docker.pkg.dev
gcloud artifacts repositories create ledgersentry --repository-format=docker --location=us-central1
docker build -t us-central1-docker.pkg.dev/PROJECT_ID/ledgersentry/ledgersentry:latest .
docker push  us-central1-docker.pkg.dev/PROJECT_ID/ledgersentry/ledgersentry:latest
```

Then apply:

```bash
cd deploy/terraform
terraform init
terraform validate
terraform apply \
  -var project_id=PROJECT_ID \
  -var image=us-central1-docker.pkg.dev/PROJECT_ID/ledgersentry/ledgersentry:latest

curl "$(terraform output -raw service_url)/health"
terraform destroy -var project_id=PROJECT_ID   # tear it down
```

Variables (`deploy/terraform/variables.tf`): `project_id` (required), `region`
(default `us-central1`), `service_name`, `image`, and `allow_unauthenticated`
(default `true` - set `false` to keep the API private).

---

## 4. Local - docker compose

Runs the API on 8000 and the Streamlit dashboard on 8501 from the one image:

```bash
docker compose up --build
curl localhost:8000/health
# dashboard at http://localhost:8501
```

---

## Validation status

- `render.yaml`, the k8s manifest, and the Dockerfile contract are checked in CI
  (`tests/test_deploy_manifests.py`): service selectors match pod labels, the
  probes point at `/health` and `/ready`, ports line up, and the k8s `runAsUser`
  matches the Dockerfile's non-root uid.
- The Terraform parses as valid HCL2 and is schema-checked against the current
  `hashicorp/google` provider docs (notably the Cloud Run v2 IAM member, which
  references the service by `name`, not the v1 `service`).
- The self-contained build-and-serve path (generate the synthetic model when no
  data file is present, then answer `/health`, `/ready`, `/predict`) is verified by
  running those exact steps locally.
- Not yet run on the build machine: `docker build`, `kubectl`, and `terraform`
  themselves are not installed here, so the container build and a live deploy are
  the one step left, and that step needs your cloud login.
