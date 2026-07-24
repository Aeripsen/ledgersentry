# Terraform: LedgerSentry on Cloud Run

Provisions the cloud infra that hosts the serving container: a Google Cloud Run v2
service running the LedgerSentry image, scaled to zero when idle, with a startup
probe on `/ready` and a liveness probe on `/health`. Optionally grants public
(`allUsers`) invoke access.

```bash
terraform init
terraform validate
terraform apply -var project_id=PROJECT_ID -var image=REGISTRY/ledgersentry:latest
curl "$(terraform output -raw service_url)/health"
terraform destroy -var project_id=PROJECT_ID
```

| Variable | Default | Notes |
|---|---|---|
| `project_id` | (required) | GCP project to deploy into |
| `region` | `us-central1` | Cloud Run region |
| `service_name` | `ledgersentry` | Cloud Run service name |
| `image` | `ghcr.io/aeripsen/ledgersentry:latest` | image Cloud Run pulls (push it first - see ../../DEPLOY.md) |
| `allow_unauthenticated` | `true` | `false` keeps the API private |

Files: `versions.tf` (provider pins), `variables.tf`, `main.tf` (the service + IAM),
`outputs.tf` (`service_url`). Full build-and-push steps: [../../DEPLOY.md](../../DEPLOY.md).
