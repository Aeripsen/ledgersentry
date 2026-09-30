provider "google" {
  project = var.project_id
  region  = var.region
}

# The cloud infra that hosts the container: a serverless Cloud Run service
# running the LedgerSentry serving image. Cloud Run is the minimal single-
# container host - it pulls the image, scales it, and terminates TLS - so this
# is the whole "where does it run" story for one FastAPI service.
resource "google_cloud_run_v2_service" "ledgersentry" {
  name     = var.service_name
  location = var.region

  # Off so `terraform destroy` can tear the demo down without a console step.
  deletion_protection = false

  template {
    scaling {
      min_instance_count = 0 # scale to zero when idle: no cost between requests
      max_instance_count = 4
    }

    containers {
      image = var.image

      ports {
        container_port = 8000
      }

      env {
        name  = "PORT"
        value = "8000"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      # Route traffic only once /ready reports the compiled scorer is built.
      startup_probe {
        http_get {
          path = "/ready"
          port = 8000
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        timeout_seconds       = 3
        failure_threshold     = 10
      }

      # /health stays 200 while the process is up, so a warming instance is never
      # killed for being slow to load its model.
      liveness_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        initial_delay_seconds = 20
        period_seconds        = 30
        timeout_seconds       = 5
        failure_threshold     = 3
      }
    }
  }
}

# Public invocation. Cloud Run v2 IAM references the service by `name`, not
# `service` (the v1 argument) - a real difference `terraform validate` enforces.
resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.allow_unauthenticated ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.ledgersentry.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
