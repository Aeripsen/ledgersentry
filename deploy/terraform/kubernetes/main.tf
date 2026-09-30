provider "kubernetes" {
  config_path    = var.kubeconfig_path
  config_context = var.kube_context
}

# The same objects as deploy/k8s/base/ledgersentry.yaml, written as typed
# Terraform resources. CI applies this module to a kind cluster, waits for the
# rollout, smoke-tests the Service, checks that a second plan is empty, and
# diffs the live Deployment against the YAML (scripts/k8s_parity.py) so the two
# definitions cannot drift apart. See DEPLOY.md for the captured run.

locals {
  labels = { app = "ledgersentry" }
}

resource "kubernetes_namespace_v1" "this" {
  metadata {
    name = var.namespace
  }
}

resource "kubernetes_config_map_v1" "settings" {
  metadata {
    name      = "ledgersentry-config"
    namespace = kubernetes_namespace_v1.this.metadata[0].name
    labels    = local.labels
  }
  data = var.settings
}

resource "kubernetes_deployment_v1" "api" {
  metadata {
    name      = "ledgersentry"
    namespace = kubernetes_namespace_v1.this.metadata[0].name
    labels    = local.labels
  }

  # apply returns only once the rollout is complete, which means every pod has
  # passed its /ready probe. A pod that cannot score fails the apply.
  wait_for_rollout = true

  spec {
    replicas = var.replicas

    selector {
      match_labels = local.labels
    }

    strategy {
      type = "RollingUpdate"
      rolling_update {
        max_surge       = "1"
        max_unavailable = "0"
      }
    }

    template {
      metadata {
        labels = local.labels
      }

      spec {
        enable_service_links             = false
        termination_grace_period_seconds = 30

        security_context {
          run_as_non_root = true
          run_as_user     = 10001
          seccomp_profile {
            type = "RuntimeDefault"
          }
        }

        container {
          name              = "api"
          image             = var.image
          image_pull_policy = "IfNotPresent"

          port {
            name           = "http"
            container_port = 8000
          }

          env {
            name  = "PORT"
            value = "8000"
          }

          env_from {
            config_map_ref {
              name = kubernetes_config_map_v1.settings.metadata[0].name
            }
          }

          readiness_probe {
            http_get {
              path = "/ready"
              port = "http"
            }
            initial_delay_seconds = 5
            period_seconds        = 5
            timeout_seconds       = 3
            failure_threshold     = 3
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = "http"
            }
            initial_delay_seconds = 20
            period_seconds        = 30
            timeout_seconds       = 5
            failure_threshold     = 3
          }

          lifecycle {
            pre_stop {
              exec {
                command = ["sleep", "5"]
              }
            }
          }

          resources {
            requests = {
              cpu    = "250m"
              memory = "256Mi"
            }
            limits = {
              cpu    = "1"
              memory = "512Mi"
            }
          }

          security_context {
            allow_privilege_escalation = false
            capabilities {
              drop = ["ALL"]
            }
          }
        }
      }
    }
  }

  # The HPA changes spec.replicas at runtime. Without this, every plan would
  # try to scale the Deployment back to var.replicas.
  lifecycle {
    ignore_changes = [spec[0].replicas]
  }
}

resource "kubernetes_service_v1" "api" {
  metadata {
    name      = "ledgersentry"
    namespace = kubernetes_namespace_v1.this.metadata[0].name
    labels    = local.labels
  }
  spec {
    type     = "ClusterIP"
    selector = local.labels
    port {
      name        = "http"
      port        = 80
      target_port = "http"
    }
  }
}

resource "kubernetes_horizontal_pod_autoscaler_v2" "api" {
  metadata {
    name      = "ledgersentry"
    namespace = kubernetes_namespace_v1.this.metadata[0].name
    labels    = local.labels
  }
  spec {
    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = kubernetes_deployment_v1.api.metadata[0].name
    }

    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = var.cpu_target_percent
        }
      }
    }
  }
}

resource "kubernetes_pod_disruption_budget_v1" "api" {
  metadata {
    name      = "ledgersentry"
    namespace = kubernetes_namespace_v1.this.metadata[0].name
    labels    = local.labels
  }
  spec {
    min_available = "1"
    selector {
      match_labels = local.labels
    }
  }
}
