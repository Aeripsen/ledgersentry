variable "kubeconfig_path" {
  description = "Kubeconfig the provider reads. The default is where kind and most tools write it."
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "Kubeconfig context to deploy into, e.g. kind-ledgersentry. Null uses the current context."
  type        = string
  default     = null
}

variable "namespace" {
  description = "Namespace created for the service."
  type        = string
  default     = "ledgersentry"
}

variable "image" {
  description = "Serving image built from the repo Dockerfile. For kind, `kind load docker-image` it first."
  type        = string
  default     = "ghcr.io/aeripsen/ledgersentry:latest"
}

variable "replicas" {
  description = "Starting replica count. After that the HPA owns it, between min_replicas and max_replicas."
  type        = number
  default     = 2
}

variable "min_replicas" {
  description = "HPA floor."
  type        = number
  default     = 2
}

variable "max_replicas" {
  description = "HPA ceiling."
  type        = number
  default     = 5
}

variable "cpu_target_percent" {
  description = "HPA target: average CPU utilization as a percent of the pod's CPU request."
  type        = number
  default     = 70
}

variable "settings" {
  description = "LEDGERSENTRY_* settings for the ConfigMap (src/ledgersentry/config.py). The defaults equal the code defaults."
  type        = map(string)
  default = {
    LEDGERSENTRY_MAX_BATCH = "10000"
    LEDGERSENTRY_PSI_BINS  = "10"
    LEDGERSENTRY_PSI_WATCH = "0.1"
    LEDGERSENTRY_PSI_ALERT = "0.25"
  }
}
