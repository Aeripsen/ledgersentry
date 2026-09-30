variable "project_id" {
  description = "GCP project ID to deploy the Cloud Run service into."
  type        = string
}

variable "region" {
  description = "Cloud Run region."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name."
  type        = string
  default     = "ledgersentry"
}

variable "image" {
  description = "Container image URL built from the repo Dockerfile (Artifact Registry or a public registry Cloud Run can pull)."
  type        = string
  default     = "ghcr.io/aeripsen/ledgersentry:latest"
}

variable "allow_unauthenticated" {
  description = "Grant allUsers the run.invoker role so the API is publicly reachable. Set false to keep it private."
  type        = bool
  default     = true
}
