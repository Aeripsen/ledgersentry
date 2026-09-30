output "service_url" {
  description = "Public HTTPS URL of the deployed Cloud Run service."
  value       = google_cloud_run_v2_service.ledgersentry.uri
}

output "service_name" {
  description = "Deployed Cloud Run service name."
  value       = google_cloud_run_v2_service.ledgersentry.name
}
