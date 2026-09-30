output "namespace" {
  description = "Namespace the service runs in."
  value       = kubernetes_namespace_v1.this.metadata[0].name
}

output "service_dns" {
  description = "In-cluster address of the Service."
  value       = "http://${kubernetes_service_v1.api.metadata[0].name}.${kubernetes_namespace_v1.this.metadata[0].name}.svc.cluster.local"
}

output "port_forward" {
  description = "Command that exposes the Service on localhost:8000."
  value       = "kubectl -n ${kubernetes_namespace_v1.this.metadata[0].name} port-forward svc/${kubernetes_service_v1.api.metadata[0].name} 8000:80"
}
