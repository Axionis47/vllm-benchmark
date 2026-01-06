output "instance_name" {
  description = "Name of the created VM instance"
  value       = google_compute_instance.l4_bench.name
}

output "external_ip" {
  description = "External IP address of the VM"
  value       = google_compute_instance.l4_bench.network_interface[0].access_config[0].nat_ip
}

output "internal_ip" {
  description = "Internal IP address of the VM"
  value       = google_compute_instance.l4_bench.network_interface[0].network_ip
}

output "zone" {
  description = "Zone where the VM is deployed"
  value       = google_compute_instance.l4_bench.zone
}

output "machine_type" {
  description = "Machine type of the VM"
  value       = google_compute_instance.l4_bench.machine_type
}

output "ssh_command" {
  description = "Command to SSH into the VM"
  value       = "gcloud compute ssh ${google_compute_instance.l4_bench.name} --zone=${google_compute_instance.l4_bench.zone}"
}

output "vllm_endpoint" {
  description = "URL for the vLLM API endpoint"
  value       = "http://${google_compute_instance.l4_bench.network_interface[0].access_config[0].nat_ip}:8000"
}

