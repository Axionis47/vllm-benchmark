variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
}

variable "zone" {
  description = "GCP zone for the VM (must have L4 availability)"
  type        = string
}

variable "vm_name" {
  description = "Name for the benchmark VM"
  type        = string
  default     = "l4-bench"
}

variable "machine_type" {
  description = "GCP machine type (must be G2 series for L4 GPU)"
  type        = string
  default     = "g2-standard-8"
}

variable "boot_disk_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 200
}

variable "my_ip_cidr" {
  description = "Your IP address in CIDR notation for SSH/API access (e.g., 1.2.3.4/32)"
  type        = string
}

variable "service_account_email" {
  description = "Service account email to attach to the VM"
  type        = string
}

