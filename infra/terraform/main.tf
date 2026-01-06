# L4 Benchmark VM Infrastructure
# G2 instances come with L4 GPU attached - no separate guest_accelerator needed

resource "google_compute_instance" "l4_bench" {
  name         = var.vm_name
  machine_type = var.machine_type
  zone         = var.zone

  # G2 instances use a specific boot disk image optimized for GPU workloads
  boot_disk {
    initialize_params {
      # Deep Learning VM image with CUDA 12.8 + NVIDIA 570 driver pre-installed
      image = "projects/deeplearning-platform-release/global/images/family/common-cu128-ubuntu-2204-nvidia-570"
      size  = var.boot_disk_gb
      type  = "pd-ssd"
    }
  }

  network_interface {
    network = "default"
    access_config {
      # Ephemeral public IP
    }
  }

  # G2 machine types include L4 GPU - scheduling must allow maintenance migration
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
    preemptible         = false
  }

  service_account {
    email  = var.service_account_email
    scopes = ["cloud-platform"]
  }

  metadata = {
    startup-script = file("${path.module}/startup.sh")
  }

  # Ensure GPU driver is installed before marking instance ready
  metadata_startup_script = file("${path.module}/startup.sh")

  tags = ["l4-bench", "allow-ssh", "allow-vllm"]

  labels = {
    purpose    = "inference-benchmark"
    gpu        = "l4"
    managed-by = "terraform"
  }
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.vm_name}-allow-ssh"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = [var.my_ip_cidr]
  target_tags   = ["allow-ssh"]

  description = "Allow SSH access from specified IP"
}

resource "google_compute_firewall" "allow_vllm" {
  name    = "${var.vm_name}-allow-vllm"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = [var.my_ip_cidr]
  target_tags   = ["allow-vllm"]

  description = "Allow vLLM API access from specified IP"
}

