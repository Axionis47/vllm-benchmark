#!/bin/bash
# Startup script for L4 benchmark VM
# This script runs on first boot to configure the VM for GPU inference benchmarking

set -e

LOG_FILE="/var/log/startup-script.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Startup script started at $(date) ==="

# Update system packages
echo "Updating system packages..."
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    systemctl enable docker
    systemctl start docker
    echo "Docker installed successfully"
else
    echo "Docker already installed"
fi

# The Deep Learning VM image should have NVIDIA drivers pre-installed
# Verify and install nvidia-container-toolkit if needed
echo "Checking NVIDIA driver..."
if nvidia-smi; then
    echo "NVIDIA driver is working"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "ERROR: nvidia-smi failed - GPU driver may not be properly installed"
    echo "On Deep Learning VM images, drivers should be pre-installed"
    echo "If this persists, try rebooting or check the VM image"
fi

# Install nvidia-container-toolkit for Docker GPU support
echo "Installing nvidia-container-toolkit..."
if ! command -v nvidia-ctk &> /dev/null; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update
    apt-get install -y nvidia-container-toolkit

    # Configure Docker to use nvidia runtime
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo "nvidia-container-toolkit installed successfully"
else
    echo "nvidia-container-toolkit already installed"
fi

# Add default user to docker group (usually 'ubuntu' on GCP)
for user in ubuntu $SUDO_USER; do
    if id "$user" &>/dev/null; then
        usermod -aG docker "$user"
        echo "Added $user to docker group"
    fi
done

# Verify Docker GPU access
echo "Testing Docker GPU access..."
if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi; then
    echo "Docker GPU access verified successfully"
else
    echo "WARNING: Docker GPU test failed - may need to reboot or check configuration"
fi

# Create working directory for benchmarks
mkdir -p /home/ubuntu/bench-workspace
chown ubuntu:ubuntu /home/ubuntu/bench-workspace

echo "=== Startup script completed at $(date) ==="
echo "VM is ready for inference benchmarking"

