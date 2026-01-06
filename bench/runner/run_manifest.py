#!/usr/bin/env python3
"""Generate run manifest with environment metadata.

Captures:
- git_sha
- vllm_version
- model_id
- docker_image_tag
- gpu_name, driver_version, cuda_version (from nvidia-smi)
- machine_type, zone, instance_name (from GCP metadata)
"""
import json
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_nvidia_info() -> dict[str, str]:
    """Get GPU info from nvidia-smi."""
    info = {
        "gpu_name": "unknown",
        "driver_version": "unknown",
        "cuda_version": "unknown",
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 2:
                info["gpu_name"] = parts[0]
                info["driver_version"] = parts[1]

        # Get CUDA version separately
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Parse CUDA version from nvidia-smi header
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split("\n"):
                if "CUDA Version:" in line:
                    cuda_part = line.split("CUDA Version:")[1].strip().split()[0]
                    info["cuda_version"] = cuda_part
                    break
    except Exception:
        pass
    return info


def get_gcp_metadata() -> dict[str, str]:
    """Get GCP instance metadata."""
    metadata = {
        "machine_type": "unknown",
        "zone": "unknown",
        "instance_name": "unknown",
    }
    base_url = "http://metadata.google.internal/computeMetadata/v1"
    headers = {"Metadata-Flavor": "Google"}

    endpoints = {
        "machine_type": "/instance/machine-type",
        "zone": "/instance/zone",
        "instance_name": "/instance/name",
    }

    for key, path in endpoints.items():
        try:
            req = urllib.request.Request(base_url + path, headers=headers)
            with urllib.request.urlopen(req, timeout=2) as resp:
                value = resp.read().decode("utf-8")
                # machine-type and zone return full path, extract just the name
                if "/" in value:
                    value = value.split("/")[-1]
                metadata[key] = value
        except Exception:
            pass

    return metadata


def get_docker_vllm_version() -> dict[str, str]:
    """Get vLLM version from running docker container."""
    info = {"vllm_version": "unknown", "docker_image_tag": "unknown"}
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=vllm", "--format", "{{.Image}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["docker_image_tag"] = result.stdout.strip()
            # Extract version from tag like vllm/vllm-openai:v0.10.0
            if ":" in info["docker_image_tag"]:
                info["vllm_version"] = info["docker_image_tag"].split(":")[-1]
    except Exception:
        pass
    return info


def generate_manifest(model_id: str = "unknown") -> dict[str, Any]:
    """Generate complete run manifest."""
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_sha": get_git_sha(),
        "model_id": model_id,
    }

    manifest.update(get_docker_vllm_version())
    manifest.update(get_nvidia_info())
    manifest.update(get_gcp_metadata())

    return manifest


def save_manifest(output_path: Path, model_id: str = "unknown") -> dict[str, Any]:
    """Generate and save manifest to file."""
    manifest = generate_manifest(model_id)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    import sys
    model_id = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("manifest.json")
    manifest = save_manifest(output, model_id)
    print(json.dumps(manifest, indent=2))

