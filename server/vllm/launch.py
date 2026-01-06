#!/usr/bin/env python3
"""vLLM Server Launcher

Reads a YAML configuration and generates/executes the vLLM serve command.
In dry-run mode, only prints the command without executing (for CI validation).
"""

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class VLLMConfig:
    """Configuration for vLLM server."""

    config_id: str
    config_name: str
    model: str
    dtype: str = "float16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    host: str = "0.0.0.0"
    port: int = 8000
    enforce_eager: bool = False
    disable_log_requests: bool = False
    description: str = ""
    extra_args: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "VLLMConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Extract known fields
        known_fields = {
            "config_id",
            "config_name",
            "model",
            "dtype",
            "max_model_len",
            "gpu_memory_utilization",
            "tensor_parallel_size",
            "host",
            "port",
            "enforce_eager",
            "disable_log_requests",
            "description",
        }

        config_kwargs = {k: v for k, v in data.items() if k in known_fields}
        extra_args = {k: v for k, v in data.items() if k not in known_fields}

        return cls(**config_kwargs, extra_args=extra_args)

    def config_hash(self) -> str:
        """Generate a hash of the configuration for reproducibility."""
        # Sort keys for deterministic hashing
        config_str = yaml.dump(self.__dict__, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def to_command(self) -> list[str]:
        """Generate vLLM serve command arguments."""
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--dtype",
            self.dtype,
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        if self.enforce_eager:
            cmd.append("--enforce-eager")

        if self.disable_log_requests:
            cmd.append("--disable-log-requests")

        return cmd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch vLLM server with configuration"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )
    parser.add_argument(
        "--print-hash",
        action="store_true",
        help="Print configuration hash and exit",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    config = VLLMConfig.from_yaml(args.config)

    if args.print_hash:
        print(f"Config ID: {config.config_id}")
        print(f"Config Hash: {config.config_hash()}")
        return 0

    cmd = config.to_command()

    print(f"Configuration: {config.config_name} ({config.config_id})")
    print(f"Config Hash: {config.config_hash()}")
    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        print("\n[Dry run - not executing]")
        return 0

    # Execute the command
    print("\nStarting vLLM server...")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())

