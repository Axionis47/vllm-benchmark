#!/usr/bin/env python3
"""Smoke test matrix runner.

For each config C1..C8:
1. Start vLLM server
2. Run smoke test
3. Stop vLLM server

Stops on first failure.
"""
import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone

# Configs to run
CONFIGS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

CONFIG_DIR = Path(__file__).parent.parent.parent / "server" / "vllm" / "configs"
LAUNCHER = Path(__file__).parent.parent.parent / "server" / "vllm" / "launcher.py"
SMOKE_RUNNER = Path(__file__).parent / "run_smoke.py"
SELECT_SMOKE = Path(__file__).parent / "select_smoke.py"


def run_command(cmd: list[str], cwd: Path | None = None, timeout: int = 1800) -> tuple[int, str]:
    """Run a command and return (returncode, output)."""
    print(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "Command timed out"


def stop_vllm() -> None:
    """Stop any running vLLM container."""
    subprocess.run(
        ["docker", "stop", "vllm-server"],
        capture_output=True,
    )
    subprocess.run(
        ["docker", "rm", "-f", "vllm-server"],
        capture_output=True,
    )


def start_vllm(config_path: Path, log_path: Path, timeout: int = 900) -> bool:
    """Start vLLM and wait for readiness."""
    stop_vllm()  # Clean up first
    
    code, output = run_command([
        sys.executable, str(LAUNCHER),
        "--config", str(config_path),
        "--action", "start",
        "--log-output", str(log_path),
        "--timeout", str(timeout),
    ], timeout=timeout + 60)
    
    if code != 0:
        print(f"  ERROR: Failed to start vLLM")
        print(output[:500])
        return False
    return True


def run_smoke_test(config_name: str, output_dir: Path, smoke_prompts: Path) -> bool:
    """Run smoke test for a config."""
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    
    code, output = run_command([
        sys.executable, str(SMOKE_RUNNER),
        "--config", str(config_path),
        "--prompts", str(smoke_prompts),
        "--output-dir", str(output_dir),
    ], timeout=600)
    
    print(output[-1000:] if len(output) > 1000 else output)
    return code == 0


def main():
    parser = argparse.ArgumentParser(description="Smoke test matrix runner")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--prompts", type=Path, default=Path("bench/datasets/processed/prompts.jsonl"))
    parser.add_argument("--configs", nargs="+", default=CONFIGS, help="Configs to test")
    parser.add_argument("--server-timeout", type=int, default=900, help="Server startup timeout")
    args = parser.parse_args()

    # Generate run ID
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_base = args.results_dir / f"smoke_matrix_{run_id}"
    results_base.mkdir(parents=True, exist_ok=True)

    # Select smoke prompts
    smoke_prompts = results_base / "smoke_prompts.jsonl"
    print(f"\n{'='*60}")
    print("SELECTING SMOKE PROMPTS")
    print(f"{'='*60}")
    code, output = run_command([
        sys.executable, str(SELECT_SMOKE),
        "--prompts", str(args.prompts),
        "--output", str(smoke_prompts),
    ])
    print(output)
    if code != 0:
        print("ERROR: Failed to select smoke prompts")
        return 1

    # Run smoke tests
    results = {}
    all_passed = True

    for config in args.configs:
        print(f"\n{'='*60}")
        print(f"SMOKE TEST: {config}")
        print(f"{'='*60}")

        config_path = CONFIG_DIR / f"{config}.yaml"
        if not config_path.exists():
            print(f"ERROR: Config not found: {config_path}")
            results[config] = {"status": "config_not_found"}
            all_passed = False
            break

        output_dir = results_base / config
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "vllm.log"

        # Start vLLM
        print(f"\n[1/3] Starting vLLM for {config}...")
        if not start_vllm(config_path, log_path, args.server_timeout):
            results[config] = {"status": "server_start_failed"}
            all_passed = False
            break

        # Run smoke test
        print(f"\n[2/3] Running smoke test...")
        passed = run_smoke_test(config, output_dir, smoke_prompts)

        # Stop vLLM
        print(f"\n[3/3] Stopping vLLM...")
        stop_vllm()

        results[config] = {
            "status": "passed" if passed else "failed",
            "output_dir": str(output_dir),
        }

        if not passed:
            all_passed = False
            print(f"\nERROR: Smoke test failed for {config}")
            break

        print(f"✓ {config} passed")
        time.sleep(5)  # Brief pause between configs

    # Save matrix results
    matrix_summary = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "configs_tested": list(results.keys()),
        "all_passed": all_passed,
        "results": results,
    }

    with open(results_base / "matrix_summary.json", "w") as f:
        json.dump(matrix_summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SMOKE MATRIX SUMMARY")
    print(f"{'='*60}")
    for config, result in results.items():
        status = result["status"]
        icon = "✓" if status == "passed" else "✗"
        print(f"  {icon} {config}: {status}")

    print(f"\nResults saved to: {results_base}")

    if all_passed:
        print(f"\n{'='*60}")
        print("ALL SMOKE TESTS PASSED")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n{'='*60}")
        print("SMOKE TESTS FAILED - STOPPING")
        print(f"{'='*60}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

