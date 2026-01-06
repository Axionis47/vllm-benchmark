#!/usr/bin/env python3
"""Full benchmark matrix runner.

First runs smoke matrix to validate all configs,
then runs full benchmark for each config C1..C8.

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
FULL_RUNNER = Path(__file__).parent / "run_full.py"
SMOKE_MATRIX = Path(__file__).parent / "run_smoke_matrix.py"


def run_command(cmd: list[str], timeout: int = 7200) -> tuple[int, str]:
    """Run a command and return (returncode, output)."""
    print(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "Command timed out"


def stop_vllm() -> None:
    """Stop any running vLLM container."""
    subprocess.run(["docker", "stop", "vllm-server"], capture_output=True)
    subprocess.run(["docker", "rm", "-f", "vllm-server"], capture_output=True)


def start_vllm(config_path: Path, log_path: Path, timeout: int = 900) -> bool:
    """Start vLLM and wait for readiness."""
    stop_vllm()
    
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


def run_full_benchmark(config_name: str, output_dir: Path, prompts_path: Path) -> bool:
    """Run full benchmark for a config."""
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    
    code, output = run_command([
        sys.executable, str(FULL_RUNNER),
        "--config", str(config_path),
        "--prompts", str(prompts_path),
        "--output-dir", str(output_dir),
        "--concurrency", "4",
    ], timeout=7200)  # 2 hour timeout for full run
    
    # Print last portion of output
    print(output[-2000:] if len(output) > 2000 else output)
    return code == 0


def main():
    parser = argparse.ArgumentParser(description="Full benchmark matrix runner")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--prompts", type=Path, default=Path("bench/datasets/processed/prompts.jsonl"))
    parser.add_argument("--configs", nargs="+", default=CONFIGS)
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke gate (dangerous!)")
    parser.add_argument("--server-timeout", type=int, default=900)
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_base = args.results_dir / f"full_matrix_{run_id}"
    results_base.mkdir(parents=True, exist_ok=True)

    # Run smoke gate first (unless skipped)
    if not args.skip_smoke:
        print(f"\n{'='*60}")
        print("PHASE 1: SMOKE GATE")
        print(f"{'='*60}")
        
        code, output = run_command([
            sys.executable, str(SMOKE_MATRIX),
            "--results-dir", str(args.results_dir),
            "--prompts", str(args.prompts),
            "--configs", *args.configs,
        ], timeout=3600)
        
        print(output[-3000:] if len(output) > 3000 else output)
        
        if code != 0:
            print("\n" + "="*60)
            print("SMOKE GATE FAILED - ABORTING FULL MATRIX")
            print("="*60)
            return 1
        
        print("\n✓ Smoke gate passed - proceeding to full benchmark")

    # Run full benchmarks
    print(f"\n{'='*60}")
    print("PHASE 2: FULL BENCHMARK MATRIX")
    print(f"{'='*60}")

    results = {}
    all_passed = True

    for config in args.configs:
        print(f"\n{'='*60}")
        print(f"FULL BENCHMARK: {config}")
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
            # Capture logs for debugging
            subprocess.run(
                ["docker", "logs", "vllm-server"],
                stdout=open(log_path, "w"),
                stderr=subprocess.STDOUT,
            )
            all_passed = False
            break

        # Run full benchmark
        print(f"\n[2/3] Running full benchmark...")
        passed = run_full_benchmark(config, output_dir, args.prompts)

        # Stop vLLM
        print(f"\n[3/3] Stopping vLLM...")
        stop_vllm()

        results[config] = {
            "status": "passed" if passed else "failed",
            "output_dir": str(output_dir),
        }

        if not passed:
            all_passed = False
            print(f"\nERROR: Full benchmark failed for {config}")
            break

        print(f"✓ {config} completed")
        time.sleep(10)  # Pause between configs

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
    print("FULL MATRIX SUMMARY")
    print(f"{'='*60}")
    for config, result in results.items():
        status = result["status"]
        icon = "✓" if status == "passed" else "✗"
        print(f"  {icon} {config}: {status}")

    print(f"\nResults saved to: {results_base}")

    if all_passed:
        print(f"\n{'='*60}")
        print("ALL FULL BENCHMARKS COMPLETED")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n{'='*60}")
        print("FULL BENCHMARK MATRIX FAILED")
        print(f"{'='*60}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

