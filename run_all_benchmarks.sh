#!/bin/bash
cd ~/bench-workspace

run_config() {
    CONFIG=$1
    echo ""
    echo "============================================================"
    echo "STARTING $CONFIG at $(date)"
    echo "============================================================"
    
    # Stop existing container
    cd ~/bench-workspace/server/vllm
    docker compose down -v 2>/dev/null
    sleep 5
    
    # Generate new compose file
    cd ~/bench-workspace
    .venv/bin/python server/vllm/launcher.py --config server/vllm/configs/${CONFIG}.yaml --action generate
    
    # Start container
    cd ~/bench-workspace/server/vllm
    docker compose up -d
    
    # Wait for ready
    echo "Waiting for vLLM to be ready..."
    for i in {1..90}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "Ready after ${i}0 seconds"
            break
        fi
        sleep 10
    done
    
    # Run benchmark
    cd ~/bench-workspace
    OUTPUT_DIR=results/full_$(echo $CONFIG | tr A-Z a-z)
    rm -rf $OUTPUT_DIR  # Clean previous partial run
    .venv/bin/python -m bench.runner.run_full --config server/vllm/configs/${CONFIG}.yaml --output-dir $OUTPUT_DIR 2>&1
    
    echo "Completed $CONFIG at $(date)"
}

# Run C2 through C8
for config in C2 C3 C4 C5 C6 C7 C8; do
    run_config $config
done

echo ""
echo "============================================================"
echo "ALL BENCHMARKS COMPLETE at $(date)"
echo "============================================================"

