# vLLM Inference Benchmark Docker Image
# Based on official vLLM image with CUDA support

FROM vllm/vllm-openai:v0.6.4.post1

LABEL maintainer="benchmark-team"
LABEL description="vLLM inference benchmark container"

# Set working directory
WORKDIR /app

# Copy requirements and install additional dependencies
COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install --no-cache-dir -r requirements.lock.txt

# Copy server code
COPY ../vllm /app/server/vllm

# Default port for vLLM API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - can be overridden with specific config
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--help"]

