# Multi-stage Dockerfile for SuisenSha (CPU-only PyTorch)
# Builder stage: install dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first to avoid massive CUDA wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch>=2.0.0 --extra-index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
COPY requirements.txt .
COPY README.md .
COPY src/ ./src/

# Install dependencies into /install
RUN pip install --no-cache-dir --prefix=/install .

# Runtime stage: minimal image with non-root user
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root user
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/ /app/src/
COPY pyproject.toml /app/

# Install the app itself in editable mode without reinstalling deps
RUN pip install --no-cache-dir --no-deps -e .

# Prepare directories with non-root permissions
RUN mkdir -p /app/data /app/artifacts /app/outputs && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000 8501

# Healthcheck testing API readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')" || exit 1

CMD ["uvicorn", "pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
