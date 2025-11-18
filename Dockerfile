# Multi-stage build for optimized production image
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn[standard] fastapi

# Copy application code
COPY service/ ./service/

# Copy DVC-tracked models and artifacts
# These must be pulled before building (via build_docker.sh)
COPY models/model_ann.keras ./models/
COPY models/model_gru.keras ./models/
COPY models/model_lstm.keras ./models/
COPY models/model_transformer.keras ./models/
COPY artifacts/ensemble/ ./artifacts/ensemble/
COPY artifacts/drift_detection/ ./artifacts/drift_detection/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://localhost:8000/', timeout=5)"

EXPOSE 8000

# Run with uvicorn (production-ready ASGI server)
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
