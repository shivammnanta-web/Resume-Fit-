# Multi-stage, production-hardened Dockerfile for ResumeFit
# Stage 1: Build dependencies & pre-download models
FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch to minimize image footprint (avoids 2GB+ CUDA bloat)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy project manifest and install runtime dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Download lightweight spaCy model
RUN python -m spacy download en_core_web_sm

# Pre-cache Sentence-Transformer embedding model for zero cold-start delay
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Stage 2: Minimal, non-root runtime image
FROM python:3.11-slim as runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    APP_CONFIG=config/default.yaml \
    HF_HOME=/home/appuser/.cache/huggingface \
    TORCH_HOME=/home/appuser/.cache/torch

WORKDIR /app

# Install runtime dependencies (libpq for postgresql, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy installed Python packages and entrypoint scripts from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache /home/appuser/.cache

# Copy application source tree
COPY --chown=appuser:appgroup config/ /app/config/
COPY --chown=appuser:appgroup core/ /app/core/
COPY --chown=appuser:appgroup resumefit/ /app/resumefit/
COPY --chown=appuser:appgroup migrations/ /app/migrations/
COPY --chown=appuser:appgroup alembic.ini /app/alembic.ini
COPY --chown=appuser:appgroup pyproject.toml /app/pyproject.toml
COPY --chown=appuser:appgroup data/aliases.yaml /app/data/aliases.yaml
COPY --chown=appuser:appgroup data/artifacts/ /app/data/artifacts/
COPY --chown=appuser:appgroup data/esco/ /app/data/esco/
COPY --chown=appuser:appgroup evaluation/results/latest/ /app/evaluation/results/latest/

# Ensure non-root ownership
RUN chown -R appuser:appgroup /app /home/appuser/.cache

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "resumefit.app:create_app()"]
