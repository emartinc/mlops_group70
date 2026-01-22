FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# Install C++ compiler for torch.compile
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# Install only core + API dependencies
RUN uv sync --frozen --no-install-project --group api

# Copy source code
COPY src src/
COPY README.md README.md
COPY LICENSE LICENSE

# Final sync with project
RUN uv sync --frozen --group api

# Create directories for volumes
RUN mkdir -p /app/models /app/data

# Set environment variable for model path
ENV MODEL_PATH=/app/models/best.ckpt

EXPOSE 8000

# Create directory for models (will be mounted as volume)
RUN mkdir -p models

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "mbti_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
