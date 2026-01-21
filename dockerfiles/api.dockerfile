#FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

# Install curl for health checks and build tools for PyTorch
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project

COPY src/ src/
COPY README.md LICENSE ./

RUN uv sync --frozen

# Create directory for models (will be mounted as volume)
RUN mkdir -p models

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "mbti_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
