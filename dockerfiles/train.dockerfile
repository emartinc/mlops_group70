#FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project

COPY src/ src/
COPY configs/ configs/
COPY README.md LICENSE ./

RUN uv sync --frozen

# Create directories for models
RUN mkdir -p models data/raw data/processed

ENTRYPOINT ["uv", "run", "python", "src/mbti_classifier/train.py"]
