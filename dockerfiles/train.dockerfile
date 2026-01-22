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

# Install only core + training dependencies
RUN uv sync --frozen --no-install-project --group train

# Copy source code and configs
COPY src src/
COPY configs configs/
COPY README.md README.md
COPY LICENSE LICENSE

# Final sync with project
RUN uv sync --frozen --group train

# Create directories for volumes
RUN mkdir -p /app/models /app/data /app/logs /app/outputs

ENTRYPOINT ["uv", "run", "python", "src/mbti_classifier/train.py"]
