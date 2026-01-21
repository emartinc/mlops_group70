#FROM ghcr.io/astral-sh/uv:python3.12-alpine AS base
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project

COPY src/ src/
COPY README.md LICENSE ./

RUN uv sync --frozen

EXPOSE 8501

ENTRYPOINT ["uv", "run", "streamlit", "run", "src/mbti_classifier/ui.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
