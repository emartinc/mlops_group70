# Docker

This guide covers using Docker and Docker Compose for containerized deployment.

## Overview

The project uses three separate Docker images:

| Image | Purpose | Port | Profile |
|-------|---------|------|---------|
| `mbti-train` | Model training | - | `training` |
| `mbti-api` | FastAPI server | 8000 | default |
| `mbti-ui` | Streamlit UI | 8501 | default |

## Docker Compose Setup

### Architecture

```
┌─────────────┐
│   train     │ (one-time job)
│   service   │
└──────┬──────┘
       │ Saves to ./models/
       ↓
┌─────────────┐        HTTP         ┌──────────┐
│     api     │ ←───────────────────│    ui    │
│  (port 8000)│                     │(port 8501│
└─────────────┘                     └──────────┘
       ↑
       │ Loads from ./models/ (read-only)
```

### Shared Volumes

```yaml
volumes:
  - ./models:/app/models    # Train writes, API reads
  - ./data:/app/data        # Dataset caching
  - ./configs:/app/configs  # Hydra configurations
```

**Why bind mounts?**
- Direct access from host
- Changes visible in real-time
- Easy debugging
- Compatible with DVC

## Basic Usage

### Build Images

```bash
# Build all images
uv run invoke docker-build

# Or manually
docker compose build

# Build specific service
docker compose build api
```

### Start Services

```bash
# Start API + UI
uv run invoke docker-up

# Or manually
docker compose up -d api ui

# Check status
docker compose ps
```

### View Logs

```bash
# All services
uv run invoke docker-logs

# Specific service
uv run invoke docker-logs --service=api
uv run invoke docker-logs --service=ui

# Follow logs in real-time
docker compose logs -f api
```

### Stop Services

```bash
# Stop all
uv run invoke docker-down

# Or manually
docker compose down

# Stop specific service
docker compose stop ui
```

## Training Workflow

### Train in Container

```bash
# Train with default config
uv run invoke docker-train

# Or manually
docker compose run --rm train

# Train with custom config
docker compose run --rm train \
  --config-name train_production
```

**What happens:**
1. Starts `mbti-train` container
2. Downloads/uses cached data from `./data/`
3. Trains model
4. Saves checkpoint to `./models/best.ckpt`
5. Container exits (`--rm` removes it)

### Use Trained Model

After training, restart API to load new model:

```bash
uv run invoke docker-restart --service=api

# Or manually
docker compose restart api
```

## Development Workflow

### Option 1: Local Development

```bash
# Develop without Docker
uv run invoke train
uv run invoke api     # Terminal 1
uv run invoke ui      # Terminal 2
```

### Option 2: Hybrid (API in Docker, dev locally)

```bash
# Start only API
docker compose up -d api

# Develop locally
code src/mbti_classifier/ui.py
uv run invoke ui
```

### Option 3: Full Docker

```bash
# Build and run everything
docker compose build
docker compose up -d
```

## Advanced Configuration

### Environment Variables

Set in `docker-compose.yaml`:

```yaml
api:
  environment:
    - PYTHONUNBUFFERED=1          # Real-time logs
    - TRANSFORMERS_CACHE=/app/.cache  # Model cache

ui:
  environment:
    - API_URL=http://api:8000     # Internal networking
```

### Health Checks

API has health check configured:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 5
  start_period: 120s  # Model loading time
```

**Why 120s start period?**
- DistilBERT downloads on first run (~250MB)
- Model loading from checkpoint (~800MB)
- Prevents false "unhealthy" status

### Service Dependencies

UI waits for API to be healthy:

```yaml
ui:
  depends_on:
    api:
      condition: service_healthy
```

This prevents connection errors on startup.

## Docker Images

### Base Image

All services use:
```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
```

**Why this image?**
- Includes `uv` pre-installed
- Debian-based (better compatibility than Alpine)
- Optimized for Python projects

### API Image

Special additions:

```dockerfile
RUN apt-get update && apt-get install -y \
    curl \           # For health checks
    build-essential \ # For PyTorch compilation
    g++ \
    && rm -rf /var/lib/apt/lists/*
```

**Why build tools?**
- PyTorch may compile optimizations at runtime
- Without g++, you get: `InvalidCxxCompiler` error

### Image Sizes

```
REPOSITORY      TAG       SIZE
mbti-train      latest    ~1.2 GB  (PyTorch + Lightning + WandB)
mbti-api        latest    ~900 MB  (PyTorch + Transformers)
mbti-ui         latest    ~600 MB  (Streamlit + Plotly)
```

## Debugging

### Enter Running Container

```bash
# Interactive shell in API container
docker compose exec api bash

# Check model exists
docker compose exec api ls -lh /app/models/

# Test tokenizer
docker compose exec api python -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('distilbert-base-uncased'))"
```

### View Container Logs

```bash
# Last 100 lines
docker compose logs --tail=100 api

# Continuous follow
docker compose logs -f ui

# Since specific time
docker compose logs --since 5m api
```

### Inspect Container

```bash
# Full container details
docker compose exec api env

# Network info
docker inspect mbti-api | grep IPAddress

# Volume mounts
docker inspect mbti-api | grep -A 10 Mounts
```

## Production Deployment

### Build for Production

```bash
# Multi-platform build (for different architectures)
docker buildx build --platform linux/amd64,linux/arm64 \
  -t mbti-api:latest \
  -f dockerfiles/api.dockerfile .
```

### Push to Registry

```bash
# Tag for registry
docker tag mbti-api:latest ghcr.io/mlopsgroup70/mbti-api:v1.0

# Push
docker push ghcr.io/mlopsgroup70/mbti-api:v1.0
```

### Deploy on Server

```bash
# On remote server
git clone https://github.com/mlopsgroup70/mlops_group70.git
cd mlops_group70

# Authenticate DVC
gcloud auth application-default login

# Pull data and models
uv run dvc pull

# Start services
docker compose up -d api ui
```

## Optimization Tips

### Reduce Build Time

```dockerfile
# Cache dependencies separately
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Then copy code
COPY src/ src/
RUN uv sync --frozen
```

### Multi-stage Builds

For even smaller images:

```dockerfile
# Build stage
FROM python:3.12 AS builder
COPY . .
RUN pip install build && python -m build

# Runtime stage
FROM python:3.12-slim
COPY --from=builder /dist/*.whl .
RUN pip install *.whl
```

### Use .dockerignore

Already configured:

```
__pycache__/
*.pyc
.git/
data/       # Don't copy data, use volumes
models/     # Don't copy models, use volumes
.venv/
tests/
```

## Troubleshooting

### Port already in use

```bash
# Find process
lsof -ti:8000

# Kill it
lsof -ti:8000 | xargs kill -9
```

### Container exits immediately

```bash
# Check exit code
docker compose ps

# View logs
docker compose logs train
```

### Out of disk space

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Full cleanup
docker system prune -a --volumes
```

### Permission denied on volumes

```bash
# Fix ownership (Linux)
sudo chown -R $USER:$USER ./models ./data
```

## Next Steps

- Learn about [DVC](dvc.md) for data versioning
- Explore [Testing](testing.md) in Docker
- Read [API Reference](../api/api.md)
