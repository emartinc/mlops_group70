import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from invoke import Context, task

load_dotenv()

WINDOWS = os.name == "nt"
PROJECT_NAME = "mbti_classifier"
PYTHON_VERSION = "3.12"

def load_data_config() -> dict:
    """Load data configuration from YAML file."""
    config_path = Path("configs/data/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    config = load_data_config()
    raw_path = config.get("raw_data_path", "data/raw")
    processed_path = config.get("processed_data_path", "data/processed")
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py {raw_path} {processed_path}", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve", echo=True, pty=not WINDOWS)

# API commands
@task
def api(ctx: Context, port: int = 8000) -> None:
    """Start FastAPI server for inference."""
    ctx.run(
        f"uv run uvicorn {PROJECT_NAME}.api:app --reload --host 0.0.0.0 --port {port}",
        echo=True,
        pty=not WINDOWS
    )

@task
def ui(ctx: Context, port: int = 8501) -> None:
    """Start Streamlit UI."""
    ctx.run(
        f"uv run streamlit run src/{PROJECT_NAME}/ui.py --server.port {port}",
        echo=True,
        pty=not WINDOWS
    )

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build all docker images."""
    ctx.run("docker compose build", echo=True, pty=not WINDOWS)

@task
def docker_train(ctx: Context) -> None:
    """Run training in Docker container."""
    ctx.run("docker compose run --rm train", echo=True, pty=not WINDOWS)

@task
def docker_up(ctx: Context) -> None:
    """Start API and UI services."""
    ctx.run("docker compose up -d api ui", echo=True, pty=not WINDOWS)

@task
def docker_down(ctx: Context) -> None:
    """Stop all services."""
    ctx.run("docker compose down", echo=True, pty=not WINDOWS)

@task
def docker_logs(ctx: Context, service: str = "") -> None:
    """View logs from services."""
    cmd = f"docker compose logs -f {service}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)

@task
def docker_full_pipeline(ctx: Context) -> None:
    """Run full pipeline: build, train, start services."""
    ctx.run("docker compose build", echo=True, pty=not WINDOWS)
    ctx.run("docker compose run --rm train", echo=True, pty=not WINDOWS)
    ctx.run("docker compose up -d api ui", echo=True, pty=not WINDOWS)
    ctx.run("docker compose logs -f", echo=True, pty=not WINDOWS)

# DVC commands
@task
def dvc_status(ctx: Context) -> None:
    """Check DVC status."""
    ctx.run("uv run dvc status", echo=True, pty=not WINDOWS)

@task
def dvc_push(ctx: Context) -> None:
    """Push data to DVC remote (GCS)."""
    ctx.run("uv run dvc push", echo=True, pty=not WINDOWS)

@task
def dvc_pull(ctx: Context) -> None:
    """Pull data from DVC remote (GCS)."""
    ctx.run("uv run dvc pull", echo=True, pty=not WINDOWS)

@task
def dvc_add_data(ctx: Context) -> None:
    """Add data/ and models/ to DVC tracking."""
    ctx.run("uv run dvc add data/raw data/processed models", echo=True, pty=not WINDOWS)
    print("\n✓ Data tracked with DVC")
    print("Remember to commit:")
    print("  git add data/*.dvc models.dvc .gitignore")
    print("  git commit -m 'Update data version'")
    print("  uv run invoke dvc-push")