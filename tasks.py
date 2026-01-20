import os

from dotenv import load_dotenv
from invoke import Context, task

load_dotenv()

WINDOWS = os.name == "nt"
PROJECT_NAME = "mbti_classifier"
PYTHON_VERSION = "3.12"

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)

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
