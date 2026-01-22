# Installation

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** - Ultra-fast Python package manager
- **Git**
- **Google Cloud SDK** (for DVC remote storage)

### Installing uv

=== "macOS"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Linux"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

### Installing Google Cloud SDK

Only required if you need to pull data/models from DVC:

=== "macOS"
    ```bash
    brew install google-cloud-sdk
    ```

=== "Linux"
    ```bash
    curl https://sdk.cloud.google.com | bash
    ```

=== "Windows"
    Download the installer from [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)

## Clone the Repository

```bash
git clone https://github.com/mlopsgroup70/mlops_group70.git
cd mlops_group70
```

## Install Dependencies

```bash
# Create virtual environment and install all dependencies
uv sync --dev

# Install pre-commit hooks for code quality
uv run pre-commit install --install-hooks
```

!!! note
    You don't need to manually activate the virtual environment. `uv run` handles this automatically.

## Pull Data and Models (Optional)

If you want to use pre-trained models and preprocessed data:

```bash
# Authenticate with Google Cloud (first time only)
gcloud auth application-default login

# Pull data and models from Google Cloud Storage
uv run dvc pull
```

This downloads:
- Raw dataset from `mlops70_bucket`
- Preprocessed data
- Trained model checkpoints

## Verify Installation

```bash
# Run a quick test
uv run pytest tests/unit/ -v

# Check that all commands are available
uv run invoke --list
```

You should see output listing all available tasks.

## What's Next?

- Continue to the [Quick Start Guide](quickstart.md) to train your first model
- Learn about [Configuration](../configuration/hydra.md) to customize training
- Explore the [Architecture](../architecture/overview.md) to understand the system

## Troubleshooting

### uv sync fails

If you encounter errors during `uv sync`:

```bash
# Clear cache and retry
uv cache clean
uv sync --dev
```

### DVC authentication issues

If `dvc pull` fails:

```bash
# Re-authenticate
gcloud auth application-default login

# Verify access to bucket
gsutil ls gs://mlops70_bucket/
```

### Import errors

Ensure you're using `uv run` to execute Python scripts:

```bash
# ❌ Wrong
python src/mbti_classifier/train.py

# ✅ Correct
uv run python src/mbti_classifier/train.py
```
