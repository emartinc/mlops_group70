````markdown
# MBTI Personality Classifier

An end-to-end MLOps pipeline for predicting Myers-Briggs personality types from text using deep learning. This project implements a multi-task DistilBERT classifier that predicts personality dimensions (Introversion/Extraversion, Intuition/Sensing, Thinking/Feeling, Judging/Perceiving) from writing samples.

## 🎯 Project Overview

This project operationalizes a complete machine learning lifecycle for MBTI personality prediction:

- **Multi-task binary classification**: Instead of treating MBTI as a 16-class problem, we implement 4 independent binary classifiers (one per dimension) for better performance
- **Transformer-based model**: Uses DistilBERT (67M parameters) for contextual understanding of linguistic patterns
- **Advanced data handling**: Random window sampling for long sequences (>512 tokens), preserving special tokens
- **Production-ready serving**: FastAPI backend with Streamlit UI featuring interactive radar plot visualizations
- **Modular configuration**: Hydra-based configs for reproducible experiments across different environments
- **Comprehensive testing**: Unit and integration tests with coverage reporting

### Dataset

- **Source**: [Kaggle MBTI Personality Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)
- **Size**: ~8,675 samples from PersonalityCafe forum posts
- **Features**: Concatenated posts (separated by "|||") with self-reported MBTI types
- **Challenge**: Severe class imbalance (75% Introverts, 85% Intuitives)
- **Format**: Unstructured text with vocabulary usage and writing style patterns

### Model Architecture

- **Base**: `distilbert-base-uncased` (Hugging Face Transformers)
- **Approach**: 4 independent binary classifiers with shared representation layer
- **Tasks**: E/I (Extraversion), N/S (Intuition), T/F (Thinking), J/P (Judging)
- **Output**: Probabilities for each dimension + combined MBTI type prediction

## 📁 Project Structure

```
├── configs/                      # Hydra configuration files
│   ├── train.yaml               # Main training config
│   ├── train_quick.yaml         # Quick 2-epoch test
│   ├── train_cpu.yaml           # CPU-only training
│   ├── train_production.yaml    # Extended 20-epoch config
│   ├── callbacks/               # Checkpoint, early stopping configs
│   ├── data/                    # DataModule configurations
│   ├── logger/                  # Logging configurations (wandb, csv)
│   ├── model/                   # Model hyperparameters
│   └── trainer/                 # PyTorch Lightning Trainer settings
├── data/
│   ├── raw/                     # Raw dataset (auto-downloaded)
│   └── processed/               # Preprocessed and cached data
├── docs/                        # MkDocs documentation
├── models/                      # Saved model checkpoints
├── src/mbti_classifier/
│   ├── api/
│   │   ├── api.py              # FastAPI inference server
│   │   └── ui.py               # Streamlit visualization UI
│   └── training/
│       ├── data.py             # DataModule with preprocessing
│       ├── model.py            # Multi-task DistilBERT model
│       └── train.py            # Hydra training script
├── tests/
│   ├── unit/                   # Unit tests (data, model, API)
│   └── integration/            # End-to-end pipeline tests
├── AGENTS.md                   # Guidance for AI coding agents
├── pyproject.toml              # Project dependencies and config
└── tasks.py                    # Invoke task definitions
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (ultra-fast Python package manager)

### Setup Instructions

Follow these steps **in order** to set up the project:

#### 1. Clone the Repository

```bash
git clone https://github.com/mlopsgroup70/mlops_group70.git
cd mlops_group70
```

#### 2. Install Dependencies

```bash
# Create virtual environment and install all dependencies
uv sync --dev

# Install pre-commit hooks for code quality
uv run pre-commit install --install-hooks
```

> **Note**: You don't need to manually activate the virtual environment. `uv run` handles this automatically.

#### 3. Prepare the Data

The data preprocessing step automatically downloads the dataset if missing and creates cached processed files:

```bash
uv run invoke preprocess-data
```

**What this does**:
- Downloads raw MBTI dataset via `mlcroissant` (if not already present in `data/raw/`)
- Preprocesses text and labels
- Saves processed data to `data/processed/processed_mbti.csv`
- Subsequent runs load from cache for faster startup

#### 4. Train the Model

```bash
# Standard training (GPU recommended)
uv run invoke train

# Quick test (2 epochs, useful for validation)
uv run python src/mbti_classifier/training/train.py --config-name train_quick

# CPU-only training
uv run python src/mbti_classifier/training/train.py --config-name train_cpu

# Production training (20 epochs)
uv run python src/mbti_classifier/training/train.py --config-name train_production
```

**Training saves only the best model** based on validation F1 score. Checkpoints are saved to `models/checkpoints/`.

#### 5. Run Tests

```bash
# Run all tests with coverage
uv run invoke test

# Run only unit tests
uv run pytest tests/unit/ -v

# Run only integration tests
uv run pytest tests/integration/ -v

# Skip slow tests
uv run pytest -m "not slow"
```

## 🌐 Serving & Inference

### Start the FastAPI Backend

```bash
# Using invoke task (recommended)
uv run invoke api

# Or directly (custom port)
uv run invoke api --port 8080
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

**Example API request**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love spending time alone reading books and thinking deeply about abstract concepts."}'
```

**Response**:
```json
{
  "mbti_type": "INFP",
  "dimensions": {
    "ei": "I",
    "sn": "N",
    "tf": "F",
    "jp": "P"
  },
  "probabilities": {
    "ei": 0.85,
    "sn": 0.72,
    "tf": 0.68,
    "jp": 0.55
  }
}
```

### Launch the Streamlit UI

```bash
# Using invoke task
uv run invoke ui

# Or with custom port
uv run invoke ui --port 8501
```

Access the interactive UI at http://localhost:8501. Features:
- Text input for personality prediction
- Real-time inference via API
- Radar plot visualization of personality dimensions
- Probability scores for each dimension

## 🔧 Configuration Management

This project uses **Hydra** for modular, composable configurations. All configs are in the `configs/` directory.

### Available Configurations

- **Main configs** (in `configs/`):
  - `train.yaml` - Default training configuration
  - `train_quick.yaml` - Fast 2-epoch test
  - `train_cpu.yaml` - CPU-only training
  - `train_production.yaml` - Extended 20-epoch training

- **Modular components** (in subdirectories):
  - `data/` - DataModule settings (batch size, num workers, data augmentation)
  - `model/` - Model architecture and hyperparameters
  - `trainer/` - PyTorch Lightning Trainer options
  - `callbacks/` - Checkpointing, early stopping, learning rate monitoring
  - `logger/` - Logging backends (WandB, CSV)

### Override Configuration

```bash
# Override specific parameters
uv run python src/mbti_classifier/training/train.py \
  trainer.max_epochs=10 \
  data.batch_size=16 \
  model.learning_rate=5e-5

# Use different config components
uv run python src/mbti_classifier/training/train.py \
  data=quick \
  trainer=cpu \
  logger=csv
```

See [configs/README.md](configs/README.md) for detailed configuration documentation.

## 📊 Available Tasks

All tasks are defined in `tasks.py` and can be run with `uv run invoke <task>`:

| Task | Command | Description |
|------|---------|-------------|
| **preprocess-data** | `uv run invoke preprocess-data` | Download and preprocess MBTI dataset |
| **train** | `uv run invoke train` | Train the model with default config |
| **test** | `uv run invoke test` | Run all tests with coverage report |
| **api** | `uv run invoke api` | Start FastAPI server (port 8000) |
| **ui** | `uv run invoke ui` | Start Streamlit UI (port 8501) |
| **docker-build** | `uv run invoke docker-build` | Build Docker images for training and API |
| **serve-docs** | `uv run invoke serve-docs` | Serve MkDocs documentation locally |
| **build-docs** | `uv run invoke build-docs` | Build documentation site |

## 🐳 Docker Support

Build containerized versions:

```bash
# Build both training and API images
uv run invoke docker-build

# Build with auto progress output
uv run invoke docker-build --progress=auto
```

## 🧪 Development Workflow

### Adding Dependencies

```bash
# Add a new package
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>
```

### Code Quality

```bash
# Format code with ruff
uv run ruff format .

# Lint and auto-fix issues
uv run ruff check . --fix

# Run pre-commit hooks manually
uv run pre-commit run --all-files
```

### Testing Best Practices

- **Unit tests** (`tests/unit/`): Test individual components in isolation
- **Integration tests** (`tests/integration/`): Test end-to-end workflows
- Mark slow tests with `@pytest.mark.slow` decorator
- Run fast tests frequently: `uv run pytest -m "not slow"`

## 📖 Documentation

Build and serve the documentation:

```bash
# Serve locally with hot reload
uv run invoke serve-docs

# Build static site
uv run invoke build-docs
```

## 🤖 For AI Coding Agents

If you're an autonomous coding agent working on this project, please read [AGENTS.md](AGENTS.md) for:
- Project-specific commands and tools
- Code style guidelines
- Configuration management patterns
- Testing conventions
- Documentation standards

## 📝 License

This project is licensed under the terms specified in [LICENSE](LICENSE).

## 🙏 Acknowledgments

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template) - a cookiecutter template for Machine Learning Operations (MLOps).
