````markdown
# mbti_classifier

# mbti_classifier

This project implements a multi-class NLP classifier utilizing machine learning to predict one of sixteen Myers-Briggs personality types based on a user's linguistic patterns and writing style.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

````

## 🚀 Project Setup Guide

Follow these steps to clone the repository, set up the environment using `uv`, and run the MLOps pipeline.

### 1. Clone the Repository

Start by cloning the project to your local machine:

```bash
git clone <YOUR_REPO_URL>
cd <REPO_NAME>

```

### 2. Install uv

This project uses `uv` for extremely fast dependency management. See [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for details on installing `uv`.

### 3. Configure the Environment

Run the following commands to create the virtual environment, install dependencies, and set up git hooks:

```bash
# 1. Sync dependencies (creates .venv and installs dev tools)
uv sync --dev
uv add transformers 
# 2. Install pre-commit hooks (ensures code quality on commit)
uv run pre-commit install --install-hooks

```

*Note: You do not need to manually activate the virtual environment. `uv run` handles this automatically.*

### 4. Run the Pipeline

Use the project's `invoke` tasks (defined in `tasks.py`) to execute the workflow:

**Step A: Download & Preprocess Data**
This checks for the dataset (downloading it via `mlcroissant` if missing) and processes it.
For adding dependencies to the project, use `uv add <package_name>`.

```bash
uv run invoke preprocess-data

```

**Step B: Train the Model**
Run the training script:

```bash
uv run invoke train

```

**Step C: Run Tests**
Validate the setup by running the test suite:

```bash
uv run invoke test

```
# mlops_group70

### a. Overall Goal of the Project
The goal of this project is to operationalize an end-to-end machine learning pipeline for MBTI Personality Type Prediction, classifying individuals into one of 16 personality types based on their written text. Beyond simple prediction, the project focuses on the MLOps lifecycle by building a robust serving API with real-time text processing, implementing a CI/CD pipeline for automated testing and deployment, and establishing continuous monitoring to detect both data drift (evolving language patterns, vocabulary changes) and concept drift (shifting personality expression online).

### b. Dataset
Data Source: Kaggle MBTI Personality Type Dataset https://www.kaggle.com/datasets/datasnaek/mbti-type.
Description: The dataset contains forum posts from PersonalityCafe users with self-reported MBTI types. Features include approximately 50 concatenated posts per person (separated by "|||"), resulting in long text sequences containing various linguistic patterns, vocabulary usage, and writing styles that correlate with personality dimensions across four axes: I/E (Introversion/Extraversion), N/S (Intuition/Sensing), T/F (Thinking/Feeling), and J/P (Judging/Perceiving).
Size: Approximately 8,675 samples. This size is chosen to allow for rapid prototyping of the infrastructure and pipeline without high computational overhead.
Modality: Unstructured text data (natural language posts). Contains severe class imbalance with Introverts (~75%) dominating Extraverts (~25%) and Intuitives (~85%) dominating Sensors (~15%).

### c. Models
Baseline Model: Logistic Regression on TF-IDF features. A simple, interpretable model to establish a performance benchmark and validate preprocessing pipeline.
Production Model: DistilBERT (or RoBERTa-base). A transformer-based language model chosen for its ability to capture complex linguistic patterns and contextual understanding from pre-training. We will implement four independent binary classifiers (one per MBTI dimension) using Hugging Face Transformers, with class-weighted loss and SMOTE to handle severe imbalance. The imbalanced-learn library will be used as an additional open-source tool for addressing class imbalance challenges specific to this dataset.
