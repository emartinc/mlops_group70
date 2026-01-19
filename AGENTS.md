> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

# Relevant commands

* The project uses `uv` for management of virtual environments. This means:
  * To install packages, use `uv add <package-name>`.
  * To run Python scripts, use `uv run <script-name>.py`.
  * To run other commands related to Python, prefix them with `uv run `, e.g., `uv run <command>`.
* The project uses `pytest` for testing. To run tests, use `uv run pytest tests/`.
  * Run only unit tests: `uv run pytest tests/unit/`
  * Run only integration tests: `uv run pytest tests/integration/`
  * Run without slow tests: `uv run pytest -m "not slow"`
* The project uses `ruff` for linting and formatting:
    * To format code, use `uv run ruff format .`.
    * To lint code, use `uv run ruff check . --fix`.
* The project uses `invoke` for task management. To see available tasks, use `uv run invoke --list` or refer to the
    `tasks.py` file.
* The project uses `pre-commit` for managing pre-commit hooks. To run all hooks on all files, use
    `uv run pre-commit run --all-files`. For more information, refer to the `.pre-commit-config.yaml` file.
* The project uses **Hydra** for configuration management. All configs are in the `configs/` directory:
    * Main config: `configs/train.yaml`
    * Quick test: `configs/train_quick.yaml`
    * CPU only: `configs/train_cpu.yaml`
    * Production: `configs/train_production.yaml`
    * See `configs/README.md` for detailed configuration guide
* Training saves **only the best model** (no last.ckpt or other checkpoints).

# Code style

* Follow existing code style.
* Keep line length within 120 characters.
* Use f-strings for formatting.
* Use type hints
* Do not add inline comments unless absolutely necessary.

# Documentation

* If the project has a `docs/` folder, update documentation there as needed.
* In this case the project will be using `mkdocs` for documentation. To build the docs locally, use
    `uv run mkdocs serve`
* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.

# Project Structure

* **Data paths are relative to project root**: All data paths in configs use relative paths (e.g., `data/raw`).
  The `MBTIDataModule` automatically resolves these relative to the project root by searching for `pyproject.toml`.
  This makes the project portable across different environments.
* **Automatic data preprocessing and caching**: The DataModule checks if processed data exists in
  `data/processed/processed_mbti.csv`. If not found, it automatically downloads raw data, preprocesses it,
  and saves the result. Subsequent runs load from cache for faster startup.
* **Configuration uses Hydra instantiate**: All components (data, model, trainer, callbacks, logger) are instantiated
  via Hydra's `instantiate()` function using `_target_` in YAML configs.
* **Multi-task binary classification**: The model treats MBTI as 4 independent binary tasks (E/I, S/N, T/F, J/P)
  instead of 16-class classification for better performance.
