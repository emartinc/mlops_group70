# Hydra Configuration

This page explains how the Hydra-based configuration system works in the project.

## What is Hydra?

[Hydra](https://hydra.cc/) is a configuration framework that enables:

* ✅ **Modular Configuration**: Splits configs into small, reusable files.
* ✅ **Composition**: Combines multiple configs.
* ✅ **CLI Overrides**: Changes parameters without editing files.
* ✅ **Multi-run**: Runs experiments with different configurations automatically.
* ✅ **Instantiation**: Creates Python objects directly from YAML.

## Configuration Structure

```
configs/
├── train.yaml              # Main config
├── train_quick.yaml        # Config for quick testing
├── train_cpu.yaml          # Config for CPU
├── train_production.yaml   # Config for production
├── data/
│   └── default.yaml        # DataModule config
├── model/
│   ├── default.yaml        # Model config
│   └── quick.yaml          # Model for quick tests
├── trainer/
│   ├── default.yaml        # Trainer config
│   ├── quick.yaml          # Quick Trainer
│   └── cpu.yaml            # Trainer for CPU
├── callbacks/
│   └── default.yaml        # Callbacks (early stopping, checkpoints)
└── logger/
    ├── csv.yaml            # CSV logger
    └── wandb.yaml          # Weights & Biases logger

```

## Main Config: `train.yaml`

```yaml
# @package _global_

defaults:
  - data: default
  - model: default
  - trainer: default
  - callbacks: default
  - logger: csv
  - _self_

# Seed for reproducibility
seed: 42

# Paths
data_dir: data
model_dir: models
log_dir: logs

# Output directory (Hydra creates this automatically)
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  job:
    chdir: false  # Do not change working directory

```

### Composition with `defaults`

```yaml
defaults:
  - data: default        # Loads configs/data/default.yaml
  - model: default       # Loads configs/model/default.yaml
  - trainer: default     # Loads configs/trainer/default.yaml
  - callbacks: default   # Loads configs/callbacks/default.yaml
  - logger: csv          # Loads configs/logger/csv.yaml
  - _self_               # Applies this file at the end

```

**Priority Order** (last one wins):

1. `configs/data/default.yaml`
2. `configs/model/default.yaml`
3. `configs/trainer/default.yaml`
4. `configs/callbacks/default.yaml`
5. `configs/logger/csv.yaml`
6. `train.yaml` (due to `_self_`)
7. CLI overrides

## Component Configs

### Data Module: `configs/data/default.yaml`

```yaml
_target_: src.mbti_classifier.data.MBTIDataModule
raw_data_path: data/raw/mbti_1.csv
processed_data_path: data/processed/processed_mbti.csv
batch_size: 16
max_length: 512
num_workers: 4

```

**`_target_`** indicates the class to instantiate:

```python
# Hydra does this automatically:
from src.mbti_classifier.data import MBTIDataModule

datamodule = MBTIDataModule(
    raw_data_path="data/raw/mbti_1.csv",
    processed_data_path="data/processed/processed_mbti.csv",
    batch_size=16,
    max_length=512,
    num_workers=4
)

```

### Model: `configs/model/default.yaml`

```yaml
_target_: src.mbti_classifier.model.MBTIModel
model_name: distilbert-base-uncased
learning_rate: 2e-5
dropout_rate: 0.1
weight_decay: 0.01

```

### Trainer: `configs/trainer/default.yaml`

```yaml
_target_: lightning.pytorch.Trainer
max_epochs: 10
accelerator: auto
devices: auto
precision: 16-mixed
log_every_n_steps: 10
enable_checkpointing: true
deterministic: true

```

**Important**: `_target_: lightning.pytorch.Trainer` directly instantiates the PyTorch Lightning Trainer.

### Callbacks: `configs/callbacks/default.yaml`

```yaml
model_checkpoint:
  _target_: lightning.pytorch.callbacks.ModelCheckpoint
  dirpath: ${model_dir}
  filename: best
  monitor: val_loss
  mode: min
  save_top_k: 1
  save_last: false
  verbose: true

early_stopping:
  _target_: lightning.pytorch.callbacks.EarlyStopping
  monitor: val_loss
  patience: 5
  mode: min
  verbose: true

```

**List of callbacks:**

```python
# Hydra instantiates each one:
callbacks = [
    ModelCheckpoint(dirpath="models", filename="best", ...),
    EarlyStopping(monitor="val_loss", patience=5, ...)
]

```

### Logger: `configs/logger/csv.yaml`

```yaml
_target_: lightning.pytorch.loggers.CSVLogger
save_dir: ${log_dir}
name: mbti_classifier

```

## Usage in Code

### Train Script: `src/mbti_classifier/train.py`

```python
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    # cfg contains the entire composed configuration
    
    # Instantiate components
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    trainer = instantiate(
        cfg.trainer,
        callbacks=[instantiate(cb) for cb in cfg.callbacks.values()],
        logger=instantiate(cfg.logger)
    )
    
    # Train
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()

```

### `@hydra.main` Decorator

```python
@hydra.main(
    version_base=None,           # No deprecation warnings
    config_path="../../configs",  # Path relative to configs/
    config_name="train"           # Name without .yaml
)

```

**Effect:**

1. Hydra loads `configs/train.yaml`.
2. Processes `defaults` (loads data, model, etc. configs).
3. Applies CLI overrides.
4. Passes the composed config to `main(cfg)`.

### `instantiate()` Function

```python
from hydra.utils import instantiate

# Example 1: Simple Class
cfg = {"_target_": "builtins.dict", "a": 1, "b": 2}
obj = instantiate(cfg)
# obj = dict(a=1, b=2)

# Example 2: Custom Class
cfg = {
    "_target_": "src.mbti_classifier.data.MBTIDataModule",
    "batch_size": 16
}
obj = instantiate(cfg)
# obj = MBTIDataModule(batch_size=16)

# Example 3: List of Objects
cfg = {
    "cb1": {"_target_": "lightning.pytorch.callbacks.EarlyStopping", "patience": 5},
    "cb2": {"_target_": "lightning.pytorch.callbacks.ModelCheckpoint", "monitor": "val_loss"}
}
callbacks = [instantiate(cb) for cb in cfg.values()]

```

## CLI Overrides

### Simple Override

```bash
# Change batch size
uv run python src/mbti_classifier/train.py data.batch_size=32

# Change learning rate
uv run python src/mbti_classifier/train.py model.learning_rate=5e-5

# Change max epochs
uv run python src/mbti_classifier/train.py trainer.max_epochs=20

# Multiple overrides
uv run python src/mbti_classifier/train.py \
    data.batch_size=32 \
    model.learning_rate=5e-5 \
    trainer.max_epochs=20

```

### Config Group Override

```bash
# Use quick config instead of default
uv run python src/mbti_classifier/train.py model=quick trainer=quick

# Use wandb logger instead of csv
uv run python src/mbti_classifier/train.py logger=wandb

# Use CPU config
uv run python src/mbti_classifier/train.py trainer=cpu

```

### Full File Override

```bash
# Use train_quick.yaml instead of train.yaml
uv run python src/mbti_classifier/train.py --config-name=train_quick

# Equivalent to:
uv run python src/mbti_classifier/train.py \
    model=quick \
    trainer=quick \
    trainer.max_epochs=1

```

## Pre-defined Configs

### 1. Quick Test: `train_quick.yaml`

```yaml
defaults:
  - train  # Inherits from train.yaml
  - override model: quick
  - override trainer: quick
  - _self_

# Additional overrides
trainer:
  max_epochs: 1  # Only 1 epoch
  limit_train_batches: 10  # Only 10 batches
  limit_val_batches: 5     # Only 5 validation batches

```

**Usage:**

```bash
uv run python src/mbti_classifier/train.py --config-name=train_quick
# Finishes in ~2 minutes to validate pipeline

```

### 2. CPU Only: `train_cpu.yaml`

```yaml
defaults:
  - train
  - override trainer: cpu
  - _self_

trainer:
  precision: 32  # No mixed precision on CPU

```

**Usage:**

```bash
uv run python src/mbti_classifier/train.py --config-name=train_cpu

```

### 3. Production: `train_production.yaml`

```yaml
defaults:
  - train
  - override logger: wandb
  - _self_

trainer:
  max_epochs: 20
  precision: 16-mixed
  
callbacks:
  model_checkpoint:
    save_top_k: 3  # Save top 3 models
  early_stopping:
    patience: 10  # More patience

```

**Usage:**

```bash
uv run python src/mbti_classifier/train.py --config-name=train_production

```

## Multi-Run

Runs multiple experiments automatically:

```bash
# Sweep over batch sizes
uv run python src/mbti_classifier/train.py -m \
    data.batch_size=8,16,32

# Sweep over learning rates
uv run python src/mbti_classifier/train.py -m \
    model.learning_rate=1e-5,2e-5,5e-5

# Grid search (9 combinations)
uv run python src/mbti_classifier/train.py -m \
    data.batch_size=8,16,32 \
    model.learning_rate=1e-5,2e-5,5e-5

```

**Output:**

```
outputs/
├── 2026-01-21/
│   ├── 10-30-15/  # batch_size=8, lr=1e-5
│   ├── 10-35-22/  # batch_size=8, lr=2e-5
│   ├── 10-40-18/  # batch_size=8, lr=5e-5
│   ├── 10-45-30/  # batch_size=16, lr=1e-5
│   └── ...

```

## Working Directory

By default, Hydra **changes** the working directory to `outputs/.../`:

```python
# Problem with chdir=true (default)
with open("data/raw/file.csv", "r") as f:  # Error: file not found
    ...

# Hydra is in outputs/2026-01-21/10-30-15/
# It looks for: outputs/2026-01-21/10-30-15/data/raw/file.csv

```

**Solution: Disable `chdir**`

```yaml
# configs/train.yaml
hydra:
  job:
    chdir: false  # Keep original working directory

```

Now relative paths work:

```python
with open("data/raw/file.csv", "r") as f:  # ✅ Works
    ...

```

## Accessing Configuration

### DictConfig vs Dict

```python
from omegaconf import DictConfig, OmegaConf

@hydra.main(...)
def main(cfg: DictConfig):
    # cfg is a DictConfig (not a normal dict)
    
    # Access with dot notation
    print(cfg.data.batch_size)  # 16
    
    # Access with []
    print(cfg["data"]["batch_size"])  # 16
    
    # Convert to normal dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(type(cfg_dict))  # <class 'dict'>

```

### String Interpolation

```yaml
data_dir: data
raw_data_path: ${data_dir}/raw/mbti_1.csv  # Interpolates data_dir
processed_data_path: ${data_dir}/processed/processed_mbti.csv

```

**Result:**

```python
cfg.data.raw_data_path  # "data/raw/mbti_1.csv"
cfg.data.processed_data_path  # "data/processed/processed_mbti.csv"

```

### Special Values

```yaml
# Current timestamp
output_dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
# output_dir = "outputs/2026-01-21/10-30-15"

# Access environment variables
api_key: ${oc.env:WANDB_API_KEY}

# Default value if it doesn't exist
api_key: ${oc.env:WANDB_API_KEY,default_key}

```

## Debugging

### View Final Configuration

```bash
# Print config without running
uv run python src/mbti_classifier/train.py --cfg job

# With overrides
uv run python src/mbti_classifier/train.py --cfg job \
    data.batch_size=32 \
    model.learning_rate=5e-5

```

### Validation

```python
from omegaconf import OmegaConf

@hydra.main(...)
def main(cfg: DictConfig):
    # Print full config
    print(OmegaConf.to_yaml(cfg))
    
    # Validate required fields
    assert cfg.data.batch_size > 0, "batch_size must be > 0"
    assert cfg.model.learning_rate > 0, "learning_rate must be > 0"

```

## Best Practices

### 1. Use Relative Paths

```yaml
# ✅ Good: relative to project root
raw_data_path: data/raw/mbti_1.csv

# ❌ Bad: absolute path
raw_data_path: /Users/username/project/data/raw/mbti_1.csv

```

### 2. Separate Concerns

```yaml
# ✅ Good: separate configs by component
configs/
├── data/default.yaml
├── model/default.yaml
└── trainer/default.yaml

# ❌ Bad: everything in one file
configs/train.yaml  # 200 lines

```

### 3. Use Smart Defaults

```yaml
# ✅ Good: reasonable defaults
batch_size: 16  # Works on most GPUs
max_length: 512  # Standard for BERT

# ❌ Bad: extreme defaults
batch_size: 256  # OOM on small GPUs
max_length: 2048  # Too long

```

### 4. Document Configs

```yaml
# Batch size for training
# Reduce if OOM, increase for better performance
batch_size: 16

# Max sequence length
# 512 is standard for DistilBERT
# Reduce for short texts
max_length: 512

```

## Full Examples

### Train with Wandb

```bash
# 1. Export API key
export WANDB_API_KEY=your_key_here

# 2. Train with wandb logger
uv run python src/mbti_classifier/train.py \
    logger=wandb \
    logger.project=mbti-classifier \
    logger.name=experiment-1

```

### Quick Test on CPU

```bash
uv run python src/mbti_classifier/train.py \
    --config-name=train_quick \
    trainer.accelerator=cpu \
    data.num_workers=0

```

### Hyperparameter Sweep

```bash
uv run python src/mbti_classifier/train.py -m \
    model.learning_rate=1e-5,2e-5,5e-5 \
    model.dropout_rate=0.1,0.2,0.3 \
    trainer.max_epochs=10

```

## References

* [Hydra Documentation](https://hydra.cc/)
* [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
* [PyTorch Lightning + Hydra](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)