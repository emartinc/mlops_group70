# Training Configuration

This page documents all configuration options for training the model.

## Available Configurations

### 1. Default Training (`train.yaml`)

Standard configuration for GPU training:

```yaml
defaults:
  - data: default
  - model: default
  - trainer: default
  - callbacks: default
  - logger: csv
  - _self_

seed: 42

data_dir: data
model_dir: models
log_dir: logs

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  job:
    chdir: false
```

**Usage:**
```bash
uv run python src/mbti_classifier/train.py
```

**Duration:** ~2 hours on V100 GPU  
**Typical results:** 75-80% total accuracy

### 2. Quick Training (`train_quick.yaml`)

For quick pipeline testing:

```yaml
defaults:
  - train
  - override model: quick
  - override trainer: quick
  - _self_

trainer:
  max_epochs: 1
  limit_train_batches: 10
  limit_val_batches: 5
```

**Usage:**
```bash
uv run python src/mbti_classifier/train.py --config-name=train_quick
```

**Duration:** ~2 minutes  
**Purpose:** Validate that everything works without errors

### 3. CPU Training (`train_cpu.yaml`)

To train without a GPU:

```yaml
defaults:
  - train
  - override trainer: cpu
  - _self_

trainer:
  accelerator: cpu
  devices: 1
  precision: 32
```

**Usage:**
```bash
uv run python src/mbti_classifier/train.py --config-name=train_cpu
```

**Duration:** ~8-10 hours  
**When to use:** You do not have a GPU available

### 4. Production Training (`train_production.yaml`)

For optimal training with Weights & Biases:

```yaml
defaults:
  - train
  - override logger: wandb
  - _self_

trainer:
  max_epochs: 20
  precision: 16-mixed
  gradient_clip_val: 1.0

callbacks:
  model_checkpoint:
    save_top_k: 3
    save_last: true
  early_stopping:
    patience: 10
    min_delta: 0.001

logger:
  project: mbti-classifier
  name: production-run
  log_model: true
```

**Usage:**
```bash
export WANDB_API_KEY=your_key
uv run python src/mbti_classifier/train.py --config-name=train_production
```

**Duration:** ~4-5 hours  
**Results:** Best possible accuracy with early stopping

## Data Configuration

### Default Data (`configs/data/default.yaml`)

```yaml
_target_: src.mbti_classifier.data.MBTIDataModule
raw_data_path: data/raw/mbti_1.csv
processed_data_path: data/processed/processed_mbti.csv
batch_size: 16
max_length: 512
num_workers: 4
```

### Parameters Explained

| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|-------------------|
| `batch_size` | 16 | Samples per batch | 8-32 (depends on GPU) |
| `max_length` | 512 | Maximum sequence length | 128-512 |
| `num_workers` | 4 | Parallel workers for DataLoader | 0-8 |

### Adjust for your Hardware

```bash
# Small GPU (8GB)
uv run python src/mbti_classifier/train.py \
    data.batch_size=8 \
    data.max_length=256

# Large GPU (16GB+)
uv run python src/mbti_classifier/train.py \
    data.batch_size=32 \
    data.max_length=512

# CPU (no GPU)
uv run python src/mbti_classifier/train.py \
    --config-name=train_cpu \
    data.batch_size=4 \
    data.num_workers=0
```

## Model Configuration

### Default Model (`configs/model/default.yaml`)

```yaml
_target_: src.mbti_classifier.model.MBTIModel
model_name: distilbert-base-uncased
learning_rate: 2e-5
dropout_rate: 0.1
weight_decay: 0.01
```

### Parameters Explained

| Parameter | Default | Description | When to change |
|-----------|---------|-------------|----------------|
| `model_name` | distilbert-base-uncased | Hugging Face base model | Change to `bert-base-uncased` for better accuracy (slower) |
| `learning_rate` | 2e-5 | Learning rate | Increase if underfitting, decrease if overfitting |
| `dropout_rate` | 0.1 | Dropout for regularization | Increase (0.2-0.3) if overfitting |
| `weight_decay` | 0.01 | L2 regularization | Increase if overfitting |

### Quick Model (`configs/model/quick.yaml`)

```yaml
_target_: src.mbti_classifier.model.MBTIModel
model_name: distilbert-base-uncased
learning_rate: 5e-5  # Higher for fast convergence
dropout_rate: 0.1
weight_decay: 0.01
```

**Difference:** Learning rate 2.5x higher for quick testing.

## Trainer Configuration

### Default Trainer (`configs/trainer/default.yaml`)

```yaml
_target_: lightning.pytorch.Trainer
max_epochs: 10
accelerator: auto
devices: auto
precision: 16-mixed
log_every_n_steps: 10
enable_checkpointing: true
deterministic: true
gradient_clip_val: null
accumulate_grad_batches: 1
```

### Parameters Explained

#### Basics

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_epochs` | 10 | Maximum number of epochs |
| `accelerator` | auto | Accelerator type (auto detects GPU/CPU/MPS) |
| `devices` | auto | Number of devices (auto uses all available) |

#### Precision

| Parameter | Default | Options | Notes |
|-----------|---------|----------|-------|
| `precision` | 16-mixed | 32, 16-mixed, bf16-mixed | 16-mixed reduces memory by 50% |

**Example:**

```bash
# Float32 (higher precision, more memory)
uv run python src/mbti_classifier/train.py trainer.precision=32

# Mixed precision (recommended)
uv run python src/mbti_classifier/train.py trainer.precision=16-mixed

# BFloat16 (if GPU supports it)
uv run python src/mbti_classifier/train.py trainer.precision=bf16-mixed
```

#### Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_every_n_steps` | 10 | Log metrics every N steps |

```bash
# Log more frequently
uv run python src/mbti_classifier/train.py trainer.log_every_n_steps=5

# Log less frequently (faster)
uv run python src/mbti_classifier/train.py trainer.log_every_n_steps=50
```

#### Gradient Clipping

Prevents exploding gradients:

```yaml
gradient_clip_val: 1.0  # Clip gradients to [-1, 1]
```

```bash
uv run python src/mbti_classifier/train.py trainer.gradient_clip_val=1.0
```

#### Gradient Accumulation

Simulates a larger batch size:

```yaml
accumulate_grad_batches: 4  # Effective batch = batch_size * 4
```

```bash
# Effective batch of 64 with small GPU
uv run python src/mbti_classifier/train.py \
    data.batch_size=16 \
    trainer.accumulate_grad_batches=4
```

#### Debugging

```yaml
# Fast training for debug
fast_dev_run: true  # Only 1 batch of train/val/test

# Limit batches
limit_train_batches: 10  # Only 10 batches per epoch
limit_val_batches: 5
limit_test_batches: 5

# Overfit on 1 batch (to test)
overfit_batches: 1
```

```bash
# Fast dev run
uv run python src/mbti_classifier/train.py trainer.fast_dev_run=true

# Limit batches
uv run python src/mbti_classifier/train.py \
    trainer.limit_train_batches=10 \
    trainer.limit_val_batches=5
```

### CPU Trainer (`configs/trainer/cpu.yaml`)

```yaml
_target_: lightning.pytorch.Trainer
max_epochs: 10
accelerator: cpu
devices: 1
precision: 32  # No mixed precision on CPU
log_every_n_steps: 10
enable_checkpointing: true
deterministic: true
```

### Quick Trainer (`configs/trainer/quick.yaml`)

```yaml
_target_: lightning.pytorch.Trainer
max_epochs: 1
accelerator: auto
devices: auto
precision: 16-mixed
log_every_n_steps: 5
enable_checkpointing: false
limit_train_batches: 10
limit_val_batches: 5
```

## Callbacks Configuration

### Default Callbacks (`configs/callbacks/default.yaml`)

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

### Model Checkpoint

Saves the best models:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dirpath` | models | Directory to save to |
| `filename` | best | Filename |
| `monitor` | val_loss | Metric to monitor |
| `mode` | min | Minimize or maximize metric |
| `save_top_k` | 1 | Save top K models |
| `save_last` | false | Save last checkpoint |

**Examples:**

```bash
# Save top 3 models
uv run python src/mbti_classifier/train.py \
    callbacks.model_checkpoint.save_top_k=3

# Monitor accuracy instead of loss
uv run python src/mbti_classifier/train.py \
    callbacks.model_checkpoint.monitor=val_acc_total \
    callbacks.model_checkpoint.mode=max

# Also save the last model
uv run python src/mbti_classifier/train.py \
    callbacks.model_checkpoint.save_last=true
```

### Early Stopping

Stops automatically if it doesn't improve:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `monitor` | val_loss | Metric to monitor |
| `patience` | 5 | Epochs without improvement before stopping |
| `mode` | min | Minimize or maximize |
| `min_delta` | 0.0 | Minimum improvement considered |

**Examples:**

```bash
# More patience
uv run python src/mbti_classifier/train.py \
    callbacks.early_stopping.patience=10

# Require minimum improvement of 0.001
uv run python src/mbti_classifier/train.py \
    callbacks.early_stopping.min_delta=0.001

# Disable early stopping
uv run python src/mbti_classifier/train.py \
    callbacks.early_stopping=null
```

## Logger Configuration

### CSV Logger (`configs/logger/csv.yaml`)

```yaml
_target_: lightning.pytorch.loggers.CSVLogger
save_dir: ${log_dir}
name: mbti_classifier
```

Saves logs to `logs/mbti_classifier/version_X/metrics.csv`.

### Wandb Logger (`configs/logger/wandb.yaml`)

```yaml
_target_: lightning.pytorch.loggers.WandbLogger
project: mbti-classifier
name: null
save_dir: ${log_dir}
log_model: false
```

**Setup:**

```bash
# 1. Install wandb
uv add wandb

# 2. Login
wandb login

# 3. Train
uv run python src/mbti_classifier/train.py logger=wandb
```

**Options:**

```bash
# Custom name
uv run python src/mbti_classifier/train.py \
    logger=wandb \
    logger.name=experiment-1

# Custom project
uv run python src/mbti_classifier/train.py \
    logger=wandb \
    logger.project=my-mbti-project

# Save model to wandb
uv run python src/mbti_classifier/train.py \
    logger=wandb \
    logger.log_model=true
```

## Full Examples

### 1. Basic Training

```bash
uv run python src/mbti_classifier/train.py
```

### 2. Quick Test

```bash
uv run python src/mbti_classifier/train.py --config-name=train_quick
```

### 3. CPU Training

```bash
uv run python src/mbti_classifier/train.py --config-name=train_cpu
```

### 4. Custom Hyperparameters

```bash
uv run python src/mbti_classifier/train.py \
    data.batch_size=32 \
    model.learning_rate=5e-5 \
    model.dropout_rate=0.2 \
    trainer.max_epochs=15 \
    callbacks.early_stopping.patience=8
```

### 5. Production with Wandb

```bash
export WANDB_API_KEY=your_key

uv run python src/mbti_classifier/train.py \
    --config-name=train_production \
    logger.name=production-v1 \
    logger.log_model=true
```

### 6. Hyperparameter Sweep

```bash
uv run python src/mbti_classifier/train.py -m \
    model.learning_rate=1e-5,2e-5,5e-5 \
    model.dropout_rate=0.1,0.2,0.3
```

Runs 9 combinations (3 x 3).

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
uv run python src/mbti_classifier/train.py data.batch_size=8

# Reduce max_length
uv run python src/mbti_classifier/train.py data.max_length=256

# Gradient accumulation (large effective batch, small memory)
uv run python src/mbti_classifier/train.py \
    data.batch_size=8 \
    trainer.accumulate_grad_batches=4
```

### Training Too Slow

```bash
# Use mixed precision
uv run python src/mbti_classifier/train.py trainer.precision=16-mixed

# Reduce logging
uv run python src/mbti_classifier/train.py trainer.log_every_n_steps=50

# Fewer workers (if CPU is the bottleneck)
uv run python src/mbti_classifier/train.py data.num_workers=2
```

### Not Converging

```bash
# Increase learning rate
uv run python src/mbti_classifier/train.py model.learning_rate=5e-5

# More epochs
uv run python src/mbti_classifier/train.py trainer.max_epochs=20

# Reduce regularization
uv run python src/mbti_classifier/train.py \
    model.dropout_rate=0.05 \
    model.weight_decay=0.001
```

### Overfitting

```bash
# Increase dropout
uv run python src/mbti_classifier/train.py model.dropout_rate=0.3

# Increase weight decay
uv run python src/mbti_classifier/train.py model.weight_decay=0.05

# More aggressive early stopping
uv run python src/mbti_classifier/train.py \
    callbacks.early_stopping.patience=3 \
    callbacks.early_stopping.min_delta=0.001
```

## References

- [Hydra Configuration](./hydra.md)
- [Model Architecture](../architecture/model.md)
- [Data Pipeline](../architecture/data-pipeline.md)
- [PyTorch Lightning Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html)