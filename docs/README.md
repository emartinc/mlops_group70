# MBTI Classifier Training Module

PyTorch Lightning implementation for fine-tuning transformer models on MBTI personality classification.

## Files

- **`data.py`**: DataModule with tokenization for transformer models
- **`model.py`**: Lightning module with DistilBERT fine-tuning
- **`train_script.py`**: Hydra-based training script with W&B logging

## Quick Start

```bash
# Install dependencies (if not already installed)
uv add hydra-core torchmetrics

# Train with default settings
uv run python src/mbti_classifier/training/train_script.py

# Train with custom config
uv run python src/mbti_classifier/training/train_script.py --config-name=train_quick

# Override specific parameters
uv run python src/mbti_classifier/training/train_script.py \
    model.learning_rate=1e-5 \
    trainer.max_epochs=15
```

## Features

✅ DistilBERT fine-tuning  
✅ Hydra configuration management  
✅ W&B experiment tracking  
✅ Model checkpointing & early stopping  
✅ Mixed precision training  
✅ Multi-GPU support  
✅ Comprehensive metrics (accuracy, F1)  

## Documentation

See [docs/lightning_module_guide.md](../../../docs/lightning_module_guide.md) for complete documentation.

## Testing

```bash
# Run tests
uv run pytest tests/test_model.py -v
uv run pytest tests/test_data_module.py -v

# Run example
uv run python examples/example_model_usage.py
```
