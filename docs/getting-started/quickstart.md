# Quick Start

This guide will get you up and running with the MBTI classifier in minutes.

## Option 1: Using Pre-trained Model

The fastest way to start making predictions.

### 1. Pull Data and Model

```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# Download pre-trained model and data
uv run dvc pull
```

### 2. Start the API

```bash
# Start FastAPI server
uv run invoke api
```

The API will be available at http://localhost:8000

### 3. Start the UI

In a new terminal:

```bash
# Start Streamlit interface
uv run invoke ui
```

Access the UI at http://localhost:8501

### 4. Make a Prediction

Open http://localhost:8501 in your browser and try:

```
I enjoy spending quiet evenings reading philosophy books and pondering 
the meaning of existence. Abstract concepts fascinate me more than 
concrete details.
```

Expected result: INFP or similar Introverted, Intuitive type.

## Option 2: Train from Scratch

For full control and experimentation.

### 1. Prepare Data

```bash
# Download and preprocess MBTI dataset
uv run invoke preprocess-data
```

This will:
- Download raw data from Kaggle
- Clean text and create binary labels
- Save to `data/processed/processed_mbti.csv`

### 2. Train the Model

```bash
# Quick training (2 epochs for testing)
uv run python src/mbti_classifier/train.py --config-name train_quick

# Full training (default config)
uv run invoke train

# Production training (20 epochs)
uv run python src/mbti_classifier/train.py --config-name train_production
```

Training progress is logged to:
- Console
- WandB (if configured)
- `logs/` directory

### 3. Verify Model

```bash
# Model saved to models/best.ckpt
ls -lh models/

# Run tests to verify
uv run pytest tests/integration/test_pipeline.py
```

### 4. Version with DVC

```bash
# Track new model
uv run dvc add models

# Commit metadata
git add models.dvc
git commit -m "Train model v1"

# Push to cloud storage
uv run dvc push
```

## Option 3: Using Docker

Production-ready deployment with Docker Compose.

### 1. Build Images

```bash
# Build all Docker images
uv run invoke docker-build
```

### 2. Train (Optional)

```bash
# Train model in container
uv run invoke docker-train
```

### 3. Start Services

```bash
# Start API + UI
uv run invoke docker-up

# View logs
uv run invoke docker-logs
```

Access:
- API: http://localhost:8000/docs
- UI: http://localhost:8501

### 4. Stop Services

```bash
uv run invoke docker-down
```

## Testing the API

### cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love meeting new people and being the center of attention at parties!"
  }'
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "I prefer facts and details over theories and possibilities."}
)

print(response.json())
# {
#   "mbti_type": "ISTJ",
#   "dimensions": [...],
#   "probabilities": {...}
# }
```

## Configuration Overrides

Customize training on the fly:

```bash
# Override learning rate and batch size
uv run python src/mbti_classifier/train.py \
  model.learning_rate=5e-5 \
  data.batch_size=32

# Use CPU instead of GPU
uv run python src/mbti_classifier/train.py \
  trainer.accelerator=cpu

# Change maximum epochs
uv run python src/mbti_classifier/train.py \
  trainer.max_epochs=15
```

## Common Workflows

### Development Iteration

```bash
# 1. Make code changes
# 2. Run fast tests
uv run pytest -m "not slow"

# 3. Format code
uv run ruff format .

# 4. Quick training test
uv run python src/mbti_classifier/train.py --config-name train_quick

# 5. Full testing
uv run invoke test
```

### Data Updates

```bash
# 1. Update data
# (modify data/raw/ or re-run preprocessing)

# 2. Track changes
uv run dvc add data/processed

# 3. Commit and push
git add data/processed.dvc
git commit -m "Update dataset"
uv run dvc push
```

## Next Steps

- Learn about [Architecture](../architecture/overview.md)
- Explore [Configuration](../configuration/hydra.md)
- Read [API Reference](../api/data.md)
- Set up [Docker](../development/docker.md) for production

## Troubleshooting

### Model not found error

```bash
# Ensure model exists
ls models/best.ckpt

# If using DVC, pull first
uv run dvc pull
```

### Port already in use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uv run invoke api --port 8080
```

### Out of memory during training

```bash
# Reduce batch size
uv run python src/mbti_classifier/train.py data.batch_size=8

# Use gradient accumulation
uv run python src/mbti_classifier/train.py \
  trainer.accumulate_grad_batches=4
```
