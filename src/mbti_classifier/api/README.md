# MBTI Personality Classifier - API and Web Interface

This directory contains the FastAPI backend and Streamlit frontend for the MBTI personality classifier.

## Quick Start

### 1. Start the API Server

```bash
# Option 1: Using invoke
uv run invoke api

# Option 2: Direct command
uv run python src/mbti_classifier/api/api.py

# Option 3: Using uvicorn (recommended for production)
uv run uvicorn mbti_classifier.api.api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 2. Start the Streamlit UI

In a **separate terminal**:

```bash
# Option 1: Using invoke
uv run invoke ui

# Option 2: Direct command
uv run streamlit run src/mbti_classifier/api/ui.py
```

The UI will open automatically in your browser at `http://localhost:8501`

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Root
```bash
curl http://localhost:8000/
```

### Predict Personality Type
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here..."
  }'
```

### Interactive API Documentation

FastAPI provides automatic interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Response Format

```json
{
  "mbti_type": "INTJ",
  "dimensions": [
    {
      "dimension": "E/I",
      "score": 0.142,
      "label": "I"
    },
    {
      "dimension": "S/N",
      "score": 0.044,
      "label": "N"
    },
    {
      "dimension": "T/F",
      "score": 0.923,
      "label": "T"
    },
    {
      "dimension": "J/P",
      "score": 0.680,
      "label": "J"
    }
  ],
  "probabilities": {
    "E": 0.142,
    "I": 0.858,
    "S": 0.044,
    "N": 0.956,
    "T": 0.923,
    "F": 0.077,
    "J": 0.680,
    "P": 0.320
  }
}
```

## Web Interface Features

The Streamlit web interface provides:

- 📝 **Text Input**: Enter your writing samples
- 🔮 **Real-time Prediction**: Get instant MBTI predictions
- 📊 **Radar Plot Visualization**: Interactive radar chart showing personality dimensions
- 💡 **Detailed Insights**: Breakdown of each personality dimension
- 📚 **MBTI Information**: Educational content about personality types

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │  (Port 8501)
│   (Frontend)    │
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐
│   FastAPI       │  (Port 8000)
│   (Backend)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DistilBERT     │
│  MBTI Model     │
└─────────────────┘
```

## Testing

Run the test script to verify the API:

```bash
./test_api.sh
```

## Tips for Best Results

1. **Text Length**: Provide at least 100-200 words for accurate predictions
2. **Natural Writing**: Use conversational, natural text
3. **Personal Content**: Include personal thoughts and opinions
4. **Variety**: Mix different topics if possible

## Model Information

- **Model**: DistilBERT (distilbert-base-uncased)
- **Architecture**: Multi-task binary classification (4 independent tasks)
- **Dimensions**: E/I, S/N, T/F, J/P
- **Max Tokens**: 512
- **Device**: Automatically uses CUDA if available, falls back to CPU

## Troubleshooting

### API Server Won't Start

1. Check if port 8000 is already in use:
   ```bash
   lsof -i :8000
   ```

2. Verify the model checkpoint exists:
   ```bash
   ls -lh models/checkpoints/best.ckpt
   ```

### Streamlit UI Can't Connect

1. Ensure API server is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Check firewall settings if running on different machines

### Slow Predictions

- First prediction is slower due to model initialization
- GPU is recommended for faster inference
- Consider batch processing for multiple predictions

## Production Deployment

For production deployment:

1. **Use proper CORS settings** in `api.py`
2. **Set up HTTPS** with SSL certificates
3. **Use a production ASGI server** like Gunicorn with Uvicorn workers
4. **Add authentication** if needed
5. **Set up monitoring** and logging
6. **Configure rate limiting** to prevent abuse

Example production command:
```bash
gunicorn mbti_classifier.api.api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```
