# MBTI Personality Classifier - Complete Setup Guide

## 🎯 What Has Been Implemented

### 1. FastAPI Backend (`src/mbti_classifier/api/api.py`)
- ✅ RESTful API for MBTI personality prediction
- ✅ Automatic model loading on startup
- ✅ CORS enabled for cross-origin requests
- ✅ Health check endpoint
- ✅ Interactive API documentation (Swagger/ReDoc)
- ✅ Request validation with Pydantic
- ✅ GPU support (automatically uses CUDA if available)

### 2. Streamlit Web Interface (`src/mbti_classifier/api/ui.py`)
- ✅ Beautiful, user-friendly web interface
- ✅ Real-time personality prediction
- ✅ Interactive radar plot visualization using Plotly
- ✅ Detailed dimension breakdown
- ✅ MBTI type descriptions
- ✅ Personality insights for each dimension
- ✅ Input validation and error handling

### 3. Additional Features
- ✅ Invoke tasks for easy server management
- ✅ API test script
- ✅ Comprehensive documentation

## 🚀 Quick Start

### Step 1: Start the API Server

Open a terminal and run:

```bash
cd /home/prg/GitHub/mlops_group70

# Start API server (choose one option)
uv run invoke api                          # Option 1: Using invoke
uv run python src/mbti_classifier/api/api.py  # Option 2: Direct
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:__main__:Using device: cuda
INFO:__main__:Loading model from models/checkpoints/best.ckpt
INFO:__main__:Model and tokenizer loaded successfully!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Start the Streamlit UI

Open a **NEW terminal** and run:

```bash
cd /home/prg/GitHub/mlops_group70

# Start Streamlit UI (choose one option)
uv run invoke ui                              # Option 1: Using invoke
uv run streamlit run src/mbti_classifier/api/ui.py  # Option 2: Direct
```

The web interface will automatically open in your browser at `http://localhost:8501`

### Step 3: Use the Web Interface

1. **Enter Text**: Write or paste text (100+ words recommended)
2. **Click Predict**: Click the "🔮 Predict Personality Type" button
3. **View Results**: See your MBTI type, radar plot, and detailed insights

## 📊 API Usage Examples

### Example 1: Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "tokenizer_loaded": true,
  "device": "cuda"
}
```

### Example 2: Predict Personality
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love spending time alone with my thoughts. I enjoy analyzing complex problems and coming up with creative solutions. I prefer to plan things ahead rather than being spontaneous. When making decisions, I rely more on logic than emotions."
  }'
```

Response:
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
    "E": 0.142, "I": 0.858,
    "S": 0.044, "N": 0.956,
    "T": 0.923, "F": 0.077,
    "J": 0.680, "P": 0.320
  }
}
```

### Example 3: Python Client
```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Your text here..."}
)

result = response.json()
print(f"Personality Type: {result['mbti_type']}")
for dim in result['dimensions']:
    print(f"{dim['dimension']}: {dim['label']} ({dim['score']:.2%})")
```

## 🎨 Web Interface Features

### Main Features:
- **Text Input Area**: Large text box for entering your writing
- **Character/Word Count**: Real-time tracking
- **Prediction Button**: Clear call-to-action
- **Results Display**: Clean, organized results

### Visualizations:
- **Radar Plot**: Interactive Plotly chart showing all 4 dimensions
- **Dimension Breakdown**: Color-coded scores for each dimension
- **Personality Insights**: Explanations of what each result means

### Additional Info:
- **MBTI Descriptions**: Detailed description of all 16 types
- **Tips Section**: Best practices for accurate predictions
- **Error Handling**: User-friendly error messages

## 📁 File Structure

```
src/mbti_classifier/api/
├── __init__.py          # Module initialization
├── api.py              # FastAPI backend server
├── ui.py               # Streamlit web interface
└── README.md           # API documentation

models/checkpoints/
├── best.ckpt           # Best model checkpoint (used by default)
├── last.ckpt           # Last training checkpoint
└── mbti_distilbert_final.ckpt  # Final model

tasks.py                # Invoke tasks (api, ui commands)
test_api.sh            # API testing script
```

## 🔧 Configuration

### API Configuration (api.py)
- **Default Port**: 8000
- **Default Model**: `models/checkpoints/best.ckpt`
- **Max Tokens**: 512
- **CORS**: Enabled for all origins (change for production)

### UI Configuration (ui.py)
- **Default Port**: 8501
- **API URL**: `http://localhost:8000`
- **Min Text Length**: 50 characters

### Changing Ports:
```bash
# API on custom port
uv run invoke api --port 9000

# Streamlit on custom port
uv run invoke ui --port 8502
```

## 📚 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check and status |
| `/predict` | POST | Predict MBTI personality type |
| `/docs` | GET | Interactive API documentation (Swagger) |
| `/redoc` | GET | Alternative API documentation (ReDoc) |

## 🎯 Tips for Best Results

1. **Text Length**:
   - Minimum: 50 characters
   - Recommended: 100-200 words
   - More text = better accuracy

2. **Text Quality**:
   - Use natural, conversational writing
   - Include personal thoughts and opinions
   - Avoid copy-pasted formal text

3. **Content**:
   - Mix different topics
   - Show how you think and make decisions
   - Express your preferences and values

## 🐛 Troubleshooting

### API Won't Start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Verify model exists
ls -lh models/checkpoints/best.ckpt
```

### UI Can't Connect
```bash
# Test API manually
curl http://localhost:8000/health

# Check if API is running
ps aux | grep uvicorn
```

### Slow Predictions
- First prediction is slower (model initialization)
- Use GPU for faster inference (currently using CUDA)
- Consider caching for repeated texts

## 🚀 Testing

Run the comprehensive API test:
```bash
./test_api.sh
```

This tests:
- ✅ Health endpoint
- ✅ Root endpoint
- ✅ Prediction with sample text

## 📦 Dependencies

All required dependencies are already installed:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `streamlit` - Web interface
- `plotly` - Interactive plots
- `pydantic` - Data validation
- `transformers` - Model and tokenizer
- `torch` - Deep learning framework

## 🎉 Demo Time!

1. **Start both servers** (API + UI)
2. **Open browser** to `http://localhost:8501`
3. **Enter sample text**:
   ```
   I love spending time alone with my thoughts. I enjoy analyzing
   complex problems and coming up with creative solutions. I prefer
   to plan things ahead rather than being spontaneous. When making
   decisions, I rely more on logic than emotions. I find deep
   conversations more interesting than small talk.
   ```
4. **Click "Predict"**
5. **Explore the results**:
   - MBTI type (e.g., INTJ)
   - Radar plot showing all dimensions
   - Detailed breakdown of each trait
   - Personality insights

## 🌟 Next Steps

Potential enhancements:
- [ ] Add user authentication
- [ ] Save prediction history
- [ ] Batch predictions
- [ ] Confidence intervals
- [ ] Compare multiple texts
- [ ] Export results as PDF
- [ ] Multi-language support

---

**Created**: January 17, 2026
**Status**: ✅ Fully Functional
**Model**: DistilBERT-based MBTI Classifier
**Framework**: FastAPI + Streamlit
