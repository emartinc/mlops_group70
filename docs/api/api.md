# API Server

Documentation for the FastAPI inference server.

## Endpoints

### Root

`GET /`

Returns service information and available endpoints.

**Response:**
```json
{
  "message": "MBTI Personality Classifier API",
  "version": "1.0.0",
  "endpoints": {
    "predict": "/predict",
    "health": "/health",
    "docs": "/docs"
  }
}
```

### Health Check

`GET /health`

Returns health status and model loading state.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "tokenizer_loaded": true,
  "device": "cpu"
}
```

### Predict

`POST /predict`

Predicts MBTI personality type from text.

**Request Body:**
```json
{
  "text": "I enjoy quiet evenings reading philosophy books..."
}
```

**Response:**
```json
{
  "mbti_type": "INFP",
  "dimensions": [
    {
      "dimension": "E/I",
      "score": 0.15,
      "label": "I"
    },
    {
      "dimension": "S/N",
      "score": 0.72,
      "label": "N"
    },
    {
      "dimension": "T/F",
      "score": 0.35,
      "label": "F"
    },
    {
      "dimension": "J/P",
      "score": 0.42,
      "label": "P"
    }
  ],
  "probabilities": {
    "E": 0.15,
    "I": 0.85,
    "S": 0.28,
    "N": 0.72,
    "T": 0.35,
    "F": 0.65,
    "J": 0.42,
    "P": 0.58
  }
}
```

**Error Responses:**

400 Bad Request:
```json
{
  "detail": "Text too short after cleaning. Please provide more text."
}
```

503 Service Unavailable:
```json
{
  "detail": "Model not loaded"
}
```

## Python Client Example

```python
import requests

# Create client
class MBTIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict(self, text: str) -> dict:
        """Predict MBTI type from text."""
        response = requests.post(
            f"{self.base_url}/predict",
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()
    
    def health(self) -> dict:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

# Usage
client = MBTIClient()

# Make prediction
result = client.predict(
    "I love abstract theories and philosophical discussions!"
)

print(f"Type: {result['mbti_type']}")
print(f"Probabilities: {result['probabilities']}")
```

## JavaScript Client Example

```javascript
class MBTIClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async predict(text) {
    const response = await fetch(`${this.baseUrl}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  }

  async health() {
    const response = await fetch(`${this.baseUrl}/health`);
    return await response.json();
  }
}

// Usage
const client = new MBTIClient();

const result = await client.predict(
  'I prefer concrete facts over abstract theories.'
);

console.log(`Type: ${result.mbti_type}`);
```

## CORS Configuration

By default, CORS is enabled for all origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, specify exact origins:

```python
allow_origins=[
    "https://your-frontend.com",
    "https://app.your-domain.com"
]
```
