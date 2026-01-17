"""
FastAPI server for MBTI personality classification inference.

Usage:
    uv run uvicorn mbti_classifier.api.api:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mbti_classifier.model import MBTIClassifier
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global variables for model and tokenizer
model = None
tokenizer = None
device = None


def clean_text(text: str) -> str:
    """
    Clean text data by removing URLs, byte artifacts, and normalizing whitespace.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    # Remove byte string prefixes
    if text.startswith("b'") or text.startswith('b"'):
        text = text[2:-1]

    # Remove post separators
    text = text.replace("|||", " ")

    # Remove URLs
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)

    # Remove markdown image syntax
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Remove excessive special characters but keep punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\'\-\:\;]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_model_and_tokenizer(checkpoint_path: str = "models/checkpoints/best.ckpt"):
    """Load the trained model and tokenizer."""
    global model, tokenizer, device

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    logger.info(f"Loading model from {checkpoint_path}")
    model = MBTIClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    # Load tokenizer
    model_name = model.hparams.model_name
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Model and tokenizer loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    try:
        load_model_and_tokenizer()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield

    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="MBTI Personality Classifier API",
    description="API for predicting Myers-Briggs personality types from text",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware to allow Streamlit to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text to analyze for personality prediction")


class DimensionScore(BaseModel):
    dimension: str
    score: float = Field(..., ge=0.0, le=1.0, description="Probability score (0-1)")
    label: str


class PredictionResponse(BaseModel):
    mbti_type: str = Field(..., description="Predicted MBTI type (e.g., 'INTJ')")
    dimensions: List[DimensionScore] = Field(..., description="Scores for each MBTI dimension")
    probabilities: Dict[str, float] = Field(..., description="Raw probabilities for each dimension")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MBTI Personality Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(device),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict MBTI personality type from text.

    Args:
        request: PredictionRequest with text to analyze

    Returns:
        PredictionResponse with MBTI type and dimension scores
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Clean the text
        text = clean_text(request.text)

        if len(text) < 10:
            raise HTTPException(
                status_code=400,
                detail="Text too short after cleaning. Please provide more text (at least 10 characters).",
            )

        # Tokenize
        encoding = tokenizer(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

        # Move to device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Predict
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [4]
            preds = (probs > 0.5).astype(int)  # [4]

        # Convert to MBTI type
        ei = "E" if preds[0] == 1 else "I"
        sn = "S" if preds[1] == 1 else "N"
        tf = "T" if preds[2] == 1 else "F"
        jp = "J" if preds[3] == 1 else "P"
        mbti_type = f"{ei}{sn}{tf}{jp}"

        # Create dimension scores
        dimensions = [
            DimensionScore(dimension="E/I", score=float(probs[0]), label=ei),
            DimensionScore(dimension="S/N", score=float(probs[1]), label=sn),
            DimensionScore(dimension="T/F", score=float(probs[2]), label=tf),
            DimensionScore(dimension="J/P", score=float(probs[3]), label=jp),
        ]

        # Create probability dict
        probabilities = {
            "E": float(probs[0]),
            "I": float(1 - probs[0]),
            "S": float(probs[1]),
            "N": float(1 - probs[1]),
            "T": float(probs[2]),
            "F": float(1 - probs[2]),
            "J": float(probs[3]),
            "P": float(1 - probs[3]),
        }

        return PredictionResponse(mbti_type=mbti_type, dimensions=dimensions, probabilities=probabilities)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
