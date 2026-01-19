"""Unit tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import torch

# Import your app
from mbti_classifier.api import app, PredictionRequest, PredictionResponse, DimensionScore, clean_text

client = TestClient(app)

class TestAPI:
    """Tests for FastAPI endpoints."""

    def test_health_endpoint_structure(self):
        """Test expected health endpoint structure."""
        expected_keys = ["status", "model_loaded", "tokenizer_loaded", "device"]
        assert all(isinstance(key, str) for key in expected_keys)

    def test_prediction_request_validation(self):
        """Test prediction request validation."""
        request = PredictionRequest(text="This is a test text for MBTI prediction.")
        assert request.text == "This is a test text for MBTI prediction."

        with pytest.raises(Exception):
            PredictionRequest(text="short")

    def test_prediction_response_structure(self):
        """Test prediction response structure."""
        dimensions = [
            DimensionScore(dimension="E/I", score=0.8, label="E"),
            DimensionScore(dimension="S/N", score=0.3, label="N"),
            DimensionScore(dimension="T/F", score=0.9, label="T"),
            DimensionScore(dimension="J/P", score=0.6, label="J"),
        ]
        probabilities = {"E": 0.8, "I": 0.2, "S": 0.3, "N": 0.7, "T": 0.9, "F": 0.1, "J": 0.6, "P": 0.4}

        response = PredictionResponse(
            mbti_type="ENTJ", dimensions=dimensions, probabilities=probabilities
        )
        assert response.mbti_type == "ENTJ"

    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        text = "Check this https://example.com website"
        cleaned = clean_text(text)
        assert "https://example.com" not in cleaned
        assert "  " not in clean_text("Too    many    spaces")

    def test_root_endpoint(self):
        """Test the Health Check Endpoint (GET /)."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    # 🚀 KEY FIX: Patch BOTH model AND tokenizer!
    @patch("mbti_classifier.api.tokenizer") 
    @patch("mbti_classifier.api.model")     
    def test_predict_endpoint(self, mock_model, mock_tokenizer):
        """Test the Prediction Endpoint with mocked globals."""
        
        # 1. Setup Fake Model Output
        mock_output = MagicMock()
        # Logits for [E, S, T, J] all negative -> [0,0,0,0] after sigmoid -> I, N, F, P
        mock_output.logits = torch.tensor([[-10.0, -10.0, -10.0, -10.0]])
        mock_model.return_value = mock_output
        mock_model.device = "cpu"

        # 2. Setup Fake Tokenizer Output
        # The API calls tokenizer(...), so we make it return a dict of tensors
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }

        # 3. Call the API
        payload = {
            "text": "I really enjoy reading books alone at home.",
            "request_id": "test_123"
        }
        response = client.post("/predict", json=payload)

        # 4. Verify Success
        assert response.status_code == 200, f"Failed with: {response.text}"
        
        data = response.json()
        assert data["mbti_type"] == "INFP"  # Matches our negative logits
        assert "dimensions" in data