"""Unit tests for API endpoints."""

import pytest


class TestAPI:
    """Tests for FastAPI endpoints."""

    def test_health_endpoint_structure(self):
        """Test expected health endpoint structure."""
        expected_keys = ["status", "model_loaded", "tokenizer_loaded", "device"]
        assert all(isinstance(key, str) for key in expected_keys)

    def test_prediction_request_validation(self):
        """Test prediction request validation."""
        from mbti_classifier.api import PredictionRequest

        # Valid request
        request = PredictionRequest(text="This is a test text for MBTI prediction.")
        assert request.text == "This is a test text for MBTI prediction."

        # Too short request should fail
        with pytest.raises(Exception):
            PredictionRequest(text="short")

    def test_prediction_response_structure(self):
        """Test prediction response structure."""
        from mbti_classifier.api import PredictionResponse, DimensionScore

        dimensions = [
            DimensionScore(dimension="E/I", score=0.8, label="E"),
            DimensionScore(dimension="S/N", score=0.3, label="N"),
            DimensionScore(dimension="T/F", score=0.9, label="T"),
            DimensionScore(dimension="J/P", score=0.6, label="J"),
        ]

        probabilities = {
            "E": 0.8,
            "I": 0.2,
            "S": 0.3,
            "N": 0.7,
            "T": 0.9,
            "F": 0.1,
            "J": 0.6,
            "P": 0.4,
        }

        response = PredictionResponse(
            mbti_type="ENTJ", dimensions=dimensions, probabilities=probabilities
        )

        assert response.mbti_type == "ENTJ"
        assert len(response.dimensions) == 4
        assert len(response.probabilities) == 8

    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        from mbti_classifier.api import clean_text

        # Test URL removal
        text = "Check this https://example.com website"
        cleaned = clean_text(text)
        assert "https://example.com" not in cleaned

        # Test whitespace normalization
        text = "Too    many    spaces"
        cleaned = clean_text(text)
        assert "  " not in cleaned

        # Test special character removal
        text = "Test @#$% text"
        cleaned = clean_text(text)
        assert "@#$%" not in cleaned
