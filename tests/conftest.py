"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path
import pytest
import torch

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def model_checkpoint_dir(project_root):
    """Return the model checkpoint directory."""
    return project_root / "models" / "checkpoints"


@pytest.fixture(scope="function")
def device():
    """Return the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def sample_texts():
    """Return sample texts for testing."""
    return [
        "I love spending time alone with my thoughts. I enjoy analyzing complex problems.",
        "I'm very outgoing and love meeting new people. Social gatherings energize me.",
        "I prefer concrete facts and practical solutions to abstract theories.",
        "I make decisions based on logic and objective analysis rather than feelings.",
    ]


@pytest.fixture(scope="session")
def sample_mbti_types():
    """Return sample MBTI types."""
    return ["INTJ", "ENFP", "ISTJ", "ENTP"]

