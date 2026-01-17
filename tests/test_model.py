"""Tests for the MBTI Classifier Lightning Module."""

import pytest
import torch
from mbti_classifier.training.model import MBTIClassifier


@pytest.fixture
def model():
    """Create a model instance for testing."""
    return MBTIClassifier(
        model_name="distilbert-base-uncased",
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        dropout=0.1,
    )


def test_model_initialization(model):
    """Test that model initializes correctly."""
    assert model.hparams.learning_rate == 2e-5
    assert model.hparams.dropout == 0.1
    assert model.encoder is not None
    assert model.ei_head is not None
    assert model.sn_head is not None
    assert model.tf_head is not None
    assert model.jp_head is not None


def test_model_forward_pass(model):
    """Test forward pass with dummy data."""
    batch_size = 4
    seq_length = 128

    input_ids = torch.randint(0, 30522, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    logits = model(input_ids, attention_mask)

    assert logits.shape == (batch_size, 4)  # 4 binary tasks
    assert logits.dtype == torch.float32


def test_model_training_step(model):
    """Test training step."""
    batch = {
        "input_ids": torch.randint(0, 30522, (4, 128)),
        "attention_mask": torch.ones(4, 128),
        "labels": torch.randint(0, 2, (4, 4)).float(),  # Binary labels for 4 tasks
    }

    loss = model.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert loss > 0


def test_parameter_counting(model):
    """Test parameter counting methods."""
    total_params = model.count_parameters()
    trainable_params = model.count_trainable_parameters()

    assert total_params > 0
    assert trainable_params > 0
    assert trainable_params <= total_params


def test_freeze_encoder(model):
    """Test that encoder freezing works."""
    # Initially all parameters should be trainable
    initial_trainable = model.count_trainable_parameters()

    # Create model with frozen encoder
    frozen_model = MBTIClassifier(
        model_name="distilbert-base-uncased",
        freeze_encoder=True,
    )

    frozen_trainable = frozen_model.count_trainable_parameters()

    # Frozen model should have fewer trainable parameters
    assert frozen_trainable < initial_trainable


def test_prediction_step(model):
    """Test prediction step for inference."""
    batch = {
        "input_ids": torch.randint(0, 30522, (4, 128)),
        "attention_mask": torch.ones(4, 128),
    }

    outputs = model.predict_step(batch, 0)

    assert "predictions" in outputs
    assert "probabilities" in outputs
    assert "mbti_types" in outputs
    assert "logits" in outputs

    assert outputs["predictions"].shape == (4, 4)  # 4 samples, 4 binary tasks
    assert outputs["probabilities"].shape == (4, 4)
    assert len(outputs["mbti_types"]) == 4
    assert outputs["logits"].shape == (4, 4)

    # Check all probabilities are between 0 and 1
    assert torch.all(outputs["probabilities"] >= 0)
    assert torch.all(outputs["probabilities"] <= 1)
    
    # Check MBTI types are valid 4-character strings
    for mbti_type in outputs["mbti_types"]:
        assert len(mbti_type) == 4
        assert mbti_type[0] in ['E', 'I']
        assert mbti_type[1] in ['S', 'N']
        assert mbti_type[2] in ['T', 'F']
        assert mbti_type[3] in ['J', 'P']


def test_metrics_creation(model):
    """Test that metrics are created correctly."""
    assert model.train_metrics is not None
    assert model.val_metrics is not None
    assert model.test_metrics is not None
    
    # Check that we have metrics for each task
    assert 'EI' in model.train_metrics
    assert 'SN' in model.train_metrics
    assert 'TF' in model.train_metrics
    assert 'JP' in model.train_metrics
    assert 'overall' in model.train_metrics


def test_binary_to_mbti_conversion(model):
    """Test binary prediction to MBTI type conversion."""
    # Test conversion
    binary_preds = torch.tensor([
        [1, 1, 1, 1],  # ESTJ
        [0, 0, 0, 0],  # INFP
        [1, 0, 1, 0],  # ENTP
        [0, 1, 0, 1],  # ISFJ
    ])
    
    mbti_types = model._binary_to_mbti(binary_preds)
    
    assert mbti_types == ['ESTJ', 'INFP', 'ENTP', 'ISFJ']


def test_configure_optimizers(model):
    """Test optimizer configuration."""
    # Need to attach a dummy trainer for this test
    from pytorch_lightning import Trainer

    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    model.trainer = trainer

    optimizer_config = model.configure_optimizers()

    assert "optimizer" in optimizer_config
    assert "lr_scheduler" in optimizer_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
