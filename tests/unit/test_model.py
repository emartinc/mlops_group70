"""Unit tests for the MBTI classifier model."""

import pytest
import torch
from mbti_classifier.model import MBTIClassifier


class TestMBTIClassifier:
    """Tests for MBTIClassifier class."""

    @pytest.fixture
    def model(self):
        """Create a model instance for testing."""
        return MBTIClassifier(
            model_name="distilbert-base-uncased",
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=100,
            dropout=0.1,
        )

    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert model.hparams.model_name == "distilbert-base-uncased"
        assert model.hparams.learning_rate == 2e-5
        assert model.hparams.dropout == 0.1
        assert model.encoder is not None

    def test_model_forward(self, model):
        """Test forward pass."""
        batch_size = 4
        seq_length = 128

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        logits = model(input_ids, attention_mask)

        # Should output [batch_size, 4] for 4 binary tasks
        assert logits.shape == (batch_size, 4)

    def test_shared_step(self, model):
        """Test shared step functionality."""
        batch_size = 4
        seq_length = 128

        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length),
            "labels": torch.randint(0, 2, (batch_size, 4)).float(),
        }

        outputs = model._shared_step(batch, "train")

        assert "loss" in outputs
        assert "logits" in outputs
        assert "preds" in outputs
        assert "labels" in outputs
        assert outputs["loss"].item() >= 0

    def test_binary_to_mbti(self, model):
        """Test binary to MBTI conversion."""
        binary_preds = torch.tensor(
            [
                [0, 0, 1, 0],  # INTP
                [1, 1, 0, 1],  # ESFJ
                [0, 0, 0, 1],  # INFJ
            ]
        )

        mbti_types = model._binary_to_mbti(binary_preds)

        assert mbti_types == ["INTP", "ESFJ", "INFJ"]
        assert len(mbti_types) == 3

    def test_predict_step(self, model):
        """Test prediction step."""
        batch_size = 2
        seq_length = 128

        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length),
        }

        predictions = model.predict_step(batch, 0)

        assert "predictions" in predictions
        assert "probabilities" in predictions
        assert "mbti_types" in predictions
        assert "logits" in predictions
        assert len(predictions["mbti_types"]) == batch_size

    def test_parameter_count(self, model):
        """Test parameter counting."""
        total_params = model.count_parameters()
        trainable_params = model.count_trainable_parameters()

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_freeze_encoder(self):
        """Test encoder freezing."""
        model = MBTIClassifier(
            model_name="distilbert-base-uncased", freeze_encoder=True
        )

        # Check that encoder parameters are frozen
        for param in model.encoder.parameters():
            assert not param.requires_grad

        # Classification head should still be trainable
        assert model.ei_head.weight.requires_grad

    def test_configure_optimizers(self, model):
        """Test optimizer configuration."""
        # Need to attach a dummy trainer
        from pytorch_lightning import Trainer

        trainer = Trainer(max_epochs=1, fast_dev_run=True)
        trainer.strategy.connect(model)
        trainer.lightning_module.trainer = trainer

        # Mock estimated_stepping_batches
        trainer.estimated_stepping_batches = 100

        optimizer_config = model.configure_optimizers()

        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
