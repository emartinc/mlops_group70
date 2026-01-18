"""Integration tests for the full training pipeline."""

import pytest
import torch
from pathlib import Path


@pytest.mark.slow
@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for training."""

    def test_quick_training_run(self, tmp_path):
        """Test a quick training run end-to-end."""
        pytest.skip("Integration test - run manually with: pytest -m integration")

        # This would test:
        # 1. Data loading
        # 2. Model initialization
        # 3. Training for 1 epoch
        # 4. Checkpoint saving
        # 5. Model loading from checkpoint

    def test_checkpoint_loading(self):
        """Test loading model from checkpoint."""
        from mbti_classifier.model import MBTIClassifier

        checkpoint_path = Path("models/checkpoints/best.ckpt")

        if not checkpoint_path.exists():
            pytest.skip("Best checkpoint not found")

        model = MBTIClassifier.load_from_checkpoint(checkpoint_path)
        assert model is not None

        # Test inference
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)

        logits = model(input_ids, attention_mask)
        assert logits.shape == (batch_size, 4)


@pytest.mark.slow
@pytest.mark.integration
class TestDataPipeline:
    """Integration tests for data pipeline."""

    def test_full_data_pipeline(self):
        """Test full data pipeline from download to DataLoader."""
        from mbti_classifier.data import MBTIDataModule

        dm = MBTIDataModule(
            raw_data_path="data/raw",
            processed_data_path="data/processed",
            batch_size=4,
            num_workers=0,
            max_length=128,
        )

        # Prepare and setup
        dm.prepare_data()
        dm.setup(stage="fit")

        # Get data loaders
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        # Get batches
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        assert train_batch["input_ids"].shape[0] == 4
        assert val_batch["labels"].shape[1] == 4


@pytest.mark.slow
@pytest.mark.integration
class TestAPIPipeline:
    """Integration tests for API."""

    def test_api_prediction_pipeline(self):
        """Test full API prediction pipeline."""
        pytest.skip("Requires running API server - test manually")

        # This would test:
        # 1. API server startup
        # 2. Model loading
        # 3. Prediction request
        # 4. Response validation
