"""Tests for the improved MBTI DataModule."""

import pytest
import torch
from mbti_classifier.training.data import MBTIDataModule
from torch.utils.data import DataLoader


@pytest.fixture
def data_module():
    """Create a DataModule instance for testing."""
    return MBTIDataModule(
        raw_data_path="data/raw",
        processed_data_path="data/processed",
        batch_size=8,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        test_size=0.15,
        val_size=0.15,
        random_seed=42,
        model_name="distilbert-base-uncased",
        max_length=128,  # Smaller for faster testing
    )


def test_datamodule_initialization(data_module):
    """Test that DataModule initializes correctly."""
    assert data_module.batch_size == 8
    assert data_module.test_size == 0.15
    assert data_module.val_size == 0.15
    assert data_module.max_length == 128
    assert data_module.model_name == "distilbert-base-uncased"
    assert data_module.tokenizer is None  # Not initialized until setup


def test_datamodule_setup(data_module):
    """Test that DataModule setup works correctly."""
    # This will download data if not present and initialize tokenizer
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Check tokenizer is initialized
    assert data_module.tokenizer is not None

    # Check datasets are created
    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None

    # Check label mappings
    assert data_module.num_classes == 16  # MBTI has 16 types
    assert len(data_module.type_to_idx) == 16
    assert len(data_module.idx_to_type) == 16


def test_datamodule_dataloaders(data_module):
    """Test that DataLoaders are created correctly."""
    data_module.prepare_data()
    data_module.setup(stage="fit")

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    # Check batch from train loader
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "label" in batch

    # Check shapes
    assert batch["input_ids"].shape == (8, 128)  # [batch_size, max_length]
    assert batch["attention_mask"].shape == (8, 128)
    assert batch["label"].shape == (8,)

    # Check types
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["label"].dtype == torch.long


def test_text_cleaning():
    """Test the text cleaning function."""
    # Test URL removal
    text = "Check this out http://example.com great stuff"
    cleaned = MBTIDataModule._clean_text(text)
    assert "http" not in cleaned

    # Test separator removal
    text = "First post|||Second post|||Third post"
    cleaned = MBTIDataModule._clean_text(text)
    assert "|||" not in cleaned

    # Test whitespace normalization
    text = "Too    many     spaces"
    cleaned = MBTIDataModule._clean_text(text)
    assert "  " not in cleaned

    # Test that punctuation is preserved
    text = "Hello, world! How are you?"
    cleaned = MBTIDataModule._clean_text(text)
    assert "," in cleaned
    assert "!" in cleaned
    assert "?" in cleaned


def test_dataset_with_custom_tokenizer(data_module):
    """Test that MBTIDataset works with a tokenizer."""
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Get a sample from the dataset
    sample = data_module.train_dataset[0]

    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "label" in sample

    # Check that tensors have correct dimensions
    assert sample["input_ids"].dim() == 1
    assert sample["attention_mask"].dim() == 1
    assert sample["label"].dim() == 0  # scalar

    # Check lengths
    assert len(sample["input_ids"]) == 128  # max_length
    assert len(sample["attention_mask"]) == 128


def test_datamodule_with_no_test_split():
    """Test DataModule with test_size=0."""
    dm = MBTIDataModule(
        batch_size=8,
        num_workers=0,
        test_size=0.0,
        val_size=0.1,
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    # test_dataset might be None or empty since test_size=0


def test_label_mappings(data_module):
    """Test that label mappings are bidirectional and complete."""
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Check that mappings are inverse of each other
    for mbti_type, idx in data_module.type_to_idx.items():
        assert data_module.idx_to_type[idx] == mbti_type

    # Check all MBTI types are present (should be 16)
    assert len(data_module.type_to_idx) == 16
    assert all(0 <= idx < 16 for idx in data_module.type_to_idx.values())


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
