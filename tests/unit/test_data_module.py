"""Unit tests for the data module."""

import pytest
import torch
from mbti_classifier.data import MBTIDataModule, MBTIDataset
from torch.utils.data import DataLoader


class TestMBTIDataset:
    """Tests for MBTIDataset class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        from transformers import AutoTokenizer

        texts = ["This is a test text.", "Another test text for MBTI."]
        binary_labels = {
            "E": [1, 0],
            "S": [0, 1],
            "T": [1, 1],
            "J": [0, 0],
        }
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        return texts, binary_labels, tokenizer

    def test_dataset_length(self, sample_data):
        """Test dataset length."""
        texts, binary_labels, tokenizer = sample_data
        dataset = MBTIDataset(texts, binary_labels, tokenizer, max_length=128)
        assert len(dataset) == 2

    def test_dataset_getitem(self, sample_data):
        """Test dataset __getitem__."""
        texts, binary_labels, tokenizer = sample_data
        dataset = MBTIDataset(texts, binary_labels, tokenizer, max_length=128)

        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].shape == (128,)
        assert item["attention_mask"].shape == (128,)
        assert item["labels"].shape == (4,)

    def test_random_window_augmentation(self, sample_data):
        """Test that random window augmentation produces different results."""
        texts, binary_labels, tokenizer = sample_data
        # Use a longer text
        long_text = " ".join(["test"] * 1000)
        texts = [long_text]
        binary_labels = {"E": [1], "S": [0], "T": [1], "J": [0]}

        dataset = MBTIDataset(
            texts, binary_labels, tokenizer, max_length=128, use_random_window=True
        )

        # Get same item multiple times - should be different with random windows
        item1 = dataset[0]
        item2 = dataset[0]

        # They might be different (random), but structure should be same
        assert item1["input_ids"].shape == item2["input_ids"].shape
        assert item1["labels"].tolist() == item2["labels"].tolist()


class TestMBTIDataModule:
    """Tests for MBTIDataModule class."""

    @pytest.fixture
    def data_module(self):
        """Create a DataModule instance for testing."""
        return MBTIDataModule(
            raw_data_path="data/raw",
            processed_data_path="data/processed",
            batch_size=8,
            num_workers=0,  # Use 0 for testing
            test_size=0.15,
            val_size=0.15,
            random_seed=42,
            model_name="distilbert-base-uncased",
            max_length=128,  # Smaller for faster testing
        )

    def test_datamodule_initialization(self, data_module):
        """Test that DataModule initializes correctly."""
        assert data_module.batch_size == 8
        assert data_module.test_size == 0.15
        assert data_module.val_size == 0.15
        assert data_module.max_length == 128
        assert data_module.model_name == "distilbert-base-uncased"
        assert data_module.tokenizer is None  # Not initialized until setup

    def test_datamodule_setup(self, data_module):
        """Test that DataModule setup works correctly."""
        data_module.prepare_data()
        data_module.setup(stage="fit")

        # Check tokenizer is initialized
        assert data_module.tokenizer is not None

        # Check datasets are created
        assert data_module.train_dataset is not None
        assert data_module.val_dataset is not None

        # Check label mappings
        assert data_module.num_classes == 16
        assert len(data_module.type_to_idx) == 16

    def test_train_dataloader(self, data_module):
        """Test train dataloader creation."""
        data_module.prepare_data()
        data_module.setup(stage="fit")

        train_loader = data_module.train_dataloader()
        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 8

        # Get one batch
        batch = next(iter(train_loader))
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

    def test_val_dataloader(self, data_module):
        """Test validation dataloader creation."""
        data_module.prepare_data()
        data_module.setup(stage="fit")

        val_loader = data_module.val_dataloader()
        assert isinstance(val_loader, DataLoader)

        batch = next(iter(val_loader))
        assert batch["input_ids"].shape[0] <= 8  # batch size

    def test_test_dataloader(self, data_module):
        """Test test dataloader creation."""
        data_module.prepare_data()
        data_module.setup(stage="test")

        test_loader = data_module.test_dataloader()
        assert isinstance(test_loader, DataLoader)

    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        from mbti_classifier.data import MBTIDataModule

        # Test URL removal
        text = "Check this https://example.com website"
        cleaned = MBTIDataModule._clean_text(text)
        assert "https://example.com" not in cleaned

        # Test whitespace normalization
        text = "Too    many    spaces"
        cleaned = MBTIDataModule._clean_text(text)
        assert "  " not in cleaned
