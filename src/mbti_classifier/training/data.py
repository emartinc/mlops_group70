import logging
import re
from pathlib import Path
from typing import Optional

import mlcroissant as mlc
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Get the project root directory by searching for pyproject.toml.
    
    Returns:
        Path to the project root directory
    """
    current_path = Path(__file__).resolve()
    
    # Search upwards for pyproject.toml
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    
    # Fallback: assume we're in src/mbti_classifier/training/
    # and go up 3 levels to reach project root
    return current_path.parent.parent.parent


class MBTIDataset(Dataset):
    """PyTorch Dataset for MBTI personality types with tokenization.

    Treats MBTI as 4 independent binary classification tasks:
    - E/I (Extraversion vs Introversion)
    - S/N (Sensing vs Intuition)
    - T/F (Thinking vs Feeling)
    - J/P (Judging vs Perceiving)
    """

    def __init__(self, texts, binary_labels, tokenizer, max_length: int = 512):
        """
        Args:
            texts: List or array of text posts
            binary_labels: Dict with keys ['E', 'S', 'T', 'J'], values are binary tensors (0 or 1)
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.binary_labels = binary_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Get binary labels for all 4 dimensions
        labels = torch.tensor(
            [
                self.binary_labels["E"][idx],
                self.binary_labels["S"][idx],
                self.binary_labels["T"][idx],
                self.binary_labels["J"][idx],
            ],
            dtype=torch.float32,
        )

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,  # [4] binary labels for E, S, T, J
        }


class MBTIDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MBTI personality classification.

    Handles data downloading, cleaning, preprocessing, tokenization, and DataLoader creation.
    Optimized for fine-tuning transformer models like DistilBERT.
    """

    def __init__(
        self,
        raw_data_path: str = "data/raw",
        processed_data_path: str = "data/processed",
        batch_size: int = 16,
        num_workers: int = 4,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_seed: int = 42,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            raw_data_path: Path to store raw downloaded data (relative to project root or absolute)
            processed_data_path: Path to store processed data (relative to project root or absolute)
            batch_size: Batch size for DataLoaders (default 16 for transformer models)
            num_workers: Number of workers for DataLoaders
            test_size: Proportion of data for test set
            val_size: Proportion of train data for validation set
            random_seed: Random seed for reproducibility
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length for tokenization
            cache_dir: Optional directory to cache tokenizer and models
        """
        super().__init__()
        
        # Get project root
        project_root = get_project_root()
        
        # Convert paths to absolute paths relative to project root if they are relative
        raw_path = Path(raw_data_path)
        if not raw_path.is_absolute():
            raw_path = project_root / raw_path
        self.raw_data_path = raw_path
        
        processed_path = Path(processed_data_path)
        if not processed_path.is_absolute():
            processed_path = project_root / processed_path
        self.processed_data_path = processed_path
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = cache_dir

        # Initialize tokenizer
        self.tokenizer = None

        # Will be populated during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.type_to_idx = None
        self.idx_to_type = None
        self.num_classes = None

    def prepare_data(self):
        """
        Download data and tokenizer if needed. Called only on 1 GPU in distributed settings.
        """
        # Download dataset
        self._ensure_data()

        # Download tokenizer (will be cached)
        logger.info(f"Loading tokenizer: {self.model_name}")
        AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)

    def setup(self, stage: Optional[str] = None):
        """
        Load and split data. Called on every GPU in distributed settings.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Initialize tokenizer if not already done
        if self.tokenizer is None:
            logger.info(f"Initializing tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)

        # Load processed data or create it
        df = self._load_or_process_data()

        # Create label mappings
        unique_types = sorted(df["type"].unique())
        self.type_to_idx = {t: i for i, t in enumerate(unique_types)}
        self.idx_to_type = {i: t for t, i in self.type_to_idx.items()}
        self.num_classes = len(unique_types)

        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Class distribution:\n{df['type'].value_counts().sort_index()}")

        df["type_idx"] = df["type"].map(self.type_to_idx)

        # Split into train+val and test
        if self.test_size > 0:
            train_val_df, test_df = train_test_split(
                df, test_size=self.test_size, random_state=self.random_seed, stratify=df["type"]
            )
        else:
            train_val_df = df
            test_df = pd.DataFrame()

        # Split train into train and validation
        if self.val_size > 0:
            train_df, val_df = train_test_split(
                train_val_df, test_size=self.val_size, random_state=self.random_seed, stratify=train_val_df["type"]
            )
        else:
            train_df = train_val_df
            val_df = pd.DataFrame()

        # Create datasets with binary labels for multi-task learning
        if stage == "fit" or stage is None:
            train_binary_labels = {
                "E": train_df["is_E"].values,
                "S": train_df["is_S"].values,
                "T": train_df["is_T"].values,
                "J": train_df["is_J"].values,
            }
            self.train_dataset = MBTIDataset(
                train_df["posts"].values,
                train_binary_labels,
                self.tokenizer,
                self.max_length,
            )
            if len(val_df) > 0:
                val_binary_labels = {
                    "E": val_df["is_E"].values,
                    "S": val_df["is_S"].values,
                    "T": val_df["is_T"].values,
                    "J": val_df["is_J"].values,
                }
                self.val_dataset = MBTIDataset(
                    val_df["posts"].values,
                    val_binary_labels,
                    self.tokenizer,
                    self.max_length,
                )

        if stage == "test" or stage is None:
            if len(test_df) > 0:
                test_binary_labels = {
                    "E": test_df["is_E"].values,
                    "S": test_df["is_S"].values,
                    "T": test_df["is_T"].values,
                    "J": test_df["is_J"].values,
                }
                self.test_dataset = MBTIDataset(
                    test_df["posts"].values,
                    test_binary_labels,
                    self.tokenizer,
                    self.max_length,
                )

        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        logger.info(f"Max sequence length: {self.max_length}")

    def train_dataloader(self):
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """Create validation DataLoader."""
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        """Create test DataLoader."""
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def _ensure_data(self):
        """Download dataset if not present locally."""
        csv_path = self.raw_data_path / "mbti_1.csv"

        if csv_path.exists():
            logger.info(f"Raw data found at: {csv_path}")
            return csv_path

        logger.info("Downloading dataset via mlcroissant...")
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        try:
            url = "https://www.kaggle.com/datasets/datasnaek/mbti-type/croissant/download"
            dataset = mlc.Dataset(url)
            record_sets = dataset.metadata.record_sets
            records = dataset.records(record_set=record_sets[0].uuid)
            df = pd.DataFrame(records)

            # Clean column names
            df.columns = [col.split("/")[-1] for col in df.columns]

            df.to_csv(csv_path, index=False)
            logger.info(f"Dataset saved to: {csv_path}")
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise

        return csv_path

    def _load_or_process_data(self):
        """
        Load processed data if available, otherwise download, process, and save it.
        
        Returns:
            Processed DataFrame with cleaned text and binary labels
        """
        processed_file = self.processed_data_path / "processed_mbti.csv"
        
        # Check if processed data exists
        if processed_file.exists():
            logger.info(f"Loading processed data from: {processed_file}")
            df = pd.read_csv(processed_file)
            logger.info(f"Loaded {len(df)} preprocessed rows")
            return df
        
        # Download and process data
        logger.info("Processed data not found. Downloading and preprocessing...")
        self._ensure_data()
        df = self._load_and_clean_data()
        
        # Save processed data
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_file, index=False)
        logger.info(f"Saved processed data to: {processed_file}")
        
        return df

    def _load_and_clean_data(self):
        """Load and clean the MBTI dataset."""
        csv_path = self.raw_data_path / "mbti_1.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Data file not found at {csv_path}. " "Run prepare_data() first or ensure the file exists."
            )

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")

        # Clean 'type' column (remove byte string artifacts)
        if df["type"].dtype == object:
            df["type"] = df["type"].astype(str).str.replace(r"^b'|'$", "", regex=True).str.upper()

        # Clean 'posts' column
        df["posts"] = df["posts"].astype(str).apply(self._clean_text)

        # Remove rows with empty posts after cleaning
        initial_len = len(df)
        df = df[df["posts"].str.len() > 0].reset_index(drop=True)
        removed = initial_len - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed} rows with empty posts after cleaning")

        # Add binary features (optional, can be used for multi-task learning)
        df["is_E"] = df["type"].apply(lambda x: 1 if "E" in x else 0)
        df["is_S"] = df["type"].apply(lambda x: 1 if "S" in x else 0)
        df["is_T"] = df["type"].apply(lambda x: 1 if "T" in x else 0)
        df["is_J"] = df["type"].apply(lambda x: 1 if "J" in x else 0)

        logger.info(f"Final dataset size: {len(df)} rows")
        return df

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean text data by removing URLs, byte artifacts, and normalizing whitespace.
        Optimized for transformer models which can handle more natural text.

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

        # Remove URLs (keep URL text is sometimes meaningful, but typically not)
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", text)

        # Remove markdown image syntax
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

        # Remove excessive special characters but keep punctuation for sentiment
        # Keep: letters, numbers, spaces, and common punctuation
        text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\'\-\:\;]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Keep original casing - transformers can learn from it
        return text



"""
Training script for MBTI personality classification using PyTorch Lightning and Hydra.

Usage:
    uv run python src/mbti_classifier/training/train.py
    uv run python src/mbti_classifier/training/train.py model.learning_rate=1e-5
    uv run python src/mbti_classifier/training/train.py trainer.max_epochs=10
"""

import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="dataset")
def main(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration
    """
    # Print configuration
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    data_module = instantiate(cfg.data)
    data_module.prepare_data()

    


# if __name__ == "__main__":
#     main()


# if __name__ == "__main__":
#     # Download and preprocess and save processed data

    
#     data_module = MBTIDataModule()
#     data_module.prepare_data()

