import argparse
import logging
import random
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MBTIDataset(Dataset):
    def __init__(self, texts, binary_labels, tokenizer, max_length: int = 512, use_random_window: bool = False):
        self.texts = texts
        self.binary_labels = binary_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_random_window = use_random_window

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = torch.tensor([
            self.binary_labels["E"][idx], self.binary_labels["S"][idx],
            self.binary_labels["T"][idx], self.binary_labels["J"][idx]
        ], dtype=torch.float32)

        if self.use_random_window:
            full_encoding = self.tokenizer(text, truncation=False, return_tensors="pt")
            input_ids = full_encoding["input_ids"].squeeze(0)
            if len(input_ids) > self.max_length:
                content_len = len(input_ids) - 2
                max_content = self.max_length - 2
                if content_len > max_content:
                    start = random.randint(0, content_len - max_content)
                    content = input_ids[1:-1][start : start + max_content]
                    input_ids = torch.cat([input_ids[0:1], content, input_ids[-1:]])

            if len(input_ids) < self.max_length:
                pad_len = self.max_length - len(input_ids)
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])

            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        else:
            encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# --- DataModule Class ---
class MBTIDataModule(pl.LightningDataModule):
    def __init__(self, raw_data_path: str = "data/raw", processed_data_path: str = "data/processed", batch_size: int = 16, num_workers: int = 4, test_size: float = 0.15, val_size: float = 0.15, random_seed: int = 42, model_name: str = "distilbert-base-uncased", max_length: int = 512, cache_dir: Optional[str] = None):
        super().__init__()
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        self.model_name = model_name
        self.max_length = max_length
        self.cache_dir = cache_dir

        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # FIX: Needed for tests
        self.num_classes = 16
        self.type_to_idx = {}

    @staticmethod
    def _clean_text(text):
        return clean_text(text)

    def prepare_data(self):
        download_dataset(self.raw_data_path)
        AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)

    def setup(self, stage: Optional[str] = None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)

        csv_path = self.raw_data_path / "mbti_1.csv"
        if not csv_path.exists():
            download_dataset(self.raw_data_path)

        df = load_and_process_df(csv_path)

        # FIX: Populate type_to_idx for tests
        unique_types = sorted(df["type"].unique())
        self.type_to_idx = {t: i for i, t in enumerate(unique_types)}

        train_val_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_seed, stratify=df["type"])
        train_df, val_df = train_test_split(train_val_df, test_size=self.val_size, random_state=self.random_seed, stratify=train_val_df["type"])

        def get_labels(d):
            return {"E": d["is_E"].values, "S": d["is_S"].values, "T": d["is_T"].values, "J": d["is_J"].values}

        if stage == "fit" or stage is None:
            self.train_dataset = MBTIDataset(train_df["posts"].values, get_labels(train_df), self.tokenizer, self.max_length, use_random_window=True)
            self.val_dataset = MBTIDataset(val_df["posts"].values, get_labels(val_df), self.tokenizer, self.max_length, use_random_window=False)
        if stage == "test" or stage is None:
            if len(test_df) > 0:
                test_binary_labels = {
                    "E": test_df["is_E"].values,
                    "S": test_df["is_S"].values,
                    "T": test_df["is_T"].values,
                    "J": test_df["is_J"].values,
                }
                # Use standard truncation for testing (consistent evaluation)
                self.test_dataset = MBTIDataset(
                    test_df["posts"].values,
                    test_binary_labels,
                    self.tokenizer,
                    self.max_length,
                    use_random_window=False,  # No augmentation for testing
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


def main():
    """Download and preprocess MBTI data."""
    parser = argparse.ArgumentParser(description="Download and preprocess MBTI dataset")
    parser.add_argument("raw_data_path", type=str, help="Path to store raw downloaded data")
    parser.add_argument("processed_data_path", type=str, help="Path to store processed data")
    args = parser.parse_args()

    logger.info(f"Raw data path: {args.raw_data_path}")
    logger.info(f"Processed data path: {args.processed_data_path}")

    # Create data module with specified paths
    data_module = MBTIDataModule(
        raw_data_path=args.raw_data_path,
        processed_data_path=args.processed_data_path,
    )

    # Download raw data if needed
    logger.info("Ensuring raw data is downloaded...")
    data_module._ensure_data()

    # Process and save data
    logger.info("Processing data...")
    df = data_module._load_or_process_data()

    logger.info("✓ Data preprocessing complete!")
    logger.info(f"  - Raw data: {data_module.raw_data_path / 'mbti_1.csv'}")
    logger.info(f"  - Processed data: {data_module.processed_data_path / 'processed_mbti.csv'}")
    logger.info(f"  - Total rows: {len(df)}")


if __name__ == "__main__":
    main()
