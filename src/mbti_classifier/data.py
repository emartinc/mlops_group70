import argparse
import logging
import random
import re
from pathlib import Path
from typing import Annotated
import pandas as pd
import typer
import mlcroissant as mlc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class MBTIDataset(Dataset):
    """PyTorch Dataset for MBTI personality types with tokenization.

    Treats MBTI as 4 independent binary classification tasks:
    - E/I (Extraversion vs Introversion)
    - S/N (Sensing vs Intuition)
    - T/F (Thinking vs Feeling)
    - J/P (Judging vs Perceiving)
    """

    def __init__(self, texts, binary_labels, tokenizer, max_length: int = 512, use_random_window: bool = False):
        """
        Args:
            texts: List or array of text posts
            binary_labels: Dict with keys ['E', 'S', 'T', 'J'], values are binary tensors (0 or 1)
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length for tokenization
            use_random_window: If True, uses random windows for data augmentation (training only)
        """
        self.texts = texts
        self.binary_labels = binary_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_random_window = use_random_window

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

        # Tokenize with random window augmentation if enabled
        if self.use_random_window:
            # First tokenize without truncation to get full length
            full_encoding = self.tokenizer(
                text,
                truncation=False,
                return_tensors="pt",
            )

            input_ids = full_encoding["input_ids"].squeeze(0)
            total_length = len(input_ids)

            # If sequence is longer than max_length, take a random window
            # BUT always keep [CLS] at start and [SEP] at end
            if total_length > self.max_length:
                # Extract [CLS] token (first token)
                cls_token = input_ids[0:1]  # [CLS]
                # Extract [SEP] token (last token)
                sep_token = input_ids[-1:]  # [SEP]

                # Content tokens (excluding [CLS] and [SEP])
                content_tokens = input_ids[1:-1]
                content_length = len(content_tokens)

                # Available space for content (max_length - 2 for [CLS] and [SEP])
                max_content_length = self.max_length - 2

                if content_length > max_content_length:
                    # Random start position in content
                    max_start = content_length - max_content_length
                    start_idx = random.randint(0, max_start)
                    end_idx = start_idx + max_content_length

                    # Extract random window from content
                    content_window = content_tokens[start_idx:end_idx]
                else:
                    # Content fits, use all of it
                    content_window = content_tokens

                # Reconstruct: [CLS] + content_window + [SEP]
                input_ids = torch.cat([cls_token, content_window, sep_token])
                attention_mask = torch.ones(len(input_ids), dtype=torch.long)

                # Pad if necessary
                current_length = len(input_ids)
                if current_length < self.max_length:
                    padding_length = self.max_length - current_length
                    input_ids = torch.cat(
                        [input_ids, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)]
                    )
                    attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            else:
                # If shorter than max_length, pad normally
                attention_mask = torch.ones(total_length, dtype=torch.long)
                padding_length = self.max_length - total_length
                if padding_length > 0:
                    input_ids = torch.cat(
                        [input_ids, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)]
                    )
                    attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
        else:
            # Standard tokenization with truncation from the beginning
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
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
            raw_data_path: Path to store raw downloaded data
            processed_data_path: Path to store processed data
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

        # Convert paths to Path objects
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
            # Use random window augmentation for training
            self.train_dataset = MBTIDataset(
                train_df["posts"].values,
                train_binary_labels,
                self.tokenizer,
                self.max_length,
                use_random_window=True,  # Enable random windows for training
            )
            if len(val_df) > 0:
                val_binary_labels = {
                    "E": val_df["is_E"].values,
                    "S": val_df["is_S"].values,
                    "T": val_df["is_T"].values,
                    "J": val_df["is_J"].values,
                }
                # Use standard truncation for validation (consistent evaluation)
                self.val_dataset = MBTIDataset(
                    val_df["posts"].values,
                    val_binary_labels,
                    self.tokenizer,
                    self.max_length,
                    use_random_window=False,  # No augmentation for validation
                )

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

        df.to_csv(csv_output_path, index=False)
        logger.info(f"Dataset downloaded and saved to: {csv_output_path}")

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise
    
    return csv_output_path

def process_data(
    raw_data_path: Annotated[Path, typer.Argument()] = Path("data/raw"),
    processed_output_path: Annotated[Path, typer.Argument()] = Path("data/processed"),
) -> None:
    """Main pipeline: Load, Clean, Feature Engineering, and Split."""
    
    # 1. Ensure and Load Data
    csv_path = ensure_data(raw_data_path)
    df = pd.read_csv(csv_path)
    logger.info(f"Processing {len(df)} rows...")

    # 2. Text Cleaning
    # Remove '|||' separators, URLs, and extra whitespace
    df["posts"] = df["posts"].str.replace(r"\|\|\|", " ", regex=True)
    df["posts"] = df["posts"].str.replace(r"http\S+", "", regex=True)
    df["posts"] = df["posts"].str.lower().str.strip()

    # 3. Feature Engineering
    # Convert MBTI letters into binary numerical columns (0 or 1)
    logger.info("Generating binary columns (is_E, is_S, is_T, is_J)...")
    
    # Axis 1: Extraversion (1) vs Introversion (0)
    df['is_E'] = df['type'].apply(lambda x: 1 if 'E' in x else 0)
    # Axis 2: Sensing (1) vs Intuition (0)
    df['is_S'] = df['type'].apply(lambda x: 1 if 'S' in x else 0)
    # Axis 3: Thinking (1) vs Feeling (0)
    df['is_T'] = df['type'].apply(lambda x: 1 if 'T' in x else 0)
    # Axis 4: Judging (1) vs Perceiving (0)
    df['is_J'] = df['type'].apply(lambda x: 1 if 'J' in x else 0)

    # 4. Train / Test Split
    logger.info("Splitting data into Train (80%) and Test (20%)...")
    # We use 'stratify' to ensure balanced personality types in both sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['type'])

    # 5. Save output files
    processed_output_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_output_path / "train.csv", index=False)
    test_df.to_csv(processed_output_path / "test.csv", index=False)

    logger.info(f"Success! Data saved to: {processed_output_path}")
    logger.info(f"Train set: {len(train_df)} rows | Test set: {len(test_df)} rows")


if __name__ == "__main__":
    typer.run(process_data)