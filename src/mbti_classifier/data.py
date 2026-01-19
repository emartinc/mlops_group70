import argparse
import logging
import random
import re
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
import mlcroissant as mlc
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def download_dataset(output_path: Path) -> Path:
    """Downloads dataset via mlcroissant if not exists."""
    csv_path = output_path / "mbti_1.csv"
    if csv_path.exists():
        logger.info(f"Raw data found at: {csv_path}")
        return csv_path

    logger.info("Downloading dataset via mlcroissant...")
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        url = "https://www.kaggle.com/datasets/datasnaek/mbti-type/croissant/download"
        dataset = mlc.Dataset(url)
        record_sets = dataset.metadata.record_sets
        records = dataset.records(record_set=record_sets[0].uuid)
        df = pd.DataFrame(records)
        df.columns = [col.split("/")[-1] for col in df.columns]
        df.to_csv(csv_path, index=False)
        return csv_path
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise e

def clean_text(text: str) -> str:
    """Cleans text by removing separators, URLs, and EXTRA SPACES."""
    text = str(text)
    text = text.replace("|||", " ")
    text = re.sub(r"http\S+", "", text)
    # FIX: Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def load_and_process_df(raw_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)
    df["posts"] = df["posts"].apply(clean_text)
    df['is_E'] = df['type'].apply(lambda x: 1 if 'E' in x else 0)
    df['is_S'] = df['type'].apply(lambda x: 1 if 'S' in x else 0)
    df['is_T'] = df['type'].apply(lambda x: 1 if 'T' in x else 0)
    df['is_J'] = df['type'].apply(lambda x: 1 if 'J' in x else 0)
    return df

# --- Dataset Class ---
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
            self.test_dataset = MBTIDataset(test_df["posts"].values, get_labels(test_df), self.tokenizer, self.max_length, use_random_window=False)

    def train_dataloader(self): return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    def val_dataloader(self): return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self): return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# --- CLI Command ---
def process_data_cli(raw_data_path: Annotated[Path, typer.Argument()] = Path("data/raw"), processed_output_path: Annotated[Path, typer.Argument()] = Path("data/processed")):
    csv_path = download_dataset(raw_data_path)
    df = load_and_process_df(csv_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['type'])
    processed_output_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_output_path / "train.csv", index=False)
    test_df.to_csv(processed_output_path / "test.csv", index=False)
    logger.info(f"Success! Data saved to: {processed_output_path}")

if __name__ == "__main__":
    typer.run(process_data_cli)