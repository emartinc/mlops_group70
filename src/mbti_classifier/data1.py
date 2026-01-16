import logging
import re
from pathlib import Path
from typing import Annotated
import pandas as pd
import torch
import typer
import mlcroissant as mlc
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_data(raw_path: Path) -> Path:
    """
    Ensures the mbti_1.csv file exists, downloading it via mlcroissant if needed.
    """
    csv_output_path = raw_path / "mbti_1.csv"

    if csv_output_path.exists():
        logger.info(f"Raw data found locally at: {csv_output_path}")
        return csv_output_path

    logger.info(f"Data not found in {raw_path}. Downloading via mlcroissant...")
    raw_path.mkdir(parents=True, exist_ok=True)

    try:
        url = "https://www.kaggle.com/datasets/datasnaek/mbti-type/croissant/download"
        dataset = mlc.Dataset(url)
        record_sets = dataset.metadata.record_sets
        records = dataset.records(record_set=record_sets[0].uuid)
        df = pd.DataFrame(records)
        
        # Clean column names (remove prefixes like 'mbti_1.csv/')
        df.columns = [col.split("/")[-1] for col in df.columns]
        
        df.to_csv(csv_output_path, index=False)
        logger.info(f"Dataset downloaded and saved to: {csv_output_path}")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise
    
    return csv_output_path

def clean_text(text: str) -> str:
    """
    Advanced text cleaning.
    Removes byte string artifacts (b'..'), URLs, and normalizes whitespace.
    """
    # 1. Remove byte string prefixes and suffixes if present
    if text.startswith("b'") or text.startswith('b"'):
        # Removes b' at the start and the quote at the end
        text = text[2:-1]
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 3. Remove pipe separators (|||) and extra whitespace
    text = text.replace('|||', ' ')
    text = text.lower()
    text = ' '.join(text.split())
    
    return text

def process_data(
    raw_data_path: Annotated[Path, typer.Argument()] = Path("data/raw"),
    processed_output_path: Annotated[Path, typer.Argument()] = Path("data/processed"),
) -> None:
    """
    Main Pipeline:
    1. Clean data.
    2. Generate CSVs (for your current training script).
    3. Generate .pt file (for PyTorch compatibility).
    """
    
    # --- 1. Load Data ---
    csv_path = ensure_data(raw_data_path)
    df = pd.read_csv(csv_path)
    logger.info(f"Processing {len(df)} rows...")

    # --- 2. Clean Data ---
    # Fix 'type' column artifacts (e.g., b'INFJ' -> INFJ)
    if df['type'].dtype == object:
         df['type'] = df['type'].astype(str).str.replace(r"^b'|'$", "", regex=True)

    # Clean the 'posts' column
    df['posts'] = df['posts'].astype(str).apply(clean_text)

    # --- 3. Feature Engineering (For train.py) ---
    logger.info("Generating binary columns (is_E, is_S, is_T, is_J)...")
    df['is_E'] = df['type'].apply(lambda x: 1 if 'E' in x else 0)
    df['is_S'] = df['type'].apply(lambda x: 1 if 'S' in x else 0)
    df['is_T'] = df['type'].apply(lambda x: 1 if 'T' in x else 0)
    df['is_J'] = df['type'].apply(lambda x: 1 if 'J' in x else 0)

    # --- 4. Indexing (For .pt file) ---
    # Map types to integers 0-15 (e.g., INFJ -> 4)
    unique_types = sorted(df['type'].unique())
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    idx_to_type = {i: t for t, i in type_to_idx.items()}
    df['type_idx'] = df['type'].map(type_to_idx)

    # --- 5. Split Data ---
    logger.info("Splitting data into Train/Test...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['type']
    )

    processed_output_path.mkdir(parents=True, exist_ok=True)

    # --- 6. Save Format 1: CSVs ---
    # Used by your current train.py
    train_df.to_csv(processed_output_path / "train.csv", index=False)
    test_df.to_csv(processed_output_path / "test.csv", index=False)
    logger.info(f"Success! CSV files saved to {processed_output_path}")

    # --- 7. Save Format 2: PyTorch .pt ---
    # Used by your teammate or pure PyTorch scripts
    logger.info("Generating PyTorch .pt file...")
    data_dict = {
        'X_train': train_df['posts'].values,   # Raw text for train
        'X_test': test_df['posts'].values,     # Raw text for test
        'y_train': torch.tensor(train_df['type_idx'].values, dtype=torch.long), # Labels 0-15
        'y_test': torch.tensor(test_df['type_idx'].values, dtype=torch.long),   # Labels 0-15
        'type_to_idx': type_to_idx,
        'idx_to_type': idx_to_type,
        'num_classes': len(unique_types)
    }
    
    pt_path = processed_output_path / "mbti_cleaned.pt"
    torch.save(data_dict, pt_path)
    logger.info(f"Success! PyTorch file saved to: {pt_path}")

if __name__ == "__main__":
    typer.run(process_data)