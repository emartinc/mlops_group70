import logging
from pathlib import Path
from typing import Annotated
import pandas as pd
import typer
import mlcroissant as mlc
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_data(raw_path: Path) -> Path:
    """Ensures the mbti_1.csv file exists, downloading it via mlcroissant if needed."""
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

        # Fix column names (e.g., 'mbti_1.csv/posts' -> 'posts')
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
