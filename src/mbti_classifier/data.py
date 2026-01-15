import logging
from pathlib import Path
from typing import Annotated

import mlcroissant as mlc
import pandas as pd
import typer
from torch.utils.data import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MBTIDataset(Dataset):
    """MBTI dataset for personality type classification."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        csv_file = data_path / "mbti_1.csv"

        if not csv_file.exists():
            raise FileNotFoundError(f"Dataset file not found at {csv_file}. Run preprocessing first.")

        self.df = pd.read_csv(csv_file)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[str, str]:
        row = self.df.iloc[index]
        return row["posts"], row["type"]

    def preprocess(self, output_folder: Path) -> None:
        """Clean dataset and save to the processed folder."""
        logger.info(f"Preprocessing data to {output_folder}...")

        # Basic cleaning: remove separator and whitespace
        self.df["posts"] = self.df["posts"].str.replace(r"\|\|\|", " ", regex=True).str.strip()

        output_folder.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_folder / "processed_mbti.csv", index=False)
        logger.info("Preprocessing complete.")


def ensure_data(raw_path: Path) -> None:
    """Download dataset using mlcroissant if not present."""
    # Target file path
    csv_output_path = raw_path / "mbti_1.csv"

    if not csv_output_path.exists():
        logger.info(f"Data not found in {raw_path}. Downloading via mlcroissant...")
        raw_path.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize Croissant dataset from the URL provided
            url = "https://www.kaggle.com/datasets/datasnaek/mbti-type/croissant/download"
            dataset = mlc.Dataset(url)

            # Get record sets from metadata
            record_sets = dataset.metadata.record_sets
            if not record_sets:
                raise ValueError("No record sets found in the Croissant metadata.")

            logger.info(f"Fetching records for: {record_sets[0].uuid}")

            # Fetch records and convert to DataFrame
            records = dataset.records(record_set=record_sets[0].uuid)
            df = pd.DataFrame(records)

            # FIX: Clean column names by removing the file prefix (e.g., 'mbti_1.csv/posts' -> 'posts')
            df.columns = [col.split("/")[-1] for col in df.columns]

            # Save to raw directory to maintain project structure and DVC compatibility
            df.to_csv(csv_output_path, index=False)
            logger.info(f"Dataset successfully saved to {csv_output_path}")

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    else:
        logger.info("Dataset already exists locally.")


def main(
    raw_data_path: Annotated[Path, typer.Argument()] = Path("data/raw"),
    processed_output_path: Annotated[Path, typer.Argument()] = Path("data/processed"),
) -> None:
    """Orchestrate data download and preprocessing."""
    ensure_data(raw_data_path)

    dataset = MBTIDataset(raw_data_path)
    dataset.preprocess(processed_output_path)


if __name__ == "__main__":
    typer.run(main)
