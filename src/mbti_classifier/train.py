import logging
import sys
import os
from pathlib import Path
import pandas as pd
import torch
import typer
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, Trainer, TrainingArguments

# Add current directory to path
sys.path.append(os.getcwd())
from src.mbti_classifier.model import build_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MBTITrainDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)

def train_models(
    processed_data_path: Path = Path("/Users/estelamartincebrian/Desktop/mlops_group70/data/processed"),
    output_model_dir: Path = Path("models"),
    epochs: int = 1,
    batch_size: int = 8 
):
    logger.info("Starting Multi-Axis Training Pipeline...")

    # 1. Load Data
    train_path = processed_data_path / "train.csv"
    test_path = processed_data_path / "test.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"File not found: {train_path}. Run data.py first.")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path).sample(100, random_state=42) 

    # 2. Tokenization
    model_name = "distilbert-base-uncased"
    logger.info(f"Loading Tokenizer: {model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    logger.info("Tokenizing text...")
    train_encodings = tokenizer(df_train['posts'].tolist(), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(df_test['posts'].tolist(), truncation=True, padding=True, max_length=128)

    # 3. Training Loop
    axes = [('IE', 'is_E'), ('NS', 'is_S'), ('TF', 'is_T'), ('JP', 'is_J')]

    for axis_name, label_col in axes:
        logger.info(f"\n=== Training Specialist Model for Axis: {axis_name} ===")
        
        train_dataset = MBTITrainDataset(train_encodings, df_train[label_col].tolist())
        test_dataset = MBTITrainDataset(test_encodings, df_test[label_col].tolist())

        model = build_model(num_labels=2)
        
        training_args = TrainingArguments(
            output_dir=f"./models/checkpoints/{axis_name}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            eval_strategy="epoch",  # <--- CHANGED HERE
            save_strategy="no",
            logging_steps=50,
            use_mps_device=torch.backends.mps.is_available() 
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        trainer.train()

        final_path = output_model_dir / axis_name
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Model {axis_name} saved at: {final_path}")

if __name__ == "__main__":
    typer.run(train_models)
