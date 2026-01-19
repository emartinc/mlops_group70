import logging
import torch
import pytorch_lightning as pl
from transformers import DistilBertForSequenceClassification

logger = logging.getLogger(__name__)

class MBTIClassifier(pl.LightningModule):
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased", 
        num_labels: int = 4, 
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01
    ):
        super().__init__()
        self.save_hyperparameters()
        
        logger.info(f"Loading architecture: {model_name}")
        
        # 1. Use the EXACT model you wanted
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Pass inputs to the Hugging Face model
        output = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )
        
        # CRITICAL FIX FOR TESTS:
        # Hugging Face returns a complex object (SequenceClassifierOutput).
        # Your tests expect a simple Tensor (logits).
        # We extract .logits here so the rest of your code works.
        return output.logits

    def training_step(self, batch, batch_idx):
        # We can implement a simple shared step here
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Forward pass
        logits = self(input_ids, attention_mask)
        
        # Calculate loss (BCEWithLogitsLoss is standard for multi-label)
        # Note: We use manual loss calculation to match your labels shape
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

        self.log("train_loss", loss)
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

        self.log("val_loss", loss)
        return {"loss": loss, "logits": logits}

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        logits = self(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        
        return {"logits": logits, "probabilities": probs}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 1. Instantiate
    model = MBTIClassifier(num_labels=4)
    
    # 2. Dummy Data
    dummy_input = torch.randint(0, 1000, (1, 10))
    dummy_mask = torch.ones((1, 10))
    
    # 3. Forward Pass
    logits = model(dummy_input, dummy_mask)
    
    print(f"Correctly loaded model: {type(model).__name__}")
    print(f"Output shape (Logits): {logits.shape}") 
    # Output: [1, 4]