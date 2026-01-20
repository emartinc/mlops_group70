import logging
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, MetricCollection
from transformers import AutoModel, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class MBTIClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for MBTI personality classification using DistilBERT.

    Treats MBTI as 4 independent binary classification tasks (multi-task learning):
    - E/I (Extraversion vs Introversion)
    - S/N (Sensing vs Intuition)
    - T/F (Thinking vs Feeling)
    - J/P (Judging vs Perceiving)

    This approach is more effective than 16-class classification and more interpretable.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        freeze_layers: int = 0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            model_name: HuggingFace model name to load
            learning_rate: Peak learning rate for optimizer
            weight_decay: Weight decay for AdamW optimizer
            warmup_steps: Number of warmup steps for learning rate scheduler
            dropout: Dropout probability for classification head
            freeze_encoder: Whether to freeze the entire encoder
            freeze_layers: Number of encoder layers to freeze (from bottom)
            pos_weight: Optional positive class weights for each task [4] to handle imbalance
        """
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])

        # Load pretrained transformer model
        logger.info(f"Loading pretrained model: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Multi-task classification heads (4 binary classifiers)
        # Shared representation
        self.shared_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Task-specific heads for E/I, S/N, T/F, J/P
        self.ei_head = nn.Linear(hidden_size, 1)  # E vs I
        self.sn_head = nn.Linear(hidden_size, 1)  # S vs N
        self.tf_head = nn.Linear(hidden_size, 1)  # T vs F
        self.jp_head = nn.Linear(hidden_size, 1)  # J vs P

        # Loss function - Binary Cross Entropy with Logits (includes sigmoid)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        # Metrics - use nn.ModuleDict to ensure proper device placement
        self.train_metrics = nn.ModuleDict(self._create_metrics(prefix="train"))
        self.val_metrics = nn.ModuleDict(self._create_metrics(prefix="val"))
        self.test_metrics = nn.ModuleDict(self._create_metrics(prefix="test"))

        # Apply freezing if specified
        self._freeze_parameters()

        logger.info(f"Model initialized with {self.count_parameters():,} parameters")
        logger.info(f"Trainable parameters: {self.count_trainable_parameters():,}")

    def _create_metrics(self, prefix: str) -> Dict[str, MetricCollection]:
        """Create metrics for each binary classification task."""
        task_names = ["EI", "SN", "TF", "JP"]
        metrics = {}

        for task_name in task_names:
            metrics[task_name] = MetricCollection(
                {
                    f"{prefix}/{task_name}_acc": Accuracy(task="binary"),
                    f"{prefix}/{task_name}_f1": F1Score(task="binary"),
                }
            )

        # Overall metrics (average across all tasks)
        metrics["overall"] = MetricCollection(
            {
                f"{prefix}/avg_acc": Accuracy(task="binary"),
                f"{prefix}/avg_f1": F1Score(task="binary"),
            }
        )

        return metrics

    def _freeze_parameters(self):
        """Freeze encoder parameters based on configuration."""
        if self.hparams.freeze_encoder:
            logger.info("Freezing entire encoder")
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif self.hparams.freeze_layers > 0:
            logger.info(f"Freezing first {self.hparams.freeze_layers} layers")
            # For DistilBERT, freeze transformer layers
            if hasattr(self.encoder, "transformer"):
                for layer in self.encoder.transformer.layer[: self.hparams.freeze_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            Logits [batch_size, 4] for E/I, S/N, T/F, J/P tasks
        """
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Pass through shared layers
        shared_repr = self.shared_layers(cls_output)

        # Task-specific predictions
        ei_logits = self.ei_head(shared_repr)  # [batch_size, 1]
        sn_logits = self.sn_head(shared_repr)  # [batch_size, 1]
        tf_logits = self.tf_head(shared_repr)  # [batch_size, 1]
        jp_logits = self.jp_head(shared_repr)  # [batch_size, 1]

        # Concatenate all task logits
        logits = torch.cat([ei_logits, sn_logits, tf_logits, jp_logits], dim=1)  # [batch_size, 4]

        return logits

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """
        Shared step for train/val/test.

        Args:
            batch: Batch dictionary with input_ids, attention_mask, labels
            stage: One of 'train', 'val', 'test'

        Returns:
            Dictionary with loss and logits
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]  # [batch_size, 4] binary labels

        # Forward pass
        logits = self(input_ids, attention_mask)  # [batch_size, 4]

        # Compute loss for each task
        task_losses = self.criterion(logits, labels)  # [batch_size, 4]
        loss = task_losses.mean()  # Average across all tasks and batch

        # Get predictions (apply sigmoid and threshold at 0.5)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # Update metrics for each task
        metrics = getattr(self, f"{stage}_metrics")
        task_names = ["EI", "SN", "TF", "JP"]

        for i, task_name in enumerate(task_names):
            metrics[task_name].update(preds[:, i], labels[:, i].long())

        # Update overall metrics (flatten all predictions)
        metrics["overall"].update(preds.flatten(), labels.flatten().long())

        return {"loss": loss, "logits": logits, "preds": preds, "labels": labels}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self._shared_step(batch, "train")
        self.log("train/loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return outputs["loss"]

    def on_train_epoch_end(self):
        """Log training metrics at epoch end."""
        all_metrics = {}
        for task_name, metric_collection in self.train_metrics.items():
            all_metrics.update(metric_collection.compute())
            metric_collection.reset()
        self.log_dict(all_metrics, sync_dist=True)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        outputs = self._shared_step(batch, "val")
        self.log("val/loss", outputs["loss"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return outputs["loss"]

    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end."""
        all_metrics = {}
        for task_name, metric_collection in self.val_metrics.items():
            all_metrics.update(metric_collection.compute())
            metric_collection.reset()
        self.log_dict(all_metrics, sync_dist=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        outputs = self._shared_step(batch, "test")
        self.log("test/loss", outputs["loss"], on_step=False, on_epoch=True, sync_dist=True)
        return outputs["loss"]

    def on_test_epoch_end(self):
        """Log test metrics at epoch end."""
        all_metrics = {}
        for task_name, metric_collection in self.test_metrics.items():
            all_metrics.update(metric_collection.compute())
            metric_collection.reset()
        self.log_dict(all_metrics, sync_dist=True)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        # Learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        logits = self(input_ids, attention_mask)  # [batch_size, 4]
        probs = torch.sigmoid(logits)  # [batch_size, 4] probabilities
        preds = (probs > 0.5).long()  # [batch_size, 4] binary predictions

        # Convert binary predictions to MBTI types
        mbti_types = self._binary_to_mbti(preds)

        return {
            "predictions": preds,  # [batch_size, 4] binary for E/I, S/N, T/F, J/P
            "probabilities": probs,  # [batch_size, 4]
            "mbti_types": mbti_types,  # List of MBTI strings like ['INTJ', 'ENFP', ...]
            "logits": logits,
        }

    def _binary_to_mbti(self, binary_preds: torch.Tensor) -> list:
        """Convert binary predictions to MBTI type strings.

        Args:
            binary_preds: [batch_size, 4] tensor with binary values

        Returns:
            List of MBTI type strings
        """
        mbti_types = []
        for pred in binary_preds:
            # pred is [E, S, T, J] where 1 means E/S/T/J, 0 means I/N/F/P
            ei = "E" if pred[0] == 1 else "I"
            sn = "S" if pred[1] == 1 else "N"
            tf = "T" if pred[2] == 1 else "F"
            jp = "J" if pred[3] == 1 else "P"
            mbti_types.append(f"{ei}{sn}{tf}{jp}")
        return mbti_types