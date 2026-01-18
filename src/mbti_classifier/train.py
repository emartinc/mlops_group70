import logging

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
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

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Initialize DataModule
    logger.info("Initializing DataModule...")
    data_module = instantiate(cfg.data)
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Initialize Model
    logger.info("Initializing Model...")
    model = instantiate(cfg.model)

    # Setup callbacks
    callbacks = [
        instantiate(cfg.callbacks.model_checkpoint),
        instantiate(cfg.callbacks.lr_monitor),
        instantiate(cfg.callbacks.progress_bar),
    ]

    # Add early stopping if enabled
    if cfg.early_stopping_enabled:
        callbacks.append(instantiate(cfg.callbacks.early_stopping))

    # Setup logger
    wandb_logger = None
    if cfg.use_wandb:
        wandb_logger = instantiate(cfg.logger)
        # Log hyperparameters
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    # Initialize Trainer
    logger.info("Initializing Trainer...")
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=wandb_logger)

    # Train
    logger.info("Starting training...")
    trainer.fit(model, data_module)

    # Test
    if cfg.run_test:
        logger.info("Running test...")
        trainer.test(model, data_module, ckpt_path="best")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
