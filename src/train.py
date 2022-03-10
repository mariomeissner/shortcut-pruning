import os
from pathlib import Path
from typing import List, Optional

import hydra
import torch
import torchinfo
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # # If this experiment already exists then warn and exit to avoid overwriting
    # work_dir = Path(config.logger.tensorboard.save_dir) / config.name
    # if os.path.exists(work_dir / config.callbacks.model_checkpoint.dirpath):
    #     log.error("This experiment has already been run. Give your experiment a different name!")
    #     exit()

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")

    # if config.get("batched_gpus"):

    #     hydra_conf = HydraConfig.get()
    #     if hydra_conf.sweeper.get("max_batch_size"):
    #         batch_size = hydra_conf.sweeper.max_batch_size
    #     elif hydra_conf.sweeper.get("n_jobs"):
    #         batch_size = hydra_conf.sweeper.n_jobs
    #     else:
    #         raise RuntimeError("Tried running batched jobs but no batch size was found.")

    #     log.info("Assining each job to a different GPU for parallel job execution.")
    #     log.info(f"Jobs batch size: {batch_size}.")
    #     log.info(f"This is job_num: {hydra_conf.job.num}.")
    #     gpu_id = hydra_conf.job.num % batch_size
    #     log.info(f"This job gets GPU:{gpu_id}.")
    #     config.trainer.gpus = [gpu_id]

    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger, _convert_="partial")

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Get metric score for hyperparameter optimization
    score = trainer.callback_metrics.get(config.get("optimized_metric"))

    # Test the model
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    log.info("Done!")

    # Return metric score for hyperparameter optimization
    return score
