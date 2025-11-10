from pathlib import Path

import logging
import pytorch_lightning as pl
import hydra
import torch

from pytorch_lightning.strategies.ddp import DDPStrategy

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone


log = logging.getLogger(__name__)

CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'config.yaml'


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Create and load model/data
    from cross_view_transformer.common import setup_model_module, setup_data_module
    model_module = setup_model_module(cfg)
    data_module = setup_data_module(cfg)

    # Load checkpoint
    ckpt_path = cfg.experiment.get('checkpoint_path', None)

    if ckpt_path is None:
        raise ValueError("Please provide checkpoint_path in config.experiment.checkpoint_path")

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log.info(f"Loading checkpoint from {ckpt_path}")
    model_module.backbone = load_backbone(str(ckpt_path))

    # Setup logger (optional for eval)
    logger = None
    if cfg.experiment.get('use_logger', False):
        logger = pl.loggers.WandbLogger(
            project=cfg.experiment.project,
            save_dir=cfg.experiment.save_dir,
            id=f"{cfg.experiment.uuid}_eval",
            name=f"eval_{cfg.experiment.uuid}"
        )

    # Eval trainer
    trainer = pl.Trainer(
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else 'auto',
        devices=cfg.trainer.get('gpus', 1) if isinstance(cfg.trainer.get('gpus'), int) else len(cfg.trainer.get('gpus', [0])),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    )

    # Run evaluation on validation set
    log.info("Starting evaluation on validation set...")
    val_metrics = trainer.validate(model_module, datamodule=data_module)

    log.info("Validation metrics:")
    for metrics in val_metrics:
        for key, value in metrics.items():
            log.info(f"  {key}: {value:.4f}")

    # Optionally run on test set
    if cfg.experiment.get('run_test', False):
        log.info("Starting evaluation on test set...")
        test_metrics = trainer.test(model_module, datamodule=data_module)

        log.info("Test metrics:")
        for metrics in test_metrics:
            for key, value in metrics.items():
                log.info(f"  {key}: {value:.4f}")

    log.info("Evaluation complete!")


if __name__ == '__main__':
    main()
