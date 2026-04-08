import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import ValveSegmentationLightningDataset
from model import PointNetSegmentationLightning
import torch
from omegaconf import DictConfig



@hydra.main(config_path=".", config_name="conf", version_base=None)
def main(cfg: DictConfig):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    dataset = ValveSegmentationLightningDataset(
        root=cfg.data.root,
        num_points=cfg.data.num_points,
        num_classes=cfg.data.num_classes,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        val_sz=cfg.data.val_sz,
        test_sz=cfg.data.test_sz,
    )



    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="pointnet-{epoch:02d}-{val_iou:.4f}",
        monitor="val_iou",
        mode="max",
        save_top_k=2,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_iou",
        patience=cfg.training.early_stopping_patience,
        mode="max",
        verbose=True,
    )

    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir, name="pointnet", version=None
    )
    # Определяем корректный девайс для Windows
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("CUDA available:", torch.cuda.is_available())
    print("Selected accelerator:", device)

    model = PointNetSegmentationLightning(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=device,  # вместо cfg.training.device
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dataset)

    #torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
    torch.serialization.add_safe_globals([DictConfig])
    trainer.test(
        model,
        datamodule=dataset,
        ckpt_path=checkpoint_callback.best_model_path,
        weights_only=False
    )


    print("\nTraining completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"TensorBoard logs at: {logger.log_dir}")


if __name__ == "__main__":
    main()
