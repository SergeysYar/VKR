import os
import torch
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

from dataset import ValveSegmentationDataset
from model import PointNetSegmentationLightning

# Для безопасной загрузки checkpoint (PyTorch 2.6+)
import typing
import collections
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from omegaconf.listconfig import ListConfig

from pytorch_lightning.loggers import TensorBoardLogger

def main(cfg: DictConfig):
    # -----------------------------
    # 0. Разрешаем все "небезопасные" глобалы для torch.load
    # -----------------------------
    torch.serialization.add_safe_globals([
        DictConfig,
        ContainerMetadata,
        Metadata,
        AnyNode,
        ListConfig,
        typing.Any,
        dict,
        list,
        int,
        float,
        bool,
        collections.defaultdict
    ])

    # -----------------------------
    # 1. Dataset + DataLoader (TEST)
    # -----------------------------
    test_dataset = ValveSegmentationDataset(
        root=cfg.data.root,
        num_points=cfg.data.num_points,
        split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=cfg.test.num_workers,
        pin_memory=True
    )

    # -----------------------------
    # 2. Trainer (ТОЛЬКО test)
    # -----------------------------
    logger = TensorBoardLogger("tb_logs", name="pointnet_test")
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=logger,
        enable_checkpointing=False
    )

    # -----------------------------
    # 3. Проверка чекпоинта
    # -----------------------------
    ckpt_path = cfg.test.checkpoint_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\nLoading checkpoint:\n{ckpt_path}\n")

    # -----------------------------
    # 4. Загрузка модели
    # -----------------------------
    model = PointNetSegmentationLightning.load_from_checkpoint(ckpt_path)
    model.cfg = cfg
    model.eval()

    # -----------------------------
    # 5. TEST (без обучения)
    # -----------------------------
    trainer.test(
        model,
        dataloaders=test_loader
    )

    # -----------------------------
    # 6. Вывод результатов
    # -----------------------------
    if hasattr(model, "test_table") and len(model.test_table) > 0:
        df = pd.DataFrame(model.test_table)
        df[["accuracy", "iou"]] = df[["accuracy", "iou"]].round(4)

        print("\n====== TEST RESULTS (per class) ======")
        print(df)

        df.to_csv("test_metrics.csv", index=False)
        print("\nSaved to test_metrics.csv")
    else:
        print("test_table is empty or not found")


if __name__ == "__main__":
    # -----------------------------
    # Загрузка конфигурации
    # -----------------------------
    cfg = OmegaConf.load("conf.yaml")

    # Разрешаем добавление новых ключей
    OmegaConf.set_struct(cfg, False)

    # --- параметры теста ---
    cfg.test = DictConfig({
        "checkpoint_path": "checkpoints/pointnet-epoch=42-val_iou=0.6050-v1.ckpt",  # ← проверь путь
        "batch_size": 8,
        "num_workers": 4
    })

    main(cfg)
