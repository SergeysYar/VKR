from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from dataset import load_ply_file, normalize_point_cloud


def _load_local_pointnet_lightning() -> type:
    model_path = Path(__file__).resolve().parents[1] / "model.py"
    if not model_path.exists():
        raise ImportError(f"Local model.py was not found: {model_path}")

    spec = importlib.util.spec_from_file_location("lesson5_local_model", str(model_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for: {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    pointnet_class = getattr(module, "PointNetSegmentationLightning", None)
    if pointnet_class is None:
        raise ImportError(
            f"PointNetSegmentationLightning is missing in local module: {model_path}"
        )
    return pointnet_class


PointNetSegmentationLightning = _load_local_pointnet_lightning()

StatusCallback = Callable[[str], None]


@dataclass
class TrainingSettings:
    dataset_root: str
    num_classes: int = 13
    label_mode: str = "per_point"
    num_points: int = 4096
    val_size: float = 0.1
    test_size: float = 0.1
    batch_size: int = 4
    num_workers: int = 0
    max_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.7
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 15
    feature_transform_reg_weight: float = 0.001
    feature_transform: bool = True
    dropout: float = 0.3
    class_weights: list[float] | None = None
    seed: int = 42
    checkpoint_dir: str = "./checkpoints_ui"
    log_dir: str = "./logs_ui"


def list_ply_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted([path for path in root_path.rglob("*.ply") if path.is_file()])


def _align_labels(vertices: np.ndarray, labels: np.ndarray | None) -> np.ndarray:
    if labels is None or len(labels) == 0:
        return np.zeros(len(vertices), dtype=np.int64)
    if len(labels) == len(vertices):
        return labels.astype(np.int64, copy=False)

    aligned = np.zeros(len(vertices), dtype=np.int64)
    count = min(len(vertices), len(labels))
    aligned[:count] = labels[:count]
    return aligned


def _sample_points(
    vertices: np.ndarray,
    labels: np.ndarray,
    num_points: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if num_points is None or num_points <= 0 or len(vertices) == num_points:
        return vertices, labels
    if len(vertices) < num_points:
        indices = rng.choice(len(vertices), num_points, replace=True)
    else:
        indices = rng.choice(len(vertices), num_points, replace=False)
    return vertices[indices], labels[indices]


def collect_dataset_statistics(root: str | Path, max_files: int = 80) -> dict:
    files = list_ply_files(root)
    if not files:
        return {
            "file_count": 0,
            "scanned_files": 0,
            "point_count_min": 0,
            "point_count_max": 0,
            "point_count_mean": 0.0,
            "labeled_files": 0,
            "class_distribution": {},
        }

    point_counts: list[int] = []
    class_hist: dict[int, int] = {}
    labeled_files = 0

    for file_path in files[:max_files]:
        vertices, labels = load_ply_file(str(file_path))
        point_counts.append(int(len(vertices)))
        if labels is not None and len(labels) > 0:
            labeled_files += 1
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                class_hist[int(label)] = class_hist.get(int(label), 0) + int(count)

    return {
        "file_count": len(files),
        "scanned_files": min(len(files), max_files),
        "point_count_min": min(point_counts),
        "point_count_max": max(point_counts),
        "point_count_mean": float(np.mean(point_counts)),
        "labeled_files": labeled_files,
        "class_distribution": dict(sorted(class_hist.items(), key=lambda item: item[0])),
    }


class StablePointCloudDataset(Dataset):
    def __init__(
        self,
        file_paths: Iterable[Path],
        num_points: int,
        label_mode: str = "per_point",
        file_class_lookup: dict[str, int] | None = None,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.file_paths = list(file_paths)
        self.num_points = num_points
        self.label_mode = str(label_mode)
        self.file_class_lookup = file_class_lookup or {}
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def random_rotate_z(point_cloud: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        theta = rng.uniform(0, 2 * np.pi)
        rotation_matrix = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return point_cloud @ rotation_matrix.T

    @staticmethod
    def random_scale(
        point_cloud: np.ndarray,
        rng: np.random.Generator,
        scale_low: float = 0.8,
        scale_high: float = 1.25,
    ) -> np.ndarray:
        scale = rng.uniform(scale_low, scale_high)
        return point_cloud * scale

    @staticmethod
    def jitter_point_cloud(
        point_cloud: np.ndarray,
        rng: np.random.Generator,
        sigma: float = 0.01,
        clip: float = 0.05,
    ) -> np.ndarray:
        noise = np.clip(sigma * rng.standard_normal(point_cloud.shape), -clip, clip)
        return point_cloud + noise

    @staticmethod
    def random_point_dropout(
        point_cloud: np.ndarray,
        rng: np.random.Generator,
        max_dropout_ratio: float = 0.2,
    ) -> np.ndarray:
        dropout_ratio = rng.random() * max_dropout_ratio
        dropout_mask = rng.random(point_cloud.shape[0]) < dropout_ratio
        if np.any(dropout_mask):
            point_cloud = point_cloud.copy()
            point_cloud[dropout_mask] = point_cloud[0]
        return point_cloud

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[index]
        vertices, labels = load_ply_file(str(file_path))
        if len(vertices) == 0:
            raise ValueError(f"Point cloud has no vertices: {file_path}")

        if self.label_mode == "folder_name":
            class_id = self.file_class_lookup.get(str(file_path))
            if class_id is None:
                raise KeyError(
                    "Class mapping for file not found in folder_name mode: "
                    f"{file_path}"
                )
            labels = np.full(len(vertices), int(class_id), dtype=np.int64)
        else:
            labels = _align_labels(vertices, labels)
        points, point_labels = _sample_points(
            vertices=vertices,
            labels=labels,
            num_points=self.num_points,
            rng=self.rng,
        )

        if self.augment:
            points = self.random_rotate_z(points, self.rng)
            points = self.random_scale(points, self.rng)
            points = self.jitter_point_cloud(points, self.rng)
            points = self.random_point_dropout(points, self.rng)

        points = normalize_point_cloud(points)
        point_tensor = torch.tensor(points, dtype=torch.float32).transpose(0, 1)
        label_tensor = torch.tensor(point_labels, dtype=torch.long)
        return point_tensor, label_tensor


def _calculate_split_sizes(
    total: int,
    val_size: float,
    test_size: float,
) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError(
            "The dataset must contain at least 3 .ply files for train/val/test split."
        )

    val_count = int(round(total * val_size)) if val_size > 0 else 0
    test_count = int(round(total * test_size)) if test_size > 0 else 0

    if val_size > 0:
        val_count = max(1, val_count)
    if test_size > 0:
        test_count = max(1, test_count)

    while val_count + test_count > total - 1:
        if test_count >= val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break

    train_count = total - val_count - test_count
    if train_count < 1:
        raise ValueError("Unable to build a valid split with at least one training sample.")
    return train_count, val_count, test_count


class StablePointCloudDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        num_points: int,
        batch_size: int,
        num_workers: int,
        val_size: float,
        test_size: float,
        seed: int,
        label_mode: str = "per_point",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.num_points = num_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.label_mode = str(label_mode)
        self.class_name_to_id: dict[str, int] = {}
        self.file_class_lookup: dict[str, int] = {}

        self.train_dataset: StablePointCloudDataset | None = None
        self.val_dataset: StablePointCloudDataset | None = None
        self.test_dataset: StablePointCloudDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        all_files = list_ply_files(self.root)
        if not all_files:
            raise FileNotFoundError(f"No .ply files found in: {self.root}")

        train_count, val_count, test_count = _calculate_split_sizes(
            total=len(all_files),
            val_size=self.val_size,
            test_size=self.test_size,
        )

        indices = np.arange(len(all_files))
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)
        shuffled_files = [all_files[index] for index in indices]

        self.class_name_to_id = {}
        self.file_class_lookup = {}
        if self.label_mode == "folder_name":
            class_names = sorted({path.parent.name for path in all_files}, key=lambda name: name.lower())
            self.class_name_to_id = {name: idx for idx, name in enumerate(class_names)}
            self.file_class_lookup = {
                str(path): int(self.class_name_to_id[path.parent.name])
                for path in all_files
            }

        train_files = shuffled_files[:train_count]
        val_files = shuffled_files[train_count : train_count + val_count]
        test_files = shuffled_files[train_count + val_count : train_count + val_count + test_count]

        self.train_dataset = StablePointCloudDataset(
            file_paths=train_files,
            num_points=self.num_points,
            label_mode=self.label_mode,
            file_class_lookup=self.file_class_lookup,
            augment=True,
            seed=self.seed,
        )
        self.val_dataset = StablePointCloudDataset(
            file_paths=val_files,
            num_points=self.num_points,
            label_mode=self.label_mode,
            file_class_lookup=self.file_class_lookup,
            augment=False,
            seed=self.seed + 1,
        )
        self.test_dataset = StablePointCloudDataset(
            file_paths=test_files,
            num_points=self.num_points,
            label_mode=self.label_mode,
            file_class_lookup=self.file_class_lookup,
            augment=False,
            seed=self.seed + 2,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule is not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule is not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule is not initialized. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def _build_lightning_cfg(settings: TrainingSettings):
    cfg = {
        "seed": settings.seed,
        "data": {
            "root": str(settings.dataset_root),
            "num_classes": settings.num_classes,
            "label_mode": settings.label_mode,
            "num_points": settings.num_points,
            "val_sz": settings.val_size,
            "test_sz": settings.test_size,
            "class_weights": settings.class_weights,
        },
        "model": {
            "feature_transform": settings.feature_transform,
            "dropout": settings.dropout,
        },
        "training": {
            "max_epochs": settings.max_epochs,
            "batch_size": settings.batch_size,
            "num_workers": settings.num_workers,
            "gradient_clip_val": settings.gradient_clip_val,
            "early_stopping_patience": settings.early_stopping_patience,
            "optimizer": {
                "lr": settings.learning_rate,
                "weight_decay": settings.weight_decay,
            },
            "scheduler": {
                "name": "StepLR",
                "step_size": settings.scheduler_step_size,
                "gamma": settings.scheduler_gamma,
            },
            "checkpoint_dir": settings.checkpoint_dir,
            "log_dir": settings.log_dir,
        },
        "loss": {
            "feature_transform_reg_weight": settings.feature_transform_reg_weight,
        },
        "evaluation": {
            "visualize_results": False,
            "visualize_samples": 2,
        },
    }
    return OmegaConf.create(cfg)


def _notify(status_cb: StatusCallback | None, message: str) -> None:
    if status_cb is not None:
        status_cb(message)


def _to_float_dict(metrics: dict) -> dict[str, float]:
    converted: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            converted[key] = float(value.detach().cpu().item())
        elif isinstance(value, (float, int)):
            converted[key] = float(value)
    return converted


def train_model(
    settings: TrainingSettings,
    status_cb: StatusCallback | None = None,
) -> dict:
    dataset_root = Path(settings.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    Path(settings.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    _register_checkpoint_safe_globals()

    _notify(status_cb, "Preparing deterministic train/val/test split...")
    pl.seed_everything(settings.seed, workers=True)

    datamodule = StablePointCloudDataModule(
        root=dataset_root,
        num_points=settings.num_points,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        val_size=settings.val_size,
        test_size=settings.test_size,
        seed=settings.seed,
        label_mode=settings.label_mode,
    )

    cfg = _build_lightning_cfg(settings)
    model = PointNetSegmentationLightning(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=settings.checkpoint_dir,
        filename="pointnet-ui-{epoch:02d}-{val_iou:.4f}",
        monitor="val_iou",
        mode="max",
        save_top_k=2,
        save_last=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_iou",
        patience=settings.early_stopping_patience,
        mode="max",
        verbose=True,
    )

    tensorboard_logger = None
    csv_logger = CSVLogger(
        save_dir=settings.log_dir,
        name="pointnet_ui_csv",
    )
    loggers: list = [csv_logger]
    try:
        tensorboard_logger = TensorBoardLogger(
            save_dir=settings.log_dir,
            name="pointnet_ui",
        )
        loggers.insert(0, tensorboard_logger)
    except ModuleNotFoundError:
        _notify(
            status_cb,
            "TensorBoard не установлен, продолжаем только с CSV-логгером.",
        )

    trainer = pl.Trainer(
        max_epochs=settings.max_epochs,
        accelerator="auto",
        devices=1,
        logger=loggers if len(loggers) > 1 else loggers[0],
        callbacks=[checkpoint_callback, early_stopping],
        gradient_clip_val=settings.gradient_clip_val,
        deterministic=True,
        log_every_n_steps=10,
        enable_progress_bar=False,
    )

    _notify(status_cb, "Starting model training...")
    trainer.fit(model, datamodule=datamodule)

    best_checkpoint = checkpoint_callback.best_model_path
    if not best_checkpoint:
        best_checkpoint = checkpoint_callback.last_model_path

    _notify(status_cb, "Running test on the best checkpoint...")
    if best_checkpoint:
        test_output = trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=best_checkpoint,
            weights_only=False,
        )
    else:
        test_output = trainer.test(model, datamodule=datamodule, weights_only=False)

    if datamodule.train_dataset is None or datamodule.val_dataset is None or datamodule.test_dataset is None:
        raise RuntimeError("DataModule failed to initialize datasets.")

    result = {
        "best_checkpoint": best_checkpoint,
        "test_metrics": _to_float_dict(test_output[0]) if test_output else {},
        "split_counts": {
            "train": len(datamodule.train_dataset),
            "val": len(datamodule.val_dataset),
            "test": len(datamodule.test_dataset),
        },
        "settings": asdict(settings),
    }
    if datamodule.class_name_to_id:
        result["class_name_to_id"] = dict(datamodule.class_name_to_id)
    _notify(status_cb, "Training workflow completed.")
    return result


def _register_checkpoint_safe_globals() -> None:
    if not hasattr(torch.serialization, "add_safe_globals"):
        return

    safe_globals = [omegaconf.dictconfig.DictConfig]
    try:
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.listconfig import ListConfig
        from omegaconf.nodes import AnyNode

        safe_globals.extend(
            [
                ContainerMetadata,
                Metadata,
                AnyNode,
                ListConfig,
                dict,
                list,
                int,
                float,
                bool,
                Any,
            ]
        )
    except Exception:
        pass

    torch.serialization.add_safe_globals(safe_globals)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_file(
    checkpoint_path: str | Path,
    input_file: str | Path,
    num_points: int = 4096,
) -> dict:
    checkpoint = Path(checkpoint_path)
    source_file = Path(input_file)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not source_file.exists():
        raise FileNotFoundError(f"Input cloud not found: {source_file}")

    _register_checkpoint_safe_globals()
    device = _get_device()

    lightning_module = PointNetSegmentationLightning.load_from_checkpoint(
        str(checkpoint),
        map_location=device,
        weights_only=False,
    )
    model = lightning_module.model.to(device)
    model.eval()

    vertices, labels = load_ply_file(str(source_file))
    if len(vertices) == 0:
        raise ValueError(f"Input file contains zero points: {source_file}")

    has_true_labels = labels is not None and len(labels) > 0
    aligned_labels = _align_labels(vertices, labels)
    points, sampled_true_labels = _sample_points(
        vertices=vertices,
        labels=aligned_labels,
        num_points=num_points,
        rng=np.random.default_rng(42),
    )

    normalized_points = normalize_point_cloud(points.copy())
    input_tensor = (
        torch.tensor(normalized_points, dtype=torch.float32)
        .transpose(0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        logits, _, _ = model(input_tensor)
        predicted_labels = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
        probabilities = torch.exp(logits).squeeze(0)
        mean_confidence = float(probabilities.max(dim=-1).values.mean().item())

    unique_labels, label_counts = np.unique(predicted_labels, return_counts=True)
    prediction_histogram = [
        {"class_id": int(label), "count": int(count)}
        for label, count in zip(unique_labels, label_counts)
    ]

    return {
        "input_file": str(source_file),
        "checkpoint": str(checkpoint),
        "points": points,
        "normalized_points": normalized_points,
        "predicted_labels": predicted_labels,
        "true_labels": sampled_true_labels if has_true_labels else np.array([], dtype=np.int64),
        "has_true_labels": bool(has_true_labels),
        "prediction_histogram": prediction_histogram,
        "mean_confidence": mean_confidence,
    }


def save_predictions_to_ply(
    output_path: str | Path,
    points: np.ndarray,
    predicted_labels: np.ndarray,
    label_property: str = "pred_label",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.transpose(0, 1)

    if len(points) != len(predicted_labels):
        raise ValueError("Points and labels must have the same length.")

    with output.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write(f"property int {label_property}\n")
        file.write("end_header\n")
        for point, label in zip(points, predicted_labels):
            x, y, z = point
            file.write(f"{x} {y} {z} {int(label)}\n")

    return output


def build_point_cloud_figure(
    points: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "Point cloud",
    max_points: int = 50000,
    point_size: int = 2,
):
    import plotly.graph_objects as go

    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.transpose(0, 1)

    if len(points) == 0:
        raise ValueError("Cannot build a figure for an empty point cloud.")

    current_points = points
    current_labels = labels if labels is not None else None
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(points), max_points, replace=False)
        current_points = points[sample_indices]
        if current_labels is not None:
            current_labels = current_labels[sample_indices]

    marker_kwargs = {
        "size": point_size,
        "opacity": 0.85,
    }
    if current_labels is not None and len(current_labels) == len(current_points):
        marker_kwargs["color"] = current_labels
        marker_kwargs["colorscale"] = "Turbo"
        marker_kwargs["colorbar"] = {"title": "Class"}
    else:
        marker_kwargs["color"] = "#1f77b4"

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=current_points[:, 0],
                y=current_points[:, 1],
                z=current_points[:, 2],
                mode="markers",
                marker=marker_kwargs,
            )
        ]
    )
    figure.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        paper_bgcolor="#f6f9ff",
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
            "bgcolor": "#eff4ff",
        },
    )
    return figure


def find_latest_checkpoint(search_paths: Iterable[str | Path]) -> str | None:
    candidate_files: list[Path] = []
    for search_path in search_paths:
        root = Path(search_path)
        if root.exists():
            candidate_files.extend(path for path in root.rglob("*.ckpt") if path.is_file())

    if not candidate_files:
        return None

    latest = max(candidate_files, key=lambda path: path.stat().st_mtime)
    return str(latest)
