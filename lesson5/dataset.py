import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

_PATH_CONTROL_TRANSLATION = {
    ord(char): None
    for char in "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff"
}


def sanitize_filesystem_path(file_path):
    raw_value = os.fspath(file_path)
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8", errors="ignore")

    cleaned = str(raw_value).translate(_PATH_CONTROL_TRANSLATION).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def load_ply_file(file_path):
    """Load .ply file and return vertices and labels"""
    normalized_path = sanitize_filesystem_path(file_path)
    with open(normalized_path, "r") as f:
        # Read header to find end of header
        line = f.readline()
        while line:
            if line.strip() == "end_header":
                break
            line = f.readline()

        # Read vertex data (x, y, z, label)
        verts = []
        labels = []
        for line in f:
            values = line.strip().split()
            if len(values) >= 4:  # x, y, z, label
                x, y, z = float(values[0]), float(values[1]), float(values[2])
                label = (
                    int(values[3]) if len(values) > 3 else 0
                )  # default label 0 if not provided
                verts.append([x, y, z])
                labels.append(label)
            elif len(values) == 3:  # x, y, z only
                x, y, z = float(values[0]), float(values[1]), float(values[2])
                verts.append([x, y, z])

        return np.array(verts), np.array(labels)


def sample_point_cloud(vertices, labels, num_points=1024):
    """Sample points from mesh vertices"""
    if num_points is None:
        if labels is None or len(labels) == 0:
            labels = np.zeros(len(vertices), dtype=np.int64)
        return vertices, labels

    if len(vertices) < num_points:
        indices = np.random.choice(len(vertices), num_points, replace=True)
    else:
        indices = np.random.choice(len(vertices), num_points, replace=False)

    point_cloud = vertices[indices]
    if labels is not None and len(labels) > 0:
        point_labels = labels[indices]
    else:
        point_labels = np.zeros(num_points, dtype=np.int64)
    return point_cloud, point_labels


def normalize_point_cloud(point_cloud):
    """Normalize point cloud to [-1, 1]"""
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid

    # Scale to [-1, 1]
    max_dist = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
    point_cloud = point_cloud / max_dist

    return point_cloud


class ValveSegmentationDataset(Dataset):
    @staticmethod
    def random_rotate_z(point_cloud):
        """Случайный поворот точки облака вокруг оси Z"""
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return point_cloud @ rotation_matrix.T

    @staticmethod
    def random_scale(point_cloud, scale_low=0.8, scale_high=1.25):
        scale = np.random.uniform(scale_low, scale_high)
        return point_cloud * scale

    @staticmethod
    def jitter_point_cloud(point_cloud, sigma=0.01, clip=0.05):
        noise = np.clip(
            sigma * np.random.randn(*point_cloud.shape),
            -clip, clip
        )
        return point_cloud + noise

    @staticmethod
    def random_point_dropout(point_cloud, max_dropout_ratio=0.2):
        dropout_ratio = np.random.rand() * max_dropout_ratio
        drop_idx = np.where(np.random.rand(point_cloud.shape[0]) < dropout_ratio)[0]

        if len(drop_idx) > 0:
            point_cloud[drop_idx] = point_cloud[0]

        return point_cloud
    def __init__(self, root, num_points=1024, split="train", val_sz=0.1, test_sz=0.1):
        self.augment = split == "train"#флаг аугментации
        self.root = root
        self.num_points = num_points
        self.split = split

        self.clouds = np.array(os.listdir(root))
        train_idx = np.random.choice(
            len(self.clouds),
            int((1 - val_sz - test_sz) * len(self.clouds)),
            replace=False,
        )
        val_idx = np.random.choice(
            list(set(range(len(self.clouds))) - set(train_idx)),
            int(val_sz * len(self.clouds)),
            replace=False,
        )
        test_idx = np.array(
            list(set(range(len(self.clouds))) - set(train_idx) - set(val_idx))
        )

        if split == "full":
            self.clouds = self.clouds
        elif split == "train":
            self.clouds = self.clouds[train_idx]
        elif split == "val":
            self.clouds = self.clouds[val_idx]
        elif split == "test":
            self.clouds = self.clouds[test_idx]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, idx):
        file_path = self.clouds[idx]
        vertices, labels = load_ply_file(os.path.join(self.root, file_path))

        # 1. Сэмплируем фиксированное число точек
        point_cloud, point_labels = sample_point_cloud(
            vertices, labels, self.num_points
        )

        # 2. Аугментация ТОЛЬКО для train
        if self.augment:
            point_cloud = ValveSegmentationDataset.random_rotate_z(point_cloud)
            point_cloud = ValveSegmentationDataset.random_scale(point_cloud)
            point_cloud = ValveSegmentationDataset.jitter_point_cloud(point_cloud)
            point_cloud = ValveSegmentationDataset.random_point_dropout(point_cloud)

        # 3. Нормализация после аугментации
        point_cloud = normalize_point_cloud(point_cloud)

        # 4. Преобразование в torch
        point_cloud = torch.FloatTensor(point_cloud)

        # 5. (3, N) — формат для PointNet
        point_cloud = point_cloud.transpose(0, 1)

        return point_cloud, torch.LongTensor(point_labels)






class ValveSegmentationLightningDataset(pl.LightningDataModule):
    def __init__(
        self,
        root,
        num_points=1024,
        num_classes=13,
        batch_size=2,
        num_workers=4,
        val_sz=0.1,
        test_sz=0.1,
    ):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_sz = val_sz
        self.test_sz = test_sz

    def setup(self, stage=None):
        self.train_dataset = ValveSegmentationDataset(
            root=self.root,
            num_points=self.num_points,
            split="train",
            val_sz=self.val_sz,
            test_sz=self.test_sz,
        )

        self.val_dataset = ValveSegmentationDataset(
            root=self.root,
            num_points=self.num_points,
            split="val",
            val_sz=self.val_sz,
            test_sz=self.test_sz,
        )

        self.test_dataset = ValveSegmentationDataset(
            root=self.root,
            num_points=self.num_points,
            split="test",
            val_sz=self.val_sz,
            test_sz=self.test_sz,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    dataset = ValveSegmentationDataset(
        root="C:\VPO\.venv\lesson5\\3011", num_points=1024, split="test"
    )
    print(f"Dataset size: {len(dataset)}")

    # Test loading a sample
    point_cloud, labels = dataset[0]
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:5]} (first 5 labels)")
