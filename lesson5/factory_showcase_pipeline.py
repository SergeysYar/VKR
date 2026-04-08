from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dataset import load_ply_file, sanitize_filesystem_path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_FACTORY_SRC = REPO_ROOT / "synthetic_factory" / "src"
LASER_SENSING_ROOT = REPO_ROOT / "laserSensing"

if str(SYNTHETIC_FACTORY_SRC) not in sys.path:
    sys.path.insert(0, str(SYNTHETIC_FACTORY_SRC))

_SYNTHETIC_IMPORT_ERROR: Exception | None = None
try:
    from synthetic_factory.generators.factory_generator import (
        FactoryGenerator,
        FactoryParams,
        RoomLayout,
    )
    from synthetic_factory.sensing.lidar_pipeline import LidarStation, LidarSurveyGenerator
except Exception as synthetic_exc:  # pragma: no cover - runtime dependency issue
    FactoryGenerator = None  # type: ignore[assignment]
    FactoryParams = None  # type: ignore[assignment]
    RoomLayout = None  # type: ignore[assignment]
    LidarStation = None  # type: ignore[assignment]
    LidarSurveyGenerator = None  # type: ignore[assignment]
    _SYNTHETIC_IMPORT_ERROR = synthetic_exc


@dataclass(frozen=True)
class DatasetClassInfo:
    class_name: str
    class_index: int
    directory: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class RoomInfo:
    room_id: str
    biome: str
    center_x: float
    center_y: float
    width: float
    depth: float
    floor_z: float
    ceiling_z: float

    @property
    def min_x(self) -> float:
        return self.center_x - self.width / 2.0

    @property
    def max_x(self) -> float:
        return self.center_x + self.width / 2.0

    @property
    def min_y(self) -> float:
        return self.center_y - self.depth / 2.0

    @property
    def max_y(self) -> float:
        return self.center_y + self.depth / 2.0


@dataclass
class PlacedObject:
    instance_id: str
    object_class: str
    class_index: int
    source_file: str
    room_id: str
    points: np.ndarray
    part_labels: np.ndarray
    position: tuple[float, float, float]
    scale: float


@dataclass
class PartClassificationResult:
    points: np.ndarray
    labels: np.ndarray
    source: str
    notes: list[str] = field(default_factory=list)

    @property
    def histogram(self) -> dict[int, int]:
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique_labels, counts)}


@dataclass
class ReconstructionResult:
    object_id: str
    backend: str
    combined_points: np.ndarray
    generated_files: list[str]
    part_summaries: list[dict[str, Any]]
    notes: list[str] = field(default_factory=list)


@dataclass
class FactoryDemoResult:
    run_id: str
    output_dir: str
    dataset_classes: list[DatasetClassInfo]
    rooms: list[RoomInfo]
    stations: list[dict[str, Any]]
    placed_objects: list[PlacedObject]
    detection_by_room: dict[str, list[dict[str, Any]]]
    notes: list[str]
    factory_points: np.ndarray
    factory_labels: np.ndarray
    factory_room_indices: np.ndarray
    combined_points: np.ndarray
    combined_source: np.ndarray
    combined_factory_label: np.ndarray
    combined_object_type: np.ndarray
    combined_object_instance: np.ndarray
    combined_part_label: np.ndarray
    combined_room_index: np.ndarray
    room_index_lookup: dict[str, int]
    class_index_lookup: dict[str, int]
    combined_ply_path: str
    metadata_path: str

    @property
    def room_count(self) -> int:
        return len(self.rooms)

    @property
    def placed_object_count(self) -> int:
        return len(self.placed_objects)


@dataclass
class FactoryCloudResult:
    run_id: str
    output_dir: str
    rooms: list[RoomInfo]
    stations: list[dict[str, Any]]
    notes: list[str]
    factory_points: np.ndarray
    factory_labels: np.ndarray
    factory_room_indices: np.ndarray
    room_index_lookup: dict[str, int]
    factory_cloud_path: str
    metadata_path: str


@dataclass
class PlacementResult:
    run_id: str
    output_dir: str
    dataset_classes: list[DatasetClassInfo]
    placed_objects: list[PlacedObject]
    detection_by_room: dict[str, list[dict[str, Any]]]
    combined_points: np.ndarray
    combined_source: np.ndarray
    combined_factory_label: np.ndarray
    combined_object_type: np.ndarray
    combined_object_instance: np.ndarray
    combined_part_label: np.ndarray
    combined_room_index: np.ndarray
    class_index_lookup: dict[str, int]
    combined_ply_path: str
    metadata_path: str

def _ensure_synthetic_factory_available() -> None:
    if _SYNTHETIC_IMPORT_ERROR is not None or FactoryGenerator is None:
        raise RuntimeError(
            "synthetic_factory modules are unavailable. "
            "Install dependencies for synthetic_factory first."
        ) from _SYNTHETIC_IMPORT_ERROR


def _numeric_sort_key(name: str) -> tuple[int, str]:
    digits = "".join(char for char in name if char.isdigit())
    if digits:
        try:
            return (int(digits), name.lower())
        except ValueError:
            return (10**9, name.lower())
    return (10**9, name.lower())


def _align_labels(points: np.ndarray, labels: np.ndarray | None) -> np.ndarray:
    if labels is None or len(labels) == 0:
        return np.zeros(len(points), dtype=np.int32)
    if len(labels) == len(points):
        return labels.astype(np.int32, copy=False)
    aligned = np.zeros(len(points), dtype=np.int32)
    size = min(len(points), len(labels))
    aligned[:size] = labels[:size].astype(np.int32, copy=False)
    return aligned


def _downsample_points(
    points: np.ndarray,
    labels: np.ndarray,
    max_points: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, labels
    indices = list(range(len(points)))
    rng.shuffle(indices)
    selected = np.array(indices[:max_points], dtype=np.int64)
    return points[selected], labels[selected]


def _normalize_object(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    centered = points - points.mean(axis=0, keepdims=True)
    span = points.max(axis=0) - points.min(axis=0)
    scale = float(max(np.max(span), 1e-6))
    return centered / scale


def _write_ply(
    path: str | Path,
    points: np.ndarray,
    labels: np.ndarray | None = None,
    label_name: str = "label",
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    label_data = None
    if labels is not None:
        label_data = labels.astype(np.int32, copy=False)
        if len(label_data) != len(points):
            raise ValueError("Points and labels lengths must match when writing PLY.")

    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        if label_data is not None:
            handle.write(f"property int {label_name}\n")
        handle.write("end_header\n")
        if label_data is None:
            for x, y, z in points:
                handle.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
        else:
            for index, (x, y, z) in enumerate(points):
                handle.write(
                    f"{float(x):.6f} {float(y):.6f} {float(z):.6f} {int(label_data[index])}\n"
                )
    return str(target)

def discover_dataset_classes(
    dataset_root: str | Path,
    max_classes: int = 5,
) -> list[DatasetClassInfo]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    candidate_dirs = [path for path in root.iterdir() if path.is_dir()]
    candidate_dirs.sort(key=lambda item: _numeric_sort_key(item.name))
    selected_dirs = candidate_dirs[: max(1, int(max_classes))]

    classes: list[DatasetClassInfo] = []
    for index, class_dir in enumerate(selected_dirs):
        files = sorted(str(path) for path in class_dir.glob("*.ply") if path.is_file())
        if not files:
            continue
        classes.append(
            DatasetClassInfo(
                class_name=class_dir.name,
                class_index=index,
                directory=str(class_dir),
                files=tuple(files),
            )
        )
    if not classes:
        raise RuntimeError(f"No .ply files found in first {max_classes} class folders.")
    return classes


def _build_factory_scene(
    seed: int,
    factory_width: float,
    factory_depth: float,
    room_count: int,
    room_height: float,
) -> tuple[Any, list[Any], Any]:
    _ensure_synthetic_factory_available()

    params = FactoryParams(
        factory_width=float(factory_width),
        factory_depth=float(factory_depth),
        number_of_rooms=int(room_count),
        room_size_range=(6.5, 22.0),
        corridor_width=4.0,
        room_height=float(room_height),
        layout_strategy="perlin",
        noise={
            "seed": int(seed),
            "octaves": 4,
            "frequency": 1.0,
            "persistence": 0.5,
            "lacunarity": 2.0,
            "sampling_scale": 0.08,
            "full_occupancy": True,
            "slot_multiplier": 2.0,
            "threshold": 0.35,
            "connectivity_bias": 0.8,
            "min_distance": 10.0,
            "aspect_strength": 0.35,
            "room_margin_ratio": 0.92,
        },
        biomes={
            "enabled": True,
            "ensure_all_types": True,
            "clustered_assignment": True,
            "height_grouping": True,
            "workshop_area_ratio": 0.72,
        },
        columns={"enabled": True, "spacing": 6.0, "radius": 0.3, "height": room_height + 1.8},
        beams={"enabled": True, "spacing": 6.0, "elevation": room_height + 1.2},
        machinery={
            "enabled": True,
            "density": 0.05,
            "conveyors_per_room": 3,
            "machines_per_room": 5,
            "auxiliary": {"enabled": True, "seed": int(seed) + 17},
        },
        exterior={"enabled": True},
        site={"enabled": True},
        seed=int(seed),
    )

    generator = FactoryGenerator(params)
    layout = generator.generate_layout()
    scene = generator.instantiate_rooms()
    scene = generator.place_corridors()
    scene = generator.place_exterior()
    scene = generator.place_site()
    scene = generator.place_factory_flow()
    scene = generator.place_columns()
    scene = generator.place_beams()
    scene = generator.place_machinery()
    scene = generator.place_infrastructure()
    scene = generator.place_auxiliary()
    return generator, layout, scene


def _layout_to_rooms(layout: Iterable[Any], room_height: float) -> list[RoomInfo]:
    rooms: list[RoomInfo] = []
    for room in layout:
        rooms.append(
            RoomInfo(
                room_id=str(room.room_id),
                biome=str(room.biome),
                center_x=float(room.center_x),
                center_y=float(room.center_y),
                width=float(room.width),
                depth=float(room.depth),
                floor_z=0.0,
                ceiling_z=float(room_height),
            )
        )
    return rooms


def _default_lidar_settings(seed: int, density_factor: float = 1.0) -> dict[str, Any]:
    density = max(0.2, float(density_factor))
    points_per_station = max(4000, int(round(38000 * density)))
    point_multiplier = max(1, int(round(2.0 * density)))
    exterior_density = min(1.0, max(0.05, 0.3 * density))
    return {
        "scan_range": 26.0,
        "angular_resolution_deg": 2.0,
        "vertical_resolution_deg": 12.0,
        "vertical_fov_up_deg": 35.0,
        "vertical_fov_down_deg": 55.0,
        "sensor_height": 1.6,
        "min_range": 0.35,
        "blind_spot_radius": 0.55,
        "station_spacing": 4.8,
        "station_margin": 0.95,
        "obstacle_clearance": 0.35,
        "ensure_blind_spot_coverage": True,
        "include_structural": False,
        "global_coverage": False,
        "include_factory_room": False,
        "point_multiplier": point_multiplier,
        "point_jitter": 0.0065,
        "exterior_point_density_factor": exterior_density,
        "points_per_station": points_per_station,
        "max_stations_per_room": 5,
        "max_stations": 42,
        "coverage_grid_step": 2.8,
        "coverage_wall_step": 4.0,
        "coverage_max_targets": 1600,
        "optimize_raycasts": True,
        "raycast_cell_size": 2.1,
        "raycast_march_step": 1.2,
        "seed": int(seed),
    }


def _find_room_for_xy(x: float, y: float, rooms: list[RoomInfo]) -> str:
    for room in rooms:
        if room.min_x <= x <= room.max_x and room.min_y <= y <= room.max_y:
            return room.room_id
    nearest = min(
        rooms,
        key=lambda room: (room.center_x - x) * (room.center_x - x)
        + (room.center_y - y) * (room.center_y - y),
    )
    return nearest.room_id


def _try_lasersensing_station_plan(
    rooms: list[RoomInfo],
    scan_range: float,
) -> tuple[list[tuple[float, float]] | None, str]:
    if str(LASER_SENSING_ROOT) not in sys.path:
        sys.path.insert(0, str(LASER_SENSING_ROOT))

    try:
        from model.geometry_manager import GeometryManager  # type: ignore
    except Exception as error:
        return None, f"laserSensing planner unavailable: {error}"

    if not rooms:
        return None, "laserSensing planner skipped: no room definitions."

    min_x = min(room.min_x for room in rooms)
    max_x = max(room.max_x for room in rooms)
    min_y = min(room.min_y for room in rooms)
    max_y = max(room.max_y for room in rooms)

    manager = GeometryManager()
    manager.region = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]
    if not manager.finish_region():
        return None, "laserSensing planner skipped: region initialization failed."

    centers, _ = manager.optimize_coverage(R_meters=float(scan_range))
    if not centers:
        return None, "laserSensing planner produced no stations."
    output = [(float(x), float(y)) for x, y in centers]
    return output, "laserSensing planner was applied."


def _flatten_lidar_points(
    circles: Iterable[Any],
    room_index_lookup: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: list[list[float]] = []
    labels: list[int] = []
    room_indices: list[int] = []
    for circle in circles:
        room_id = str(circle.room_id)
        room_index = int(room_index_lookup.get(room_id, -1))
        for point in circle.points:
            points.append([float(point.x), float(point.y), float(point.z)])
            labels.append(int(point.label))
            room_indices.append(room_index)
    if not points:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(room_indices, dtype=np.int32),
    )


def _pick_non_overlapping_position(
    room: RoomInfo,
    radius: float,
    occupancy: list[tuple[float, float, float]],
    rng: random.Random,
) -> tuple[float, float]:
    min_x = room.min_x + radius
    max_x = room.max_x - radius
    min_y = room.min_y + radius
    max_y = room.max_y - radius
    if min_x >= max_x or min_y >= max_y:
        return room.center_x, room.center_y

    for _ in range(80):
        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)
        conflict = False
        for ox, oy, oradius in occupancy:
            distance = math.hypot(x - ox, y - oy)
            if distance < (oradius + radius):
                conflict = True
                break
        if not conflict:
            return x, y
    return room.center_x, room.center_y


def _place_object_items(
    object_items: list[dict[str, Any]],
    rooms: list[RoomInfo],
    max_points_per_object: int,
    rng: random.Random,
    *,
    global_scale: float = 1.0,
) -> tuple[list[PlacedObject], dict[str, int], dict[str, list[str]]]:
    if not rooms:
        return [], {}, {}

    placed: list[PlacedObject] = []
    class_index_lookup: dict[str, int] = {}
    class_files: dict[str, list[str]] = {}
    occupancy_by_room: dict[str, list[tuple[float, float, float]]] = {
        room.room_id: [] for room in rooms
    }

    room_cycle = list(rooms)
    rng.shuffle(room_cycle)
    room_cycle_index = 0
    instance_index = 1
    scale_factor = max(0.2, float(global_scale))

    for item in object_items:
        normalized_file_path = sanitize_filesystem_path(str(item.get("file_path", "")))
        file_path = Path(normalized_file_path).expanduser()
        if not file_path.exists() or not file_path.is_file():
            continue

        class_name_raw = str(item.get("class_name", "")).strip()
        class_name = class_name_raw or file_path.parent.name or file_path.stem
        if class_name not in class_index_lookup:
            class_index_lookup[class_name] = len(class_index_lookup)
        class_files.setdefault(class_name, []).append(str(file_path))
        class_index = class_index_lookup[class_name]

        points, raw_labels = load_ply_file(str(file_path))
        if len(points) < 8:
            continue

        labels = _align_labels(points, raw_labels)
        points, labels = _downsample_points(points, labels, max_points_per_object, rng)
        normalized = _normalize_object(points)

        room = room_cycle[room_cycle_index % len(room_cycle)]
        room_cycle_index += 1
        min_room_size = min(room.width, room.depth)
        target_span = min_room_size * rng.uniform(0.14, 0.22) * scale_factor
        scale = float(max(target_span, 0.8))

        angle = rng.uniform(0.0, 2.0 * math.pi)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        rotation = np.array(
            [
                [cos_angle, -sin_angle, 0.0],
                [sin_angle, cos_angle, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        transformed = (normalized @ rotation.T) * scale
        transformed[:, 2] -= transformed[:, 2].min()

        span_xy = transformed[:, :2].max(axis=0) - transformed[:, :2].min(axis=0)
        radius = float(max(span_xy) / 2.0 + 0.6)
        occupancy = occupancy_by_room[room.room_id]
        tx, ty = _pick_non_overlapping_position(room, radius, occupancy, rng)
        tz = room.floor_z + rng.uniform(0.06, 0.22)
        transformed[:, 0] += tx
        transformed[:, 1] += ty
        transformed[:, 2] += tz
        occupancy.append((tx, ty, radius))

        placed.append(
            PlacedObject(
                instance_id=f"obj_{instance_index:03d}",
                object_class=class_name,
                class_index=class_index,
                source_file=str(file_path),
                room_id=room.room_id,
                points=transformed.astype(np.float32),
                part_labels=labels.astype(np.int32),
                position=(float(tx), float(ty), float(tz)),
                scale=scale,
            )
        )
        instance_index += 1

    return placed, class_index_lookup, class_files


def _place_dataset_objects(
    classes: list[DatasetClassInfo],
    rooms: list[RoomInfo],
    objects_per_class: int,
    max_points_per_object: int,
    rng: random.Random,
) -> list[PlacedObject]:
    object_items: list[dict[str, Any]] = []
    for class_info in classes:
        file_pool = list(class_info.files)
        if not file_pool:
            continue
        for _ in range(max(1, int(objects_per_class))):
            object_items.append(
                {
                    "file_path": str(Path(rng.choice(file_pool))),
                    "class_name": class_info.class_name,
                }
            )
    placed, _, _ = _place_object_items(
        object_items=object_items,
        rooms=rooms,
        max_points_per_object=max_points_per_object,
        rng=rng,
        global_scale=1.0,
    )
    return placed

def _build_detection_by_room(placed_objects: list[PlacedObject]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for obj in placed_objects:
        output.setdefault(obj.room_id, []).append(
            {
                "instance_id": obj.instance_id,
                "object_class": obj.object_class,
                "source_file": obj.source_file,
                "point_count": int(len(obj.points)),
                "confidence": 1.0,
            }
        )
    for room_id in output:
        output[room_id] = sorted(output[room_id], key=lambda item: item["instance_id"])
    return output


def _compose_combined_cloud(
    factory_points: np.ndarray,
    factory_labels: np.ndarray,
    factory_room_indices: np.ndarray,
    placed_objects: list[PlacedObject],
    room_index_lookup: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    combined_points = [factory_points]
    combined_source = [np.zeros(len(factory_points), dtype=np.int32)]
    combined_factory_label = [factory_labels.astype(np.int32, copy=False)]
    combined_object_type = [np.full(len(factory_points), -1, dtype=np.int32)]
    combined_object_instance = [np.full(len(factory_points), -1, dtype=np.int32)]
    combined_part_label = [np.full(len(factory_points), -1, dtype=np.int32)]
    combined_room_index = [factory_room_indices.astype(np.int32, copy=False)]

    for obj_index, obj in enumerate(placed_objects):
        point_count = len(obj.points)
        combined_points.append(obj.points.astype(np.float32, copy=False))
        combined_source.append(np.ones(point_count, dtype=np.int32))
        combined_factory_label.append(np.full(point_count, -1, dtype=np.int32))
        combined_object_type.append(np.full(point_count, int(obj.class_index), dtype=np.int32))
        combined_object_instance.append(np.full(point_count, obj_index, dtype=np.int32))
        combined_part_label.append(obj.part_labels.astype(np.int32, copy=False))
        room_index = int(room_index_lookup.get(obj.room_id, -1))
        combined_room_index.append(np.full(point_count, room_index, dtype=np.int32))

    return (
        np.concatenate(combined_points, axis=0),
        np.concatenate(combined_source, axis=0),
        np.concatenate(combined_factory_label, axis=0),
        np.concatenate(combined_object_type, axis=0),
        np.concatenate(combined_object_instance, axis=0),
        np.concatenate(combined_part_label, axis=0),
        np.concatenate(combined_room_index, axis=0),
    )


def load_point_cloud_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    normalized_path = sanitize_filesystem_path(path)
    points, labels = load_ply_file(normalized_path)
    if len(points) == 0:
        raise ValueError(f"Point cloud is empty: {normalized_path}")
    aligned = _align_labels(points, labels)
    return points.astype(np.float32, copy=False), aligned.astype(np.int32, copy=False)


def build_single_room_from_cloud(points: np.ndarray) -> list[RoomInfo]:
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    center = (min_corner + max_corner) / 2.0
    width = float(max(1.0, max_corner[0] - min_corner[0]))
    depth = float(max(1.0, max_corner[1] - min_corner[1]))
    room = RoomInfo(
        room_id="factory_room_loaded",
        biome="loaded_factory",
        center_x=float(center[0]),
        center_y=float(center[1]),
        width=width,
        depth=depth,
        floor_z=float(min_corner[2]),
        ceiling_z=float(max_corner[2]),
    )
    return [room]


def generate_factory_cloud(
    *,
    output_root: str | Path | None = None,
    seed: int = 42,
    factory_width: float = 120.0,
    factory_depth: float = 90.0,
    room_count: int = 9,
    room_height: float = 6.0,
    lidar_density: float = 1.0,
    use_lasersensing: bool = False,
) -> FactoryCloudResult:
    _ensure_synthetic_factory_available()

    _, layout, scene = _build_factory_scene(
        seed=int(seed),
        factory_width=factory_width,
        factory_depth=factory_depth,
        room_count=room_count,
        room_height=room_height,
    )
    rooms = _layout_to_rooms(layout, room_height=room_height)
    room_index_lookup = {room.room_id: index for index, room in enumerate(rooms)}

    lidar_settings = _default_lidar_settings(seed=seed, density_factor=lidar_density)
    lidar_generator = LidarSurveyGenerator(lidar_settings)
    notes: list[str] = []
    stations = None

    if use_lasersensing:
        candidate_centers, planner_message = _try_lasersensing_station_plan(
            rooms=rooms,
            scan_range=float(lidar_settings["scan_range"]),
        )
        notes.append(planner_message)
        if candidate_centers:
            stations = []
            for index, (x, y) in enumerate(candidate_centers, start=1):
                room_id = _find_room_for_xy(x, y, rooms)
                stations.append(
                    LidarStation(
                        id=f"lidar_station_{index}",
                        room_id=room_id,
                        x=float(x),
                        y=float(y),
                        z=float(room_height * 0.28),
                    )
                )

    if not stations:
        stations = lidar_generator.plan_stations(scene)

    circles = lidar_generator.generate_scans(scene, stations)
    factory_points, factory_labels, factory_room_indices = _flatten_lidar_points(
        circles=circles,
        room_index_lookup=room_index_lookup,
    )
    if len(factory_points) == 0:
        raise RuntimeError("Factory LiDAR stage produced an empty point cloud.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(output_root)
        if output_root is not None
        else (Path(__file__).resolve().parent / "factory_demo_runs" / run_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    factory_cloud_path = _write_factory_cloud_ply(
        output_dir / "factory_cloud.ply",
        factory_points,
        factory_labels,
        factory_room_indices,
    )
    metadata = {
        "run_id": run_id,
        "seed": int(seed),
        "lidar_density": float(lidar_density),
        "rooms": [
            {
                "room_id": room.room_id,
                "biome": room.biome,
                "center_x": room.center_x,
                "center_y": room.center_y,
                "width": room.width,
                "depth": room.depth,
                "floor_z": room.floor_z,
                "ceiling_z": room.ceiling_z,
            }
            for room in rooms
        ],
        "room_index_lookup": room_index_lookup,
        "stations": [
            {"id": station.id, "room_id": station.room_id, "x": station.x, "y": station.y, "z": station.z}
            for station in stations
        ],
        "factory_cloud_path": factory_cloud_path,
        "notes": notes,
    }
    metadata_path = output_dir / "factory_cloud_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return FactoryCloudResult(
        run_id=run_id,
        output_dir=str(output_dir),
        rooms=rooms,
        stations=metadata["stations"],
        notes=notes,
        factory_points=factory_points,
        factory_labels=factory_labels,
        factory_room_indices=factory_room_indices,
        room_index_lookup=room_index_lookup,
        factory_cloud_path=factory_cloud_path,
        metadata_path=str(metadata_path),
    )


def place_objects_in_factory_cloud(
    *,
    factory_points: np.ndarray,
    factory_labels: np.ndarray | None,
    factory_room_indices: np.ndarray | None,
    rooms: list[RoomInfo] | None,
    room_index_lookup: dict[str, int] | None,
    dataset_root: str | Path,
    output_root: str | Path | None = None,
    max_classes: int = 5,
    objects_per_class: int = 1,
    max_points_per_object: int = 16000,
    seed: int = 42,
) -> PlacementResult:
    rng = random.Random(int(seed))
    classes = discover_dataset_classes(dataset_root, max_classes=max_classes)

    normalized_points = factory_points.astype(np.float32, copy=False)
    normalized_labels = (
        factory_labels.astype(np.int32, copy=False)
        if factory_labels is not None and len(factory_labels) == len(factory_points)
        else np.zeros(len(factory_points), dtype=np.int32)
    )
    if factory_room_indices is not None and len(factory_room_indices) == len(factory_points):
        room_indices = factory_room_indices.astype(np.int32, copy=False)
    else:
        room_indices = np.zeros(len(factory_points), dtype=np.int32)

    if rooms is None or not rooms:
        rooms = build_single_room_from_cloud(normalized_points)
    if room_index_lookup is None or not room_index_lookup:
        room_index_lookup = {room.room_id: index for index, room in enumerate(rooms)}

    placed_objects = _place_dataset_objects(
        classes=classes,
        rooms=rooms,
        objects_per_class=objects_per_class,
        max_points_per_object=max_points_per_object,
        rng=rng,
    )
    if not placed_objects:
        raise RuntimeError("No objects were placed from the selected dataset folder.")

    detection_by_room = _build_detection_by_room(placed_objects)
    (
        combined_points,
        combined_source,
        combined_factory_label,
        combined_object_type,
        combined_object_instance,
        combined_part_label,
        combined_room_index,
    ) = _compose_combined_cloud(
        factory_points=normalized_points,
        factory_labels=normalized_labels,
        factory_room_indices=room_indices,
        placed_objects=placed_objects,
        room_index_lookup=room_index_lookup,
    )

    class_index_lookup = {item.class_name: int(item.class_index) for item in classes}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(output_root)
        if output_root is not None
        else (Path(__file__).resolve().parent / "factory_demo_runs" / run_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_ply_path = _write_combined_ply(
        path=output_dir / "factory_objects_combined.ply",
        points=combined_points,
        source=combined_source,
        factory_label=combined_factory_label,
        object_type=combined_object_type,
        object_instance=combined_object_instance,
        part_label=combined_part_label,
        room_index=combined_room_index,
    )

    placed_dir = output_dir / "placed_objects"
    placed_dir.mkdir(parents=True, exist_ok=True)
    for obj in placed_objects:
        _write_ply(
            placed_dir / f"{obj.instance_id}_{obj.object_class}.ply",
            obj.points,
            obj.part_labels,
            "part_label",
        )

    metadata = {
        "run_id": run_id,
        "dataset_root": str(Path(dataset_root)),
        "class_index_lookup": class_index_lookup,
        "rooms": [room.room_id for room in rooms],
        "detection_by_room": detection_by_room,
        "combined_ply_path": combined_ply_path,
    }
    metadata_path = output_dir / "placement_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return PlacementResult(
        run_id=run_id,
        output_dir=str(output_dir),
        dataset_classes=classes,
        placed_objects=placed_objects,
        detection_by_room=detection_by_room,
        combined_points=combined_points,
        combined_source=combined_source,
        combined_factory_label=combined_factory_label,
        combined_object_type=combined_object_type,
        combined_object_instance=combined_object_instance,
        combined_part_label=combined_part_label,
        combined_room_index=combined_room_index,
        class_index_lookup=class_index_lookup,
        combined_ply_path=combined_ply_path,
        metadata_path=str(metadata_path),
    )


def place_object_files_in_factory_cloud(
    *,
    factory_points: np.ndarray,
    factory_labels: np.ndarray | None,
    factory_room_indices: np.ndarray | None,
    rooms: list[RoomInfo] | None,
    room_index_lookup: dict[str, int] | None,
    object_items: list[dict[str, Any]],
    output_root: str | Path | None = None,
    max_points_per_object: int = 16000,
    seed: int = 42,
    global_scale: float = 1.0,
) -> PlacementResult:
    rng = random.Random(int(seed))
    normalized_points = factory_points.astype(np.float32, copy=False)
    normalized_labels = (
        factory_labels.astype(np.int32, copy=False)
        if factory_labels is not None and len(factory_labels) == len(factory_points)
        else np.zeros(len(factory_points), dtype=np.int32)
    )
    if factory_room_indices is not None and len(factory_room_indices) == len(factory_points):
        room_indices = factory_room_indices.astype(np.int32, copy=False)
    else:
        room_indices = np.zeros(len(factory_points), dtype=np.int32)

    if rooms is None or not rooms:
        rooms = build_single_room_from_cloud(normalized_points)
    if room_index_lookup is None or not room_index_lookup:
        room_index_lookup = {room.room_id: index for index, room in enumerate(rooms)}

    placed_objects, class_index_lookup, class_files = _place_object_items(
        object_items=object_items,
        rooms=rooms,
        max_points_per_object=max_points_per_object,
        rng=rng,
        global_scale=global_scale,
    )
    if not placed_objects:
        raise RuntimeError("No valid objects were provided for placement.")

    dataset_classes: list[DatasetClassInfo] = []
    for class_name, class_index in sorted(class_index_lookup.items(), key=lambda item: item[1]):
        files = tuple(sorted(class_files.get(class_name, [])))
        directory = str(Path(files[0]).parent) if files else ""
        dataset_classes.append(
            DatasetClassInfo(
                class_name=class_name,
                class_index=int(class_index),
                directory=directory,
                files=files,
            )
        )

    detection_by_room = _build_detection_by_room(placed_objects)
    (
        combined_points,
        combined_source,
        combined_factory_label,
        combined_object_type,
        combined_object_instance,
        combined_part_label,
        combined_room_index,
    ) = _compose_combined_cloud(
        factory_points=normalized_points,
        factory_labels=normalized_labels,
        factory_room_indices=room_indices,
        placed_objects=placed_objects,
        room_index_lookup=room_index_lookup,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(output_root)
        if output_root is not None
        else (Path(__file__).resolve().parent / "factory_demo_runs" / run_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_ply_path = _write_combined_ply(
        path=output_dir / "factory_objects_combined.ply",
        points=combined_points,
        source=combined_source,
        factory_label=combined_factory_label,
        object_type=combined_object_type,
        object_instance=combined_object_instance,
        part_label=combined_part_label,
        room_index=combined_room_index,
    )

    placed_dir = output_dir / "placed_objects"
    placed_dir.mkdir(parents=True, exist_ok=True)
    for obj in placed_objects:
        _write_ply(placed_dir / f"{obj.instance_id}_{obj.object_class}.ply", obj.points, obj.part_labels, "part_label")

    metadata = {
        "run_id": run_id,
        "class_index_lookup": class_index_lookup,
        "rooms": [room.room_id for room in rooms],
        "detection_by_room": detection_by_room,
        "combined_ply_path": combined_ply_path,
        "object_items": object_items,
        "global_scale": float(global_scale),
    }
    metadata_path = output_dir / "placement_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return PlacementResult(
        run_id=run_id,
        output_dir=str(output_dir),
        dataset_classes=dataset_classes,
        placed_objects=placed_objects,
        detection_by_room=detection_by_room,
        combined_points=combined_points,
        combined_source=combined_source,
        combined_factory_label=combined_factory_label,
        combined_object_type=combined_object_type,
        combined_object_instance=combined_object_instance,
        combined_part_label=combined_part_label,
        combined_room_index=combined_room_index,
        class_index_lookup=class_index_lookup,
        combined_ply_path=combined_ply_path,
        metadata_path=str(metadata_path),
    )


def _write_combined_ply(
    path: str | Path,
    points: np.ndarray,
    source: np.ndarray,
    factory_label: np.ndarray,
    object_type: np.ndarray,
    object_instance: np.ndarray,
    part_label: np.ndarray,
    room_index: np.ndarray,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (
        len(points)
        == len(source)
        == len(factory_label)
        == len(object_type)
        == len(object_instance)
        == len(part_label)
        == len(room_index)
    ):
        raise ValueError("Combined cloud columns have mismatched lengths.")

    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("comment source 0=factory 1=dataset_object\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property int source\n")
        handle.write("property int factory_label\n")
        handle.write("property int object_type\n")
        handle.write("property int object_instance\n")
        handle.write("property int part_label\n")
        handle.write("property int room_index\n")
        handle.write("end_header\n")
        for index, (x, y, z) in enumerate(points):
            handle.write(
                (
                    f"{float(x):.6f} {float(y):.6f} {float(z):.6f} "
                    f"{int(source[index])} {int(factory_label[index])} "
                    f"{int(object_type[index])} {int(object_instance[index])} "
                    f"{int(part_label[index])} {int(room_index[index])}\n"
                )
            )
    return str(target)


def _write_factory_cloud_ply(
    path: str | Path,
    points: np.ndarray,
    labels: np.ndarray,
    room_indices: np.ndarray,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (len(points) == len(labels) == len(room_indices)):
        raise ValueError("Factory cloud arrays must have the same length.")

    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property int label\n")
        handle.write("property int room_index\n")
        handle.write("end_header\n")
        for index, (x, y, z) in enumerate(points):
            handle.write(
                (
                    f"{float(x):.6f} {float(y):.6f} {float(z):.6f} "
                    f"{int(labels[index])} {int(room_indices[index])}\n"
                )
            )
    return str(target)


def save_factory_cloud_as(
    *,
    path: str | Path,
    points: np.ndarray,
    labels: np.ndarray,
    room_indices: np.ndarray,
) -> str:
    return _write_factory_cloud_ply(
        path=path,
        points=points,
        labels=labels.astype(np.int32, copy=False),
        room_indices=room_indices.astype(np.int32, copy=False),
    )


def save_combined_cloud_as(
    *,
    path: str | Path,
    points: np.ndarray,
    source: np.ndarray,
    factory_label: np.ndarray,
    object_type: np.ndarray,
    object_instance: np.ndarray,
    part_label: np.ndarray,
    room_index: np.ndarray,
) -> str:
    return _write_combined_ply(
        path=path,
        points=points,
        source=source.astype(np.int32, copy=False),
        factory_label=factory_label.astype(np.int32, copy=False),
        object_type=object_type.astype(np.int32, copy=False),
        object_instance=object_instance.astype(np.int32, copy=False),
        part_label=part_label.astype(np.int32, copy=False),
        room_index=room_index.astype(np.int32, copy=False),
    )


def run_factory_demo(
    dataset_root: str | Path,
    output_root: str | Path | None = None,
    *,
    max_classes: int = 5,
    objects_per_class: int = 1,
    max_points_per_object: int = 16000,
    seed: int = 42,
    factory_width: float = 120.0,
    factory_depth: float = 90.0,
    room_count: int = 9,
    room_height: float = 6.0,
    lidar_density: float = 1.0,
    use_lasersensing: bool = False,
) -> FactoryDemoResult:
    _ensure_synthetic_factory_available()
    rng = random.Random(int(seed))

    classes = discover_dataset_classes(dataset_root, max_classes=max_classes)
    _, layout, scene = _build_factory_scene(
        seed=int(seed),
        factory_width=factory_width,
        factory_depth=factory_depth,
        room_count=room_count,
        room_height=room_height,
    )
    rooms = _layout_to_rooms(layout, room_height=room_height)
    room_index_lookup = {room.room_id: index for index, room in enumerate(rooms)}
    class_index_lookup = {item.class_name: int(item.class_index) for item in classes}

    lidar_settings = _default_lidar_settings(seed=seed, density_factor=lidar_density)
    lidar_generator = LidarSurveyGenerator(lidar_settings)
    stations = None
    notes: list[str] = []

    if use_lasersensing:
        candidate_centers, planner_message = _try_lasersensing_station_plan(
            rooms=rooms,
            scan_range=float(lidar_settings["scan_range"]),
        )
        notes.append(planner_message)
        if candidate_centers:
            stations = []
            for index, (x, y) in enumerate(candidate_centers, start=1):
                room_id = _find_room_for_xy(x, y, rooms)
                stations.append(
                    LidarStation(
                        id=f"lidar_station_{index}",
                        room_id=room_id,
                        x=float(x),
                        y=float(y),
                        z=float(room_height * 0.28),
                    )
                )

    if not stations:
        stations = lidar_generator.plan_stations(scene)

    circles = lidar_generator.generate_scans(scene, stations)
    factory_points, factory_labels, factory_room_indices = _flatten_lidar_points(
        circles=circles,
        room_index_lookup=room_index_lookup,
    )
    if len(factory_points) == 0:
        raise RuntimeError("Factory LiDAR stage produced an empty point cloud.")

    placed_objects = _place_dataset_objects(
        classes=classes,
        rooms=rooms,
        objects_per_class=objects_per_class,
        max_points_per_object=max_points_per_object,
        rng=rng,
    )
    if not placed_objects:
        raise RuntimeError("No dataset objects were placed into factory rooms.")

    detection_by_room = _build_detection_by_room(placed_objects)
    (
        combined_points,
        combined_source,
        combined_factory_label,
        combined_object_type,
        combined_object_instance,
        combined_part_label,
        combined_room_index,
    ) = _compose_combined_cloud(
        factory_points=factory_points,
        factory_labels=factory_labels,
        factory_room_indices=factory_room_indices,
        placed_objects=placed_objects,
        room_index_lookup=room_index_lookup,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(output_root)
        if output_root is not None
        else (Path(__file__).resolve().parent / "factory_demo_runs" / run_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_ply_path = _write_combined_ply(
        path=output_dir / "factory_combined_scene.ply",
        points=combined_points,
        source=combined_source,
        factory_label=combined_factory_label,
        object_type=combined_object_type,
        object_instance=combined_object_instance,
        part_label=combined_part_label,
        room_index=combined_room_index,
    )

    placed_dir = output_dir / "placed_objects"
    placed_dir.mkdir(parents=True, exist_ok=True)
    for obj in placed_objects:
        _write_ply(
            path=placed_dir / f"{obj.instance_id}_{obj.object_class}.ply",
            points=obj.points,
            labels=obj.part_labels,
            label_name="part_label",
        )

    metadata_payload = {
        "run_id": run_id,
        "seed": int(seed),
        "lidar_density": float(lidar_density),
        "dataset_root": str(Path(dataset_root)),
        "combined_ply_path": combined_ply_path,
        "class_index_lookup": class_index_lookup,
        "room_index_lookup": room_index_lookup,
        "dataset_classes": [
            {
                "class_name": info.class_name,
                "class_index": info.class_index,
                "directory": info.directory,
                "file_count": len(info.files),
            }
            for info in classes
        ],
        "rooms": [
            {
                "room_id": room.room_id,
                "biome": room.biome,
                "center_x": room.center_x,
                "center_y": room.center_y,
                "width": room.width,
                "depth": room.depth,
                "floor_z": room.floor_z,
                "ceiling_z": room.ceiling_z,
            }
            for room in rooms
        ],
        "stations": [
            {
                "id": station.id,
                "room_id": station.room_id,
                "x": station.x,
                "y": station.y,
                "z": station.z,
            }
            for station in stations
        ],
        "placed_objects": [
            {
                "instance_id": obj.instance_id,
                "object_class": obj.object_class,
                "class_index": obj.class_index,
                "source_file": obj.source_file,
                "room_id": obj.room_id,
                "point_count": int(len(obj.points)),
                "position": list(obj.position),
                "scale": obj.scale,
            }
            for obj in placed_objects
        ],
        "detection_by_room": detection_by_room,
        "notes": notes,
    }
    metadata_path = output_dir / "pipeline_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return FactoryDemoResult(
        run_id=run_id,
        output_dir=str(output_dir),
        dataset_classes=classes,
        rooms=rooms,
        stations=[
            {
                "id": station.id,
                "room_id": station.room_id,
                "x": station.x,
                "y": station.y,
                "z": station.z,
            }
            for station in stations
        ],
        placed_objects=placed_objects,
        detection_by_room=detection_by_room,
        notes=notes,
        factory_points=factory_points,
        factory_labels=factory_labels,
        factory_room_indices=factory_room_indices,
        combined_points=combined_points,
        combined_source=combined_source,
        combined_factory_label=combined_factory_label,
        combined_object_type=combined_object_type,
        combined_object_instance=combined_object_instance,
        combined_part_label=combined_part_label,
        combined_room_index=combined_room_index,
        room_index_lookup=room_index_lookup,
        class_index_lookup=class_index_lookup,
        combined_ply_path=combined_ply_path,
        metadata_path=str(metadata_path),
    )


def get_object_by_instance(
    result: FactoryDemoResult,
    instance_id: str,
) -> PlacedObject:
    for obj in result.placed_objects:
        if obj.instance_id == instance_id:
            return obj
    raise KeyError(f"Object with instance id '{instance_id}' not found.")


def build_virtual_object(
    points: np.ndarray,
    *,
    instance_id: str,
    object_class: str = "detected_object",
    class_index: int = -1,
    room_id: str = "room_unknown",
    source_file: str = "",
) -> PlacedObject:
    if len(points) == 0:
        raise ValueError("Cannot build virtual object from empty points.")
    normalized_points = points.astype(np.float32, copy=False)
    part_labels = np.zeros(len(points), dtype=np.int32)
    center = normalized_points.mean(axis=0)
    return PlacedObject(
        instance_id=instance_id,
        object_class=object_class,
        class_index=int(class_index),
        source_file=source_file,
        room_id=room_id,
        points=normalized_points,
        part_labels=part_labels,
        position=(float(center[0]), float(center[1]), float(center[2])),
        scale=1.0,
    )


def recognize_cloud_with_model(
    *,
    points: np.ndarray,
    checkpoint_path: str,
    output_dir: str | Path | None = None,
    num_points: int = 4096,
) -> dict[str, Any]:
    from engineering_classifier import predict_file

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
        temp_input = Path(tmp_file.name)
    try:
        _write_ply(temp_input, points, np.zeros(len(points), dtype=np.int32), "scalar_Label")
        prediction = predict_file(
            checkpoint_path=checkpoint_path,
            input_file=str(temp_input),
            num_points=min(int(num_points), max(256, len(points))),
        )
    finally:
        temp_input.unlink(missing_ok=True)

    sampled_points = prediction["points"].astype(np.float32, copy=False)
    predicted_labels = prediction["predicted_labels"].astype(np.int32, copy=False)
    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    histogram = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    saved_path = None
    if output_dir is not None:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_path = _write_ply(
            target_dir / "recognized_labels.ply",
            sampled_points,
            predicted_labels,
            "pred_label",
        )

    return {
        "points": sampled_points,
        "predicted_labels": predicted_labels,
        "histogram": histogram,
        "mean_confidence": float(prediction["mean_confidence"]),
        "saved_path": saved_path,
    }


def _emit_status(status_cb: Any, message: str) -> None:
    if status_cb is None:
        return
    try:
        status_cb(str(message))
    except Exception:
        pass


def _scan_dataset_label_space(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(sanitize_filesystem_path(dataset_root))
    if not root.exists():
        raise FileNotFoundError(f"Папка датасета не найдена: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Ожидалась папка датасета, но получен файл: {root}")

    files = sorted(path for path in root.rglob("*.ply") if path.is_file())
    if not files:
        raise RuntimeError(f"В папке {root} не найдено .ply файлов для дообучения.")

    min_label: int | None = None
    max_label: int | None = None
    unique_labels: set[int] = set()
    labeled_files = 0

    for file_path in files:
        _, labels = load_ply_file(str(file_path))
        if labels is None or len(labels) == 0:
            continue
        labels_int = labels.astype(np.int64, copy=False)
        file_min = int(labels_int.min())
        file_max = int(labels_int.max())
        min_label = file_min if min_label is None else min(min_label, file_min)
        max_label = file_max if max_label is None else max(max_label, file_max)
        for label in np.unique(labels_int):
            unique_labels.add(int(label))
        labeled_files += 1

    if labeled_files == 0 or min_label is None or max_label is None:
        raise RuntimeError(
            "В выбранной папке нет корректных меток в .ply файлах. "
            "Для дообучения требуются point cloud файлы с целочисленными метками частей."
        )

    return {
        "file_count": int(len(files)),
        "labeled_files": int(labeled_files),
        "min_label": int(min_label),
        "max_label": int(max_label),
        "unique_count": int(len(unique_labels)),
        "required_num_classes": int(max_label + 1),
    }


def _scan_folder_class_space(dataset_root: str | Path) -> dict[str, Any]:
    root = Path(sanitize_filesystem_path(dataset_root))
    if not root.exists():
        raise FileNotFoundError(f"Папка датасета не найдена: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Ожидалась папка датасета, но получен файл: {root}")

    class_names: list[str] = []
    class_file_counts: dict[str, int] = {}

    for candidate in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
        ply_files = [path for path in candidate.rglob("*.ply") if path.is_file()]
        if not ply_files:
            continue
        class_name = candidate.name
        class_names.append(class_name)
        class_file_counts[class_name] = int(len(ply_files))

    # Fallback: allow a single class dataset in the root directory itself.
    if not class_names:
        root_files = [path for path in root.rglob("*.ply") if path.is_file()]
        if not root_files:
            raise RuntimeError(f"В папке {root} не найдено .ply файлов для дообучения.")
        class_name = root.name or "object"
        class_names = [class_name]
        class_file_counts[class_name] = int(len(root_files))

    total_files = int(sum(class_file_counts.values()))
    return {
        "class_names": class_names,
        "class_file_counts": class_file_counts,
        "class_count": int(len(class_names)),
        "file_count": total_files,
        "required_num_classes": int(len(class_names)),
    }


def finetune_object_model(
    *,
    dataset_root: str | Path,
    object_name: str,
    weights_root: str | Path = r"D:\vesa",
    num_classes: int = 13,
    num_points: int = 4096,
    batch_size: int = 4,
    max_epochs: int = 30,
    learning_rate: float = 1e-3,
    seed: int = 42,
    label_mode: str = "per_point",
    status_cb: Any = None,
) -> dict[str, Any]:
    from engineering_classifier import TrainingSettings, train_model

    requested_num_classes = int(num_classes)
    mode_raw = str(label_mode).strip().lower()
    mode = "folder_name" if mode_raw in {"folder_name", "folder_as_class"} else "per_point"

    if mode == "folder_name":
        label_space = _scan_folder_class_space(dataset_root)
        required_num_classes = int(label_space["required_num_classes"])
        preview_names = ", ".join(label_space["class_names"][:8])
        if len(label_space["class_names"]) > 8:
            preview_names += ", ..."
        _emit_status(
            status_cb,
            (
                "Проверка датасета распознавания завершена: "
                f"классов(папок)={label_space['class_count']}, "
                f"файлов={label_space['file_count']}. "
                f"Классы: {preview_names}"
            ),
        )
    else:
        label_space = _scan_dataset_label_space(dataset_root)
        min_label = int(label_space["min_label"])
        required_num_classes = int(label_space["required_num_classes"])
        _emit_status(
            status_cb,
            (
                "Проверка меток завершена: "
                f"файлов={label_space['file_count']}, "
                f"с метками={label_space['labeled_files']}, "
                f"диапазон={label_space['min_label']}..{label_space['max_label']}."
            ),
        )
        if min_label < 0:
            raise ValueError(
                "Обнаружены отрицательные метки классов. "
                "Дообучение ожидает метки в диапазоне [0, num_classes-1]."
            )

    effective_num_classes = requested_num_classes

    if requested_num_classes < required_num_classes:
        effective_num_classes = required_num_classes
        _emit_status(
            status_cb,
            (
                "Автокоррекция num_classes: "
                f"запрошено {requested_num_classes}, требуется минимум {required_num_classes}. "
                f"Используем {effective_num_classes}."
            ),
        )
    else:
        _emit_status(
            status_cb,
            (
                "Проверка num_classes пройдена: "
                f"запрошено {requested_num_classes}, требуется минимум {required_num_classes}."
            ),
        )

    normalized_name = "".join(
        ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in str(object_name)
    ).strip("_")
    if not normalized_name:
        normalized_name = "object_model"

    model_root = Path(weights_root) / normalized_name
    checkpoint_dir = model_root / "checkpoints"
    log_dir = model_root / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    settings = TrainingSettings(
        dataset_root=str(dataset_root),
        num_classes=int(effective_num_classes),
        label_mode=mode,
        num_points=int(num_points),
        val_size=0.1,
        test_size=0.1,
        batch_size=int(batch_size),
        num_workers=0,
        max_epochs=int(max_epochs),
        learning_rate=float(learning_rate),
        weight_decay=1e-4,
        scheduler_step_size=max(1, int(max_epochs // 3)),
        scheduler_gamma=0.7,
        early_stopping_patience=max(5, int(max_epochs // 4)),
        class_weights=None,
        seed=int(seed),
        checkpoint_dir=str(checkpoint_dir),
        log_dir=str(log_dir),
    )

    result = train_model(settings, status_cb=status_cb)
    result["weights_root"] = str(model_root)
    result["object_name"] = normalized_name
    result["label_space"] = label_space
    result["requested_num_classes"] = int(requested_num_classes)
    result["effective_num_classes"] = int(effective_num_classes)
    result["num_classes_auto_adjusted"] = bool(effective_num_classes != requested_num_classes)
    result["label_mode"] = mode
    return result


def finetune_recognition_model(
    *,
    dataset_root: str | Path,
    model_name: str,
    weights_root: str | Path = r"D:\vesa\recognition",
    num_classes: int = 5,
    num_points: int = 4096,
    batch_size: int = 4,
    max_epochs: int = 30,
    learning_rate: float = 1e-3,
    seed: int = 42,
    status_cb: Any = None,
) -> dict[str, Any]:
    return finetune_object_model(
        dataset_root=dataset_root,
        object_name=model_name,
        weights_root=weights_root,
        num_classes=num_classes,
        num_points=num_points,
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        seed=seed,
        label_mode="folder_name",
        status_cb=status_cb,
    )

def _auto_partition_labels(points: np.ndarray, part_count: int = 4) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=np.int32)
    if len(points) < part_count * 3:
        return np.zeros((len(points),), dtype=np.int32)

    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    principal = centered @ vh[0]
    secondary = centered @ vh[1]
    tertiary = centered @ vh[2]
    radial = np.sqrt(secondary * secondary + tertiary * tertiary)

    q_low, q_high = np.quantile(principal, [0.33, 0.66])
    radial_q = float(np.quantile(radial, 0.62))
    labels = np.zeros((len(points),), dtype=np.int32)
    labels[principal > q_low] = 1
    labels[principal > q_high] = 2
    labels[radial > radial_q] += 1
    labels = labels % max(2, int(part_count))
    return labels


def classify_object_parts(
    obj: PlacedObject,
    checkpoint_path: str | None = None,
    num_points: int = 4096,
) -> PartClassificationResult:
    notes: list[str] = []
    if checkpoint_path:
        try:
            from engineering_classifier import predict_file

            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
            try:
                _write_ply(
                    tmp_path,
                    obj.points,
                    np.zeros(len(obj.points), dtype=np.int32),
                    "scalar_Label",
                )
                prediction = predict_file(
                    checkpoint_path=checkpoint_path,
                    input_file=str(tmp_path),
                    num_points=min(int(num_points), max(256, len(obj.points))),
                )
            finally:
                tmp_path.unlink(missing_ok=True)
            predicted_points = prediction["points"].astype(np.float32, copy=False)
            predicted_labels = prediction["predicted_labels"].astype(np.int32, copy=False)
            return PartClassificationResult(
                points=predicted_points,
                labels=predicted_labels,
                source="lesson5_model",
                notes=notes,
            )
        except Exception as model_error:
            notes.append(f"Model inference unavailable, fallback to demo labels: {model_error}")

    labels = obj.part_labels.astype(np.int32, copy=False)
    if len(np.unique(labels)) <= 1:
        labels = _auto_partition_labels(obj.points, part_count=4)
        source = "auto_geometry_partition"
    else:
        source = "dataset_part_labels"
    return PartClassificationResult(
        points=obj.points.astype(np.float32, copy=False),
        labels=labels,
        source=source,
        notes=notes,
    )


def _surface_type(points: np.ndarray) -> str:
    if len(points) < 20:
        return "small"
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 1e-10)
    l1, l2, l3 = eigvals
    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    scattering = l3 / l1
    if planarity > 0.55 and scattering < 0.12:
        return "flat"
    if linearity > 0.62 and planarity < 0.35:
        return "cylindrical"
    if scattering > 0.28:
        return "solid"
    return "complex"


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(axis, helper)) > 0.92:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(axis, helper)
    u = u / (np.linalg.norm(u) + 1e-10)
    v = np.cross(axis, u)
    v = v / (np.linalg.norm(v) + 1e-10)
    return u, v


def _reconstruct_flat(points: np.ndarray, resolution: int = 28) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_u = vh[0]
    axis_v = vh[1]
    center = points.mean(axis=0)
    u_proj = centered @ axis_u
    v_proj = centered @ axis_v
    u_min, u_max = float(np.min(u_proj)), float(np.max(u_proj))
    v_min, v_max = float(np.min(v_proj)), float(np.max(v_proj))
    uu = np.linspace(u_min, u_max, num=resolution)
    vv = np.linspace(v_min, v_max, num=resolution)
    grid_u, grid_v = np.meshgrid(uu, vv)
    reconstructed = (
        center.reshape(1, 3)
        + grid_u.reshape(-1, 1) * axis_u.reshape(1, 3)
        + grid_v.reshape(-1, 1) * axis_v.reshape(1, 3)
    )
    return reconstructed.astype(np.float32)


def _reconstruct_cylindrical(points: np.ndarray, axial_steps: int = 32, angle_steps: int = 36) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    center = points.mean(axis=0)
    t = centered @ axis
    radial_vectors = centered - np.outer(t, axis)
    radii = np.linalg.norm(radial_vectors, axis=1)
    radius = float(np.median(radii))
    if radius <= 1e-6:
        return points.astype(np.float32, copy=True)

    u, v = _orthonormal_basis(axis)
    t_values = np.linspace(float(np.min(t)), float(np.max(t)), num=axial_steps)
    angles = np.linspace(0.0, 2.0 * math.pi, num=angle_steps, endpoint=False)
    output: list[np.ndarray] = []
    for t_value in t_values:
        ring_center = center + t_value * axis
        for angle in angles:
            point = ring_center + radius * math.cos(angle) * u + radius * math.sin(angle) * v
            output.append(point.astype(np.float32))
    return np.asarray(output, dtype=np.float32)


def _reconstruct_solid(points: np.ndarray, resolution: int = 18) -> np.ndarray:
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    xs = np.linspace(float(min_corner[0]), float(max_corner[0]), num=resolution)
    ys = np.linspace(float(min_corner[1]), float(max_corner[1]), num=resolution)
    zs = np.linspace(float(min_corner[2]), float(max_corner[2]), num=resolution)

    surface_points: list[list[float]] = []
    for x in xs:
        for y in ys:
            surface_points.append([x, y, float(min_corner[2])])
            surface_points.append([x, y, float(max_corner[2])])
    for x in xs:
        for z in zs:
            surface_points.append([x, float(min_corner[1]), z])
            surface_points.append([x, float(max_corner[1]), z])
    for y in ys:
        for z in zs:
            surface_points.append([float(min_corner[0]), y, z])
            surface_points.append([float(max_corner[0]), y, z])
    return np.asarray(surface_points, dtype=np.float32)


def _reconstruct_complex(points: np.ndarray, target_points: int = 4500) -> np.ndarray:
    if len(points) <= target_points:
        return points.astype(np.float32, copy=True)
    indices = np.linspace(0, len(points) - 1, num=target_points, dtype=np.int64)
    return points[indices].astype(np.float32, copy=False)


def _run_surface_reconstructor_module(
    classification: PartClassificationResult,
    output_dir: Path,
    checkpoint_path: str | None,
    min_points: int,
) -> ReconstructionResult | None:
    module_path = Path(__file__).resolve().parent / "SurfaceReconstructor.py"
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("surface_reconstructor_mod", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "SmartReconstructor"):
        return None

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    _write_ply(
        path=temp_path,
        points=classification.points,
        labels=classification.labels,
        label_name="scalar_Label",
    )

    try:
        reconstructor = module.SmartReconstructor(
            str(temp_path),
            ckpt_path=checkpoint_path,
            output_dir=str(output_dir),
            min_points=int(min_points),
        )
        generated_files: list[str] = []
        part_summaries: list[dict[str, Any]] = []
        reconstructed_points: list[np.ndarray] = []

        for class_id, class_points in reconstructor.classes.items():
            result = reconstructor.reconstruct_class(class_id, class_points)
            if not result:
                continue
            mesh = result["mesh"]
            mesh_path = output_dir / f"class_{int(class_id):02d}_{result['type']}_mesh.ply"
            module.o3d.io.write_triangle_mesh(str(mesh_path), mesh)
            generated_files.append(str(mesh_path))
            part_summaries.append(
                {
                    "part_label": int(class_id),
                    "surface_type": str(result["type"]),
                    "triangle_count": int(result.get("triangles", 0)),
                    "point_count": int(len(class_points)),
                }
            )
            vertices = np.asarray(mesh.vertices)
            if len(vertices) > 0:
                reconstructed_points.append(vertices.astype(np.float32))

        if not part_summaries:
            return None

        if reconstructed_points:
            combined_points = np.concatenate(reconstructed_points, axis=0)
        else:
            combined_points = np.zeros((0, 3), dtype=np.float32)

        return ReconstructionResult(
            object_id="surface_module",
            backend="surface_reconstructor",
            combined_points=combined_points,
            generated_files=generated_files,
            part_summaries=part_summaries,
            notes=[],
        )
    finally:
        temp_path.unlink(missing_ok=True)


def reconstruct_object_surfaces(
    object_id: str,
    classification: PartClassificationResult,
    output_dir: str | Path,
    *,
    min_points_per_part: int = 45,
    prefer_surface_module: bool = False,
    checkpoint_path: str | None = None,
) -> ReconstructionResult:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    if prefer_surface_module:
        try:
            module_result = _run_surface_reconstructor_module(
                classification=classification,
                output_dir=target_dir,
                checkpoint_path=checkpoint_path,
                min_points=min_points_per_part,
            )
            if module_result is not None:
                return ReconstructionResult(
                    object_id=object_id,
                    backend=module_result.backend,
                    combined_points=module_result.combined_points,
                    generated_files=module_result.generated_files,
                    part_summaries=module_result.part_summaries,
                    notes=module_result.notes,
                )
            notes.append("SurfaceReconstructor module fallback was used.")
        except Exception as module_error:
            notes.append(f"SurfaceReconstructor module failed: {module_error}")

    unique_labels = sorted(int(label) for label in np.unique(classification.labels))
    generated_files: list[str] = []
    part_summaries: list[dict[str, Any]] = []
    all_reconstructed_points: list[np.ndarray] = []

    for label in unique_labels:
        mask = classification.labels == label
        part_points = classification.points[mask]
        if len(part_points) < int(min_points_per_part):
            continue

        surface_type = _surface_type(part_points)
        if surface_type == "flat":
            reconstructed = _reconstruct_flat(part_points)
        elif surface_type == "cylindrical":
            reconstructed = _reconstruct_cylindrical(part_points)
        elif surface_type == "solid":
            reconstructed = _reconstruct_solid(part_points)
        else:
            reconstructed = _reconstruct_complex(part_points)

        part_file = target_dir / f"{object_id}_part_{label:02d}_{surface_type}.ply"
        _write_ply(part_file, reconstructed, None, label_name="label")
        generated_files.append(str(part_file))
        all_reconstructed_points.append(reconstructed.astype(np.float32, copy=False))
        part_summaries.append(
            {
                "part_label": int(label),
                "surface_type": surface_type,
                "input_points": int(len(part_points)),
                "reconstructed_points": int(len(reconstructed)),
            }
        )

    if all_reconstructed_points:
        combined_points = np.concatenate(all_reconstructed_points, axis=0)
        combined_path = target_dir / f"{object_id}_reconstruction_combined.ply"
        _write_ply(combined_path, combined_points, None, label_name="label")
        generated_files.append(str(combined_path))
    else:
        combined_points = np.zeros((0, 3), dtype=np.float32)
        notes.append("No part had enough points for reconstruction.")

    return ReconstructionResult(
        object_id=object_id,
        backend="lightweight_surface_reconstructor",
        combined_points=combined_points,
        generated_files=generated_files,
        part_summaries=part_summaries,
        notes=notes,
    )
