from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..scene.scene_graph import Scene, SceneObject, Transform

try:
    import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    torch = None

Matrix4 = list[list[float]]
GLOBAL_FACTORY_ROOM_ID = "__factory__"
UNBOUNDED_STATION_LIMIT = 1_000_000_000


def _to_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_positive_float(value: object, default: float, label: str) -> float:
    if value is None:
        return default
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{label} must be > 0.")
    return number


def _to_non_negative_float(value: object, default: float, label: str) -> float:
    if value is None:
        return default
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{label} must be >= 0.")
    return number


def _identity_matrix() -> Matrix4:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat_mul(a: Matrix4, b: Matrix4) -> Matrix4:
    result: Matrix4 = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    for row in range(4):
        for col in range(4):
            result[row][col] = (
                a[row][0] * b[0][col]
                + a[row][1] * b[1][col]
                + a[row][2] * b[2][col]
                + a[row][3] * b[3][col]
            )
    return result


def _translation_matrix(x: float, y: float, z: float) -> Matrix4:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _scale_matrix(x: float, y: float, z: float) -> Matrix4:
    return [
        [x, 0.0, 0.0, 0.0],
        [0.0, y, 0.0, 0.0],
        [0.0, 0.0, z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _rotation_matrix_euler_deg(x: float, y: float, z: float) -> Matrix4:
    rx = math.radians(x)
    ry = math.radians(y)
    rz = math.radians(z)

    cx = math.cos(rx)
    sx = math.sin(rx)
    cy = math.cos(ry)
    sy = math.sin(ry)
    cz = math.cos(rz)
    sz = math.sin(rz)

    rot_x = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cx, -sx, 0.0],
        [0.0, sx, cx, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rot_y = [
        [cy, 0.0, sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0, cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rot_z = [
        [cz, -sz, 0.0, 0.0],
        [sz, cz, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return _mat_mul(rot_z, _mat_mul(rot_y, rot_x))


def _local_matrix(transform: Transform) -> Matrix4:
    tx, ty, tz = transform.position
    rx, ry, rz = transform.rotation
    sx, sy, sz = transform.scale

    translation = _translation_matrix(tx, ty, tz)
    rotation = _rotation_matrix_euler_deg(rx, ry, rz)
    scale = _scale_matrix(sx, sy, sz)
    return _mat_mul(translation, _mat_mul(rotation, scale))


def _transform_vertex(matrix: Matrix4, vertex: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = vertex
    tx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3]
    ty = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3]
    tz = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3]
    tw = matrix[3][0] * x + matrix[3][1] * y + matrix[3][2] * z + matrix[3][3]
    if abs(tw) < 1e-12:
        return (tx, ty, tz)
    return (tx / tw, ty / tw, tz / tw)


def _frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        return []
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(current)
        current += step
    return values


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _normalize_type(value: str) -> str:
    return value.strip().lower()


def _sanitize_filename_token(value: str) -> str:
    cleaned = "".join(char if (char.isalnum() or char in {"_", "-"}) else "_" for char in value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def _label_to_rgb(label: int) -> tuple[int, int, int]:
    palette: tuple[tuple[int, int, int], ...] = (
        (160, 160, 160),
        (220, 76, 70),
        (70, 150, 235),
        (170, 170, 170),
        (122, 88, 58),
        (209, 209, 209),
        (250, 184, 20),
        (40, 181, 120),
        (150, 88, 220),
        (244, 120, 52),
        (242, 212, 84),
        (96, 112, 126),
        (54, 184, 191),
    )
    if label < 0:
        return (160, 160, 160)
    return palette[label % len(palette)]


@dataclass(frozen=True)
class LidarStation:
    id: str
    room_id: str
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "room_id": self.room_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass(frozen=True)
class LidarPoint:
    x: float
    y: float
    z: float
    label: int
    class_name: str
    object_id: str
    instance_id: int
    station_id: str
    angle_deg: float
    elevation_deg: float
    distance: float
    domain: str = "unknown"

    def to_dict(self) -> dict[str, object]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "label": self.label,
            "class_name": self.class_name,
            "object_id": self.object_id,
            "instance_id": self.instance_id,
            "station_id": self.station_id,
            "angle_deg": self.angle_deg,
            "elevation_deg": self.elevation_deg,
            "distance": self.distance,
            "domain": self.domain,
        }


@dataclass
class LidarCircleScan:
    station_id: str
    room_id: str
    center: tuple[float, float, float]
    radius: float
    points: list[LidarPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "station_id": self.station_id,
            "room_id": self.room_id,
            "center": list(self.center),
            "radius": self.radius,
            "point_count": len(self.points),
            "points": [point.to_dict() for point in self.points],
        }


@dataclass
class LidarSurveyResult:
    stations: list[LidarStation]
    circles: list[LidarCircleScan]
    label_map: dict[str, int]
    source_obj_path: str | None = None

    def station_coordinates(self) -> list[tuple[float, float, float]]:
        return [station.as_tuple() for station in self.stations]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_obj_path": self.source_obj_path,
            "station_count": len(self.stations),
            "circle_count": len(self.circles),
            "labels": dict(sorted(self.label_map.items(), key=lambda item: item[1])),
            "stations": [station.to_dict() for station in self.stations],
            "circles": [circle.to_dict() for circle in self.circles],
        }

    def write_json(self, path: str) -> str:
        target = self._next_available_path(Path(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        sorted_labels = dict(sorted(self.label_map.items(), key=lambda item: item[1]))
        with target.open("w", encoding="utf-8") as handle:
            handle.write("{\n")
            handle.write(
                f'  "source_obj_path": {json.dumps(self.source_obj_path, ensure_ascii=False)},\n'
            )
            handle.write(f'  "station_count": {len(self.stations)},\n')
            handle.write(f'  "circle_count": {len(self.circles)},\n')
            handle.write(
                f'  "labels": {json.dumps(sorted_labels, ensure_ascii=False)},\n'
            )
            handle.write('  "stations": [\n')
            for index, station in enumerate(self.stations):
                payload = json.dumps(station.to_dict(), ensure_ascii=False)
                suffix = "," if index < len(self.stations) - 1 else ""
                handle.write(f"    {payload}{suffix}\n")
            handle.write("  ],\n")
            handle.write('  "circles": [\n')
            for index, circle in enumerate(self.circles):
                self._write_circle_json(handle, circle, index == len(self.circles) - 1)
            handle.write("  ]\n")
            handle.write("}\n")
        return str(target)

    def _write_circle_json(
        self,
        handle: object,
        circle: LidarCircleScan,
        is_last: bool,
    ) -> None:
        handle.write("    {\n")
        handle.write(
            f'      "station_id": {json.dumps(circle.station_id, ensure_ascii=False)},\n'
        )
        handle.write(
            f'      "room_id": {json.dumps(circle.room_id, ensure_ascii=False)},\n'
        )
        handle.write(
            f'      "center": {json.dumps(list(circle.center), ensure_ascii=False)},\n'
        )
        handle.write(f'      "radius": {circle.radius},\n')
        handle.write(f'      "point_count": {len(circle.points)},\n')
        handle.write('      "points": [\n')
        for point_index, point in enumerate(circle.points):
            payload = json.dumps(point.to_dict(), ensure_ascii=False)
            suffix = "," if point_index < len(circle.points) - 1 else ""
            handle.write(f"        {payload}{suffix}\n")
        handle.write("      ]\n")
        suffix = "" if is_last else ","
        handle.write(f"    }}{suffix}\n")

    def write_circle_ply_files(
        self,
        output_dir: str,
        file_prefix: str = "scan_circle",
    ) -> list[str]:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        station_lookup = self._station_index_lookup()

        written_paths: list[str] = []
        safe_prefix = _sanitize_filename_token(file_prefix)
        for index, circle in enumerate(self.circles, start=1):
            room_token = _sanitize_filename_token(circle.room_id)
            station_token = _sanitize_filename_token(circle.station_id)
            filename = f"{safe_prefix}_{index:04d}_{room_token}_{station_token}.ply"
            target = self._next_available_path(target_dir / filename)
            self._write_points_to_ply(
                path=target,
                points=circle.points,
                station_lookup=station_lookup,
                circle_index=index,
            )
            written_paths.append(str(target))
        return written_paths

    def write_stitched_ply(
        self,
        path: str,
        blind_spot_radius: float | None = None,
        fill_blind_spots: bool = True,
    ) -> str:
        all_points: list[LidarPoint] = []
        for circle in self.circles:
            all_points.extend(circle.points)
        if fill_blind_spots and blind_spot_radius is not None and blind_spot_radius > 0.0:
            all_points = self._fill_stitched_blind_spots(all_points, blind_spot_radius)
        target = self._next_available_path(Path(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        self._write_points_to_ply(
            path=target,
            points=all_points,
            station_lookup=self._station_index_lookup(),
            circle_index=-1,
        )
        return str(target)

    def write_split_stitched_ply(
        self,
        output_root_dir: str,
        combined_path: str | None = None,
        blind_spot_radius: float | None = None,
        fill_blind_spots: bool = True,
    ) -> dict[str, str]:
        root = Path(output_root_dir)
        root.mkdir(parents=True, exist_ok=True)

        all_points: list[LidarPoint] = []
        for circle in self.circles:
            all_points.extend(circle.points)
        if fill_blind_spots and blind_spot_radius is not None and blind_spot_radius > 0.0:
            all_points = self._fill_stitched_blind_spots(all_points, blind_spot_radius)

        interior_points = [point for point in all_points if point.domain == "interior"]
        exterior_points = [point for point in all_points if point.domain == "exterior"]

        if combined_path:
            combined_target = Path(combined_path)
        else:
            combined_target = root / "combined" / "factory_stitched.ply"
        interior_target = root / "interior" / "factory_stitched_interior.ply"
        exterior_target = root / "exterior" / "factory_stitched_exterior.ply"
        combined_target = self._next_available_path(combined_target)
        interior_target = self._next_available_path(interior_target)
        exterior_target = self._next_available_path(exterior_target)

        combined_target.parent.mkdir(parents=True, exist_ok=True)
        interior_target.parent.mkdir(parents=True, exist_ok=True)
        exterior_target.parent.mkdir(parents=True, exist_ok=True)

        station_lookup = self._station_index_lookup()
        self._write_points_to_ply(
            path=combined_target,
            points=all_points,
            station_lookup=station_lookup,
            circle_index=-1,
        )
        self._write_points_to_ply(
            path=interior_target,
            points=interior_points,
            station_lookup=station_lookup,
            circle_index=-2,
        )
        self._write_points_to_ply(
            path=exterior_target,
            points=exterior_points,
            station_lookup=station_lookup,
            circle_index=-3,
        )
        return {
            "combined": str(combined_target),
            "interior": str(interior_target),
            "exterior": str(exterior_target),
        }

    def write_grouped_stitched_ply(
        self,
        output_dir: str,
        grouped_points: Mapping[str, Sequence[LidarPoint]],
        file_prefix: str = "group",
        circle_index_base: int = -1000,
    ) -> dict[str, str]:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        station_lookup = self._station_index_lookup()
        written: dict[str, str] = {}
        safe_prefix = _sanitize_filename_token(file_prefix)
        for group_name, points in grouped_points.items():
            if not points:
                continue
            token = _sanitize_filename_token(group_name)
            target = self._next_available_path(target_dir / f"{safe_prefix}_{token}.ply")
            self._write_points_to_ply(
                path=target,
                points=list(points),
                station_lookup=station_lookup,
                circle_index=circle_index_base,
            )
            written[group_name] = str(target)
            circle_index_base -= 1
        return written

    @staticmethod
    def _next_available_path(path: Path) -> Path:
        if not path.exists():
            return path

        suffix = "".join(path.suffixes)
        stem = path.name[:-len(suffix)] if suffix else path.name
        index = 1
        while True:
            candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
            if not candidate.exists():
                return candidate
            index += 1

    def _fill_stitched_blind_spots(
        self,
        points: list[LidarPoint],
        blind_spot_radius: float,
    ) -> list[LidarPoint]:
        if not points:
            return points

        station_by_id = {station.id: station for station in self.stations}
        output = list(points)
        for station in self.stations:
            room_donors = [
                point
                for point in points
                if point.station_id != station.id
                and station_by_id.get(point.station_id) is not None
                and station_by_id[point.station_id].room_id == station.room_id
            ]
            if not room_donors:
                continue

            has_cover = any(
                point.z < station.z + 1e-6
                and math.hypot(point.x - station.x, point.y - station.y) < blind_spot_radius * 0.8
                for point in room_donors
            )
            if has_cover:
                continue

            floor_estimate = min(point.z for point in room_donors)
            fill_z = min(
                floor_estimate,
                station.z - max(0.2, blind_spot_radius * 0.35),
            )

            fill_offsets = [
                (0.0, 0.0),
                (blind_spot_radius * 0.35, 0.0),
                (-blind_spot_radius * 0.35, 0.0),
                (0.0, blind_spot_radius * 0.35),
                (0.0, -blind_spot_radius * 0.35),
            ]
            for offset_x, offset_y in fill_offsets:
                target_x = station.x + offset_x
                target_y = station.y + offset_y
                source = min(
                    room_donors,
                    key=lambda point: math.hypot(point.x - target_x, point.y - target_y),
                )
                output.append(
                    LidarPoint(
                        x=target_x,
                        y=target_y,
                        z=fill_z,
                        label=source.label,
                        class_name=source.class_name,
                        object_id=source.object_id,
                        instance_id=source.instance_id,
                        station_id=source.station_id,
                        angle_deg=source.angle_deg,
                        elevation_deg=source.elevation_deg,
                        distance=math.hypot(target_x - station.x, target_y - station.y),
                        domain=source.domain,
                    )
                )
        return output

    def _station_index_lookup(self) -> dict[str, int]:
        lookup: dict[str, int] = {}
        for index, station in enumerate(self.stations, start=1):
            lookup[station.id] = index
        return lookup

    def _write_points_to_ply(
        self,
        path: Path,
        points: Iterable[LidarPoint],
        station_lookup: Mapping[str, int],
        circle_index: int,
    ) -> None:
        point_list = list(points)
        sorted_labels = sorted(self.label_map.items(), key=lambda item: item[1])

        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write("comment generated_by synthetic_factory\n")
            handle.write("comment semantic_label_property label\n")
            for class_name, label_id in sorted_labels:
                handle.write(f"comment label {label_id} {class_name}\n")
            handle.write(f"element vertex {len(point_list)}\n")
            handle.write("property float x\n")
            handle.write("property float y\n")
            handle.write("property float z\n")
            handle.write("property int label\n")
            handle.write("property int instance_id\n")
            handle.write("property uchar red\n")
            handle.write("property uchar green\n")
            handle.write("property uchar blue\n")
            handle.write("property int station_index\n")
            handle.write("property int circle_index\n")
            handle.write("property float elevation_deg\n")
            handle.write("end_header\n")

            for point in point_list:
                red, green, blue = _label_to_rgb(point.label)
                station_index = station_lookup.get(point.station_id, -1)
                handle.write(
                    (
                        f"{point.x:.6f} {point.y:.6f} {point.z:.6f} "
                        f"{point.label} {point.instance_id} {red} {green} {blue} "
                        f"{station_index} {circle_index} {point.elevation_deg:.6f}\n"
                    )
                )


@dataclass(frozen=True)
class _SemanticObject:
    object_id: str
    room_id: str
    object_type: str
    label_id: int
    class_name: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    center_x: float
    center_y: float
    center_z: float
    scan_domain: str


@dataclass(frozen=True)
class _RoomBounds:
    room_id: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    floor_z: float


@dataclass(frozen=True)
class _CoverageTarget:
    target_id: str
    room_id: str
    x: float
    y: float
    z: float
    object_id: str | None = None


@dataclass(frozen=True)
class _RaycastRoomCache:
    objects: tuple[_SemanticObject, ...]
    origin_x: float
    origin_y: float
    cell_size: float
    cells: dict[tuple[int, int], tuple[int, ...]]
    cuda_min_bounds: object | None = None
    cuda_max_bounds: object | None = None


class LidarSurveyGenerator:
    """
    Factory LiDAR survey module.

    Workflow:
    1) optimize scanner standing stations (minimum set with full target coverage),
    2) simulate circular scans for each station,
    3) return labeled point clouds with class IDs.
    """

    def __init__(self, settings: Mapping[str, object] | None = None) -> None:
        cfg = dict(settings or {})
        self.settings = cfg
        self.scan_range = _to_positive_float(cfg.get("scan_range"), 24.0, "lidar.scan_range")
        self.scan_pattern = str(cfg.get("scan_pattern", "circular")).strip().lower()
        if self.scan_pattern != "circular":
            raise ValueError(
                "Only circular lidar scanning is supported. Set lidar.scan_pattern='circular'."
            )
        self.angular_resolution_deg = _to_positive_float(
            cfg.get("angular_resolution_deg"),
            2.0,
            "lidar.angular_resolution_deg",
        )
        self.horizontal_fov_deg = _to_positive_float(
            cfg.get("horizontal_fov_deg"),
            360.0,
            "lidar.horizontal_fov_deg",
        )
        self.horizontal_fov_deg = _clamp(self.horizontal_fov_deg, 1.0, 360.0)
        self.vertical_resolution_deg = _to_positive_float(
            cfg.get("vertical_resolution_deg"),
            12.0,
            "lidar.vertical_resolution_deg",
        )
        self.vertical_fov_up_deg = _to_positive_float(
            cfg.get("vertical_fov_up_deg"),
            35.0,
            "lidar.vertical_fov_up_deg",
        )
        self.vertical_fov_down_deg = _to_positive_float(
            cfg.get("vertical_fov_down_deg"),
            55.0,
            "lidar.vertical_fov_down_deg",
        )
        self.sensor_height = _to_positive_float(cfg.get("sensor_height"), 1.6, "lidar.sensor_height")
        self.min_range = _to_non_negative_float(cfg.get("min_range"), 0.35, "lidar.min_range")
        self.blind_spot_radius = _to_non_negative_float(
            cfg.get("blind_spot_radius"),
            0.55,
            "lidar.blind_spot_radius",
        )
        self.station_spacing = _to_positive_float(cfg.get("station_spacing"), 4.0, "lidar.station_spacing")
        self.station_margin = _to_positive_float(cfg.get("station_margin"), 0.9, "lidar.station_margin")
        self.obstacle_clearance = _to_positive_float(
            cfg.get("obstacle_clearance"),
            0.35,
            "lidar.obstacle_clearance",
        )
        self.ensure_blind_spot_coverage = _to_bool(
            cfg.get("ensure_blind_spot_coverage", True),
            default=True,
        )
        # Physical mode by default: no synthetic far "no-hit" returns through opaque walls.
        self.emit_no_hit_returns = _to_bool(
            cfg.get("emit_no_hit_returns", False),
            default=False,
        )
        # Structural geometry (walls/floor/ceiling) must participate in ray casting by default.
        self.include_structural = _to_bool(cfg.get("include_structural", True), default=True)
        self.global_coverage = _to_bool(cfg.get("global_coverage", False), default=False)
        self.factory_room_id = str(cfg.get("factory_room_id", GLOBAL_FACTORY_ROOM_ID)).strip() or GLOBAL_FACTORY_ROOM_ID
        self.include_factory_room = _to_bool(cfg.get("include_factory_room", False), default=False)
        self.point_multiplier = max(1, int(cfg.get("point_multiplier", 1)))
        self.point_jitter = _to_non_negative_float(
            cfg.get("point_jitter"),
            0.0075,
            "lidar.point_jitter",
        )
        self.exterior_point_density_factor = _to_non_negative_float(
            cfg.get("exterior_point_density_factor"),
            0.25,
            "lidar.exterior_point_density_factor",
        )
        self.points_per_station = max(0, int(cfg.get("points_per_station", 0)))
        self.max_stations_per_room = self._parse_station_limit(
            cfg.get("max_stations_per_room", 64),
            "lidar.max_stations_per_room",
        )
        self.max_stations = self._parse_station_limit(
            cfg.get("max_stations", self.max_stations_per_room),
            "lidar.max_stations",
        )
        self.coverage_grid_step = _to_positive_float(
            cfg.get("coverage_grid_step"),
            max(1.2, self.station_spacing * 0.7),
            "lidar.coverage_grid_step",
        )
        self.coverage_wall_step = _to_positive_float(
            cfg.get("coverage_wall_step"),
            max(self.coverage_grid_step * 1.25, self.station_spacing),
            "lidar.coverage_wall_step",
        )
        self.coverage_max_targets = max(96, int(cfg.get("coverage_max_targets", 1400)))
        self.optimize_raycasts = _to_bool(cfg.get("optimize_raycasts", True), default=True)
        self.compute_backend = str(cfg.get("compute_backend", "auto")).strip().lower()
        if self.compute_backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("lidar.compute_backend must be one of: auto, cpu, cuda.")
        self.cuda_ray_batch_size = max(128, int(cfg.get("cuda_ray_batch_size", 1024)))
        self.cuda_object_batch_size = max(64, int(cfg.get("cuda_object_batch_size", 1024)))
        self.raycast_cell_size = _to_positive_float(
            cfg.get("raycast_cell_size"),
            max(0.9, min(3.2, self.station_spacing * 0.85)),
            "lidar.raycast_cell_size",
        )
        self.raycast_march_step = _to_positive_float(
            cfg.get("raycast_march_step"),
            max(0.55, self.raycast_cell_size * 0.65),
            "lidar.raycast_march_step",
        )
        self.azimuth_phase_jitter_deg = _to_non_negative_float(
            cfg.get("azimuth_phase_jitter_deg"),
            max(0.1, self.angular_resolution_deg * 0.5),
            "lidar.azimuth_phase_jitter_deg",
        )
        self._label_map = self._build_label_map(cfg.get("label_classes"))
        self._station_jitter_cache: dict[str, tuple[tuple[float, float, float], ...]] = {}
        self._instance_id_by_object: dict[str, int] = {}
        self._next_instance_id = 1
        self._torch_mod = torch
        self._torch_device = None
        self._cuda_runtime_failed = False
        self._cuda_enabled = False
        if self.compute_backend != "cpu" and self._torch_mod is not None and self._torch_mod.cuda.is_available():
            self._torch_device = self._torch_mod.device("cuda")
            self._cuda_enabled = True
        elif self.compute_backend == "cuda":
            raise ValueError(
                "lidar.compute_backend='cuda' requested, but PyTorch CUDA is unavailable."
            )

    def _parse_station_limit(self, value: object, field_name: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer.") from exc
        if parsed <= 0:
            return UNBOUNDED_STATION_LIMIT
        return parsed

    @property
    def label_map(self) -> dict[str, int]:
        return dict(self._label_map)

    def run(
        self,
        scene: Scene,
        obj_path: str | None = None,
        output_path: str | None = None,
    ) -> LidarSurveyResult:
        stations = self.plan_stations(scene)
        circles = self.generate_scans(scene, stations)
        result = LidarSurveyResult(
            stations=stations,
            circles=circles,
            label_map=self._label_map,
            source_obj_path=obj_path,
        )
        if output_path:
            result.write_json(output_path)
        return result

    def plan_stations(self, scene: Scene) -> list[LidarStation]:
        semantic_objects, room_bounds = self._collect_scene_semantics(scene)
        if not semantic_objects:
            return []

        objects_by_room: dict[str, list[_SemanticObject]] = {}
        for item in semantic_objects:
            objects_by_room.setdefault(item.room_id, []).append(item)

        planned: list[LidarStation] = []
        station_counter = 1

        if self.global_coverage:
            global_bounds = self._merge_room_bounds(room_bounds.values())
            if global_bounds is None:
                return []

            global_targets = self._build_coverage_targets(global_bounds, semantic_objects)
            if not global_targets:
                center_x = (global_bounds.min_x + global_bounds.max_x) / 2.0
                center_y = (global_bounds.min_y + global_bounds.max_y) / 2.0
                return [
                    LidarStation(
                        id="lidar_station_1",
                        room_id=self.factory_room_id,
                        x=center_x,
                        y=center_y,
                        z=global_bounds.floor_z + self.sensor_height,
                    )
                ]

            candidates = self._generate_candidate_stations(global_bounds, semantic_objects)
            if not candidates:
                center_x = (global_bounds.min_x + global_bounds.max_x) / 2.0
                center_y = (global_bounds.min_y + global_bounds.max_y) / 2.0
                candidates = [(center_x, center_y)]
            candidates = self._augment_candidates_from_targets(
                bounds=global_bounds,
                room_objects=semantic_objects,
                candidates=candidates,
                targets=global_targets,
            )
            candidates = self._ensure_minimum_station_candidates(
                bounds=global_bounds,
                room_objects=semantic_objects,
                candidates=candidates,
            )

            coverage = self._build_coverage_sets(
                candidates=candidates,
                targets=global_targets,
                blockers=semantic_objects,
                sensor_z=global_bounds.floor_z + self.sensor_height,
            )
            selected_indices = self._greedy_set_cover(coverage, len(global_targets))
            if not selected_indices:
                selected_indices = [0]
            selected_indices = self._complete_target_coverage(
                selected_indices=selected_indices,
                coverage=coverage,
                target_count=len(global_targets),
                station_limit=self.max_stations,
            )
            selected_indices = self._augment_for_blind_spot_coverage(
                selected_indices=selected_indices,
                candidates=candidates,
                station_limit=self.max_stations,
            )

            selected_points = [candidates[index] for index in selected_indices[: self.max_stations]]
            selected_points = self._augment_factory_room_station_points(
                bounds=global_bounds,
                room_objects=semantic_objects,
                selected_points=selected_points,
                station_limit=self.max_stations,
            )

            for sx, sy in selected_points[: self.max_stations]:
                planned.append(
                    LidarStation(
                        id=f"lidar_station_{station_counter}",
                        room_id=self.factory_room_id,
                        x=sx,
                        y=sy,
                        z=global_bounds.floor_z + self.sensor_height,
                    )
                )
                station_counter += 1
            return planned

        for room_id, bounds in room_bounds.items():
            if room_id == self.factory_room_id and not self.include_factory_room:
                continue
            room_objects = objects_by_room.get(room_id, [])
            coverage_targets = self._build_coverage_targets(bounds, room_objects)
            station_limit = (
                self.max_stations
                if room_id == self.factory_room_id
                else self.max_stations_per_room
            )
            if not coverage_targets:
                center_x = (bounds.min_x + bounds.max_x) / 2.0
                center_y = (bounds.min_y + bounds.max_y) / 2.0
                selected_points = [(center_x, center_y)]
                if room_id == self.factory_room_id:
                    selected_points = self._augment_factory_room_station_points(
                        bounds=bounds,
                        room_objects=room_objects,
                        selected_points=selected_points,
                        station_limit=station_limit,
                    )
                for sx, sy in selected_points[:station_limit]:
                    planned.append(
                        LidarStation(
                            id=f"lidar_station_{station_counter}",
                            room_id=room_id,
                            x=sx,
                            y=sy,
                            z=bounds.floor_z + self.sensor_height,
                        )
                    )
                    station_counter += 1
                continue

            candidates = self._generate_candidate_stations(bounds, room_objects)
            if not candidates:
                center_x = (bounds.min_x + bounds.max_x) / 2.0
                center_y = (bounds.min_y + bounds.max_y) / 2.0
                candidates = [(center_x, center_y)]
            candidates = self._augment_candidates_from_targets(
                bounds=bounds,
                room_objects=room_objects,
                candidates=candidates,
                targets=coverage_targets,
            )
            candidates = self._ensure_minimum_station_candidates(
                bounds=bounds,
                room_objects=room_objects,
                candidates=candidates,
            )

            coverage = self._build_coverage_sets(
                candidates=candidates,
                targets=coverage_targets,
                blockers=room_objects,
                sensor_z=bounds.floor_z + self.sensor_height,
            )
            selected_indices = self._greedy_set_cover(coverage, len(coverage_targets))
            if not selected_indices:
                selected_indices = [0]
            selected_indices = self._complete_target_coverage(
                selected_indices=selected_indices,
                coverage=coverage,
                target_count=len(coverage_targets),
                station_limit=station_limit,
            )
            selected_indices = self._augment_for_blind_spot_coverage(
                selected_indices=selected_indices,
                candidates=candidates,
                station_limit=station_limit,
            )

            selected_points = [candidates[index] for index in selected_indices[:station_limit]]
            if room_id == self.factory_room_id:
                selected_points = self._augment_factory_room_station_points(
                    bounds=bounds,
                    room_objects=room_objects,
                    selected_points=selected_points,
                    station_limit=station_limit,
                )

            for sx, sy in selected_points[:station_limit]:
                planned.append(
                    LidarStation(
                        id=f"lidar_station_{station_counter}",
                        room_id=room_id,
                        x=sx,
                        y=sy,
                        z=bounds.floor_z + self.sensor_height,
                    )
                )
                station_counter += 1

        return planned

    def generate_scans(self, scene: Scene, stations: Iterable[LidarStation]) -> list[LidarCircleScan]:
        semantic_objects, room_bounds = self._collect_scene_semantics(scene)
        room_floor_by_id = {room.room_id: room.floor_z for room in room_bounds.values()}
        objects_by_room: dict[str, list[_SemanticObject]] = {}
        for item in semantic_objects:
            objects_by_room.setdefault(item.room_id, []).append(item)
        station_list = list(stations)
        stations_by_room: dict[str, list[LidarStation]] = {}
        for station in station_list:
            stations_by_room.setdefault(station.room_id, []).append(station)
        global_bounds = self._merge_room_bounds(room_bounds.values())
        global_floor_z = (
            global_bounds.floor_z if global_bounds is not None else (self.sensor_height * -1.0)
        )

        circles: list[LidarCircleScan] = []
        azimuth_span_deg = _clamp(self.horizontal_fov_deg, 1.0, 360.0)
        azimuth_step_count = max(1, int(round(azimuth_span_deg / self.angular_resolution_deg)))
        angle_step = azimuth_span_deg / azimuth_step_count
        elevation_angles = self._build_elevation_angles()
        elevation_data = [
            (
                elevation_deg,
                math.cos(math.radians(elevation_deg)),
                math.sin(math.radians(elevation_deg)),
            )
            for elevation_deg in elevation_angles
        ]
        room_cache_by_id: dict[str, _RaycastRoomCache] = {
            room_id: self._build_raycast_cache(room_objects)
            for room_id, room_objects in objects_by_room.items()
        }
        global_cache = self._build_raycast_cache(semantic_objects)

        for station in station_list:
            azimuth_data = self._build_station_azimuth_data(
                station_id=station.id,
                azimuth_step_count=azimuth_step_count,
                angle_step=angle_step,
            )
            if self.global_coverage:
                room_objects = semantic_objects
                room_stations = station_list
                room_cache = global_cache
            else:
                room_objects = objects_by_room.get(station.room_id, [])
                room_stations = stations_by_room.get(station.room_id, [])
                room_cache = room_cache_by_id.get(station.room_id)
                if room_cache is None:
                    room_cache = self._build_raycast_cache(room_objects)
                    room_cache_by_id[station.room_id] = room_cache
            sensor_z = station.z
            floor_z = room_floor_by_id.get(station.room_id, global_floor_z)
            if sensor_z < floor_z:
                sensor_z = floor_z + self.sensor_height

            points = self._generate_station_points(
                station=station,
                sensor_z=sensor_z,
                floor_z=floor_z,
                room_stations=room_stations,
                room_objects=room_objects,
                room_cache=room_cache,
                azimuth_data=azimuth_data,
                elevation_data=elevation_data,
            )

            circles.append(
                LidarCircleScan(
                    station_id=station.id,
                    room_id=station.room_id,
                    center=(station.x, station.y, sensor_z),
                    radius=self.scan_range,
                    points=points,
                )
            )
        return circles

    def _generate_station_points(
        self,
        station: LidarStation,
        sensor_z: float,
        floor_z: float,
        room_stations: list[LidarStation],
        room_objects: list[_SemanticObject],
        room_cache: _RaycastRoomCache,
        azimuth_data: list[tuple[float, float, float]],
        elevation_data: list[tuple[float, float, float]],
    ) -> list[LidarPoint]:
        point_limit = self.points_per_station if self.points_per_station > 0 else None
        if (
            self._cuda_enabled
            and not self._cuda_runtime_failed
            and room_cache.cuda_min_bounds is not None
            and room_cache.cuda_max_bounds is not None
        ):
            try:
                return self._generate_station_points_cuda(
                    station=station,
                    sensor_z=sensor_z,
                    floor_z=floor_z,
                    room_stations=room_stations,
                    room_objects=room_objects,
                    room_cache=room_cache,
                    azimuth_data=azimuth_data,
                    elevation_data=elevation_data,
                    point_limit=point_limit,
                )
            except Exception:
                # Any CUDA runtime issue falls back to the CPU path.
                self._cuda_runtime_failed = True
        return self._generate_station_points_cpu(
            station=station,
            sensor_z=sensor_z,
            floor_z=floor_z,
            room_stations=room_stations,
            room_objects=room_objects,
            room_cache=room_cache,
            azimuth_data=azimuth_data,
            elevation_data=elevation_data,
            point_limit=point_limit,
        )

    def _generate_station_points_cpu(
        self,
        station: LidarStation,
        sensor_z: float,
        floor_z: float,
        room_stations: list[LidarStation],
        room_objects: list[_SemanticObject],
        room_cache: _RaycastRoomCache,
        azimuth_data: list[tuple[float, float, float]],
        elevation_data: list[tuple[float, float, float]],
        point_limit: int | None,
    ) -> list[LidarPoint]:
        azimuth_candidates: list[tuple[int, ...] | None] = []
        for _, unit_x, unit_y in azimuth_data:
            azimuth_candidates.append(
                self._raycast_candidates_for_direction(
                    origin_x=station.x,
                    origin_y=station.y,
                    dir_x=unit_x,
                    dir_y=unit_y,
                    max_distance=self.scan_range,
                    cache=room_cache,
                )
            )

        points: list[LidarPoint] = []
        for ray_index, (angle_deg, unit_x, unit_y) in enumerate(azimuth_data):
            candidate_indices = azimuth_candidates[ray_index]
            for elevation_deg, cos_elevation, sin_elevation in elevation_data:
                dir_x = cos_elevation * unit_x
                dir_y = cos_elevation * unit_y
                dir_z = sin_elevation

                hit = self._raycast_nearest_3d(
                    origin_x=station.x,
                    origin_y=station.y,
                    origin_z=sensor_z,
                    dir_x=dir_x,
                    dir_y=dir_y,
                    dir_z=dir_z,
                    objects=room_cache.objects,
                    max_distance=self.scan_range,
                    candidate_indices=candidate_indices,
                )
                if hit is None:
                    if self.emit_no_hit_returns:
                        self._append_no_hit_sample(
                            points=points,
                            station=station,
                            sensor_z=sensor_z,
                            dir_x=dir_x,
                            dir_y=dir_y,
                            dir_z=dir_z,
                            angle_deg=angle_deg,
                            elevation_deg=elevation_deg,
                            max_points=None,
                        )
                    continue

                hit_obj, hit_distance = hit
                if hit_distance < self.min_range:
                    continue

                px = station.x + dir_x * hit_distance
                py = station.y + dir_y * hit_distance
                pz = sensor_z + dir_z * hit_distance
                radial_distance = math.hypot(px - station.x, py - station.y)
                if pz <= sensor_z + 1e-6 and radial_distance < self.blind_spot_radius:
                    continue

                self._append_point_samples(
                    points=points,
                    station=station,
                    x=px,
                    y=py,
                    z=pz,
                    label=hit_obj.label_id,
                    class_name=hit_obj.class_name,
                    object_id=hit_obj.object_id,
                    angle_deg=angle_deg,
                    elevation_deg=elevation_deg,
                    distance=hit_distance,
                    domain=hit_obj.scan_domain,
                    max_points=None,
                )
        self._cast_neighbor_blind_spot_rays(
            station=station,
            sensor_z=sensor_z,
            floor_z=floor_z,
            room_stations=room_stations,
            objects=room_objects,
            points=points,
            raycast_cache=room_cache,
            max_points=None,
        )
        points = self._limit_points_per_station(points, station.id)
        points = self._apply_domain_density(points)
        return points

    def _build_station_azimuth_data(
        self,
        station_id: str,
        azimuth_step_count: int,
        angle_step: float,
    ) -> list[tuple[float, float, float]]:
        if azimuth_step_count <= 0:
            return []
        station_seed = self._station_numeric_seed(station_id)
        phase = (station_seed % max(1, azimuth_step_count)) * 0.61803398875
        phase_deg = (phase % 1.0) * angle_step
        if self.azimuth_phase_jitter_deg > 0.0:
            signed = ((station_seed * 31) % 2001) / 1000.0 - 1.0
            phase_deg += signed * self.azimuth_phase_jitter_deg

        span = _clamp(self.horizontal_fov_deg, 1.0, 360.0)
        start_angle = 0.0 if span >= 359.999 else ((360.0 - span) * 0.5)
        azimuth_data: list[tuple[float, float, float]] = []
        for ray_index in range(azimuth_step_count):
            angle_deg = (start_angle + ray_index * angle_step + phase_deg) % 360.0
            angle_rad = math.radians(angle_deg)
            azimuth_data.append((angle_deg, math.cos(angle_rad), math.sin(angle_rad)))
        return azimuth_data

    def _generate_station_points_cuda(
        self,
        station: LidarStation,
        sensor_z: float,
        floor_z: float,
        room_stations: list[LidarStation],
        room_objects: list[_SemanticObject],
        room_cache: _RaycastRoomCache,
        azimuth_data: list[tuple[float, float, float]],
        elevation_data: list[tuple[float, float, float]],
        point_limit: int | None,
    ) -> list[LidarPoint]:
        ray_dirs: list[tuple[float, float, float]] = []
        ray_angles: list[float] = []
        ray_elevations: list[float] = []
        for angle_deg, unit_x, unit_y in azimuth_data:
            for elevation_deg, cos_elevation, sin_elevation in elevation_data:
                ray_dirs.append((cos_elevation * unit_x, cos_elevation * unit_y, sin_elevation))
                ray_angles.append(angle_deg)
                ray_elevations.append(elevation_deg)

        hits = self._raycast_hits_cuda(
            origin_x=station.x,
            origin_y=station.y,
            origin_z=sensor_z,
            directions=ray_dirs,
            room_cache=room_cache,
            max_distance=self.scan_range,
        )

        points: list[LidarPoint] = []
        for ray_index, hit in enumerate(hits):
            if hit is None:
                if self.emit_no_hit_returns:
                    dir_x, dir_y, dir_z = ray_dirs[ray_index]
                    self._append_no_hit_sample(
                        points=points,
                        station=station,
                        sensor_z=sensor_z,
                        dir_x=dir_x,
                        dir_y=dir_y,
                        dir_z=dir_z,
                        angle_deg=ray_angles[ray_index],
                        elevation_deg=ray_elevations[ray_index],
                        max_points=None,
                    )
                continue
            hit_obj_index, hit_distance = hit
            if hit_obj_index < 0 or hit_obj_index >= len(room_cache.objects):
                continue
            if hit_distance < self.min_range:
                continue

            dir_x, dir_y, dir_z = ray_dirs[ray_index]
            hit_obj = room_cache.objects[hit_obj_index]
            px = station.x + dir_x * hit_distance
            py = station.y + dir_y * hit_distance
            pz = sensor_z + dir_z * hit_distance
            radial_distance = math.hypot(px - station.x, py - station.y)
            if pz <= sensor_z + 1e-6 and radial_distance < self.blind_spot_radius:
                continue

            self._append_point_samples(
                points=points,
                station=station,
                x=px,
                y=py,
                z=pz,
                label=hit_obj.label_id,
                class_name=hit_obj.class_name,
                object_id=hit_obj.object_id,
                angle_deg=ray_angles[ray_index],
                elevation_deg=ray_elevations[ray_index],
                distance=hit_distance,
                domain=hit_obj.scan_domain,
                max_points=None,
            )

        self._cast_neighbor_blind_spot_rays(
            station=station,
            sensor_z=sensor_z,
            floor_z=floor_z,
            room_stations=room_stations,
            objects=room_objects,
            points=points,
            raycast_cache=room_cache,
            max_points=None,
        )
        points = self._limit_points_per_station(points, station.id)
        points = self._apply_domain_density(points)
        return points

    def _raycast_hits_cuda(
        self,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        directions: list[tuple[float, float, float]],
        room_cache: _RaycastRoomCache,
        max_distance: float,
    ) -> list[tuple[int, float] | None]:
        if (
            not self._cuda_enabled
            or self._cuda_runtime_failed
            or self._torch_mod is None
            or room_cache.cuda_min_bounds is None
            or room_cache.cuda_max_bounds is None
            or not directions
            or not room_cache.objects
        ):
            return [None for _ in directions]

        torch_mod = self._torch_mod
        device = self._torch_device
        if device is None:
            return [None for _ in directions]

        dir_tensor = torch_mod.tensor(directions, dtype=torch_mod.float32, device=device)
        min_bounds = room_cache.cuda_min_bounds
        max_bounds = room_cache.cuda_max_bounds
        if min_bounds.shape[0] == 0:
            return [None for _ in directions]

        origin_tensor = torch_mod.tensor(
            [origin_x, origin_y, origin_z],
            dtype=torch_mod.float32,
            device=device,
        )
        object_origin_outside = (origin_tensor[None, :] < min_bounds) | (origin_tensor[None, :] > max_bounds)

        ray_count = int(dir_tensor.shape[0])
        object_count = int(min_bounds.shape[0])
        ray_batch = max(128, min(self.cuda_ray_batch_size, ray_count))
        object_batch = max(64, min(self.cuda_object_batch_size, object_count))
        epsilon = 1e-8
        inf = float("inf")

        nearest_distances = torch_mod.full((ray_count,), inf, dtype=torch_mod.float32, device=device)
        nearest_indices = torch_mod.full((ray_count,), -1, dtype=torch_mod.int64, device=device)

        for ray_start in range(0, ray_count, ray_batch):
            ray_end = min(ray_start + ray_batch, ray_count)
            ray_dirs = dir_tensor[ray_start:ray_end]
            local_count = int(ray_dirs.shape[0])
            local_best_distances = torch_mod.full((local_count,), inf, dtype=torch_mod.float32, device=device)
            local_best_indices = torch_mod.full((local_count,), -1, dtype=torch_mod.int64, device=device)

            ray_dirs_expanded = ray_dirs[:, None, :]
            ray_abs_expanded = torch_mod.abs(ray_dirs_expanded)

            for object_start in range(0, object_count, object_batch):
                object_end = min(object_start + object_batch, object_count)
                obj_min = min_bounds[object_start:object_end]
                obj_max = max_bounds[object_start:object_end]
                outside_mask = object_origin_outside[:, object_start:object_end]

                zero_mask = ray_abs_expanded <= epsilon
                safe_dirs = torch_mod.where(
                    zero_mask,
                    torch_mod.ones_like(ray_dirs_expanded),
                    ray_dirs_expanded,
                )
                t1 = (obj_min[None, :, :] - origin_tensor[None, None, :]) / safe_dirs
                t2 = (obj_max[None, :, :] - origin_tensor[None, None, :]) / safe_dirs
                t_near = torch_mod.minimum(t1, t2).amax(dim=2)
                t_far = torch_mod.maximum(t1, t2).amin(dim=2)

                invalid_parallel = (zero_mask & outside_mask[None, :, :]).any(dim=2)
                t_hit = torch_mod.where(t_near > 1e-5, t_near, t_far)
                valid = (~invalid_parallel) & (t_far >= t_near) & (t_hit > 1e-5) & (t_hit <= max_distance)
                masked_hits = torch_mod.where(valid, t_hit, torch_mod.full_like(t_hit, inf))
                chunk_best_distances, chunk_best_rel_indices = torch_mod.min(masked_hits, dim=1)

                update_mask = chunk_best_distances < local_best_distances
                if bool(update_mask.any().item()):
                    local_best_distances = torch_mod.where(
                        update_mask,
                        chunk_best_distances,
                        local_best_distances,
                    )
                    local_best_indices = torch_mod.where(
                        update_mask,
                        chunk_best_rel_indices + object_start,
                        local_best_indices,
                    )

            nearest_distances[ray_start:ray_end] = local_best_distances
            nearest_indices[ray_start:ray_end] = local_best_indices

        distance_values = nearest_distances.detach().cpu().tolist()
        index_values = nearest_indices.detach().cpu().tolist()
        hits: list[tuple[int, float] | None] = []
        for object_index, distance in zip(index_values, distance_values):
            if object_index < 0:
                hits.append(None)
                continue
            if not math.isfinite(distance) or distance <= 1e-5 or distance > max_distance:
                hits.append(None)
                continue
            hits.append((int(object_index), float(distance)))
        return hits

    def stations_from_coordinates(
        self,
        room_id: str,
        coordinates: Iterable[tuple[float, float, float]],
    ) -> list[LidarStation]:
        output: list[LidarStation] = []
        for index, (x, y, z) in enumerate(coordinates, start=1):
            output.append(
                LidarStation(
                    id=f"lidar_station_{index}",
                    room_id=room_id,
                    x=float(x),
                    y=float(y),
                    z=float(z),
                )
            )
        return output

    def _build_elevation_angles(self) -> list[float]:
        start = -self.vertical_fov_down_deg
        stop = self.vertical_fov_up_deg
        values = _frange(start, stop, self.vertical_resolution_deg)
        if not values:
            return [0.0]
        return values

    def _append_point_samples(
        self,
        points: list[LidarPoint],
        station: LidarStation,
        x: float,
        y: float,
        z: float,
        label: int,
        class_name: str,
        object_id: str,
        angle_deg: float,
        elevation_deg: float,
        distance: float,
        domain: str,
        max_points: int | None = None,
    ) -> None:
        if max_points is not None and len(points) >= max_points:
            return
        points.append(
            LidarPoint(
                x=x,
                y=y,
                z=z,
                label=label,
                class_name=class_name,
                object_id=object_id,
                instance_id=self._instance_id_for_object(object_id),
                station_id=station.id,
                angle_deg=angle_deg,
                elevation_deg=elevation_deg,
                distance=distance,
                domain=domain,
            )
        )

    def _append_no_hit_sample(
        self,
        points: list[LidarPoint],
        station: LidarStation,
        sensor_z: float,
        dir_x: float,
        dir_y: float,
        dir_z: float,
        angle_deg: float,
        elevation_deg: float,
        max_points: int | None,
    ) -> None:
        if max_points is not None and len(points) >= max_points:
            return
        px = station.x + dir_x * self.scan_range
        py = station.y + dir_y * self.scan_range
        pz = sensor_z + dir_z * self.scan_range
        radial_distance = math.hypot(px - station.x, py - station.y)
        if pz <= sensor_z + 1e-6 and radial_distance < self.blind_spot_radius:
            return
        points.append(
            LidarPoint(
                x=px,
                y=py,
                z=pz,
                label=-1,
                class_name="no_return",
                object_id="no_return",
                instance_id=-1,
                station_id=station.id,
                angle_deg=angle_deg,
                elevation_deg=elevation_deg,
                distance=self.scan_range,
                domain="unknown",
            )
        )

    def _instance_id_for_object(self, object_id: str) -> int:
        normalized = object_id.strip()
        if not normalized or normalized == "no_return":
            return -1
        existing = self._instance_id_by_object.get(normalized)
        if existing is not None:
            return existing
        assigned = self._next_instance_id
        self._next_instance_id += 1
        self._instance_id_by_object[normalized] = assigned
        return assigned

    def _station_jitter_offsets(self, station_id: str) -> tuple[tuple[float, float, float], ...]:
        cached = self._station_jitter_cache.get(station_id)
        if cached is not None:
            return cached
        if self.point_multiplier <= 1 or self.point_jitter <= 0.0:
            self._station_jitter_cache[station_id] = ()
            return ()

        station_seed = self._station_numeric_seed(station_id)
        rotation_deg = float((station_seed * 17) % 360)
        rotation_rad = math.radians(rotation_deg)
        rot_cos = math.cos(rotation_rad)
        rot_sin = math.sin(rotation_rad)

        offsets: list[tuple[float, float, float]] = []
        golden_angle_deg = 137.50776405003785
        for sample_index in range(1, self.point_multiplier):
            ring = 1.0 + (sample_index // 6) * 0.62
            jitter_radius = self.point_jitter * ring
            base_angle_rad = math.radians((sample_index * golden_angle_deg) % 360.0)
            base_x = math.cos(base_angle_rad) * jitter_radius
            base_y = math.sin(base_angle_rad) * jitter_radius
            offset_x = base_x * rot_cos - base_y * rot_sin
            offset_y = base_x * rot_sin + base_y * rot_cos
            offset_z = ((sample_index % 5) - 2) * self.point_jitter * 0.11
            offsets.append((offset_x, offset_y, offset_z))

        cached_offsets = tuple(offsets)
        self._station_jitter_cache[station_id] = cached_offsets
        return cached_offsets

    def _station_numeric_seed(self, station_id: str) -> int:
        value = 0
        for index, char in enumerate(station_id, start=1):
            value += index * ord(char)
        return value

    def _limit_points_per_station(
        self,
        points: list[LidarPoint],
        station_id: str,
    ) -> list[LidarPoint]:
        limit = self.points_per_station
        if limit <= 0:
            return points
        if len(points) <= limit:
            return points
        if limit == 1:
            return [points[0]]
        return self._spherical_stratified_limit(points=points, limit=limit, station_id=station_id)

    def _spherical_stratified_limit(
        self,
        points: list[LidarPoint],
        limit: int,
        station_id: str,
    ) -> list[LidarPoint]:
        if limit >= len(points):
            return points

        az_bins = max(24, int(round(360.0 / max(self.angular_resolution_deg, 0.1))))
        v_span = self.vertical_fov_up_deg + self.vertical_fov_down_deg
        el_bins = max(8, int(round(v_span / max(self.vertical_resolution_deg, 0.1))))

        buckets: dict[tuple[int, int], list[LidarPoint]] = {}
        for point in points:
            az = point.angle_deg % 360.0
            el = point.elevation_deg + self.vertical_fov_down_deg
            az_i = int((az / 360.0) * az_bins) % az_bins
            el_norm = _clamp(el / max(v_span, 1e-6), 0.0, 0.999999)
            el_i = int(el_norm * el_bins)
            buckets.setdefault((az_i, el_i), []).append(point)

        for bucket_points in buckets.values():
            bucket_points.sort(key=lambda p: p.distance)

        keys = sorted(buckets.keys())
        if not keys:
            return points[:limit]

        seed = self._station_numeric_seed(station_id)
        start = seed % len(keys)
        ordered_keys = keys[start:] + keys[:start]

        selected: list[LidarPoint] = []
        depth = 0
        while len(selected) < limit:
            progressed = False
            for key in ordered_keys:
                bucket_points = buckets[key]
                if depth < len(bucket_points):
                    selected.append(bucket_points[depth])
                    progressed = True
                    if len(selected) >= limit:
                        break
            if not progressed:
                break
            depth += 1

        if len(selected) < limit:
            need = limit - len(selected)
            selected.extend(points[:need])
        return selected[:limit]

    def _apply_domain_density(self, points: list[LidarPoint]) -> list[LidarPoint]:
        factor = self.exterior_point_density_factor
        if not points or factor >= 1.0:
            return points

        exterior_indices = [index for index, point in enumerate(points) if point.domain == "exterior"]
        if not exterior_indices:
            return points

        if factor <= 0.0:
            keep_exterior_indices: set[int] = set()
        else:
            keep_count = int(round(len(exterior_indices) * factor))
            if keep_count <= 0:
                keep_count = 1
            if keep_count >= len(exterior_indices):
                return points
            keep_exterior_indices = {
                exterior_indices[source_index]
                for source_index in self._stratified_index_selection(
                    total=len(exterior_indices),
                    limit=keep_count,
                )
            }

        filtered: list[LidarPoint] = []
        for index, point in enumerate(points):
            if point.domain != "exterior" or index in keep_exterior_indices:
                filtered.append(point)
        return filtered

    def _stratified_index_selection(self, total: int, limit: int) -> list[int]:
        if limit <= 0 or total <= 0:
            return []
        if limit >= total:
            return list(range(total))

        selected: list[int] = []
        step = total / float(limit)
        for index in range(limit):
            source_index = int(index * step)
            if source_index >= total:
                source_index = total - 1
            selected.append(source_index)
        return selected

    def _cast_neighbor_blind_spot_rays(
        self,
        station: LidarStation,
        sensor_z: float,
        floor_z: float,
        room_stations: list[LidarStation],
        objects: list[_SemanticObject],
        points: list[LidarPoint],
        raycast_cache: _RaycastRoomCache | None = None,
        max_points: int | None = None,
    ) -> None:
        if self.blind_spot_radius <= 0.0:
            return
        if len(room_stations) <= 1:
            return
        if max_points is not None and len(points) >= max_points:
            return

        offset = self.blind_spot_radius * 0.45
        offsets = [
            (0.0, 0.0),
            (offset, 0.0),
            (-offset, 0.0),
            (0.0, offset),
            (0.0, -offset),
        ]
        for neighbor in room_stations:
            if neighbor.id == station.id:
                continue
            if max_points is not None and len(points) >= max_points:
                break
            for offset_x, offset_y in offsets:
                if max_points is not None and len(points) >= max_points:
                    break
                target_x = neighbor.x + offset_x
                target_y = neighbor.y + offset_y
                target_z = floor_z + 1e-3

                dir_x = target_x - station.x
                dir_y = target_y - station.y
                dir_z = target_z - sensor_z
                norm = math.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
                if norm <= 1e-9:
                    continue
                if norm > self.scan_range:
                    continue
                if norm < self.min_range:
                    continue
                dir_x /= norm
                dir_y /= norm
                dir_z /= norm

                hit = self._raycast_nearest_3d(
                    origin_x=station.x,
                    origin_y=station.y,
                    origin_z=sensor_z,
                    dir_x=dir_x,
                    dir_y=dir_y,
                    dir_z=dir_z,
                    objects=(
                        raycast_cache.objects
                        if raycast_cache is not None
                        else tuple(objects)
                    ),
                    max_distance=self.scan_range,
                )
                if hit is None:
                    continue
                hit_obj, hit_distance = hit
                if hit_distance < self.min_range:
                    continue

                px = station.x + dir_x * hit_distance
                py = station.y + dir_y * hit_distance
                pz = sensor_z + dir_z * hit_distance
                radial_distance = math.hypot(px - station.x, py - station.y)
                if pz <= sensor_z + 1e-6 and radial_distance < self.blind_spot_radius:
                    continue

                azimuth = (math.degrees(math.atan2(dir_y, dir_x)) + 360.0) % 360.0
                elevation = math.degrees(math.asin(_clamp(dir_z, -1.0, 1.0)))
                self._append_point_samples(
                    points=points,
                    station=station,
                    x=px,
                    y=py,
                    z=pz,
                    label=hit_obj.label_id,
                    class_name=hit_obj.class_name,
                    object_id=hit_obj.object_id,
                    angle_deg=azimuth,
                    elevation_deg=elevation,
                    distance=hit_distance,
                    domain=hit_obj.scan_domain,
                    max_points=max_points,
                )

    def _augment_for_blind_spot_coverage(
        self,
        selected_indices: list[int],
        candidates: list[tuple[float, float]],
        station_limit: int | None = None,
    ) -> list[int]:
        if not self.ensure_blind_spot_coverage:
            return selected_indices
        if self.blind_spot_radius <= 0.0 or len(candidates) <= 1:
            return selected_indices

        limit = station_limit if station_limit is not None else self.max_stations_per_room
        selected_unique = list(dict.fromkeys(selected_indices))
        if len(selected_unique) >= max(4, int(limit * 0.7)):
            return selected_unique[:limit]
        min_neighbor_distance = max(self.blind_spot_radius * 1.15, self.min_range * 1.25, 0.8)
        max_neighbor_distance = max(min_neighbor_distance + 0.1, self.scan_range * 0.9)

        if len(selected_unique) == 1 and len(candidates) > 1 and len(selected_unique) < limit:
            initial_neighbor = self._find_neighbor_candidate(
                index=selected_unique[0],
                selected_indices=selected_unique,
                candidates=candidates,
                min_distance=min_neighbor_distance,
                max_distance=max_neighbor_distance,
            )
            if initial_neighbor is not None:
                selected_unique.append(initial_neighbor)

        changed = True
        while changed and len(selected_unique) < limit:
            changed = False
            for index in list(selected_unique):
                if self._has_station_neighbor(
                    index=index,
                    selected_indices=selected_unique,
                    candidates=candidates,
                    min_distance=min_neighbor_distance,
                    max_distance=max_neighbor_distance,
                ):
                    continue
                replacement = self._find_neighbor_candidate(
                    index=index,
                    selected_indices=selected_unique,
                    candidates=candidates,
                    min_distance=min_neighbor_distance,
                    max_distance=max_neighbor_distance,
                )
                if replacement is None:
                    continue
                selected_unique.append(replacement)
                changed = True
                if len(selected_unique) >= limit:
                    break
        return selected_unique

    def _has_station_neighbor(
        self,
        index: int,
        selected_indices: list[int],
        candidates: list[tuple[float, float]],
        min_distance: float,
        max_distance: float,
    ) -> bool:
        for other in selected_indices:
            if other == index:
                continue
            distance = self._candidate_distance(candidates[index], candidates[other])
            if min_distance <= distance <= max_distance:
                return True
        return False

    def _find_neighbor_candidate(
        self,
        index: int,
        selected_indices: list[int],
        candidates: list[tuple[float, float]],
        min_distance: float,
        max_distance: float,
    ) -> int | None:
        existing = set(selected_indices)
        source = candidates[index]
        best_index: int | None = None
        best_score = float("inf")
        fallback_index: int | None = None
        fallback_score = float("inf")
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in existing:
                continue
            distance = self._candidate_distance(source, candidate)
            if distance >= max(self.min_range, 0.2) and distance < fallback_score:
                fallback_score = distance
                fallback_index = candidate_index
            if distance < min_distance or distance > max_distance:
                continue
            if distance < best_score:
                best_score = distance
                best_index = candidate_index
        if best_index is not None:
            return best_index
        return fallback_index

    def _candidate_distance(
        self,
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    def _build_label_map(self, custom_mapping: object) -> dict[str, int]:
        base = {
            "unknown": 0,
            "pipe": 1,
            "wire": 2,
            "wall": 3,
            "floor": 4,
            "ceiling": 5,
            "machine": 6,
            "desk": 7,
            "rack": 8,
            "boiler": 9,
            "conveyor": 10,
            "structure": 11,
            "infrastructure": 12,
            "roof": 13,
            "window": 14,
            "door": 15,
            "gate": 16,
            "terrain": 17,
            "facade": 18,
        }
        if not isinstance(custom_mapping, Mapping):
            return base
        merged = dict(base)
        for key, value in custom_mapping.items():
            try:
                merged[str(key).strip().lower()] = int(value)
            except (TypeError, ValueError):
                continue
        return merged

    def _class_for_object_type(self, object_type: str) -> tuple[str, int]:
        normalized = _normalize_type(object_type)
        tokens = set(part for part in normalized.replace("-", "_").split("_") if part)
        has_cable_token = "cable" in normalized or "wire" in normalized
        is_exterior_like = (
            normalized.startswith("exterior_")
            or normalized.startswith("site_")
            or normalized.startswith("shell_")
        )
        has_opening_token = "opening" in tokens

        if (
            "roof" in tokens
            or normalized in {"flat", "sawtooth", "gabled"}
            or normalized.startswith("roof_")
        ) and not any(token in tokens for token in {"pipe", "tube", "duct", "chimney"}):
            class_name = "roof"
        elif "window" in tokens and not has_opening_token:
            class_name = "window"
        elif "door" in tokens and not has_opening_token:
            class_name = "door"
        elif "gate" in tokens and not has_opening_token:
            class_name = "gate"
        elif (
            is_exterior_like
            and any(
                token in tokens
                for token in {
                    "terrain",
                    "ground",
                    "road",
                    "parking",
                    "yard",
                    "apron",
                    "dock",
                    "loading",
                    "foundation",
                    "zone",
                }
            )
        ):
            class_name = "terrain"
        elif "facade" in tokens:
            class_name = "facade"
        elif (
            normalized in {"ceiling", "shell_ceiling"}
            or normalized.endswith("_ceiling")
            or "_ceiling_" in normalized
            or normalized.startswith("shell_ceiling")
            or normalized.startswith("ceiling_")
            or "ceiling" in tokens
            or "light" in tokens
        ):
            class_name = "ceiling"
        elif (
            normalized in {"wall", "door_opening", "window_opening", "shell_wall"}
            or normalized.endswith("_wall")
            or "_wall_" in normalized
            or normalized.startswith("shell_wall")
            or normalized.startswith("room_") and "_wall" in normalized
            or "fence" in tokens
        ):
            class_name = "wall"
        elif (
            normalized in {"floor", "corridor", "corridor_connector", "room_link", "shell_floor"}
            or normalized.endswith("_floor")
            or "_floor_" in normalized
            or normalized.startswith("floor_")
            or normalized.startswith("corridor_")
            or "corridor" in tokens
            or (
                "connector" in tokens
                and "ceiling" not in tokens
                and "wall" not in tokens
            )
            or any(token in tokens for token in {"walkway", "aisle", "ground", "road", "dock"})
            or normalized.endswith("_zone")
            or normalized in {"service_zone", "access_corridor", "operator_zone"}
        ):
            class_name = "floor"
        elif (
            "pipe" in normalized
            or "chimney" in normalized
            or "tube" in normalized
            or "duct" in normalized
        ):
            class_name = "pipe"
        elif (
            has_cable_token
            and "tray" not in tokens
            and "junction" not in tokens
            and "node" not in tokens
            and "support" not in tokens
        ):
            class_name = "wire"
        elif (
            normalized.startswith("infra_")
            or any(token in tokens for token in {"tray", "junction", "node"})
            or "vent" in tokens
            or ("drop" in tokens and not has_cable_token)
        ):
            class_name = "infrastructure"
        elif "boiler" in normalized:
            class_name = "boiler"
        elif "conveyor" in normalized:
            class_name = "conveyor"
        elif any(
            token in tokens
            for token in {
                "desk",
                "bench",
                "table",
                "chair",
                "seat",
                "console",
                "keyboard",
                "monitor",
                "screen",
                "display",
                "workstation",
            }
        ):
            class_name = "desk"
        elif any(token in tokens for token in {"rack", "shelf"}):
            class_name = "rack"
        elif any(
            token in tokens
            for token in {
                "beam",
                "column",
                "rail",
                "bridge",
                "support",
                "ladder",
                "hook",
                "platform",
                "canopy",
                "parapet",
            }
        ):
            class_name = "structure"
        elif any(
            token in tokens
            for token in {
                "machine",
                "cabinet",
                "panel",
                "pump",
                "tank",
                "valve",
                "toolbox",
                "part",
                "bunker",
                "exchanger",
                "cooling",
                "transformer",
                "battery",
                "instrument",
                "fume",
                "waste",
                "technical",
                "block",
            }
        ):
            class_name = "machine"
        else:
            class_name = "unknown"
        return class_name, self._label_map.get(class_name, 0)

    def _is_exterior_scan_type(self, object_type: str) -> bool:
        normalized = _normalize_type(object_type)
        tokens = set(part for part in normalized.replace("-", "_").split("_") if part)
        if normalized.startswith("exterior_") or normalized.startswith("site_"):
            return True
        if normalized.startswith("shell_"):
            return True
        if any(
            token in tokens
            for token in {
                "roof",
                "facade",
                "terrain",
                "road",
                "parking",
                "fence",
                "foundation",
                "apron",
                "canopy",
                "yard",
                "dock",
                "gate",
            }
        ):
            return True
        return False

    def _scan_domain_for_object(
        self,
        object_type: str,
        room_id: str,
    ) -> str:
        if room_id != self.factory_room_id:
            return "interior"
        if self._is_exterior_scan_type(object_type):
            return "exterior"
        return "interior"

    def _is_target_for_coverage(self, object_type: str) -> bool:
        normalized = _normalize_type(object_type)
        if normalized.startswith("room_"):
            return False
        if normalized.endswith("_group"):
            return False
        if normalized in {"door_opening", "window_opening"}:
            return False
        if not self.include_structural and normalized in {"floor", "wall", "ceiling"}:
            return False
        return True

    def _collect_scene_semantics(self, scene: Scene) -> tuple[list[_SemanticObject], dict[str, _RoomBounds]]:
        semantics: list[_SemanticObject] = []
        floors_by_room: dict[str, list[tuple[float, float, float, float, float]]] = {}

        def walk(
            node: SceneObject,
            parent_matrix: Matrix4,
            room_id: str | None,
        ) -> None:
            local = _local_matrix(node.transform)
            world = _mat_mul(parent_matrix, local)

            current_room_id = room_id
            if node.type.startswith("room_"):
                current_room_id = node.id

            if node.mesh is not None and node.mesh.vertices and current_room_id is not None:
                transformed = [_transform_vertex(world, vertex) for vertex in node.mesh.vertices]
                xs = [item[0] for item in transformed]
                ys = [item[1] for item in transformed]
                zs = [item[2] for item in transformed]
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                min_z = min(zs)
                max_z = max(zs)
                center_x = (min_x + max_x) / 2.0
                center_y = (min_y + max_y) / 2.0
                center_z = (min_z + max_z) / 2.0

                class_name, label_id = self._class_for_object_type(node.type)
                semantics.append(
                    _SemanticObject(
                        object_id=node.id,
                        room_id=current_room_id,
                        object_type=node.type,
                        label_id=label_id,
                        class_name=class_name,
                        min_x=min_x,
                        max_x=max_x,
                        min_y=min_y,
                        max_y=max_y,
                        min_z=min_z,
                        max_z=max_z,
                        center_x=center_x,
                        center_y=center_y,
                        center_z=center_z,
                        scan_domain=self._scan_domain_for_object(node.type, current_room_id),
                    )
                )
                if class_name == "floor":
                    floors_by_room.setdefault(current_room_id, []).append(
                        (min_x, max_x, min_y, max_y, min_z)
                    )

            for child in node.children:
                walk(child, world, current_room_id)

        for root in scene.objects:
            walk(root, _identity_matrix(), self.factory_room_id)

        room_bounds: dict[str, _RoomBounds] = {}
        room_ids = {item.room_id for item in semantics}
        for room_id in room_ids:
            floor_items = floors_by_room.get(room_id, [])
            if floor_items:
                min_x = min(item[0] for item in floor_items)
                max_x = max(item[1] for item in floor_items)
                min_y = min(item[2] for item in floor_items)
                max_y = max(item[3] for item in floor_items)
                floor_z = min(item[4] for item in floor_items)
            else:
                room_items = [item for item in semantics if item.room_id == room_id]
                min_x = min(item.min_x for item in room_items)
                max_x = max(item.max_x for item in room_items)
                min_y = min(item.min_y for item in room_items)
                max_y = max(item.max_y for item in room_items)
                floor_z = min(item.min_z for item in room_items)

            room_bounds[room_id] = _RoomBounds(
                room_id=room_id,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                floor_z=floor_z,
            )
        return semantics, room_bounds

    def _merge_room_bounds(self, bounds: Iterable[_RoomBounds]) -> _RoomBounds | None:
        bound_list = list(bounds)
        if not bound_list:
            return None
        return _RoomBounds(
            room_id=self.factory_room_id,
            min_x=min(item.min_x for item in bound_list),
            max_x=max(item.max_x for item in bound_list),
            min_y=min(item.min_y for item in bound_list),
            max_y=max(item.max_y for item in bound_list),
            floor_z=min(item.floor_z for item in bound_list),
        )

    def _build_coverage_targets(
        self,
        bounds: _RoomBounds,
        room_objects: list[_SemanticObject],
    ) -> list[_CoverageTarget]:
        targets: list[_CoverageTarget] = []
        dedup: dict[tuple[float, float, float], _CoverageTarget] = {}

        def register(target: _CoverageTarget) -> None:
            key = (round(target.x, 3), round(target.y, 3), round(target.z, 3))
            dedup[key] = target

        coverage_objects = [
            obj for obj in room_objects if self._is_target_for_coverage(obj.object_type)
        ]
        for obj in coverage_objects:
            register(
                _CoverageTarget(
                    target_id=f"obj:{obj.object_id}:center",
                    room_id=bounds.room_id,
                    x=obj.center_x,
                    y=obj.center_y,
                    z=obj.center_z,
                    object_id=obj.object_id,
                )
            )
            if obj.max_z - obj.min_z > 2.0:
                register(
                    _CoverageTarget(
                        target_id=f"obj:{obj.object_id}:top",
                        room_id=bounds.room_id,
                        x=obj.center_x,
                        y=obj.center_y,
                        z=obj.max_z - 0.08,
                        object_id=obj.object_id,
                    )
                )

        if self.include_structural:
            margin = min(self.station_margin, 0.2)
            min_x = bounds.min_x + margin
            max_x = bounds.max_x - margin
            min_y = bounds.min_y + margin
            max_y = bounds.max_y - margin
            if min_x > max_x:
                min_x, max_x = bounds.min_x, bounds.max_x
            if min_y > max_y:
                min_y, max_y = bounds.min_y, bounds.max_y

            area = max(1.0, (max_x - min_x) * (max_y - min_y))
            area_based_step = math.sqrt(area / max(40.0, float(self.coverage_max_targets) * 0.42))
            grid_step = max(self.coverage_grid_step, area_based_step)
            wall_step = max(self.coverage_wall_step, grid_step * 1.15)

            xs = _frange(min_x, max_x, grid_step)
            ys = _frange(min_y, max_y, grid_step)
            if not xs:
                xs = [(min_x + max_x) / 2.0]
            if not ys:
                ys = [(min_y + max_y) / 2.0]

            floor_z = bounds.floor_z + 0.02
            ceiling_z = self._estimate_room_ceiling(bounds, room_objects) - 0.02
            if ceiling_z < floor_z + 0.08:
                ceiling_z = floor_z + 0.08

            for x in xs:
                for y in ys:
                    register(
                        _CoverageTarget(
                            target_id=f"floor:{bounds.room_id}:{x:.3f}:{y:.3f}",
                            room_id=bounds.room_id,
                            x=x,
                            y=y,
                            z=floor_z,
                            object_id=None,
                        )
                    )
            ceiling_step = grid_step * 1.35
            xs_ceiling = _frange(min_x, max_x, ceiling_step)
            ys_ceiling = _frange(min_y, max_y, ceiling_step)
            if not xs_ceiling:
                xs_ceiling = [(min_x + max_x) / 2.0]
            if not ys_ceiling:
                ys_ceiling = [(min_y + max_y) / 2.0]
            for x in xs_ceiling:
                for y in ys_ceiling:
                    register(
                        _CoverageTarget(
                            target_id=f"ceiling:{bounds.room_id}:{x:.3f}:{y:.3f}",
                            room_id=bounds.room_id,
                            x=x,
                            y=y,
                            z=ceiling_z,
                            object_id=None,
                        )
                    )

            room_height = max(0.8, ceiling_z - floor_z)
            z_levels = [
                floor_z + min(1.2, room_height * 0.33),
                floor_z + min(room_height - 0.15, room_height * 0.66),
            ]
            wall_xs = _frange(min_x, max_x, wall_step)
            wall_ys = _frange(min_y, max_y, wall_step)
            if not wall_xs:
                wall_xs = [(min_x + max_x) / 2.0]
            if not wall_ys:
                wall_ys = [(min_y + max_y) / 2.0]
            wall_offset = 0.015
            for z in z_levels:
                for x in wall_xs:
                    register(
                        _CoverageTarget(
                            target_id=f"wall:south:{bounds.room_id}:{x:.3f}:{z:.3f}",
                            room_id=bounds.room_id,
                            x=x,
                            y=min_y + wall_offset,
                            z=z,
                            object_id=None,
                        )
                    )
                    register(
                        _CoverageTarget(
                            target_id=f"wall:north:{bounds.room_id}:{x:.3f}:{z:.3f}",
                            room_id=bounds.room_id,
                            x=x,
                            y=max_y - wall_offset,
                            z=z,
                            object_id=None,
                        )
                    )
                for y in wall_ys:
                    register(
                        _CoverageTarget(
                            target_id=f"wall:west:{bounds.room_id}:{y:.3f}:{z:.3f}",
                            room_id=bounds.room_id,
                            x=min_x + wall_offset,
                            y=y,
                            z=z,
                            object_id=None,
                        )
                    )
                    register(
                        _CoverageTarget(
                            target_id=f"wall:east:{bounds.room_id}:{y:.3f}:{z:.3f}",
                            room_id=bounds.room_id,
                            x=max_x - wall_offset,
                            y=y,
                            z=z,
                            object_id=None,
                        )
                    )

        targets.extend(dedup.values())
        if len(targets) <= self.coverage_max_targets:
            return targets

        sampled: list[_CoverageTarget] = []
        step = len(targets) / float(self.coverage_max_targets)
        for index in range(self.coverage_max_targets):
            source_index = int(index * step)
            if source_index >= len(targets):
                source_index = len(targets) - 1
            sampled.append(targets[source_index])
        return sampled

    def _estimate_room_ceiling(
        self,
        bounds: _RoomBounds,
        room_objects: list[_SemanticObject],
    ) -> float:
        ceiling_levels = [obj.max_z for obj in room_objects if obj.class_name == "ceiling"]
        if ceiling_levels:
            return max(ceiling_levels)
        if room_objects:
            return max(obj.max_z for obj in room_objects)
        return bounds.floor_z + max(self.sensor_height * 1.8, 2.6)

    def _augment_candidates_from_targets(
        self,
        bounds: _RoomBounds,
        room_objects: list[_SemanticObject],
        candidates: list[tuple[float, float]],
        targets: list[_CoverageTarget],
    ) -> list[tuple[float, float]]:
        if not targets:
            return candidates

        dedup: dict[tuple[float, float], tuple[float, float]] = {}
        for x, y in candidates:
            dedup[(round(x, 3), round(y, 3))] = (x, y)

        margin = min(self.station_margin, 0.2)
        min_x = bounds.min_x + margin
        max_x = bounds.max_x - margin
        min_y = bounds.min_y + margin
        max_y = bounds.max_y - margin
        if min_x > max_x:
            min_x, max_x = bounds.min_x, bounds.max_x
        if min_y > max_y:
            min_y, max_y = bounds.min_y, bounds.max_y

        max_candidates = max(len(candidates), self.max_stations_per_room * 20)
        candidate_min_gap = max(0.5, self.station_spacing * 0.55)
        probe_offset = max(self.blind_spot_radius * 1.35, self.min_range * 1.2, 0.55)
        sensor_z = bounds.floor_z + self.sensor_height

        def can_place(x: float, y: float) -> bool:
            if any(math.hypot(x - existing_x, y - existing_y) < candidate_min_gap for existing_x, existing_y in dedup.values()):
                return False
            return self._point_is_free(
                x,
                y,
                room_objects,
                clearance=self.obstacle_clearance * 0.45,
                sensor_z=sensor_z,
            )

        for target in targets:
            if len(dedup) >= max_candidates:
                break
            if any(
                math.hypot(target.x - existing_x, target.y - existing_y) < candidate_min_gap
                for existing_x, existing_y in dedup.values()
            ):
                continue

            placed = False
            for angle_deg in (0.0, 90.0, 180.0, 270.0):
                angle_rad = math.radians(angle_deg)
                px = _clamp(target.x + probe_offset * math.cos(angle_rad), min_x, max_x)
                py = _clamp(target.y + probe_offset * math.sin(angle_rad), min_y, max_y)
                if not can_place(px, py):
                    continue
                dedup[(round(px, 3), round(py, 3))] = (px, py)
                placed = True
                break
            if placed:
                continue
            fallback_x = _clamp(target.x, min_x, max_x)
            fallback_y = _clamp(target.y, min_y, max_y)
            if can_place(fallback_x, fallback_y):
                dedup[(round(fallback_x, 3), round(fallback_y, 3))] = (fallback_x, fallback_y)

        return list(dedup.values())

    def _complete_target_coverage(
        self,
        selected_indices: list[int],
        coverage: list[set[int]],
        target_count: int,
        station_limit: int,
    ) -> list[int]:
        selected_unique = list(dict.fromkeys(selected_indices))
        if not coverage or target_count <= 0:
            return selected_unique[:station_limit]

        uncovered = set(range(target_count))
        for index in selected_unique:
            if 0 <= index < len(coverage):
                uncovered -= coverage[index]
        used = set(selected_unique)

        while uncovered and len(selected_unique) < station_limit:
            best_idx = -1
            best_gain = 0
            for index, covered in enumerate(coverage):
                if index in used:
                    continue
                gain = len(uncovered & covered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = index
            if best_idx < 0 or best_gain <= 0:
                break
            selected_unique.append(best_idx)
            used.add(best_idx)
            uncovered -= coverage[best_idx]
        return selected_unique[:station_limit]

    def _generate_candidate_stations(
        self,
        bounds: _RoomBounds,
        room_objects: list[_SemanticObject],
    ) -> list[tuple[float, float]]:
        min_x = bounds.min_x + self.station_margin
        max_x = bounds.max_x - self.station_margin
        min_y = bounds.min_y + self.station_margin
        max_y = bounds.max_y - self.station_margin
        if min_x > max_x:
            min_x = bounds.min_x
            max_x = bounds.max_x
        if min_y > max_y:
            min_y = bounds.min_y
            max_y = bounds.max_y

        step = max(0.8, self.station_spacing)
        offsets = [(0.0, 0.0), (step * 0.5, 0.0), (0.0, step * 0.5), (step * 0.5, step * 0.5)]
        candidates: list[tuple[float, float]] = []
        for offset_x, offset_y in offsets:
            xs = _frange(min_x + offset_x, max_x, step)
            ys = _frange(min_y + offset_y, max_y, step)
            sensor_z = bounds.floor_z + self.sensor_height
            for x in xs:
                for y in ys:
                    if self._point_is_free(x, y, room_objects, sensor_z=sensor_z):
                        candidates.append((x, y))

        dedup: dict[tuple[float, float], tuple[float, float]] = {}
        for x, y in candidates:
            key = (round(x, 3), round(y, 3))
            dedup[key] = (x, y)

        strategic_points = [
            ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5),
            (min_x, min_y),
            (min_x, max_y),
            (max_x, min_y),
            (max_x, max_y),
            ((min_x * 0.75) + (max_x * 0.25), (min_y + max_y) * 0.5),
            ((min_x * 0.25) + (max_x * 0.75), (min_y + max_y) * 0.5),
            ((min_x + max_x) * 0.5, (min_y * 0.75) + (max_y * 0.25)),
            ((min_x + max_x) * 0.5, (min_y * 0.25) + (max_y * 0.75)),
        ]
        for px, py in strategic_points:
            if self._point_is_free(px, py, room_objects, sensor_z=sensor_z):
                dedup[(round(px, 3), round(py, 3))] = (px, py)
        if self.ensure_blind_spot_coverage and len(dedup) < 2:
            dense_step = max(0.6, step * 0.5)
            dense_offsets = [
                (0.0, 0.0),
                (dense_step * 0.5, 0.0),
                (0.0, dense_step * 0.5),
                (dense_step * 0.5, dense_step * 0.5),
            ]
            for offset_x, offset_y in dense_offsets:
                xs = _frange(min_x + offset_x, max_x, dense_step)
                ys = _frange(min_y + offset_y, max_y, dense_step)
                for x in xs:
                    for y in ys:
                        if self._point_is_free(x, y, room_objects, sensor_z=sensor_z):
                            key = (round(x, 3), round(y, 3))
                            dedup[key] = (x, y)
        if dedup:
            return list(dedup.values())
        return [((bounds.min_x + bounds.max_x) / 2.0, (bounds.min_y + bounds.max_y) / 2.0)]

    def _augment_factory_room_station_points(
        self,
        bounds: _RoomBounds,
        room_objects: list[_SemanticObject],
        selected_points: list[tuple[float, float]],
        station_limit: int,
    ) -> list[tuple[float, float]]:
        if station_limit <= 0:
            return []

        corridor_points = self._corridor_surface_station_points(bounds, room_objects)
        exterior_points = self._exterior_surface_station_points(bounds, room_objects)
        if not corridor_points and not exterior_points:
            return selected_points[:station_limit]

        dedup: dict[tuple[float, float], tuple[float, float]] = {}
        augmented: list[tuple[float, float]] = []
        min_gap = max(1.2, min(self.station_spacing * 0.7, max(self.scan_range * 0.2, 3.2)))
        clearance = max(0.0, self.obstacle_clearance * 0.35)
        sensor_z = bounds.floor_z + self.sensor_height
        min_exterior_points = 0
        if exterior_points:
            min_exterior_points = min(len(exterior_points), max(1, station_limit // 3))
            min_exterior_points = min(min_exterior_points, station_limit)

        def try_add(point_x: float, point_y: float) -> None:
            if len(augmented) >= station_limit:
                return
            if not self._point_is_free(
                point_x,
                point_y,
                room_objects,
                clearance=clearance,
                sensor_z=sensor_z,
            ):
                return
            if any(math.hypot(point_x - sx, point_y - sy) < min_gap for sx, sy in augmented):
                return
            key = (round(point_x, 3), round(point_y, 3))
            if key in dedup:
                return
            dedup[key] = (point_x, point_y)
            augmented.append((point_x, point_y))

        # For the factory-wide room we prioritize several stations outside the shell
        # to guarantee facade/roof/terrain coverage in the exterior domain.
        for sx, sy in exterior_points:
            if len(augmented) >= min_exterior_points:
                break
            try_add(sx, sy)
        for sx, sy in selected_points:
            try_add(sx, sy)
        for sx, sy in exterior_points:
            try_add(sx, sy)
        for sx, sy in corridor_points:
            try_add(sx, sy)

        if augmented:
            return augmented
        return selected_points[:station_limit]

    def _corridor_surface_station_points(
        self,
        bounds: _RoomBounds,
        room_objects: list[_SemanticObject],
    ) -> list[tuple[float, float]]:
        corridor_objects = [
            obj for obj in room_objects if self._is_corridor_surface_type(obj.object_type)
        ]
        if not corridor_objects:
            return []

        stride = max(1.8, min(self.scan_range * 0.32, self.station_spacing * 1.35))
        room_margin = min(self.station_margin, 0.3)
        min_x = bounds.min_x + room_margin
        max_x = bounds.max_x - room_margin
        min_y = bounds.min_y + room_margin
        max_y = bounds.max_y - room_margin
        if min_x > max_x:
            min_x = bounds.min_x
            max_x = bounds.max_x
        if min_y > max_y:
            min_y = bounds.min_y
            max_y = bounds.max_y

        sampled: list[tuple[float, float]] = []
        for obj in corridor_objects:
            span_x = max(0.0, obj.max_x - obj.min_x)
            span_y = max(0.0, obj.max_y - obj.min_y)
            if span_x <= 1e-6 or span_y <= 1e-6:
                continue
            if max(span_x, span_y) <= stride * 1.2:
                sampled.append((_clamp(obj.center_x, min_x, max_x), _clamp(obj.center_y, min_y, max_y)))
                continue
            if span_x >= span_y:
                edge_offset = min(stride * 0.5, span_x / 2.0)
                start = obj.min_x + edge_offset
                stop = obj.max_x - edge_offset
                xs = _frange(start, stop, stride)
                if not xs:
                    xs = [obj.center_x]
                for x in xs:
                    sampled.append((_clamp(x, min_x, max_x), _clamp(obj.center_y, min_y, max_y)))
            else:
                edge_offset = min(stride * 0.5, span_y / 2.0)
                start = obj.min_y + edge_offset
                stop = obj.max_y - edge_offset
                ys = _frange(start, stop, stride)
                if not ys:
                    ys = [obj.center_y]
                for y in ys:
                    sampled.append((_clamp(obj.center_x, min_x, max_x), _clamp(y, min_y, max_y)))

        dedup: dict[tuple[float, float], tuple[float, float]] = {}
        for x, y in sampled:
            key = (round(x, 3), round(y, 3))
            dedup[key] = (x, y)
        return list(dedup.values())

    def _exterior_surface_station_points(
        self,
        bounds: _RoomBounds,
        room_objects: list[_SemanticObject],
    ) -> list[tuple[float, float]]:
        exterior_objects = [
            obj for obj in room_objects if self._is_exterior_scan_type(obj.object_type)
        ]
        if not exterior_objects:
            return []

        shell_objects = []
        terrain_objects = []
        for obj in exterior_objects:
            normalized = _normalize_type(obj.object_type)
            tokens = set(part for part in normalized.replace("-", "_").split("_") if part)
            if (
                normalized.startswith("exterior_wall")
                or normalized.startswith("exterior_roof")
                or normalized.startswith("exterior_facade")
                or normalized.startswith("shell_")
                or "facade" in tokens
                or "roof" in tokens
                or "wall" in tokens
            ):
                shell_objects.append(obj)
            if (
                obj.class_name == "terrain"
                or normalized.startswith("site_")
                or any(
                    token in tokens
                    for token in {
                        "terrain",
                        "ground",
                        "road",
                        "parking",
                        "yard",
                        "zone",
                        "apron",
                        "dock",
                    }
                )
            ):
                terrain_objects.append(obj)

        if not shell_objects:
            shell_objects = exterior_objects
        if not terrain_objects:
            terrain_objects = exterior_objects

        shell_min_x = min(obj.min_x for obj in shell_objects)
        shell_max_x = max(obj.max_x for obj in shell_objects)
        shell_min_y = min(obj.min_y for obj in shell_objects)
        shell_max_y = max(obj.max_y for obj in shell_objects)

        terrain_min_x = min(obj.min_x for obj in terrain_objects)
        terrain_max_x = max(obj.max_x for obj in terrain_objects)
        terrain_min_y = min(obj.min_y for obj in terrain_objects)
        terrain_max_y = max(obj.max_y for obj in terrain_objects)

        station_margin = max(0.25, min(self.station_margin, 0.5))
        outer_min_x = bounds.min_x + station_margin
        outer_max_x = bounds.max_x - station_margin
        outer_min_y = bounds.min_y + station_margin
        outer_max_y = bounds.max_y - station_margin
        if outer_min_x > outer_max_x:
            outer_min_x, outer_max_x = bounds.min_x, bounds.max_x
        if outer_min_y > outer_max_y:
            outer_min_y, outer_max_y = bounds.min_y, bounds.max_y

        perimeter_offset = max(1.0, self.station_margin + 0.9)
        terrain_offset = max(perimeter_offset + 1.2, min(self.scan_range * 0.32, 5.5))
        shell_line_step = max(2.2, min(self.scan_range * 0.45, self.station_spacing * 1.25))
        terrain_line_step = max(shell_line_step * 1.15, 3.0)

        shell_south_y = _clamp(shell_min_y - perimeter_offset, outer_min_y, outer_max_y)
        shell_north_y = _clamp(shell_max_y + perimeter_offset, outer_min_y, outer_max_y)
        shell_west_x = _clamp(shell_min_x - perimeter_offset, outer_min_x, outer_max_x)
        shell_east_x = _clamp(shell_max_x + perimeter_offset, outer_min_x, outer_max_x)

        terrain_south_y = _clamp(shell_min_y - terrain_offset, terrain_min_y + station_margin, terrain_max_y - station_margin)
        terrain_north_y = _clamp(shell_max_y + terrain_offset, terrain_min_y + station_margin, terrain_max_y - station_margin)
        terrain_west_x = _clamp(shell_min_x - terrain_offset, terrain_min_x + station_margin, terrain_max_x - station_margin)
        terrain_east_x = _clamp(shell_max_x + terrain_offset, terrain_min_x + station_margin, terrain_max_x - station_margin)

        points: list[tuple[float, float]] = []

        def add_ring_samples(
            min_x: float,
            max_x: float,
            min_y: float,
            max_y: float,
            south_y: float,
            north_y: float,
            west_x: float,
            east_x: float,
            step: float,
        ) -> None:
            xs = _frange(min_x, max_x, step)
            ys = _frange(min_y, max_y, step)
            if not xs:
                xs = [(min_x + max_x) / 2.0]
            if not ys:
                ys = [(min_y + max_y) / 2.0]
            for x in xs:
                points.append((x, south_y))
                points.append((x, north_y))
            for y in ys:
                points.append((west_x, y))
                points.append((east_x, y))

        add_ring_samples(
            min_x=shell_min_x,
            max_x=shell_max_x,
            min_y=shell_min_y,
            max_y=shell_max_y,
            south_y=shell_south_y,
            north_y=shell_north_y,
            west_x=shell_west_x,
            east_x=shell_east_x,
            step=shell_line_step,
        )
        add_ring_samples(
            min_x=terrain_min_x,
            max_x=terrain_max_x,
            min_y=terrain_min_y,
            max_y=terrain_max_y,
            south_y=terrain_south_y,
            north_y=terrain_north_y,
            west_x=terrain_west_x,
            east_x=terrain_east_x,
            step=terrain_line_step,
        )

        # Extra anchors around corners/midpoints help capture box silhouette and yard.
        midpoint_x = (shell_min_x + shell_max_x) / 2.0
        midpoint_y = (shell_min_y + shell_max_y) / 2.0
        points.extend(
            [
                (shell_west_x, shell_south_y),
                (shell_east_x, shell_south_y),
                (shell_west_x, shell_north_y),
                (shell_east_x, shell_north_y),
                (midpoint_x, shell_south_y),
                (midpoint_x, shell_north_y),
                (shell_west_x, midpoint_y),
                (shell_east_x, midpoint_y),
                (terrain_west_x, terrain_south_y),
                (terrain_east_x, terrain_south_y),
                (terrain_west_x, terrain_north_y),
                (terrain_east_x, terrain_north_y),
            ]
        )

        dedup: dict[tuple[float, float], tuple[float, float]] = {}
        for x, y in points:
            px = _clamp(x, outer_min_x, outer_max_x)
            py = _clamp(y, outer_min_y, outer_max_y)
            dedup[(round(px, 3), round(py, 3))] = (px, py)
        return list(dedup.values())

    def _is_corridor_surface_type(self, object_type: str) -> bool:
        normalized = _normalize_type(object_type)
        tokens = set(part for part in normalized.replace("-", "_").split("_") if part)
        return (
            normalized in {"corridor", "corridor_connector", "room_link"}
            or normalized.startswith("corridor_")
            or "corridor" in tokens
            or (
                "connector" in tokens
                and "ceiling" not in tokens
                and "wall" not in tokens
            )
            or normalized.startswith("factory_flow_link")
        )

    def _ensure_minimum_station_candidates(
        self,
        bounds: _RoomBounds,
        room_objects: list[_SemanticObject],
        candidates: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        if not self.ensure_blind_spot_coverage:
            return candidates
        if self.blind_spot_radius <= 0.0:
            return candidates
        if len(candidates) >= 2:
            return candidates
        if not candidates:
            return candidates

        base_x, base_y = candidates[0]
        probe_radius = max(self.blind_spot_radius * 1.35, 0.9)
        room_margin = min(self.station_margin, 0.25)
        min_x = bounds.min_x + room_margin
        max_x = bounds.max_x - room_margin
        min_y = bounds.min_y + room_margin
        max_y = bounds.max_y - room_margin
        if min_x > max_x:
            min_x = bounds.min_x
            max_x = bounds.max_x
        if min_y > max_y:
            min_y = bounds.min_y
            max_y = bounds.max_y

        angles = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
        relaxed_clearance = self.obstacle_clearance * 0.25
        sensor_z = bounds.floor_z + self.sensor_height
        fallback: tuple[float, float] | None = None
        for angle in angles:
            rad = math.radians(angle)
            px = _clamp(base_x + probe_radius * math.cos(rad), min_x, max_x)
            py = _clamp(base_y + probe_radius * math.sin(rad), min_y, max_y)
            if math.hypot(px - base_x, py - base_y) < max(self.min_range, 0.2):
                continue
            if fallback is None:
                fallback = (px, py)
            if self._point_is_free(
                px,
                py,
                room_objects,
                clearance=relaxed_clearance,
                sensor_z=sensor_z,
            ):
                return [candidates[0], (px, py)]

        if fallback is not None:
            return [candidates[0], fallback]
        return candidates

    def _point_is_free(
        self,
        x: float,
        y: float,
        room_objects: list[_SemanticObject],
        clearance: float | None = None,
        sensor_z: float | None = None,
    ) -> bool:
        buffer = self.obstacle_clearance if clearance is None else max(0.0, clearance)
        probe_z = self.sensor_height if sensor_z is None else sensor_z
        for obj in room_objects:
            normalized = _normalize_type(obj.object_type)
            if normalized in {"floor", "ceiling", "door_opening", "window_opening"}:
                continue
            if obj.max_z <= probe_z - max(0.4, self.sensor_height * 0.75):
                continue
            if not (obj.min_z - 1e-6 <= probe_z <= obj.max_z + 1e-6):
                continue
            if (
                obj.min_x - buffer <= x <= obj.max_x + buffer
                and obj.min_y - buffer <= y <= obj.max_y + buffer
            ):
                return False
        return True

    def _build_coverage_sets(
        self,
        candidates: list[tuple[float, float]],
        targets: list[_CoverageTarget],
        blockers: list[_SemanticObject],
        sensor_z: float,
    ) -> list[set[int]]:
        target_index = {target.target_id: idx for idx, target in enumerate(targets)}
        visibility_blockers = self._visibility_blockers(blockers)
        cover_sets: list[set[int]] = []
        for sx, sy in candidates:
            covered: set[int] = set()
            for target in targets:
                dx = target.x - sx
                dy = target.y - sy
                dz = target.z - sensor_z
                planar_distance = math.hypot(dx, dy)
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                if distance > self.scan_range:
                    continue
                if distance < self.min_range:
                    continue
                if target.z <= sensor_z + 1e-6 and planar_distance < self.blind_spot_radius:
                    continue
                if target.object_id is not None:
                    if self._segment_blocked(
                        sx,
                        sy,
                        target.x,
                        target.y,
                        visibility_blockers,
                        ignored_id=target.object_id,
                        sensor_z=sensor_z,
                        target_z=target.z,
                    ):
                        continue
                covered.add(target_index[target.target_id])
            cover_sets.append(covered)
        return cover_sets

    def _visibility_blockers(
        self,
        objects: list[_SemanticObject],
    ) -> list[_SemanticObject]:
        blockers: list[_SemanticObject] = []
        for obj in objects:
            normalized = _normalize_type(obj.object_type)
            if normalized in {"floor", "ceiling", "door_opening", "window_opening"}:
                continue
            if obj.class_name == "wire":
                continue
            if obj.class_name == "pipe":
                thickness = min(
                    obj.max_x - obj.min_x,
                    obj.max_y - obj.min_y,
                    obj.max_z - obj.min_z,
                )
                if thickness <= 0.3:
                    continue
            blockers.append(obj)
        return blockers

    def _greedy_set_cover(self, coverage: list[set[int]], target_count: int) -> list[int]:
        uncovered = set(range(target_count))
        selected: list[int] = []
        used: set[int] = set()

        while uncovered:
            best_idx = -1
            best_gain = 0
            for index, covered in enumerate(coverage):
                if index in used:
                    continue
                gain = len(uncovered & covered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = index
            if best_idx < 0 or best_gain == 0:
                break
            selected.append(best_idx)
            used.add(best_idx)
            uncovered -= coverage[best_idx]
        return selected

    def _build_raycast_cache(self, objects: list[_SemanticObject]) -> _RaycastRoomCache:
        scan_objects = tuple(
            obj for obj in objects if not self._should_skip_scan_object(obj.object_type)
        )
        if not scan_objects:
            return _RaycastRoomCache(
                objects=(),
                origin_x=0.0,
                origin_y=0.0,
                cell_size=max(0.5, self.raycast_cell_size),
                cells={},
                cuda_min_bounds=None,
                cuda_max_bounds=None,
            )

        min_x = min(obj.min_x for obj in scan_objects)
        max_x = max(obj.max_x for obj in scan_objects)
        min_y = min(obj.min_y for obj in scan_objects)
        max_y = max(obj.max_y for obj in scan_objects)
        margin = max(self.raycast_cell_size * 0.6, 0.5)
        origin_x = min_x - margin
        origin_y = min_y - margin
        cell_size = max(0.55, self.raycast_cell_size)

        cells: dict[tuple[int, int], list[int]] = {}
        for index, obj in enumerate(scan_objects):
            start_x = int(math.floor((obj.min_x - origin_x) / cell_size))
            end_x = int(math.floor((obj.max_x - origin_x) / cell_size))
            start_y = int(math.floor((obj.min_y - origin_y) / cell_size))
            end_y = int(math.floor((obj.max_y - origin_y) / cell_size))
            for ix in range(start_x, end_x + 1):
                for iy in range(start_y, end_y + 1):
                    cells.setdefault((ix, iy), []).append(index)

        frozen_cells = {key: tuple(value) for key, value in cells.items()}
        cuda_min_bounds: object | None = None
        cuda_max_bounds: object | None = None
        if self._cuda_enabled and not self._cuda_runtime_failed and self._torch_mod is not None:
            try:
                min_bounds_data = [
                    [obj.min_x, obj.min_y, obj.min_z]
                    for obj in scan_objects
                ]
                max_bounds_data = [
                    [obj.max_x, obj.max_y, obj.max_z]
                    for obj in scan_objects
                ]
                cuda_min_bounds = self._torch_mod.tensor(
                    min_bounds_data,
                    dtype=self._torch_mod.float32,
                    device=self._torch_device,
                )
                cuda_max_bounds = self._torch_mod.tensor(
                    max_bounds_data,
                    dtype=self._torch_mod.float32,
                    device=self._torch_device,
                )
            except Exception:
                # Keep CPU mode if CUDA tensor upload fails for any reason.
                self._cuda_runtime_failed = True
                cuda_min_bounds = None
                cuda_max_bounds = None

        return _RaycastRoomCache(
            objects=scan_objects,
            origin_x=origin_x,
            origin_y=origin_y,
            cell_size=cell_size,
            cells=frozen_cells,
            cuda_min_bounds=cuda_min_bounds,
            cuda_max_bounds=cuda_max_bounds,
        )

    def _raycast_candidates_for_direction(
        self,
        origin_x: float,
        origin_y: float,
        dir_x: float,
        dir_y: float,
        max_distance: float,
        cache: _RaycastRoomCache,
    ) -> tuple[int, ...] | None:
        if not self.optimize_raycasts:
            return None
        if not cache.cells or not cache.objects:
            return None
        planar_length = math.hypot(dir_x, dir_y)
        if planar_length <= 1e-9:
            return None

        unit_x = dir_x / planar_length
        unit_y = dir_y / planar_length
        step = max(0.45, min(cache.cell_size, self.raycast_march_step))
        step_count = max(1, int(math.ceil(max_distance / step)))

        seen: set[int] = set()
        for step_index in range(step_count + 1):
            sample_x = origin_x + unit_x * step * step_index
            sample_y = origin_y + unit_y * step * step_index
            ix = int(math.floor((sample_x - cache.origin_x) / cache.cell_size))
            iy = int(math.floor((sample_y - cache.origin_y) / cache.cell_size))
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    entries = cache.cells.get((ix + ox, iy + oy))
                    if not entries:
                        continue
                    for object_index in entries:
                        seen.add(object_index)

        if not seen:
            return None
        return tuple(sorted(seen))

    def _segment_blocked(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        objects: list[_SemanticObject],
        ignored_id: str | None = None,
        sensor_z: float | None = None,
        target_z: float | None = None,
    ) -> bool:
        dx = x1 - x0
        dy = y1 - y0
        distance = math.hypot(dx, dy)
        if distance <= 1e-9:
            return False
        dir_x = dx / distance
        dir_y = dy / distance
        max_distance = max(0.0, distance - 1e-3)
        seg_min_x = min(x0, x1) - 1e-5
        seg_max_x = max(x0, x1) + 1e-5
        seg_min_y = min(y0, y1) - 1e-5
        seg_max_y = max(y0, y1) + 1e-5
        sensor_height = self.sensor_height if sensor_z is None else sensor_z
        target_height = sensor_height if target_z is None else target_z
        ray_min_z = min(sensor_height, target_height) - 0.05
        ray_max_z = max(sensor_height, target_height) + 0.05
        for obj in objects:
            if obj.object_id == ignored_id:
                continue
            normalized = _normalize_type(obj.object_type)
            if normalized in {"floor", "ceiling", "door_opening", "window_opening"}:
                continue
            if obj.max_x < seg_min_x or obj.min_x > seg_max_x:
                continue
            if obj.max_y < seg_min_y or obj.min_y > seg_max_y:
                continue
            if obj.max_z < ray_min_z or obj.min_z > ray_max_z:
                continue
            hit_t = self._ray_rect_intersection(
                origin_x=x0,
                origin_y=y0,
                dir_x=dir_x,
                dir_y=dir_y,
                min_x=obj.min_x - 1e-5,
                max_x=obj.max_x + 1e-5,
                min_y=obj.min_y - 1e-5,
                max_y=obj.max_y + 1e-5,
            )
            if hit_t is None:
                continue
            if 1e-5 < hit_t < max_distance:
                return True
        return False

    def _raycast_nearest_3d(
        self,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        dir_x: float,
        dir_y: float,
        dir_z: float,
        objects: Sequence[_SemanticObject],
        max_distance: float,
        candidate_indices: tuple[int, ...] | None = None,
    ) -> tuple[_SemanticObject, float] | None:
        nearest_obj: _SemanticObject | None = None
        nearest_t = max_distance + 1e-9

        if candidate_indices is None:
            candidate_iterable: Iterable[_SemanticObject] = objects
        else:
            candidate_iterable = (
                objects[index]
                for index in candidate_indices
                if 0 <= index < len(objects)
            )

        for obj in candidate_iterable:
            if self._should_skip_scan_object(obj.object_type):
                continue
            hit_t = self._ray_box_intersection(
                origin_x=origin_x,
                origin_y=origin_y,
                origin_z=origin_z,
                dir_x=dir_x,
                dir_y=dir_y,
                dir_z=dir_z,
                min_x=obj.min_x,
                max_x=obj.max_x,
                min_y=obj.min_y,
                max_y=obj.max_y,
                min_z=obj.min_z,
                max_z=obj.max_z,
            )
            if hit_t is None:
                continue
            if hit_t <= 1e-5 or hit_t > max_distance:
                continue
            if hit_t < nearest_t:
                nearest_t = hit_t
                nearest_obj = obj

        if nearest_obj is None:
            return None
        return (nearest_obj, nearest_t)

    def _should_skip_scan_object(self, object_type: str) -> bool:
        normalized = _normalize_type(object_type)
        if normalized in {"door_opening", "window_opening"}:
            return True
        if normalized.endswith("_group"):
            return True
        if not self.include_structural and normalized in {"floor", "wall", "ceiling"}:
            return True
        return False

    def _ray_box_intersection(
        self,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        dir_x: float,
        dir_y: float,
        dir_z: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        min_z: float,
        max_z: float,
    ) -> float | None:
        t_min = -float("inf")
        t_max = float("inf")

        if abs(dir_x) <= 1e-12:
            if origin_x < min_x or origin_x > max_x:
                return None
        else:
            tx1 = (min_x - origin_x) / dir_x
            tx2 = (max_x - origin_x) / dir_x
            t_min = max(t_min, min(tx1, tx2))
            t_max = min(t_max, max(tx1, tx2))

        if abs(dir_y) <= 1e-12:
            if origin_y < min_y or origin_y > max_y:
                return None
        else:
            ty1 = (min_y - origin_y) / dir_y
            ty2 = (max_y - origin_y) / dir_y
            t_min = max(t_min, min(ty1, ty2))
            t_max = min(t_max, max(ty1, ty2))

        if abs(dir_z) <= 1e-12:
            if origin_z < min_z or origin_z > max_z:
                return None
        else:
            tz1 = (min_z - origin_z) / dir_z
            tz2 = (max_z - origin_z) / dir_z
            t_min = max(t_min, min(tz1, tz2))
            t_max = min(t_max, max(tz1, tz2))

        if t_max < t_min:
            return None
        if t_min > 1e-9:
            return t_min
        if t_max > 1e-9:
            return t_max
        return None

    def _ray_rect_intersection(
        self,
        origin_x: float,
        origin_y: float,
        dir_x: float,
        dir_y: float,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> float | None:
        t_min = -float("inf")
        t_max = float("inf")

        if abs(dir_x) <= 1e-12:
            if origin_x < min_x or origin_x > max_x:
                return None
        else:
            tx1 = (min_x - origin_x) / dir_x
            tx2 = (max_x - origin_x) / dir_x
            t_min = max(t_min, min(tx1, tx2))
            t_max = min(t_max, max(tx1, tx2))

        if abs(dir_y) <= 1e-12:
            if origin_y < min_y or origin_y > max_y:
                return None
        else:
            ty1 = (min_y - origin_y) / dir_y
            ty2 = (max_y - origin_y) / dir_y
            t_min = max(t_min, min(ty1, ty2))
            t_max = min(t_max, max(ty1, ty2))

        if t_max < t_min:
            return None

        if t_min > 1e-9:
            return t_min
        if t_max > 1e-9:
            return t_max
        return None
