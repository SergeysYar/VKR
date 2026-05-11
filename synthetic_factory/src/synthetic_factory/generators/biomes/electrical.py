from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, sqrt
from random import Random
from typing import Mapping

from ...parametric.electrical_primitives import (
    create_backup_battery,
    create_cable_bundle,
    create_ceiling_tray_support,
    create_cooling_unit,
    create_electrical_cabinet,
    create_electrical_cable_tray,
    create_floor_cable_entry,
    create_junction_box,
    create_small_control_panel,
    create_transformer_unit,
)
from ...parametric.primitives import create_box
from .shared import (
    AddObjectFn,
    BiomeRoom,
    room_area_scale,
    symmetric_positions,
    to_bool,
    to_mapping,
    to_non_negative_int,
    to_positive_float,
)

Vector3 = tuple[float, float, float]

PRIMITIVES = (
    "electrical_cabinet",
    "electrical_cable_tray",
    "cable_bundle",
    "floor_cable_entry",
    "transformer_unit",
    "backup_battery",
    "small_control_panel",
    "junction_box",
    "ceiling_tray_support",
    "cooling_unit",
)
RULES: dict[str, object] = {
    "layout": "modular cabinet rows + strict aisles + tray backbone + vertical cable drops + bundled tray trunks",
    "dominance": "linear rows and repeated modules",
    "orientation": "all cabinets aligned to one global direction",
    "patterns": [
        "single_row",
        "parallel_rows",
        "back_to_back",
        "dense_grid",
        "sparse_technical",
    ],
    "rules": [
        "ClearanceRule",
        "AccessRule",
        "CableRoutingRule",
        "AlignmentRule",
        "HeightRule",
        "CoolingAccessRule",
    ],
    "auto_fix": True,
}


@dataclass(frozen=True)
class MinSpacingRule:
    min_clearance: float

    def enforce(
        self,
        cabinet_width: float,
        cabinet_depth: float,
        cabinet_step: float,
        row_step: float,
        aisle_width: float,
        access_depth: float,
    ) -> tuple[float, float]:
        min_cabinet_step = cabinet_width + self.min_clearance
        min_row_step = cabinet_depth + 2.0 * access_depth + aisle_width + self.min_clearance * 0.25
        return (max(cabinet_step, min_cabinet_step), max(row_step, min_row_step))


@dataclass(frozen=True)
class ClearanceRule:
    min_front_clearance: float

    def enforce_access_depth(self, access_depth: float) -> float:
        return max(access_depth, self.min_front_clearance)

    def enforce_row_step(
        self,
        row_step: float,
        cabinet_depth: float,
        aisle_width: float,
    ) -> float:
        min_step = cabinet_depth + self.min_front_clearance + aisle_width * 0.5
        return max(row_step, min_step)


@dataclass(frozen=True)
class WalkwayRule:
    row_aisle_width: float
    perimeter_aisle_width: float

    def inner_limits(
        self,
        room: BiomeRoom,
        cabinet_width: float,
        cabinet_depth: float,
    ) -> tuple[float, float]:
        x_limit = room.width / 2.0 - self.perimeter_aisle_width - cabinet_width / 2.0
        y_limit = room.depth / 2.0 - self.perimeter_aisle_width - cabinet_depth / 2.0
        return (max(x_limit, cabinet_width / 2.0), max(y_limit, cabinet_depth / 2.0))


@dataclass(frozen=True)
class AccessRule:
    access_depth: float

    def access_center(
        self,
        cabinet_position: tuple[float, float],
        forward: tuple[float, float],
        cabinet_depth: float,
    ) -> tuple[float, float]:
        offset = cabinet_depth / 2.0 + self.access_depth / 2.0
        return (
            cabinet_position[0] + forward[0] * offset,
            cabinet_position[1] + forward[1] * offset,
        )

    def clamp_access_center(
        self,
        center: tuple[float, float],
        x_limit: float,
        y_limit: float,
    ) -> tuple[float, float]:
        return (
            max(-x_limit, min(x_limit, center[0])),
            max(-y_limit, min(y_limit, center[1])),
        )


@dataclass(frozen=True)
class CableRoutingRule:
    enforce_tray_routing: bool = True

    def align_drop_to_tray(self, x: float, y: float, tray_rows: list[float]) -> tuple[float, float]:
        if not self.enforce_tray_routing or not tray_rows:
            return (x, y)
        closest_row = min(tray_rows, key=lambda row_y: abs(row_y - y))
        return (x, closest_row)


@dataclass(frozen=True)
class AlignmentRule:
    grid_step: float
    strict: bool = True

    def snap(self, value: float) -> float:
        if not self.strict:
            return value
        return round(value / self.grid_step) * self.grid_step

    def snap_xy(self, x: float, y: float) -> tuple[float, float]:
        return (self.snap(x), self.snap(y))


@dataclass(frozen=True)
class HeightRule:
    tray_clearance: float

    def enforce_tray_height(
        self,
        tray_z: float,
        tray_height: float,
        cabinet_height: float,
        room_height: float,
    ) -> float:
        min_tray_center = cabinet_height + self.tray_clearance + tray_height / 2.0
        max_tray_center = max(min_tray_center, room_height - tray_height / 2.0 - 0.05)
        return max(min_tray_center, min(tray_z, max_tray_center))


@dataclass(frozen=True)
class CoolingAccessRule:
    min_cooling_clearance: float

    def filter_candidates(
        self,
        candidates: list[tuple[float, float]],
        cabinet_points: list[tuple[float, float]],
        cabinet_width: float,
        cabinet_depth: float,
        cooling_width: float,
    ) -> list[tuple[float, float]]:
        if not cabinet_points:
            return candidates

        min_distance_sq = (
            cooling_width / 2.0
            + max(cabinet_width, cabinet_depth) / 2.0
            + self.min_cooling_clearance
        ) ** 2
        allowed: list[tuple[float, float]] = []
        for x, y in candidates:
            best_sq = min((x - cx) ** 2 + (y - cy) ** 2 for cx, cy in cabinet_points)
            if best_sq + 1e-9 >= min_distance_sq:
                allowed.append((x, y))
        return allowed


@dataclass(frozen=True)
class CabinetEndpoint:
    x: float
    y: float
    forward: tuple[float, float]


def _normalize(vx: float, vy: float) -> tuple[float, float]:
    length = sqrt(vx * vx + vy * vy)
    if length <= 1e-9:
        return (0.0, 1.0)
    return (vx / length, vy / length)


def _yaw_from_forward(forward: tuple[float, float]) -> float:
    fx, fy = _normalize(forward[0], forward[1])
    return degrees(atan2(-fx, fy))


def _parse_orientation(value: str) -> tuple[float, float]:
    normalized = value.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if normalized in {"x+", "+x", "right", "positivex"}:
        return (1.0, 0.0)
    if normalized in {"x-", "-x", "left", "negativex"}:
        return (-1.0, 0.0)
    if normalized in {"y-", "-y", "back", "negativey"}:
        return (0.0, -1.0)
    return (0.0, 1.0)


def _snap(value: float, module_size: float) -> float:
    return round(value / module_size) * module_size


def _snap_positions(values: list[float], module_size: float) -> list[float]:
    seen: set[int] = set()
    snapped: list[float] = []
    for value in values:
        snapped_value = _snap(value, module_size)
        key = int(round(snapped_value * 1000.0))
        if key in seen:
            continue
        seen.add(key)
        snapped.append(snapped_value)
    return sorted(snapped)


def _stable_hash(text: str) -> int:
    value = 0
    for idx, char in enumerate(text):
        value = (value * 131 + (idx + 1) * ord(char)) & 0xFFFFFFFF
    return value


def _sample_float(value_range: tuple[float, float], rng: Random, enabled: bool) -> float:
    low, high = value_range
    if not enabled or abs(high - low) <= 1e-12:
        return (low + high) / 2.0
    return rng.uniform(low, high)


def _sample_int(value_range: tuple[int, int], rng: Random, enabled: bool) -> int:
    low, high = value_range
    if not enabled or low == high:
        return int(round((low + high) / 2.0))
    return rng.randint(low, high)


def _parse_float_range(
    value: object,
    default_min: float,
    default_max: float,
    label: str,
) -> tuple[float, float]:
    if value is None:
        return (default_min, default_max)

    if isinstance(value, Mapping):
        if "min" not in value or "max" not in value:
            raise ValueError(f"{label} mapping must contain min/max.")
        minimum = float(value["min"])
        maximum = float(value["max"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        minimum = float(value[0])
        maximum = float(value[1])
    else:
        number = float(value)
        minimum = number
        maximum = number

    if minimum <= 0.0 or maximum <= 0.0:
        raise ValueError(f"{label} values must be > 0.")
    if minimum > maximum:
        raise ValueError(f"{label} min cannot be greater than max.")
    return (minimum, maximum)


def _parse_int_range(
    value: object,
    default_min: int,
    default_max: int,
    label: str,
) -> tuple[int, int]:
    if value is None:
        return (default_min, default_max)

    if isinstance(value, Mapping):
        if "min" not in value or "max" not in value:
            raise ValueError(f"{label} mapping must contain min/max.")
        minimum = int(value["min"])
        maximum = int(value["max"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        minimum = int(value[0])
        maximum = int(value[1])
    else:
        number = int(value)
        minimum = number
        maximum = number

    if minimum < 0 or maximum < 0:
        raise ValueError(f"{label} values must be >= 0.")
    if minimum > maximum:
        raise ValueError(f"{label} min cannot be greater than max.")
    return (minimum, maximum)


PATTERN_ALIASES = {
    "single_row": "single_row",
    "single row": "single_row",
    "single-row": "single_row",
    "single": "single_row",
    "parallel_rows": "parallel_rows",
    "parallel rows": "parallel_rows",
    "parallel-rows": "parallel_rows",
    "parallel": "parallel_rows",
    "back_to_back": "back_to_back",
    "back to back": "back_to_back",
    "back-to-back": "back_to_back",
    "b2b": "back_to_back",
    "dense_grid": "dense_grid",
    "dense grid": "dense_grid",
    "dense-grid": "dense_grid",
    "dense": "dense_grid",
    "sparse_technical": "sparse_technical",
    "sparse technical": "sparse_technical",
    "sparse-technical": "sparse_technical",
    "sparse": "sparse_technical",
}


def _normalize_pattern(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return "parallel_rows"
    return PATTERN_ALIASES.get(normalized, "parallel_rows")


def populate(room: BiomeRoom, add_object: AddObjectFn, settings: Mapping[str, object]) -> None:
    cfg = to_mapping(settings.get("electrical"))
    rules_cfg = to_mapping(cfg.get("rules"))
    room_cfg = to_mapping(cfg.get("room"))
    cabinets_cfg = to_mapping(cfg.get("cabinets"))
    walkways_cfg = to_mapping(cfg.get("walkways"))
    trays_cfg = to_mapping(cfg.get("cable_trays"))
    cables_cfg = to_mapping(cfg.get("cables"))
    cooling_cfg = to_mapping(cfg.get("cooling"))
    grid_cfg = to_mapping(cfg.get("grid"))
    secondary_cfg = to_mapping(cfg.get("secondary"))

    seed = int(cfg.get("seed", 0)) + _stable_hash(room.room_id)
    rng = Random(seed)
    random_variation = to_bool(cfg.get("random_variation", True), default=True)
    auto_fix = to_bool(cfg.get("auto_fix", True), default=True)
    pattern = _normalize_pattern(str(cfg.get("pattern", "parallel_rows")))
    grid_snapping = to_bool(
        cfg.get("grid_snapping", cfg.get("snap_to_grid", auto_fix)),
        default=auto_fix,
    )
    fixed_grid_step_source = (
        cfg.get("fixed_grid_step")
        if cfg.get("fixed_grid_step") is not None
        else grid_cfg.get("step", cfg.get("module_size"))
    )
    module_size = to_positive_float(
        fixed_grid_step_source,
        0.2,
        "machinery.electrical.fixed_grid_step",
    )

    orientation = _parse_orientation(str(cfg.get("orientation", "y+")))

    cabinet_width = to_positive_float(cfg.get("cabinet_width"), 0.9, "machinery.electrical.cabinet_width")
    cabinet_depth = to_positive_float(cfg.get("cabinet_depth"), 0.6, "machinery.electrical.cabinet_depth")
    cabinet_height = to_positive_float(cfg.get("cabinet_height"), 2.2, "machinery.electrical.cabinet_height")
    door_type = str(cfg.get("cabinet_door_type", "single")).strip().lower() or "single"

    min_clearance = to_positive_float(
        rules_cfg.get("min_spacing", cfg.get("min_clearance")),
        0.35,
        "machinery.electrical.min_clearance",
    )
    access_depth = to_positive_float(
        cfg.get("access_depth"),
        0.85,
        "machinery.electrical.access_depth",
    )
    clearance_front = to_positive_float(
        rules_cfg.get("clearance_front", cfg.get("clearance_front")),
        access_depth,
        "machinery.electrical.rules.clearance_front",
    )
    tray_height_clearance = to_positive_float(
        rules_cfg.get("tray_height_clearance"),
        0.25,
        "machinery.electrical.rules.tray_height_clearance",
    )
    cooling_access_clearance = to_positive_float(
        rules_cfg.get("cooling_access_clearance"),
        0.7,
        "machinery.electrical.rules.cooling_access_clearance",
    )
    strict_alignment = to_bool(
        rules_cfg.get("strict_alignment", grid_snapping),
        default=grid_snapping,
    )
    enforce_cable_routing = to_bool(
        rules_cfg.get("enforce_cable_routing", True),
        default=True,
    )

    room_width_range = _parse_float_range(
        room_cfg.get("width", cfg.get("room_width_range", [10.0, 50.0])),
        default_min=10.0,
        default_max=50.0,
        label="machinery.electrical.room.width",
    )
    room_depth_range = _parse_float_range(
        room_cfg.get("depth", cfg.get("room_depth_range", [10.0, 60.0])),
        default_min=10.0,
        default_max=60.0,
        label="machinery.electrical.room.depth",
    )
    room_height_range = _parse_float_range(
        room_cfg.get("height", cfg.get("room_height_range", [3.0, 6.0])),
        default_min=3.0,
        default_max=6.0,
        label="machinery.electrical.room.height",
    )

    rows_range = _parse_int_range(
        cabinets_cfg.get("rows", cfg.get("rows", [1, 6])),
        default_min=1,
        default_max=6,
        label="machinery.electrical.cabinets.rows",
    )
    per_row_range = _parse_int_range(
        cabinets_cfg.get("per_row", cfg.get("cabinets_per_row", [3, 20])),
        default_min=3,
        default_max=20,
        label="machinery.electrical.cabinets.per_row",
    )
    spacing_range = _parse_float_range(
        cabinets_cfg.get("spacing", cfg.get("cabinet_step", [0.8, 1.5])),
        default_min=0.8,
        default_max=1.5,
        label="machinery.electrical.cabinets.spacing",
    )

    walkway_width_range = _parse_float_range(
        walkways_cfg.get("width", cfg.get("row_aisle_width", [0.8, 2.0])),
        default_min=0.8,
        default_max=2.0,
        label="machinery.electrical.walkways.width",
    )

    tray_elevation_range = _parse_float_range(
        trays_cfg.get("height", cfg.get("tray_elevation_range", [2.2, 3.5])),
        default_min=2.2,
        default_max=3.5,
        label="machinery.electrical.cable_trays.height",
    )
    tray_density_range = _parse_float_range(
        trays_cfg.get("density", cfg.get("tray_density_range", [1.0, 1.0])),
        default_min=1.0,
        default_max=1.0,
        label="machinery.electrical.cable_trays.density",
    )
    cable_density_range = _parse_float_range(
        cables_cfg.get("density", cfg.get("cable_density_range", [1.0, 1.0])),
        default_min=1.0,
        default_max=1.0,
        label="machinery.electrical.cables.density",
    )
    cooling_units_range = _parse_int_range(
        cooling_cfg.get("units", cfg.get("cooling_count", [0, 4])),
        default_min=0,
        default_max=4,
        label="machinery.electrical.cooling.units",
    )

    sampled_room_width = _sample_float(room_width_range, rng, random_variation)
    sampled_room_depth = _sample_float(room_depth_range, rng, random_variation)
    sampled_room_height = _sample_float(room_height_range, rng, random_variation)
    sampled_rows = max(1, _sample_int(rows_range, rng, random_variation))
    sampled_per_row = max(1, _sample_int(per_row_range, rng, random_variation))
    sampled_spacing = _sample_float(spacing_range, rng, random_variation)
    sampled_walkway_width = _sample_float(walkway_width_range, rng, random_variation)
    sampled_tray_elevation = _sample_float(tray_elevation_range, rng, random_variation)
    sampled_tray_density = _sample_float(tray_density_range, rng, random_variation)
    sampled_cable_density = _sample_float(cable_density_range, rng, random_variation)
    sampled_cooling_count = _sample_int(cooling_units_range, rng, random_variation)

    # Scale room filling with area to keep large electrical halls meaningfully populated.
    area_scale = room_area_scale(room, reference_area=260.0, min_scale=0.75, max_scale=3.4)
    sampled_rows = max(1, int(round(sampled_rows * (0.8 + 0.3 * area_scale))))
    sampled_per_row = max(1, int(round(sampled_per_row * area_scale)))
    sampled_cooling_count = max(0, int(round(sampled_cooling_count * (0.7 + 0.35 * area_scale))))
    sampled_tray_density = sampled_tray_density * (0.85 + 0.2 * area_scale)
    sampled_cable_density = sampled_cable_density * (0.85 + 0.25 * area_scale)

    optimize_bundles = to_bool(cables_cfg.get("optimize_bundles", True), default=True)
    cable_variation = float(cables_cfg.get("variation", cfg.get("cable_variation", 0.0)))
    if cable_variation < 0.0:
        raise ValueError("machinery.electrical.cables.variation must be >= 0.")
    if cable_variation <= 1e-12 and sampled_cable_density > 1.05 and random_variation:
        cable_variation = min(cabinet_width, cabinet_depth) * 0.04
    cable_variation = min(cable_variation, min(cabinet_width, cabinet_depth) * 0.25)

    if pattern == "single_row":
        sampled_rows = 1
        sampled_per_row = max(sampled_per_row, 2)
    elif pattern == "dense_grid":
        sampled_rows = max(sampled_rows, rows_range[1])
        sampled_per_row = max(sampled_per_row, per_row_range[1])
        sampled_spacing = spacing_range[0]
        sampled_walkway_width = walkway_width_range[0]
    elif pattern == "sparse_technical":
        sampled_rows = max(1, int(round(sampled_rows * 0.55)))
        sampled_per_row = max(1, int(round(sampled_per_row * 0.6)))
        sampled_spacing = min(spacing_range[1], sampled_spacing * 1.35)
        sampled_walkway_width = min(walkway_width_range[1], sampled_walkway_width * 1.35)
    elif pattern == "back_to_back":
        sampled_rows = max(2, sampled_rows)
        sampled_per_row = max(sampled_per_row, 3)

    effective_room_width = min(room.width, sampled_room_width)
    effective_room_depth = min(room.depth, sampled_room_depth)
    effective_room_height = min(room.height, sampled_room_height)
    if effective_room_height <= 0.5:
        effective_room_height = min(room.height, 0.5)

    row_aisle_width = sampled_walkway_width
    perimeter_aisle_width = to_positive_float(
        walkways_cfg.get(
            "perimeter_width",
            cfg.get("perimeter_aisle_width", max(row_aisle_width * 0.75, 0.8)),
        ),
        max(row_aisle_width * 0.75, 0.8),
        "machinery.electrical.walkways.perimeter_width",
    )
    cabinet_step = sampled_spacing
    row_step = to_positive_float(
        cfg.get("row_step"),
        cabinet_depth + row_aisle_width,
        "machinery.electrical.row_step",
    )
    row_step = max(row_step, cabinet_depth + row_aisle_width)
    if pattern == "single_row":
        row_step = max(row_step, cabinet_depth + row_aisle_width * 1.2)
    elif pattern == "back_to_back":
        row_step = min(
            row_step,
            max(cabinet_depth + min_clearance * 0.2, cabinet_depth * 1.05),
        )
    elif pattern == "dense_grid":
        row_step = min(row_step, max(cabinet_depth + row_aisle_width * 0.75, cabinet_depth * 1.1))
    elif pattern == "sparse_technical":
        row_step = max(row_step, cabinet_depth + row_aisle_width * 1.5)

    tray_mode = str(cfg.get("tray_mode", "overhead")).strip().lower()
    tray_width = to_positive_float(cfg.get("tray_width"), 0.35, "machinery.electrical.tray_width")
    tray_profile_height = to_positive_float(cfg.get("tray_height"), 0.12, "machinery.electrical.tray_height")
    tray_height_ratio = float(cfg.get("tray_height_ratio", 0.78))
    tray_height_ratio = max(0.55, min(0.93, tray_height_ratio))
    support_spacing = to_positive_float(
        cfg.get("tray_support_spacing"),
        3.0,
        "machinery.electrical.tray_support_spacing",
    )
    base_cross_tray_count = to_non_negative_int(
        cfg.get("cross_tray_count"),
        2,
        "machinery.electrical.cross_tray_count",
    )

    cable_radius = to_positive_float(cfg.get("cable_radius"), 0.03, "machinery.electrical.cable_radius")
    junction_box_size = to_positive_float(
        cfg.get("junction_box_size"),
        0.3,
        "machinery.electrical.junction_box_size",
    )
    junction_mode = str(cfg.get("junction_mode", "walls")).strip().lower()
    junction_count = to_non_negative_int(cfg.get("junction_count"), 0, "machinery.electrical.junction_count")

    cooling_width = to_positive_float(cfg.get("cooling_width"), 0.9, "machinery.electrical.cooling_width")
    cooling_height = to_positive_float(cfg.get("cooling_height"), 2.0, "machinery.electrical.cooling_height")
    cooling_height = min(cooling_height, max(effective_room_height - 0.15, 0.6))
    cooling_airflow_direction = str(cfg.get("cooling_airflow_direction", "front_to_back")).strip().lower()
    cabinet_height = min(cabinet_height, max(effective_room_height - 0.45, 1.0))

    secondary_enabled = to_bool(secondary_cfg.get("enabled", True), default=True)
    transformer_per_rows = max(
        1,
        to_non_negative_int(
            secondary_cfg.get("transformer_per_rows"),
            2,
            "machinery.electrical.secondary.transformer_per_rows",
        ),
    )
    transformer_width = to_positive_float(
        secondary_cfg.get("transformer_width"),
        max(cabinet_width * 0.9, 0.7),
        "machinery.electrical.secondary.transformer_width",
    )
    transformer_depth = to_positive_float(
        secondary_cfg.get("transformer_depth"),
        max(cabinet_depth * 1.05, 0.7),
        "machinery.electrical.secondary.transformer_depth",
    )
    transformer_height = to_positive_float(
        secondary_cfg.get("transformer_height"),
        min(max(cabinet_height * 0.65, 1.0), max(effective_room_height - 0.25, 0.9)),
        "machinery.electrical.secondary.transformer_height",
    )
    transformer_height = min(transformer_height, max(effective_room_height - 0.2, 0.6))

    floor_entry_radius = to_positive_float(
        secondary_cfg.get("floor_entry_radius"),
        max(cable_radius * 1.35, 0.02),
        "machinery.electrical.secondary.floor_entry_radius",
    )
    floor_entry_height = to_positive_float(
        secondary_cfg.get("floor_entry_height"),
        max(cabinet_height * 0.06, 0.06),
        "machinery.electrical.secondary.floor_entry_height",
    )

    battery_count = to_non_negative_int(
        secondary_cfg.get("battery_count"),
        0,
        "machinery.electrical.secondary.battery_count",
    )
    battery_width = to_positive_float(
        secondary_cfg.get("battery_width"),
        max(cabinet_width * 0.75, 0.55),
        "machinery.electrical.secondary.battery_width",
    )
    battery_depth = to_positive_float(
        secondary_cfg.get("battery_depth"),
        max(cabinet_depth * 0.8, 0.45),
        "machinery.electrical.secondary.battery_depth",
    )
    battery_height = to_positive_float(
        secondary_cfg.get("battery_height"),
        max(cabinet_height * 0.48, 0.8),
        "machinery.electrical.secondary.battery_height",
    )
    battery_height = min(battery_height, max(effective_room_height - 0.15, 0.45))

    panel_width = to_positive_float(
        secondary_cfg.get("small_panel_width"),
        max(transformer_width * 0.42, 0.35),
        "machinery.electrical.secondary.small_panel_width",
    )
    panel_height = to_positive_float(
        secondary_cfg.get("small_panel_height"),
        max(transformer_height * 0.55, 0.6),
        "machinery.electrical.secondary.small_panel_height",
    )
    panel_depth = to_positive_float(
        secondary_cfg.get("small_panel_depth"),
        max(transformer_depth * 0.12, 0.06),
        "machinery.electrical.secondary.small_panel_depth",
    )

    clearance_rule = ClearanceRule(min_front_clearance=clearance_front)
    spacing_rule = MinSpacingRule(min_clearance=min_clearance)
    walkway_rule = WalkwayRule(
        row_aisle_width=row_aisle_width,
        perimeter_aisle_width=perimeter_aisle_width,
    )
    routing_rule = CableRoutingRule(enforce_tray_routing=enforce_cable_routing)
    alignment_rule = AlignmentRule(grid_step=module_size, strict=strict_alignment)
    height_rule = HeightRule(tray_clearance=tray_height_clearance)
    cooling_access_rule = CoolingAccessRule(min_cooling_clearance=cooling_access_clearance)
    access_depth = clearance_rule.enforce_access_depth(access_depth)
    access_rule = AccessRule(access_depth=access_depth)

    if auto_fix:
        row_step = clearance_rule.enforce_row_step(
            row_step=row_step,
            cabinet_depth=cabinet_depth,
            aisle_width=row_aisle_width,
        )
        cabinet_step, row_step = spacing_rule.enforce(
            cabinet_width=cabinet_width,
            cabinet_depth=cabinet_depth,
            cabinet_step=cabinet_step,
            row_step=row_step,
            aisle_width=row_aisle_width,
            access_depth=access_depth,
        )
    if alignment_rule.strict:
        cabinet_step = max(module_size, alignment_rule.snap(cabinet_step))
        row_step = max(module_size, alignment_rule.snap(row_step))

    working_room = BiomeRoom(
        room_id=room.room_id,
        width=effective_room_width,
        depth=effective_room_depth,
        height=effective_room_height,
        biome=room.biome,
    )
    half_w = working_room.width / 2.0
    half_d = working_room.depth / 2.0
    x_limit, y_limit = walkway_rule.inner_limits(
        room=working_room,
        cabinet_width=cabinet_width,
        cabinet_depth=cabinet_depth,
    )

    max_columns_by_step = max(1, int((2.0 * x_limit) // max(cabinet_step, 0.1)) + 1)
    max_rows_by_step = max(1, int((2.0 * y_limit) // max(row_step, 0.1)) + 1)

    # Dependencies:
    # width >= rows * (cabinet_depth + walkway_width)
    # depth >= per_row * cabinet_width
    rows_dependency_denominator = max(cabinet_depth + row_aisle_width, 0.1)
    max_rows_by_width = max(1, int(working_room.width // rows_dependency_denominator))
    max_per_row_by_depth = max(1, int(working_room.depth // max(cabinet_width, 0.1)))

    cabinets_per_row = max(
        1,
        min(sampled_per_row, max_columns_by_step, max_per_row_by_depth),
    )
    row_count = max(
        1,
        min(sampled_rows, max_rows_by_step, max_rows_by_width),
    )

    if pattern == "single_row":
        row_count = 1
    elif pattern == "back_to_back":
        row_count = min(2, max_rows_by_step, max_rows_by_width)
        row_count = max(1, row_count)
    elif pattern == "dense_grid":
        cabinets_per_row = max(1, min(max_columns_by_step, max_per_row_by_depth))
        row_count = max(1, min(max_rows_by_step, max_rows_by_width))
    elif pattern == "sparse_technical":
        cabinets_per_row = max(1, int(round(cabinets_per_row * 0.65)))
        row_count = max(1, int(round(row_count * 0.6)))

    while cabinets_per_row > 1 and (cabinets_per_row - 1) * cabinet_step > 2.0 * x_limit + 1e-6:
        cabinets_per_row -= 1
    while row_count > 1 and (row_count - 1) * row_step > 2.0 * y_limit + 1e-6:
        row_count -= 1

    x_span = min(x_limit, (cabinets_per_row - 1) * cabinet_step / 2.0)
    y_span = min(y_limit, (row_count - 1) * row_step / 2.0)

    if pattern == "sparse_technical":
        x_span *= 0.72
        y_span *= 0.7
    elif pattern == "dense_grid":
        if cabinets_per_row > 1:
            x_span = min(x_limit, max(x_span, x_limit * 0.95))
        if row_count > 1:
            y_span = min(y_limit, max(y_span, y_limit * 0.92))

    x_positions = symmetric_positions(cabinets_per_row, x_span)
    if pattern == "single_row":
        row_positions = [0.0]
    elif pattern == "back_to_back" and row_count > 1:
        back_to_back_offset = min(
            y_limit,
            max(cabinet_depth * 0.58 + min_clearance * 0.25, row_step / 2.0),
        )
        row_positions = [-back_to_back_offset, back_to_back_offset]
    else:
        row_positions = symmetric_positions(row_count, y_span)

    if alignment_rule.strict:
        x_positions = _snap_positions(x_positions, module_size)
        row_positions = _snap_positions(row_positions, module_size)

    if not x_positions:
        x_positions = [0.0]
    if not row_positions:
        row_positions = [0.0]

    row_positions = sorted(row_positions)
    cabinet_points: list[tuple[float, float]] = []
    cabinet_endpoints: list[CabinetEndpoint] = []
    counters: dict[str, int] = {}

    def emit(
        object_type: str,
        mesh: object,
        position: Vector3,
        rotation: Vector3 = (0.0, 0.0, 0.0),
        align_xy: bool = True,
    ) -> None:
        final_position = position
        if align_xy and alignment_rule.strict:
            sx, sy = alignment_rule.snap_xy(position[0], position[1])
            final_position = (sx, sy, position[2])
        counters[object_type] = counters.get(object_type, 0) + 1
        add_object(
            object_type,
            counters[object_type],
            mesh,  # type: ignore[arg-type]
            final_position,
            rotation,
        )

    def emit_bundle_segment(
        object_type: str,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        radius: float,
        z: float,
        curvature: float = 0.0,
    ) -> bool:
        dx = end_xy[0] - start_xy[0]
        dy = end_xy[1] - start_xy[1]
        length = sqrt(dx * dx + dy * dy)
        if length <= 1e-6:
            return False
        yaw = degrees(atan2(dy, dx))
        emit(
            object_type,
            create_cable_bundle(radius=radius, length=length, curvature=curvature),
            ((start_xy[0] + end_xy[0]) / 2.0, (start_xy[1] + end_xy[1]) / 2.0, z),
            (0.0, 0.0, yaw),
        )
        return True

    # 1) Walkway structure: between rows + perimeter.
    floor_marker_height = 0.02
    inner_width = max(working_room.width - 2.0 * perimeter_aisle_width, module_size)
    inner_depth = max(working_room.depth - 2.0 * perimeter_aisle_width, module_size)

    if len(row_positions) > 1:
        for idx in range(len(row_positions) - 1):
            aisle_y = (row_positions[idx] + row_positions[idx + 1]) / 2.0
            emit(
                "electrical_aisle",
                create_box(width=inner_width, height=floor_marker_height, depth=row_aisle_width),
                (0.0, aisle_y, floor_marker_height / 2.0),
            )

    emit(
        "electrical_perimeter_aisle",
        create_box(width=working_room.width, height=floor_marker_height, depth=perimeter_aisle_width),
        (0.0, -half_d + perimeter_aisle_width / 2.0, floor_marker_height / 2.0),
    )
    emit(
        "electrical_perimeter_aisle",
        create_box(width=working_room.width, height=floor_marker_height, depth=perimeter_aisle_width),
        (0.0, half_d - perimeter_aisle_width / 2.0, floor_marker_height / 2.0),
    )
    emit(
        "electrical_perimeter_aisle",
        create_box(width=perimeter_aisle_width, height=floor_marker_height, depth=inner_depth),
        (-half_w + perimeter_aisle_width / 2.0, 0.0, floor_marker_height / 2.0),
    )
    emit(
        "electrical_perimeter_aisle",
        create_box(width=perimeter_aisle_width, height=floor_marker_height, depth=inner_depth),
        (half_w - perimeter_aisle_width / 2.0, 0.0, floor_marker_height / 2.0),
    )

    # 2) Cabinet rows with uniform orientation + 3) access zones.
    for row_y in row_positions:
        row_forward = orientation
        if pattern == "back_to_back" and len(row_positions) > 1 and row_y >= 0.0:
            row_forward = (-orientation[0], -orientation[1])
        row_rotation: Vector3 = (0.0, 0.0, _yaw_from_forward(row_forward))
        for x in x_positions:
            cabinet_points.append((x, row_y))
            cabinet_endpoints.append(
                CabinetEndpoint(
                    x=x,
                    y=row_y,
                    forward=row_forward,
                )
            )
            emit(
                "electrical_cabinet",
                create_electrical_cabinet(
                    width=cabinet_width,
                    depth=cabinet_depth,
                    height=cabinet_height,
                    door_type=door_type,
                ),
                (x, row_y, cabinet_height / 2.0),
                row_rotation,
            )

            access_x, access_y = access_rule.access_center(
                cabinet_position=(x, row_y),
                forward=row_forward,
                cabinet_depth=cabinet_depth,
            )
            if auto_fix:
                access_x, access_y = access_rule.clamp_access_center(
                    center=(access_x, access_y),
                    x_limit=max(half_w - perimeter_aisle_width / 2.0, 0.0),
                    y_limit=max(half_d - perimeter_aisle_width / 2.0, 0.0),
                )
            emit(
                "electrical_access_zone",
                create_box(
                    width=cabinet_width,
                    height=floor_marker_height,
                    depth=access_depth,
                ),
                (access_x, access_y, floor_marker_height / 2.0),
                row_rotation,
            )

    use_tray_density = ("density" in trays_cfg) or (cfg.get("tray_density_range") is not None)
    if use_tray_density:
        support_spacing = max(module_size, support_spacing / max(sampled_tray_density, 0.1))
        cross_tray_count = max(
            0,
            int(round(max(len(row_positions) - 1, 0) * sampled_tray_density)),
        )
    else:
        cross_tray_count = base_cross_tray_count

    # 4) Cable trays over rows + cross connections.
    if tray_mode == "underground":
        tray_z = max(tray_profile_height * 0.8, 0.06)
    else:
        if ("height" in trays_cfg) or (cfg.get("tray_elevation_range") is not None):
            tray_z_target = sampled_tray_elevation
        else:
            tray_z_target = working_room.height * tray_height_ratio
        tray_z = max(cabinet_height + 0.35, min(tray_z_target, working_room.height - 0.25))
        if auto_fix:
            tray_z = height_rule.enforce_tray_height(
                tray_z=tray_z,
                tray_height=tray_profile_height,
                cabinet_height=cabinet_height,
                room_height=working_room.height,
            )

    if x_positions:
        row_tray_length = max(
            abs(max(x_positions) - min(x_positions)) + cabinet_width + module_size,
            cabinet_width + module_size,
        )
    else:
        row_tray_length = cabinet_width + module_size

    for row_y in row_positions:
        emit(
            "cable_tray_row",
            create_electrical_cable_tray(width=tray_width, height=tray_profile_height, length=row_tray_length),
            (0.0, row_y, tray_z),
        )

        if tray_mode != "underground":
            support_count = max(1, int(row_tray_length // max(support_spacing, module_size)) + 1)
            support_span = max(row_tray_length / 2.0 - module_size, 0.0)
            for sx in symmetric_positions(support_count, support_span):
                emit(
                    "tray_support",
                    create_ceiling_tray_support(height=tray_z, spacing=tray_width * 1.35),
                    (sx, row_y, 0.0),
                )

    cross_tray_positions: list[float] = []
    if len(row_positions) > 1 and cross_tray_count > 0:
        row_center_y = (row_positions[0] + row_positions[-1]) / 2.0
        row_span_y = abs(row_positions[-1] - row_positions[0]) + tray_width
        cross_span_x = max(abs(max(x_positions)) if x_positions else 0.0, module_size)
        cross_tray_positions = symmetric_positions(cross_tray_count, cross_span_x)
        for cross_x in cross_tray_positions:
            emit(
                "cable_tray_cross",
                create_electrical_cable_tray(width=tray_width, height=tray_profile_height, length=row_span_y),
                (cross_x, row_center_y, tray_z),
                (0.0, 0.0, 90.0),
            )

    # 5) Cable network:
    # cabinet connection point -> vertical rise to tray -> tray bundles/backbone.
    if tray_z >= cabinet_height:
        drop_length = max(tray_z - cabinet_height, 0.12)
        drop_center_z = cabinet_height + drop_length / 2.0
    else:
        drop_length = max(cabinet_height - tray_z, 0.12)
        drop_center_z = tray_z + drop_length / 2.0

    tray_drop_points: dict[float, list[float]] = {}

    def register_drop(drop_x: float, drop_y: float) -> None:
        final_x = drop_x
        final_y = drop_y
        if alignment_rule.strict:
            final_x, final_y = alignment_rule.snap_xy(drop_x, drop_y)
        tray_drop_points.setdefault(final_y, []).append(final_x)

    for endpoint in cabinet_endpoints:
        connection_x = endpoint.x
        connection_y = endpoint.y
        if cable_variation > 1e-12:
            connection_x -= endpoint.forward[0] * cabinet_depth * 0.08
            connection_y -= endpoint.forward[1] * cabinet_depth * 0.08
            connection_x += rng.uniform(-cable_variation * 0.55, cable_variation * 0.55)
            connection_y += rng.uniform(-cable_variation * 0.35, cable_variation * 0.35)
            if auto_fix:
                connection_x = max(-x_limit, min(x_limit, connection_x))
                connection_y = max(-y_limit, min(y_limit, connection_y))
        drop_x, drop_y = routing_rule.align_drop_to_tray(
            connection_x,
            connection_y,
            row_positions,
        )
        drop_curvature = 0.0
        if cable_variation > 1e-12 and sampled_cable_density > 1.05:
            drop_curvature = rng.uniform(-6.0, 6.0)

        emit(
            "cable_drop",
            create_cable_bundle(radius=cable_radius, length=drop_length, curvature=drop_curvature),
            (drop_x, drop_y, drop_center_z),
            (0.0, -90.0, 0.0),
        )
        register_drop(drop_x, drop_y)

    extra_exact = len(cabinet_endpoints) * max(sampled_cable_density - 1.0, 0.0)
    extra_cable_count = int(extra_exact)
    if rng.random() < (extra_exact - extra_cable_count):
        extra_cable_count += 1

    for idx in range(extra_cable_count):
        if not cabinet_endpoints:
            break
        endpoint = cabinet_endpoints[idx % len(cabinet_endpoints)]
        if cable_variation > 1e-12:
            jitter_x = rng.uniform(-cable_variation * 1.6, cable_variation * 1.6)
            jitter_y = rng.uniform(-cable_variation * 0.9, cable_variation * 0.9)
        else:
            jitter_x = rng.uniform(-cabinet_width * 0.18, cabinet_width * 0.18)
            jitter_y = rng.uniform(-cabinet_depth * 0.18, cabinet_depth * 0.18)

        drop_x, drop_y = routing_rule.align_drop_to_tray(
            endpoint.x + jitter_x,
            endpoint.y + jitter_y,
            row_positions,
        )
        if auto_fix:
            drop_x = max(-x_limit, min(x_limit, drop_x))
            drop_y = max(-y_limit, min(y_limit, drop_y))

        extra_curvature = 0.0
        if cable_variation > 1e-12:
            extra_curvature = rng.uniform(-10.0, 10.0)
        emit(
            "cable_drop",
            create_cable_bundle(
                radius=max(cable_radius * 0.92, 0.01),
                length=drop_length,
                curvature=extra_curvature,
            ),
            (drop_x, drop_y, drop_center_z),
            (0.0, -90.0, 0.0),
        )
        register_drop(drop_x, drop_y)

    row_ranges: list[tuple[float, float, float]] = []
    row_keys = sorted(tray_drop_points.keys())

    for row_y in row_keys:
        row_xs = sorted(tray_drop_points[row_y])
        row_min_x = row_xs[0]
        row_max_x = row_xs[-1]
        row_ranges.append((row_y, row_min_x, row_max_x))

        if optimize_bundles:
            # A single row-level trunk replaces many parallel tray cables.
            if row_max_x - row_min_x <= 1e-6:
                stub_length = max(module_size * 0.8, cabinet_width * 0.4)
                row_min_x -= stub_length / 2.0
                row_max_x += stub_length / 2.0
                row_ranges[-1] = (row_y, row_min_x, row_max_x)

            row_radius = max(
                cable_radius * min(2.4, 1.0 + 0.16 * sqrt(len(row_xs))),
                cable_radius,
            )
            row_curvature = 0.0
            if cable_variation > 1e-12 and row_max_x - row_min_x > module_size:
                row_curvature = rng.uniform(-8.0, 8.0)
            emit_bundle_segment(
                "cable_bundle_row",
                (row_min_x, row_y),
                (row_max_x, row_y),
                row_radius,
                tray_z,
                row_curvature,
            )

    if len(row_ranges) > 1:
        row_centers = sorted((row_min + row_max) / 2.0 for _, row_min, row_max in row_ranges)
        mid = len(row_centers) // 2
        if len(row_centers) % 2 == 1:
            backbone_x = row_centers[mid]
        else:
            backbone_x = (row_centers[mid - 1] + row_centers[mid]) / 2.0
        if alignment_rule.strict:
            backbone_x = alignment_rule.snap(backbone_x)
        backbone_x = max(-x_limit, min(x_limit, backbone_x))

        top_y = row_ranges[-1][0]
        bottom_y = row_ranges[0][0]
        backbone_curvature = 0.0
        if cable_variation > 1e-12 and abs(top_y - bottom_y) > module_size:
            backbone_curvature = rng.uniform(-6.0, 6.0)
        emit_bundle_segment(
            "cable_bundle_backbone",
            (backbone_x, bottom_y),
            (backbone_x, top_y),
            max(cable_radius * 1.2, 0.012),
            tray_z,
            backbone_curvature,
        )

        for row_y, row_min_x, row_max_x in row_ranges:
            row_xs = tray_drop_points[row_y]
            if optimize_bundles and row_min_x - 1e-6 <= backbone_x <= row_max_x + 1e-6:
                continue

            anchor_x = min(row_xs, key=lambda value: abs(value - backbone_x))
            branch_curvature = 0.0
            if cable_variation > 1e-12 and abs(anchor_x - backbone_x) > module_size:
                branch_curvature = rng.uniform(-5.0, 5.0)
            emit_bundle_segment(
                "cable_bundle_branch",
                (anchor_x, row_y),
                (backbone_x, row_y),
                max(cable_radius * 1.05, 0.01),
                tray_z,
                branch_curvature,
            )

    # 6) Junction boxes on walls or near cabinets.
    effective_junction_count = junction_count if junction_count > 0 else max(2, len(row_positions))
    if junction_mode in {"near_cabinets", "near", "local"} and cabinet_points:
        step = max(1, len(cabinet_points) // max(effective_junction_count, 1))
        selected_points = cabinet_points[::step][:effective_junction_count]
        for x, y in selected_points:
            emit(
                "junction_box",
                create_junction_box(size=junction_box_size),
                (x, y - orientation[1] * 0.45, cabinet_height * 0.62),
            )
    else:
        wall_x = max(half_w - perimeter_aisle_width * 0.5, junction_box_size / 2.0)
        y_span = max(half_d - perimeter_aisle_width - junction_box_size * 0.6, 0.0)
        wall_positions = symmetric_positions(effective_junction_count, y_span)
        for idx, wall_y in enumerate(wall_positions):
            side = -1.0 if idx % 2 == 0 else 1.0
            emit(
                "junction_box",
                create_junction_box(size=junction_box_size),
                (side * wall_x, wall_y, cabinet_height * 0.62),
                (0.0, 0.0, 90.0 if side < 0.0 else -90.0),
            )

    # 7) Cooling units at room edges.
    if sampled_cooling_count > 0:
        edge_offset_x = max(half_w - perimeter_aisle_width * 0.55, cooling_width / 2.0)
        edge_offset_y = max(half_d - perimeter_aisle_width * 0.55, cooling_width * 0.35)
        candidates = [
            (-edge_offset_x, -edge_offset_y),
            (edge_offset_x, -edge_offset_y),
            (-edge_offset_x, edge_offset_y),
            (edge_offset_x, edge_offset_y),
            (-edge_offset_x, 0.0),
            (edge_offset_x, 0.0),
        ]
        if auto_fix:
            filtered_candidates = cooling_access_rule.filter_candidates(
                candidates=candidates,
                cabinet_points=cabinet_points,
                cabinet_width=cabinet_width,
                cabinet_depth=cabinet_depth,
                cooling_width=cooling_width,
            )
            if filtered_candidates:
                candidates = filtered_candidates

        if cabinet_points:
            candidates.sort(
                key=lambda point: min(
                    (point[0] - cab_x) ** 2 + (point[1] - cab_y) ** 2
                    for cab_x, cab_y in cabinet_points
                ),
                reverse=True,
            )

        for x, y in candidates[:sampled_cooling_count]:
            emit(
                "cooling_unit",
                create_cooling_unit(
                    width=cooling_width,
                    height=cooling_height,
                    airflow_direction=cooling_airflow_direction,
                ),
                (x, y, cooling_height / 2.0),
            )

    # 8) Secondary electrical infrastructure details.
    if secondary_enabled:
        # 8.1) Transformer units near cabinet rows.
        transformer_positions: list[tuple[float, float]] = []
        if row_positions:
            row_stride = max(1, transformer_per_rows)
            selected_rows = row_positions[::row_stride]
            if row_positions[-1] not in selected_rows:
                selected_rows.append(row_positions[-1])

            wall_limit_x = max(
                half_w - perimeter_aisle_width * 0.45 - transformer_width / 2.0,
                transformer_width / 2.0,
            )
            base_transformer_x = (max(x_positions) if x_positions else 0.0) + cabinet_width / 2.0 + transformer_width / 2.0 + module_size * 0.6
            base_transformer_x = min(max(base_transformer_x, transformer_width / 2.0), wall_limit_x)

            for idx, row_y in enumerate(selected_rows):
                side = 1.0 if idx % 2 == 0 else -1.0
                tx = side * base_transformer_x
                ty = max(-y_limit, min(y_limit, row_y))
                facing = (-1.0, 0.0) if tx >= 0.0 else (1.0, 0.0)
                emit(
                    "transformer_unit",
                    create_transformer_unit(
                        width=transformer_width,
                        depth=transformer_depth,
                        height=transformer_height,
                    ),
                    (tx, ty, transformer_height / 2.0),
                    (0.0, 0.0, _yaw_from_forward(facing)),
                )
                transformer_positions.append((tx, ty))

        # 8.2) Extra cable bundles following tray routes.
        secondary_bundle_radius = max(cable_radius * 1.18, 0.012)
        for row_y in row_positions:
            row_bundle_curvature = 0.0
            if random_variation and cable_variation > 1e-12:
                row_bundle_curvature = rng.uniform(-5.0, 5.0)
            emit(
                "cable_bundle",
                create_cable_bundle(
                    radius=secondary_bundle_radius,
                    length=max(row_tray_length * 0.94, module_size),
                    curvature=row_bundle_curvature,
                ),
                (0.0, row_y, tray_z + tray_profile_height * 0.28),
            )

        if len(row_positions) > 1 and cross_tray_positions:
            cross_bundle_length = abs(row_positions[-1] - row_positions[0]) + tray_width
            for cross_x in cross_tray_positions:
                cross_curvature = 0.0
                if random_variation and cable_variation > 1e-12:
                    cross_curvature = rng.uniform(-4.0, 4.0)
                emit(
                    "cable_bundle",
                    create_cable_bundle(
                        radius=max(secondary_bundle_radius * 0.95, 0.01),
                        length=max(cross_bundle_length, module_size),
                        curvature=cross_curvature,
                    ),
                    (cross_x, (row_positions[0] + row_positions[-1]) / 2.0, tray_z + tray_profile_height * 0.22),
                    (0.0, 0.0, 90.0),
                )

        # 8.3) Floor cable entries under cabinets.
        for cabinet_x, cabinet_y in cabinet_points:
            emit(
                "floor_cable_entry",
                create_floor_cable_entry(
                    radius=floor_entry_radius,
                    height=floor_entry_height,
                ),
                (cabinet_x, cabinet_y, floor_entry_height / 2.0),
                align_xy=False,
            )

        # 8.4) Backup batteries along walls.
        target_battery_count = battery_count if battery_count > 0 else max(2, min(6, len(row_positions) + 1))
        battery_wall_x = max(
            half_w - perimeter_aisle_width * 0.55 - battery_width / 2.0,
            battery_width / 2.0,
        )
        battery_span_y = max(half_d - perimeter_aisle_width - battery_depth, module_size)
        battery_rows = symmetric_positions(target_battery_count, battery_span_y)
        for idx, battery_y in enumerate(battery_rows):
            side = -1.0 if idx % 2 == 0 else 1.0
            emit(
                "backup_battery",
                create_backup_battery(
                    width=battery_width,
                    depth=battery_depth,
                    height=battery_height,
                ),
                (side * battery_wall_x, battery_y, battery_height / 2.0),
                (0.0, 0.0, 90.0 if side < 0.0 else -90.0),
            )

        # 8.5) Small local control panels near transformers.
        if not transformer_positions:
            fallback_x = max(
                half_w - perimeter_aisle_width * 0.6 - panel_width / 2.0,
                panel_width / 2.0,
            )
            fallback_y = row_positions[0] if row_positions else 0.0
            transformer_positions = [(fallback_x, fallback_y)]

        for tx, ty in transformer_positions:
            if tx >= 0.0:
                panel_x = tx - (transformer_width / 2.0 + panel_depth / 2.0 + module_size * 0.25)
                panel_forward = (-1.0, 0.0)
            else:
                panel_x = tx + (transformer_width / 2.0 + panel_depth / 2.0 + module_size * 0.25)
                panel_forward = (1.0, 0.0)
            panel_x = max(-half_w + panel_width / 2.0, min(half_w - panel_width / 2.0, panel_x))
            panel_y = max(-half_d + panel_depth / 2.0, min(half_d - panel_depth / 2.0, ty))
            emit(
                "small_control_panel",
                create_small_control_panel(
                    width=panel_width,
                    height=panel_height,
                    depth=panel_depth,
                ),
                (panel_x, panel_y, panel_height / 2.0),
                (0.0, 0.0, _yaw_from_forward(panel_forward)),
            )
