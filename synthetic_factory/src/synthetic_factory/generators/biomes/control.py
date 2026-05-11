from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from random import Random
from typing import Callable, Mapping

from ...parametric.control_primitives import (
    create_cable_tray,
    create_chair,
    create_console_block,
    create_console_cable_bundle,
    create_control_panel,
    create_desk,
    create_keyboard,
    create_light_panel,
    create_monitor,
    create_rack,
    create_small_screen,
    create_ups_unit,
    create_wall_display,
    create_wall_panel,
)
from ...parametric.primitives import create_box
from .shared import (
    AddObjectFn,
    BiomeRoom,
    room_area_scale,
    symmetric_positions,
    to_bool,
    to_mapping,
    to_positive_float,
)

Vector3 = tuple[float, float, float]
EmitFn = Callable[[str, object, Vector3, Vector3, float | None], None]

PRIMITIVES = (
    "desk",
    "chair",
    "monitor",
    "keyboard",
    "console_block",
    "small_screen",
    "desk_cable_bundle",
    "wall_display",
    "ups_unit",
    "control_panel",
    "rack",
    "cable_tray",
    "wall_panel",
    "light_panel",
)
RULES: dict[str, object] = {
    "layout": "room-driven grid + control panels + desk rows + monitors/chairs + racks + cable trays",
    "dominance": "horizontal (width >> height), low visual ceiling",
    "orientation": "all workstations face panel wall",
    "patterns": [
        "linear_control_room",
        "semi_circular_control",
        "clustered_workstations",
        "minimal_control",
        "high_density",
    ],
    "visibility_priority": True,
    "rules": [
        "VisibilityRule",
        "MinSpacingRule",
        "WalkwayRule",
        "AlignmentRule",
        "ErgonomicsRule",
        "HeightConstraint",
    ],
    "auto_fix": True,
}


@dataclass(frozen=True)
class HeightConstraint:
    max_height: float
    margin: float = 0.02

    def clamp_size(self, value: float, minimum: float) -> float:
        top = max(minimum, self.max_height - 2.0 * self.margin)
        return max(minimum, min(float(value), top))

    def clamp_center(self, z_center: float, object_height: float) -> float:
        half = max(object_height / 2.0, 0.0)
        low = self.margin + half
        high = self.max_height - self.margin - half
        if low > high:
            return self.max_height / 2.0
        return max(low, min(float(z_center), high))


@dataclass(frozen=True)
class MinSpacingRule:
    min_spacing: float

    def enforce_pitch(
        self,
        desk_width: float,
        desk_depth: float,
        pitch_x: float,
        pitch_y: float,
    ) -> tuple[float, float]:
        required_x = desk_width + self.min_spacing
        required_y = desk_depth + self.min_spacing
        return (max(pitch_x, required_x), max(pitch_y, required_y))

    def enforce_positions(
        self,
        positions: list[tuple[float, float]],
        desk_width: float,
        desk_depth: float,
    ) -> list[tuple[float, float]]:
        required_x = desk_width + self.min_spacing
        required_y = desk_depth + self.min_spacing
        accepted: list[tuple[float, float]] = []
        for x, y in sorted(positions, key=lambda item: (-item[1], abs(item[0]))):
            conflict = False
            for other_x, other_y in accepted:
                if abs(x - other_x) < required_x and abs(y - other_y) < required_y:
                    conflict = True
                    break
            if not conflict:
                accepted.append((x, y))
        return accepted


@dataclass(frozen=True)
class WalkwayRule:
    main_aisle_width: float
    side_aisle_width: float
    front_clearance: float

    def blocked_half(self, desk_width: float) -> float:
        return self.main_aisle_width / 2.0 + desk_width / 2.0

    def filter_main_aisle_positions(
        self,
        x_positions: list[float],
        desk_width: float,
    ) -> list[float]:
        blocked_half = self.blocked_half(desk_width)
        return [x for x in x_positions if abs(x) >= blocked_half]

    def filter_positions(
        self,
        positions: list[tuple[float, float]],
        desk_width: float,
        x_limit: float,
    ) -> list[tuple[float, float]]:
        blocked_half = self.blocked_half(desk_width)
        return [
            (x, y)
            for x, y in positions
            if abs(x) >= blocked_half - 1e-6 and abs(x) <= x_limit + 1e-6
        ]


@dataclass(frozen=True)
class VisibilityRule:
    corridor_width: float
    blocker_height_threshold: float
    max_monitor_view_angle_deg: float = 40.0

    def allows_blocker(
        self,
        blocker: _Blocker,
        desk_positions: list[tuple[float, float]],
        panel_y: float,
    ) -> bool:
        if blocker.height <= self.blocker_height_threshold:
            return True

        bx_min = blocker.x - blocker.width / 2.0
        bx_max = blocker.x + blocker.width / 2.0
        by_min = blocker.y - blocker.depth / 2.0
        by_max = blocker.y + blocker.depth / 2.0

        for dx, dy in desk_positions:
            if abs(dx - blocker.x) > (self.corridor_width / 2.0 + blocker.width / 2.0):
                continue
            sy_min = min(dy, panel_y)
            sy_max = max(dy, panel_y)
            if by_max < sy_min or by_min > sy_max:
                continue
            if bx_max < dx - self.corridor_width / 2.0 or bx_min > dx + self.corridor_width / 2.0:
                continue
            return False

        return True

    def monitor_visible_from_chair(
        self,
        chair_x: float,
        chair_y: float,
        monitor_x: float,
        monitor_y: float,
        view_direction: tuple[float, float],
    ) -> bool:
        dx = monitor_x - chair_x
        dy = monitor_y - chair_y
        forward_x, forward_y = view_direction
        longitudinal = dx * forward_x + dy * forward_y
        if longitudinal <= 0.05:
            return False
        right_x, right_y = (forward_y, -forward_x)
        lateral = abs(dx * right_x + dy * right_y)
        angle = degrees(atan2(lateral, longitudinal))
        return angle <= self.max_monitor_view_angle_deg + 1e-6


@dataclass(frozen=True)
class AlignmentRule:
    grid_step: float

    def snap(self, value: float) -> float:
        return round(value / self.grid_step) * self.grid_step

    def snap_xy(self, x: float, y: float) -> tuple[float, float]:
        return (self.snap(x), self.snap(y))

    def snap_positions(self, positions: list[tuple[float, float]]) -> list[tuple[float, float]]:
        snapped: list[tuple[float, float]] = []
        seen: set[tuple[int, int]] = set()
        for x, y in positions:
            sx, sy = self.snap_xy(x, y)
            key = (int(round(sx * 1000.0)), int(round(sy * 1000.0)))
            if key in seen:
                continue
            seen.add(key)
            snapped.append((sx, sy))
        return snapped


@dataclass(frozen=True)
class ErgonomicsRule:
    desk_height_min: float = 0.72
    desk_height_max: float = 1.15
    seat_height_min: float = 0.38
    seat_height_max: float = 0.58
    monitor_center_offset_min: float = 0.22
    monitor_center_offset_max: float = 0.65

    def clamp_desk_height(self, value: float) -> float:
        return max(self.desk_height_min, min(self.desk_height_max, value))

    def clamp_seat_level(self, value: float) -> float:
        return max(self.seat_height_min, min(self.seat_height_max, value))

    def clamp_monitor_base_height(
        self,
        desk_height: float,
        monitor_height: float,
        monitor_base_height: float,
    ) -> float:
        current_center = monitor_base_height + monitor_height * 0.65
        min_center = desk_height + self.monitor_center_offset_min
        max_center = desk_height + self.monitor_center_offset_max
        center = max(min_center, min(max_center, current_center))
        return center - monitor_height * 0.65


@dataclass(frozen=True)
class _Blocker:
    x: float
    y: float
    width: float
    depth: float
    height: float


PATTERN_ALIASES = {
    "linear": "linear_control_room",
    "linear_control_room": "linear_control_room",
    "linear control room": "linear_control_room",
    "linear-control-room": "linear_control_room",
    "semi": "semi_circular_control",
    "semicircle": "semi_circular_control",
    "semi-circle": "semi_circular_control",
    "semi_circular_control": "semi_circular_control",
    "semi circular control": "semi_circular_control",
    "semi-circular control": "semi_circular_control",
    "semi-circular-control": "semi_circular_control",
    "clustered": "clustered_workstations",
    "clustered_workstations": "clustered_workstations",
    "clustered workstations": "clustered_workstations",
    "clustered-workstations": "clustered_workstations",
    "minimal": "minimal_control",
    "minimal_control": "minimal_control",
    "minimal control": "minimal_control",
    "minimal-control": "minimal_control",
    "high_density": "high_density",
    "high density": "high_density",
    "high-density": "high_density",
    "dense": "high_density",
}


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


def _sample_string(options: tuple[str, ...], rng: Random, enabled: bool) -> str:
    if not options:
        return ""
    if not enabled or len(options) == 1:
        return options[0]
    return options[rng.randint(0, len(options) - 1)]


def _parse_string_options(
    value: object,
    default: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError(f"{label} cannot be empty.")
        return (normalized,)
    if isinstance(value, (list, tuple)):
        normalized_values = [
            str(item).strip().lower()
            for item in value
            if str(item).strip()
        ]
        if not normalized_values:
            raise ValueError(f"{label} cannot be empty.")
        return tuple(normalized_values)
    raise TypeError(f"{label} must be string or list/tuple of strings.")


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


def _normalize_pattern(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return "linear_control_room"
    return PATTERN_ALIASES.get(normalized, "linear_control_room")


def _normalize_direction(x: float, y: float) -> tuple[float, float]:
    length = sqrt(x * x + y * y)
    if length <= 1e-9:
        return (0.0, 1.0)
    return (x / length, y / length)


def _yaw_from_view_direction(view_direction: tuple[float, float]) -> float:
    vx, vy = _normalize_direction(view_direction[0], view_direction[1])
    return degrees(atan2(-vx, vy))


def populate(room: BiomeRoom, add_object: AddObjectFn, settings: Mapping[str, object]) -> None:
    cfg = to_mapping(settings.get("control"))
    room_cfg = to_mapping(cfg.get("room"))
    desks_cfg = to_mapping(cfg.get("desks"))
    monitors_cfg = to_mapping(cfg.get("monitors"))
    panels_cfg = to_mapping(cfg.get("panels"))
    racks_cfg = to_mapping(cfg.get("racks"))
    lighting_cfg = to_mapping(cfg.get("lighting"))
    secondary_cfg = to_mapping(cfg.get("secondary"))

    seed = int(cfg.get("seed", 0)) + _stable_hash(room.room_id)
    rng = Random(seed)
    random_variation = to_bool(cfg.get("random_variation", True), default=True)
    pattern = _normalize_pattern(str(cfg.get("pattern", "linear_control_room")))
    auto_fix = to_bool(cfg.get("auto_fix", True), default=True)

    wall_margin = to_positive_float(cfg.get("wall_margin"), 0.8, "machinery.control.wall_margin")
    desk_width = to_positive_float(cfg.get("desk_width"), 1.4, "machinery.control.desk_width")
    desk_depth = to_positive_float(cfg.get("desk_depth"), 0.8, "machinery.control.desk_depth")
    desk_height = to_positive_float(cfg.get("desk_height"), 0.92, "machinery.control.desk_height")
    seat_width = to_positive_float(cfg.get("seat_width"), 0.55, "machinery.control.seat_width")
    seat_depth = to_positive_float(cfg.get("seat_depth"), 0.55, "machinery.control.seat_depth")
    seat_height = to_positive_float(cfg.get("seat_height"), 0.95, "machinery.control.seat_height")
    panel_height = to_positive_float(cfg.get("panel_height"), 1.6, "machinery.control.panel_height")
    panel_depth = to_positive_float(cfg.get("panel_depth"), 0.22, "machinery.control.panel_depth")
    panel_zone_depth = to_positive_float(cfg.get("panel_zone_depth"), 2.2, "machinery.control.panel_zone_depth")
    panel_width = to_positive_float(cfg.get("panel_width"), 1.5, "machinery.control.panel_width")
    panel_tilt_angle = float(cfg.get("panel_tilt_angle", 24.0))
    if abs(panel_tilt_angle) >= 89.0:
        panel_tilt_angle = 88.0 if panel_tilt_angle >= 0.0 else -88.0
    panel_layout = str(cfg.get("panel_layout", "auto")).strip().lower()
    desk_clearance = to_positive_float(
        cfg.get("min_desk_clearance"),
        0.9,
        "machinery.control.min_desk_clearance",
    )
    main_aisle_width = to_positive_float(
        cfg.get("main_aisle_width"),
        1.35,
        "machinery.control.main_aisle_width",
    )
    side_aisle_width = to_positive_float(
        cfg.get("side_aisle_width"),
        0.9,
        "machinery.control.side_aisle_width",
    )
    front_clearance = to_positive_float(
        cfg.get("front_clearance"),
        1.0,
        "machinery.control.front_clearance",
    )
    sight_corridor_width = to_positive_float(
        cfg.get("sight_corridor_width"),
        0.9,
        "machinery.control.sight_corridor_width",
    )
    blocker_height_threshold = to_positive_float(
        cfg.get("visibility_blocker_height"),
        1.1,
        "machinery.control.visibility_blocker_height",
    )
    monitor_width = to_positive_float(cfg.get("monitor_width"), 0.62, "machinery.control.monitor_width")
    monitor_height = to_positive_float(cfg.get("monitor_height"), 0.36, "machinery.control.monitor_height")
    monitor_thickness = to_positive_float(
        cfg.get("monitor_thickness"),
        0.06,
        "machinery.control.monitor_thickness",
    )
    secondary_enabled = to_bool(secondary_cfg.get("enabled", True), default=True)
    keyboard_width = to_positive_float(
        secondary_cfg.get("keyboard_width"),
        max(desk_width * 0.36, 0.42),
        "machinery.control.secondary.keyboard_width",
    )
    keyboard_depth = to_positive_float(
        secondary_cfg.get("keyboard_depth"),
        max(desk_depth * 0.22, 0.16),
        "machinery.control.secondary.keyboard_depth",
    )
    keyboard_height = to_positive_float(
        secondary_cfg.get("keyboard_height"),
        0.03,
        "machinery.control.secondary.keyboard_height",
    )
    console_block_width = to_positive_float(
        secondary_cfg.get("console_block_width"),
        max(desk_width * 0.28, 0.3),
        "machinery.control.secondary.console_block_width",
    )
    console_block_depth = to_positive_float(
        secondary_cfg.get("console_block_depth"),
        max(desk_depth * 0.25, 0.2),
        "machinery.control.secondary.console_block_depth",
    )
    console_block_height = to_positive_float(
        secondary_cfg.get("console_block_height"),
        0.12,
        "machinery.control.secondary.console_block_height",
    )
    small_screen_width = to_positive_float(
        secondary_cfg.get("small_screen_width"),
        max(monitor_width * 0.45, 0.24),
        "machinery.control.secondary.small_screen_width",
    )
    small_screen_height = to_positive_float(
        secondary_cfg.get("small_screen_height"),
        max(monitor_height * 0.45, 0.14),
        "machinery.control.secondary.small_screen_height",
    )
    small_screen_thickness = to_positive_float(
        secondary_cfg.get("small_screen_thickness"),
        max(monitor_thickness * 0.85, 0.03),
        "machinery.control.secondary.small_screen_thickness",
    )
    small_screen_per_desk = max(0, int(secondary_cfg.get("small_screen_per_desk", 1)))
    wall_display_width = to_positive_float(
        secondary_cfg.get("wall_display_width"),
        max(panel_width * 0.95, 1.2),
        "machinery.control.secondary.wall_display_width",
    )
    wall_display_height = to_positive_float(
        secondary_cfg.get("wall_display_height"),
        max(panel_height * 0.38, 0.45),
        "machinery.control.secondary.wall_display_height",
    )
    wall_display_thickness = to_positive_float(
        secondary_cfg.get("wall_display_thickness"),
        0.04,
        "machinery.control.secondary.wall_display_thickness",
    )
    wall_display_count = max(0, int(secondary_cfg.get("wall_display_count", 0)))
    ups_width = to_positive_float(
        secondary_cfg.get("ups_width"),
        max(seat_width * 0.4, 0.22),
        "machinery.control.secondary.ups_width",
    )
    ups_depth = to_positive_float(
        secondary_cfg.get("ups_depth"),
        max(seat_depth * 0.48, 0.26),
        "machinery.control.secondary.ups_depth",
    )
    ups_height = to_positive_float(
        secondary_cfg.get("ups_height"),
        0.55,
        "machinery.control.secondary.ups_height",
    )
    cable_bundle_radius = to_positive_float(
        secondary_cfg.get("cable_bundle_radius"),
        0.016,
        "machinery.control.secondary.cable_bundle_radius",
    )
    rack_layout = str(cfg.get("rack_layout", "walls")).strip().lower()
    cable_tray_height = to_positive_float(
        cfg.get("cable_tray_height"),
        0.16,
        "machinery.control.cable_tray_height",
    )
    cable_tray_ratio = float(cfg.get("cable_tray_elevation_ratio", 0.82))
    cable_tray_ratio = max(0.55, min(0.96, cable_tray_ratio))
    ceiling_min = to_positive_float(cfg.get("ceiling_min"), 2.8, "machinery.control.ceiling_min")
    ceiling_max = to_positive_float(cfg.get("ceiling_max"), 4.2, "machinery.control.ceiling_max")
    zone_thickness = to_positive_float(cfg.get("zone_thickness"), 0.02, "machinery.control.zone_thickness")
    alignment_grid_step = to_positive_float(
        cfg.get("alignment_grid_step"),
        0.1,
        "machinery.control.alignment_grid_step",
    )
    monitor_view_angle_deg = float(cfg.get("monitor_view_angle_deg", 40.0))
    monitor_view_angle_deg = max(15.0, min(75.0, monitor_view_angle_deg))
    ergonomic_desk_height_min = to_positive_float(
        cfg.get("ergonomic_desk_height_min"),
        0.72,
        "machinery.control.ergonomic_desk_height_min",
    )
    ergonomic_desk_height_max = to_positive_float(
        cfg.get("ergonomic_desk_height_max"),
        1.15,
        "machinery.control.ergonomic_desk_height_max",
    )
    if ergonomic_desk_height_max < ergonomic_desk_height_min:
        ergonomic_desk_height_max = ergonomic_desk_height_min
    ergonomic_seat_height_min = to_positive_float(
        cfg.get("ergonomic_seat_height_min"),
        0.38,
        "machinery.control.ergonomic_seat_height_min",
    )
    ergonomic_seat_height_max = to_positive_float(
        cfg.get("ergonomic_seat_height_max"),
        0.58,
        "machinery.control.ergonomic_seat_height_max",
    )
    if ergonomic_seat_height_max < ergonomic_seat_height_min:
        ergonomic_seat_height_max = ergonomic_seat_height_min
    ergonomic_monitor_center_min = to_positive_float(
        cfg.get("ergonomic_monitor_center_min"),
        0.22,
        "machinery.control.ergonomic_monitor_center_min",
    )
    ergonomic_monitor_center_max = to_positive_float(
        cfg.get("ergonomic_monitor_center_max"),
        0.65,
        "machinery.control.ergonomic_monitor_center_max",
    )
    if ergonomic_monitor_center_max < ergonomic_monitor_center_min:
        ergonomic_monitor_center_max = ergonomic_monitor_center_min

    room_width_range = _parse_float_range(
        room_cfg.get("width", [10.0, 40.0]),
        default_min=10.0,
        default_max=40.0,
        label="machinery.control.room.width",
    )
    room_depth_range = _parse_float_range(
        room_cfg.get("depth", [8.0, 30.0]),
        default_min=8.0,
        default_max=30.0,
        label="machinery.control.room.depth",
    )
    room_height_range = _parse_float_range(
        room_cfg.get("height", [2.5, 5.0]),
        default_min=2.5,
        default_max=5.0,
        label="machinery.control.room.height",
    )

    desks_rows_range = _parse_int_range(
        desks_cfg.get("rows", [1, 5]),
        default_min=1,
        default_max=5,
        label="machinery.control.desks.rows",
    )
    desks_cols_range = _parse_int_range(
        desks_cfg.get("cols", [2, 10]),
        default_min=2,
        default_max=10,
        label="machinery.control.desks.cols",
    )
    desks_spacing_x_range = _parse_float_range(
        desks_cfg.get("spacing_x", [1.2, 2.0]),
        default_min=1.2,
        default_max=2.0,
        label="machinery.control.desks.spacing_x",
    )
    desks_spacing_y_range = _parse_float_range(
        desks_cfg.get("spacing_y", [1.5, 3.0]),
        default_min=1.5,
        default_max=3.0,
        label="machinery.control.desks.spacing_y",
    )

    monitor_per_desk_range = _parse_int_range(
        monitors_cfg.get("per_desk", [1, 4]),
        default_min=1,
        default_max=4,
        label="machinery.control.monitors.per_desk",
    )
    panel_type_options = _parse_string_options(
        panels_cfg.get("type", ["flat", "curved"]),
        default=("flat", "curved"),
        label="machinery.control.panels.type",
    )
    panel_height_range = _parse_float_range(
        panels_cfg.get("height", [1.5, 3.0]),
        default_min=1.5,
        default_max=3.0,
        label="machinery.control.panels.height",
    )
    racks_count_range = _parse_int_range(
        racks_cfg.get("count", [0, 10]),
        default_min=0,
        default_max=10,
        label="machinery.control.racks.count",
    )
    lighting_density_range = _parse_float_range(
        lighting_cfg.get("grid_density", [0.5, 2.0]),
        default_min=0.5,
        default_max=2.0,
        label="machinery.control.lighting.grid_density",
    )

    sampled_room_width = _sample_float(room_width_range, rng, random_variation)
    sampled_room_depth = _sample_float(room_depth_range, rng, random_variation)
    sampled_room_height = _sample_float(room_height_range, rng, random_variation)
    sampled_rows = _sample_int(desks_rows_range, rng, random_variation)
    sampled_cols = _sample_int(desks_cols_range, rng, random_variation)
    sampled_spacing_x = _sample_float(desks_spacing_x_range, rng, random_variation)
    sampled_spacing_y = _sample_float(desks_spacing_y_range, rng, random_variation)
    sampled_monitor_per_desk = _sample_int(monitor_per_desk_range, rng, random_variation)
    sampled_panel_type = _sample_string(panel_type_options, rng, random_variation)
    sampled_panel_height = _sample_float(panel_height_range, rng, random_variation)
    sampled_rack_count = _sample_int(racks_count_range, rng, random_variation)
    sampled_light_density = _sample_float(lighting_density_range, rng, random_variation)

    # Larger control rooms should contain more workstations and equipment.
    area_scale = room_area_scale(room, reference_area=220.0, min_scale=0.75, max_scale=3.0)
    sampled_rows = max(1, int(round(sampled_rows * (0.8 + 0.25 * area_scale))))
    sampled_cols = max(1, int(round(sampled_cols * area_scale)))
    sampled_rack_count = max(0, int(round(sampled_rack_count * (0.75 + 0.35 * area_scale))))
    sampled_monitor_per_desk = max(1, int(round(sampled_monitor_per_desk * (0.85 + 0.2 * area_scale))))
    sampled_light_density = sampled_light_density * (0.85 + 0.2 * area_scale)

    density_profile = str(cfg.get("density_profile", "medium")).strip().lower()
    density_factor_map = {"low": 0.26, "medium": 0.36, "high": 0.48}
    density_factor = density_factor_map.get(density_profile, 0.36)
    supported_panel_layouts = {"wall", "semicircle", "arc", "radial"}
    panel_layout_is_auto = panel_layout not in supported_panel_layouts
    preferred_panel_layout = "semicircle" if sampled_panel_type == "curved" else "wall"

    if pattern == "semi_circular_control":
        sampled_panel_type = "curved"
        preferred_panel_layout = "semicircle"
        sampled_rows = max(sampled_rows, 2)
        sampled_cols = max(sampled_cols, 4)
    elif pattern == "clustered_workstations":
        preferred_panel_layout = "wall"
        sampled_rows = max(sampled_rows, 2)
        sampled_cols = max(sampled_cols, 4)
    elif pattern == "minimal_control":
        preferred_panel_layout = "wall"
        sampled_rows = max(1, min(sampled_rows, 2))
        sampled_cols = max(1, min(sampled_cols, 3))
        sampled_monitor_per_desk = 1
        sampled_rack_count = min(sampled_rack_count, 2)
        sampled_light_density = min(sampled_light_density, 0.9)
        density_factor = min(density_factor, 0.18)
    elif pattern == "high_density":
        preferred_panel_layout = "wall"
        sampled_rows = max(sampled_rows, desks_rows_range[1] + 1)
        sampled_cols = max(sampled_cols, desks_cols_range[1] + 1)
        sampled_monitor_per_desk = max(sampled_monitor_per_desk, monitor_per_desk_range[1])
        sampled_rack_count = max(sampled_rack_count, max(2, racks_count_range[1] // 2))
        sampled_light_density = max(sampled_light_density, max(1.4, lighting_density_range[1] * 0.75))
        density_factor = max(density_factor, 0.72)
        main_aisle_width = max(0.7, main_aisle_width * 0.65)
        side_aisle_width = max(0.55, side_aisle_width * 0.7)
        desk_clearance = max(0.35, desk_clearance * 0.5)

    if pattern == "semi_circular_control":
        panel_layout = "semicircle"
    elif panel_layout_is_auto:
        panel_layout = preferred_panel_layout
    elif panel_layout not in supported_panel_layouts:
        panel_layout = preferred_panel_layout

    target_visual_height = min(room.height * 0.95, sampled_room_height)
    visual_ceiling = min(ceiling_max, max(ceiling_min, target_visual_height))
    visual_ceiling = min(visual_ceiling, room.height * 0.95)
    if visual_ceiling < 1.6:
        visual_ceiling = room.height * 0.95
    height_rule = HeightConstraint(max_height=visual_ceiling, margin=0.02)
    if secondary_enabled:
        console_block_height = height_rule.clamp_size(console_block_height, minimum=0.04)
        small_screen_height = height_rule.clamp_size(small_screen_height, minimum=0.08)
        wall_display_height = height_rule.clamp_size(wall_display_height, minimum=0.25)
        ups_height = height_rule.clamp_size(ups_height, minimum=0.22)

    min_spacing_rule = MinSpacingRule(min_spacing=desk_clearance)
    walkway_rule = WalkwayRule(
        main_aisle_width=main_aisle_width,
        side_aisle_width=side_aisle_width,
        front_clearance=front_clearance,
    )
    alignment_rule = AlignmentRule(grid_step=alignment_grid_step)
    ergonomics_rule = ErgonomicsRule(
        desk_height_min=ergonomic_desk_height_min,
        desk_height_max=ergonomic_desk_height_max,
        seat_height_min=ergonomic_seat_height_min,
        seat_height_max=ergonomic_seat_height_max,
        monitor_center_offset_min=ergonomic_monitor_center_min,
        monitor_center_offset_max=ergonomic_monitor_center_max,
    )
    visibility_rule = VisibilityRule(
        corridor_width=sight_corridor_width,
        blocker_height_threshold=blocker_height_threshold,
        max_monitor_view_angle_deg=monitor_view_angle_deg,
    )
    if auto_fix:
        desk_height = ergonomics_rule.clamp_desk_height(desk_height)

    counters: dict[str, int] = {}

    def emit(
        object_type: str,
        mesh: object,
        position: Vector3,
        rotation: Vector3 = (0.0, 0.0, 0.0),
        nominal_height: float | None = None,
    ) -> None:
        final_position = position
        if nominal_height is not None:
            final_position = (
                position[0],
                position[1],
                height_rule.clamp_center(position[2], nominal_height),
            )
        counters[object_type] = counters.get(object_type, 0) + 1
        add_object(
            object_type,
            counters[object_type],
            mesh,
            final_position,
            rotation,
        )

    half_w = room.width / 2.0
    half_d = room.depth / 2.0

    panel_y = half_d - wall_margin - panel_depth / 2.0
    panel_zone_start = panel_y - panel_depth / 2.0 - panel_zone_depth
    panel_zone_center = panel_zone_start + panel_zone_depth / 2.0
    operator_zone_start = -half_d + wall_margin
    operator_zone_end = panel_zone_start - front_clearance

    if operator_zone_end <= operator_zone_start + desk_depth:
        operator_zone_end = operator_zone_start + desk_depth + 0.2

    operator_zone_depth = max(operator_zone_end - operator_zone_start, desk_depth + 0.2)
    operator_zone_center = (operator_zone_start + operator_zone_end) / 2.0

    zone_width = max(room.width - 2.0 * wall_margin, 0.6)
    emit(
        "operator_zone",
        create_box(width=zone_width, height=zone_thickness, depth=operator_zone_depth),
        (0.0, operator_zone_center, zone_thickness / 2.0),
        nominal_height=zone_thickness,
    )
    emit(
        "panel_zone",
        create_box(width=zone_width, height=zone_thickness, depth=panel_zone_depth),
        (0.0, panel_zone_center, zone_thickness / 2.0),
        nominal_height=zone_thickness,
    )

    aisle_depth = operator_zone_depth
    aisle_center = operator_zone_center
    emit(
        "aisle_zone",
        create_box(width=walkway_rule.main_aisle_width, height=zone_thickness, depth=aisle_depth),
        (0.0, aisle_center, zone_thickness / 2.0),
        nominal_height=zone_thickness,
    )
    side_aisle_x = max(half_w - wall_margin - walkway_rule.side_aisle_width / 2.0, 0.0)
    if side_aisle_x > walkway_rule.side_aisle_width / 2.0 + 0.05:
        for side in (-1.0, 1.0):
            emit(
                "aisle_zone",
                create_box(
                    width=walkway_rule.side_aisle_width,
                    height=zone_thickness,
                    depth=aisle_depth,
                ),
                (side * side_aisle_x, aisle_center, zone_thickness / 2.0),
                nominal_height=zone_thickness,
            )

    panel_wall_width = max(2.2, min(zone_width * 0.9, room.width - 2.0 * wall_margin))
    panel_height = height_rule.clamp_size(sampled_panel_height, minimum=0.8)
    emit(
        "monitor_wall",
        create_wall_panel(width=panel_wall_width, height=panel_height),
        (0.0, panel_y, panel_height / 2.0 + 0.25),
        nominal_height=panel_height,
    )

    panel_anchor_points: list[tuple[float, float]] = []
    if panel_layout in {"semicircle", "arc", "radial"}:
        panel_count = max(3, int(panel_wall_width // max(panel_width * 0.9, 0.8)))
        angle_offsets = symmetric_positions(panel_count, 60.0)
        arc_radius = max(min(panel_wall_width * 0.34, panel_zone_depth * 0.78), panel_width * 0.75)
        arc_center_y = panel_y - panel_depth / 2.0 - max(arc_radius * 0.25, 0.25)
        for angle in angle_offsets:
            theta = radians(angle)
            px = arc_radius * sin(theta)
            py = arc_center_y + arc_radius * cos(theta)
            if auto_fix:
                px, py = alignment_rule.snap_xy(px, py)
            panel_anchor_points.append((px, py))
    else:
        panel_count = max(2, int(panel_wall_width // max(panel_width * 0.92, 0.8)))
        x_positions_panels = symmetric_positions(
            panel_count,
            max(panel_wall_width / 2.0 - panel_width / 2.0, 0.0),
        )
        panel_y_front = panel_y - panel_depth / 2.0 - 0.18
        for px in x_positions_panels:
            panel_x = px
            panel_y_current = panel_y_front
            if auto_fix:
                panel_x, panel_y_current = alignment_rule.snap_xy(panel_x, panel_y_current)
            panel_anchor_points.append((panel_x, panel_y_current))

    if not panel_anchor_points:
        panel_anchor_points = [(0.0, panel_y)]

    panel_center_x = sum(point[0] for point in panel_anchor_points) / len(panel_anchor_points)
    panel_center_y = sum(point[1] for point in panel_anchor_points) / len(panel_anchor_points)
    view_direction = _normalize_direction(
        panel_center_x - 0.0,
        panel_center_y - operator_zone_center,
    )
    if view_direction[1] <= 0.0:
        view_direction = (0.0, 1.0)
    station_rotation: Vector3 = (0.0, 0.0, _yaw_from_view_direction(view_direction))
    right_direction = (view_direction[1], -view_direction[0])

    for panel_x, panel_y_current in panel_anchor_points:
        emit(
            "control_panel",
            create_control_panel(width=panel_width, height=panel_height, tilt_angle=panel_tilt_angle),
            (panel_x, panel_y_current, 0.0),
            station_rotation,
            nominal_height=panel_height,
        )

    if secondary_enabled:
        effective_wall_display_count = wall_display_count
        if effective_wall_display_count <= 0:
            effective_wall_display_count = max(2, min(6, len(panel_anchor_points)))
        display_x_span = max(panel_wall_width / 2.0 - wall_display_width / 2.0 - 0.08, 0.0)
        display_positions = (
            [0.0]
            if effective_wall_display_count == 1
            else symmetric_positions(effective_wall_display_count, display_x_span)
        )
        display_center_z = min(
            visual_ceiling - wall_display_height / 2.0 - 0.08,
            panel_height + 0.35 + wall_display_height / 2.0,
        )
        display_center_y = panel_y - panel_depth / 2.0 - wall_display_thickness / 2.0 - 0.03
        for dx in display_positions:
            emit(
                "wall_display",
                create_wall_display(
                    width=wall_display_width,
                    height=wall_display_height,
                    thickness=wall_display_thickness,
                ),
                (dx, display_center_y, display_center_z),
                station_rotation,
                nominal_height=wall_display_height,
            )

    # 1) Grid step scales with room dimensions, then corrected by clearance rules.
    usable_width = room.width - 2.0 * (wall_margin + walkway_rule.side_aisle_width)
    usable_depth = max(operator_zone_end - operator_zone_start, desk_depth + 0.25)
    control_width_limit = max(
        min(sampled_room_width, room.width) - 2.0 * (wall_margin + walkway_rule.side_aisle_width),
        desk_width + desk_clearance,
    )
    control_depth_limit = max(
        min(sampled_room_depth, room.depth) - 2.0 * wall_margin,
        desk_depth + desk_clearance,
    )
    usable_width = max(min(usable_width, control_width_limit), desk_width + desk_clearance)
    usable_depth = max(min(usable_depth, control_depth_limit), desk_depth + desk_clearance)

    grid_step_x = sampled_spacing_x
    grid_step_y = sampled_spacing_y

    if auto_fix:
        grid_step_x, grid_step_y = min_spacing_rule.enforce_pitch(
            desk_width=desk_width,
            desk_depth=desk_depth,
            pitch_x=grid_step_x,
            pitch_y=grid_step_y,
        )

    max_cols_by_room = max(1, int(usable_width // max(grid_step_x, 0.1)))
    max_rows_by_room = max(1, int(usable_depth // max(grid_step_y, 0.1)))
    min_cols = 2 if max_cols_by_room >= 2 else 1
    cols = max(min_cols, min(sampled_cols, max_cols_by_room))
    rows = max(1, min(sampled_rows, max_rows_by_room))
    if pattern == "high_density":
        cols = max(min_cols, max_cols_by_room)
        rows = max(1, max_rows_by_room)

    # Dependencies:
    # room.width >= cols * spacing_x
    # room.depth >= rows * spacing_y
    while cols > 1 and cols * grid_step_x > usable_width + 1e-9:
        cols -= 1
    while rows > 1 and rows * grid_step_y > usable_depth + 1e-9:
        rows -= 1

    x_limit = max(half_w - wall_margin - desk_width / 2.0, desk_width / 2.0)
    front_row_y = operator_zone_end - desk_depth / 2.0
    min_row_y = operator_zone_start + desk_depth / 2.0
    blocked_half = walkway_rule.blocked_half(desk_width)

    workstation_positions: list[tuple[float, float]] = []
    seen_positions: set[tuple[int, int]] = set()

    def append_workstation(x: float, y: float) -> None:
        if abs(x) > x_limit + 1e-6:
            return
        if y < min_row_y - 1e-6 or y > front_row_y + 1e-6:
            return
        if auto_fix and abs(x) < blocked_half - 1e-6:
            return
        key = (int(round(x * 1000.0)), int(round(y * 1000.0)))
        if key in seen_positions:
            return
        seen_positions.add(key)
        workstation_positions.append((x, y))

    if pattern in {"linear_control_room", "high_density"}:
        x_span = min((cols - 1) * grid_step_x / 2.0, x_limit)
        x_positions = symmetric_positions(cols, x_span)
        if auto_fix:
            x_positions = walkway_rule.filter_main_aisle_positions(
                x_positions=x_positions,
                desk_width=desk_width,
            )

        if not x_positions:
            fallback_x = max(main_aisle_width / 2.0 + desk_width / 2.0 + 0.15, desk_width / 2.0)
            fallback_x = min(fallback_x, x_limit)
            x_positions = [-fallback_x, fallback_x] if fallback_x > 0.05 else [0.0]

        y_positions = [front_row_y - idx * grid_step_y for idx in range(rows)]
        y_positions = [y for y in y_positions if y >= min_row_y - 1e-6]
        if not y_positions:
            y_positions = [operator_zone_center]

        for y in y_positions:
            for x in x_positions:
                append_workstation(x, y)

    elif pattern == "semi_circular_control":
        row_count = max(1, rows)
        arc_center_y = panel_y - panel_depth / 2.0 - max(panel_zone_depth * 0.35, 0.6)
        radius_start = max(front_clearance + desk_depth * 0.85, grid_step_y * 0.8)
        radius_limit = max(arc_center_y - min_row_y, radius_start + 0.2)
        radius_step = (radius_limit - radius_start) / max(row_count - 1, 1)
        radius_step = max(radius_step, max(grid_step_y * 0.55, 0.28))

        for row_idx in range(row_count):
            radius = radius_start + row_idx * radius_step
            if radius <= 0.1:
                continue
            ratio = min(0.995, max((x_limit - 0.05) / radius, 0.0))
            angle_half = max(24.0, min(74.0, degrees(asin(ratio)))) if ratio > 0.0 else 24.0
            row_cols = max(2, cols + row_idx)
            row_angles = symmetric_positions(row_cols, angle_half)
            for angle in row_angles:
                theta = radians(angle)
                x = radius * sin(theta)
                y = arc_center_y - radius * cos(theta)
                append_workstation(x, y)

    elif pattern == "clustered_workstations":
        target_desks = max(4, rows * cols)
        cluster_count = max(2, min(4, int(round(target_desks / 5.0))))
        cluster_x_span = min(x_limit * 0.72, max(x_limit - 0.2, 0.0))
        x_centers = symmetric_positions(2, cluster_x_span) if cluster_count > 1 else [0.0]
        y_near = operator_zone_start + operator_zone_depth * 0.4
        y_far = operator_zone_start + operator_zone_depth * 0.72
        left_center = x_centers[0] if x_centers else -x_limit * 0.55
        right_center = x_centers[-1] if len(x_centers) > 1 else x_limit * 0.55
        candidate_centers = [
            (left_center, y_near),
            (right_center, y_near),
            (left_center, y_far),
            (right_center, y_far),
        ]
        cluster_centers = candidate_centers[:cluster_count]

        if not cluster_centers:
            cluster_centers = [(-x_limit * 0.55, operator_zone_center), (x_limit * 0.55, operator_zone_center)]

        island_dx = max((desk_width + desk_clearance) * 0.48, 0.45)
        island_dy = max((desk_depth + desk_clearance) * 0.48, 0.45)
        island_offsets = [
            (0.0, 0.0),
            (-island_dx, -island_dy),
            (island_dx, -island_dy),
            (-island_dx, island_dy),
            (island_dx, island_dy),
            (0.0, -island_dy * 1.3),
        ]

        per_cluster = max(2, target_desks // len(cluster_centers))
        for idx, (cx, cy) in enumerate(cluster_centers):
            target_local = per_cluster + (1 if idx < target_desks % len(cluster_centers) else 0)
            local_limit = max(2, min(target_local, len(island_offsets)))
            for ox, oy in island_offsets[:local_limit]:
                append_workstation(cx + ox, cy + oy)

    else:  # minimal_control
        sparse_x = min(max(blocked_half + 0.4, desk_width * 0.85), max(x_limit * 0.72, desk_width / 2.0))
        first_row_y = operator_zone_start + operator_zone_depth * 0.58
        append_workstation(-sparse_x, first_row_y)
        append_workstation(sparse_x, first_row_y)
        if rows > 1 and cols > 2:
            second_row_y = max(min_row_y, first_row_y - max(grid_step_y * 0.95, 1.2))
            append_workstation(-sparse_x * 0.85, second_row_y)
            append_workstation(sparse_x * 0.85, second_row_y)

    if pattern == "minimal_control" and len(workstation_positions) > 2:
        workstation_positions = sorted(
            workstation_positions,
            key=lambda item: (-item[1], -abs(item[0])),
        )[:2]

    if auto_fix:
        workstation_positions = alignment_rule.snap_positions(workstation_positions)
        workstation_positions = walkway_rule.filter_positions(
            positions=workstation_positions,
            desk_width=desk_width,
            x_limit=x_limit,
        )
        workstation_positions = min_spacing_rule.enforce_positions(
            positions=workstation_positions,
            desk_width=desk_width,
            desk_depth=desk_depth,
        )

    if pattern == "high_density":
        dense_target = max(6, len(workstation_positions))
        if len(workstation_positions) < dense_target:
            extra_cols = 3 if x_limit > desk_width * 1.4 else 2
            x_span = min(x_limit * 0.8, max(desk_width * 0.9, 1.0))
            x_candidates = symmetric_positions(extra_cols, x_span)
            y_step = max(desk_depth + max(desk_clearance * 0.45, 0.25), 0.9)
            y_candidates: list[float] = []
            y_value = front_row_y
            while y_value >= min_row_y - 1e-6 and len(y_candidates) < 5:
                y_candidates.append(y_value)
                y_value -= y_step

            merged_positions = list(workstation_positions)
            for y in y_candidates:
                for x in x_candidates:
                    if abs(x) <= x_limit + 1e-6:
                        merged_positions.append((x, y))

            if auto_fix:
                merged_positions = alignment_rule.snap_positions(merged_positions)
                merged_positions = [
                    (x, y)
                    for x, y in merged_positions
                    if abs(x) <= x_limit + 1e-6 and min_row_y - 1e-6 <= y <= front_row_y + 1e-6
                ]

            workstation_positions = sorted(
                merged_positions,
                key=lambda item: (-item[1], abs(item[0])),
            )[:dense_target]

    if not workstation_positions:
        fallback_x = min(max(blocked_half + 0.25, desk_width / 2.0), x_limit)
        fallback_y = min(max(operator_zone_center, min_row_y), front_row_y)
        if fallback_x > 0.05:
            workstation_positions = [(-fallback_x, fallback_y), (fallback_x, fallback_y)]
        else:
            workstation_positions = [(0.0, fallback_y)]

    if auto_fix and pattern not in {"high_density", "clustered_workstations"}:
        area = max(room.width * room.depth, 1.0)
        max_desks_by_density = max(
            1,
            int(area * density_factor / max(desk_width * desk_depth * 2.2, 0.2)),
        )
        if len(workstation_positions) > max_desks_by_density:
            ranked_positions = sorted(
                workstation_positions,
                key=lambda item: (-item[1], abs(item[0])),
            )
            workstation_positions = ranked_positions[:max_desks_by_density]

    seat_back_offset = desk_depth / 2.0 + seat_depth / 2.0 + 0.2
    chair_seat_level = max(seat_height * 0.52, min(seat_width * 0.85, seat_height * 0.75), 0.35)
    if auto_fix:
        chair_seat_level = ergonomics_rule.clamp_seat_level(chair_seat_level)
    chair_back_height = max(seat_height - chair_seat_level, 0.35)
    monitor_nominal_height = max(monitor_height * 1.2, 0.2)
    monitor_z_base = desk_height
    if auto_fix:
        monitor_z_base = ergonomics_rule.clamp_monitor_base_height(
            desk_height=desk_height,
            monitor_height=monitor_height,
            monitor_base_height=monitor_z_base,
        )
    monitor_forward_offset = max(desk_depth * 0.18, 0.08)
    monitor_count = max(1, sampled_monitor_per_desk)
    monitor_x_max = max(desk_width / 2.0 - monitor_width / 2.0 - 0.03, 0.0)
    monitor_span_half = min(
        monitor_x_max,
        max((monitor_count - 1) * (monitor_width * 0.58), 0.0),
    )
    monitor_offsets = (
        [0.0] if monitor_count == 1 else symmetric_positions(monitor_count, monitor_span_half)
    )
    desk_positions: list[tuple[float, float]] = []
    for x, y in workstation_positions:
        desk_positions.append((x, y))
        emit(
            "console",
            create_desk(width=desk_width, depth=desk_depth, height=desk_height),
            (x, y, 0.0),
            station_rotation,
            nominal_height=desk_height,
        )
        if secondary_enabled:
            keyboard_x = x + view_direction[0] * (desk_depth * 0.08)
            keyboard_y = y + view_direction[1] * (desk_depth * 0.08)
            emit(
                "keyboard",
                create_keyboard(
                    width=keyboard_width,
                    depth=keyboard_depth,
                    height=keyboard_height,
                ),
                (keyboard_x, keyboard_y, desk_height + keyboard_height / 2.0 + 0.005),
                station_rotation,
                nominal_height=keyboard_height,
            )

            console_lateral = max(min(desk_width * 0.22, desk_width / 2.0 - console_block_width / 2.0), 0.0)
            console_block_x = x - right_direction[0] * console_lateral
            console_block_y = y - right_direction[1] * console_lateral
            console_block_y += view_direction[1] * desk_depth * 0.05
            console_block_x += view_direction[0] * desk_depth * 0.05
            emit(
                "console_block",
                create_console_block(
                    width=console_block_width,
                    depth=console_block_depth,
                    height=console_block_height,
                ),
                (console_block_x, console_block_y, desk_height + console_block_height / 2.0 + 0.015),
                station_rotation,
                nominal_height=console_block_height,
            )

            cable_drop_len = max(desk_height * 0.72, 0.38)
            cable_anchor_side = 1.0 if x >= 0.0 else -1.0
            cable_lateral = max(min(desk_width * 0.24, desk_width / 2.0 - 0.05), 0.0)
            cable_x = x + right_direction[0] * cable_anchor_side * cable_lateral
            cable_y = y + right_direction[1] * cable_anchor_side * cable_lateral
            cable_y -= view_direction[1] * desk_depth * 0.18
            cable_x -= view_direction[0] * desk_depth * 0.18
            emit(
                "desk_cable_bundle",
                create_console_cable_bundle(length=cable_drop_len, radius=cable_bundle_radius),
                (cable_x, cable_y, cable_drop_len / 2.0),
                (0.0, -90.0, 0.0),
                nominal_height=cable_drop_len,
            )

            ups_side = 1.0 if x >= 0.0 else -1.0
            ups_offset = desk_width / 2.0 + ups_width / 2.0 + 0.08
            ups_x = x + right_direction[0] * ups_side * ups_offset
            ups_y = y + right_direction[1] * ups_side * ups_offset
            ups_x -= view_direction[0] * desk_depth * 0.14
            ups_y -= view_direction[1] * desk_depth * 0.14
            ups_x = max(-half_w + wall_margin + ups_width / 2.0, min(half_w - wall_margin - ups_width / 2.0, ups_x))
            ups_y = max(-half_d + wall_margin + ups_depth / 2.0, min(half_d - wall_margin - ups_depth / 2.0, ups_y))
            emit(
                "ups_unit",
                create_ups_unit(width=ups_width, depth=ups_depth, height=ups_height),
                (ups_x, ups_y, ups_height / 2.0),
                station_rotation,
                nominal_height=ups_height,
            )

        seat_x = x - view_direction[0] * seat_back_offset
        seat_y = y - view_direction[1] * seat_back_offset
        desk_monitor_offsets = list(monitor_offsets)
        if auto_fix:
            desk_monitor_offsets = [
                monitor_dx
                for monitor_dx in desk_monitor_offsets
                if visibility_rule.monitor_visible_from_chair(
                    chair_x=seat_x,
                    chair_y=seat_y,
                    monitor_x=x + view_direction[0] * monitor_forward_offset + right_direction[0] * monitor_dx,
                    monitor_y=y + view_direction[1] * monitor_forward_offset + right_direction[1] * monitor_dx,
                    view_direction=view_direction,
                )
            ]
        if not desk_monitor_offsets:
            desk_monitor_offsets = [0.0]

        emitted_monitor_positions: list[tuple[float, float]] = []
        for monitor_dx in desk_monitor_offsets:
            monitor_x = x + view_direction[0] * monitor_forward_offset + right_direction[0] * monitor_dx
            monitor_y = y + view_direction[1] * monitor_forward_offset + right_direction[1] * monitor_dx
            emitted_monitor_positions.append((monitor_x, monitor_y))
            emit(
                "desk_monitor",
                create_monitor(
                    width=monitor_width,
                    height=monitor_height,
                    thickness=monitor_thickness,
                ),
                (monitor_x, monitor_y, monitor_z_base),
                station_rotation,
                nominal_height=monitor_nominal_height,
            )

        if secondary_enabled and small_screen_per_desk > 0:
            small_count = max(1, small_screen_per_desk)
            small_span = min(
                max(desk_width * 0.22, 0.08),
                max(desk_width / 2.0 - small_screen_width / 2.0 - 0.02, 0.0),
            )
            small_offsets = [0.0] if small_count == 1 else symmetric_positions(small_count, small_span)
            for idx, sx in enumerate(small_offsets):
                base_x = x + right_direction[0] * sx + view_direction[0] * max(desk_depth * 0.05, 0.03)
                base_y = y + right_direction[1] * sx + view_direction[1] * max(desk_depth * 0.05, 0.03)
                if emitted_monitor_positions:
                    nearest_x, nearest_y = min(
                        emitted_monitor_positions,
                        key=lambda point: abs(point[0] - base_x) + abs(point[1] - base_y),
                    )
                    if abs(nearest_x - base_x) < small_screen_width * 0.55:
                        base_x += right_direction[0] * (0.18 if idx % 2 == 0 else -0.18)
                    if abs(nearest_y - base_y) < small_screen_height * 0.55:
                        base_y += right_direction[1] * (0.18 if idx % 2 == 0 else -0.18)
                emit(
                    "small_screen",
                    create_small_screen(
                        width=small_screen_width,
                        height=small_screen_height,
                        thickness=small_screen_thickness,
                    ),
                    (base_x, base_y, desk_height + small_screen_height * 0.12),
                    station_rotation,
                    nominal_height=max(small_screen_height * 1.2, 0.16),
                )

        if seat_y >= operator_zone_start + seat_depth / 2.0:
            emit(
                "operator_seat",
                create_chair(seat_height=chair_seat_level, back_height=chair_back_height),
                (seat_x, seat_y, 0.0),
                station_rotation,
                nominal_height=chair_seat_level + chair_back_height,
            )

    rack_width = to_positive_float(cfg.get("rack_width"), 0.52, "machinery.control.rack_width")
    rack_depth = to_positive_float(cfg.get("rack_depth"), 0.48, "machinery.control.rack_depth")
    rack_height = to_positive_float(cfg.get("rack_height"), 1.45, "machinery.control.rack_height")
    rack_height = height_rule.clamp_size(rack_height, minimum=0.7)

    rack_candidates: list[_Blocker] = []
    rack_count = max(0, sampled_rack_count)
    if rack_count > 0:
        if rack_layout in {"zone", "dedicated_zone"}:
            zone_x = -half_w + wall_margin + walkway_rule.side_aisle_width * 0.35 + rack_width / 2.0
            zone_x = min(zone_x, -main_aisle_width / 2.0 - rack_width / 2.0 - 0.05)
            zone_y_center = operator_zone_start + min(operator_zone_depth * 0.25, 1.4)
            zone_span = max((rack_count - 1) * (rack_depth + 0.2) / 2.0, 0.0)
            for offset in symmetric_positions(rack_count, zone_span):
                rack_candidates.append(
                    _Blocker(
                        x=zone_x,
                        y=zone_y_center + offset,
                        width=rack_width,
                        depth=rack_depth,
                        height=rack_height,
                    )
                )
        else:
            rack_x = max(half_w - wall_margin - rack_width / 2.0, rack_width / 2.0)
            left_count = (rack_count + 1) // 2
            right_count = rack_count // 2
            left_values = symmetric_positions(left_count, max((operator_zone_depth / 2.0) * 0.82, 0.0))
            right_values = symmetric_positions(right_count, max((operator_zone_depth / 2.0) * 0.82, 0.0))
            for y in left_values:
                rack_candidates.append(
                    _Blocker(
                        x=-rack_x,
                        y=operator_zone_center + y,
                        width=rack_width,
                        depth=rack_depth,
                        height=rack_height,
                    )
                )
            for y in right_values:
                rack_candidates.append(
                    _Blocker(
                        x=rack_x,
                        y=operator_zone_center + y,
                        width=rack_width,
                        depth=rack_depth,
                        height=rack_height,
                    )
                )

    if auto_fix:
        aligned_candidates: list[_Blocker] = []
        seen_candidates: set[tuple[int, int]] = set()
        for blocker in rack_candidates:
            ax, ay = alignment_rule.snap_xy(blocker.x, blocker.y)
            key = (int(round(ax * 1000.0)), int(round(ay * 1000.0)))
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            aligned_candidates.append(
                _Blocker(
                    x=ax,
                    y=ay,
                    width=blocker.width,
                    depth=blocker.depth,
                    height=blocker.height,
                )
            )
        rack_candidates = aligned_candidates
        rack_candidates = [
            blocker
            for blocker in rack_candidates
            if visibility_rule.allows_blocker(
                blocker=blocker,
                desk_positions=desk_positions,
                panel_y=panel_y,
            )
        ]

    rack_points: list[tuple[float, float]] = []
    for blocker in rack_candidates:
        emit(
            "control_rack",
            create_rack(height=blocker.height, width=blocker.width, depth=blocker.depth),
            (blocker.x, blocker.y, 0.0),
            nominal_height=blocker.height,
        )
        rack_points.append((blocker.x, blocker.y))

    tray_z = height_rule.clamp_center(
        max(visual_ceiling * cable_tray_ratio, desk_height + monitor_nominal_height + 0.35),
        cable_tray_height,
    )
    main_tray_y = panel_y - panel_depth / 2.0 - 0.22
    main_tray_length = max(panel_wall_width, panel_width * 2.0)
    emit(
        "cable_tray",
        create_cable_tray(length=main_tray_length, height=cable_tray_height),
        (0.0, main_tray_y, tray_z),
        nominal_height=cable_tray_height,
    )

    if not panel_anchor_points:
        panel_anchor_points = [(0.0, panel_y)]
    for px, py in panel_anchor_points:
        tray_len = abs(py - main_tray_y)
        if tray_len <= 0.14:
            continue
        emit(
            "cable_tray",
            create_cable_tray(length=tray_len, height=cable_tray_height),
            (px, (py + main_tray_y) / 2.0, tray_z),
            (0.0, 0.0, 90.0),
            nominal_height=cable_tray_height,
        )

    for rx, ry in rack_points:
        tray_len = abs(ry - main_tray_y)
        if tray_len <= 0.14:
            continue
        emit(
            "cable_tray",
            create_cable_tray(length=tray_len, height=cable_tray_height),
            (rx, (ry + main_tray_y) / 2.0, tray_z),
            (0.0, 0.0, 90.0),
            nominal_height=cable_tray_height,
        )

    ceiling_thickness = to_positive_float(
        cfg.get("ceiling_thickness"),
        0.06,
        "machinery.control.ceiling_thickness",
    )
    ceiling_thickness = height_rule.clamp_size(ceiling_thickness, minimum=0.03)
    ceiling_width = max(room.width - 2.0 * wall_margin, 0.5)
    ceiling_depth = max(room.depth - 2.0 * wall_margin, 0.5)
    emit(
        "suspended_ceiling",
        create_box(width=ceiling_width, height=ceiling_thickness, depth=ceiling_depth),
        (0.0, 0.0, visual_ceiling - ceiling_thickness / 2.0),
        nominal_height=ceiling_thickness,
    )

    light_spacing = max(1.2, 3.0 / max(sampled_light_density, 0.1))
    light_cols = max(1, int(max(ceiling_width - 0.2, 0.2) // light_spacing))
    light_rows = max(1, int(max(ceiling_depth - 0.2, 0.2) // light_spacing))
    light_cols = min(light_cols, 10)
    light_rows = min(light_rows, 10)
    light_x_span = max((light_cols - 1) * light_spacing / 2.0, 0.0)
    light_y_span = max((light_rows - 1) * light_spacing / 2.0, 0.0)
    light_x_positions = symmetric_positions(light_cols, min(light_x_span, max(ceiling_width / 2.0 - 0.3, 0.0)))
    light_y_positions = symmetric_positions(light_rows, min(light_y_span, max(ceiling_depth / 2.0 - 0.3, 0.0)))
    light_size = max(0.35, min(light_spacing * 0.42, 0.95))
    light_intensity = 320.0 * sampled_light_density
    light_thickness = max(light_size * 0.08, 0.015)
    light_z = visual_ceiling - ceiling_thickness - light_thickness / 2.0 - 0.01
    for lx in light_x_positions:
        for ly in light_y_positions:
            emit(
                "light_panel",
                create_light_panel(size=light_size, intensity=light_intensity),
                (lx, ly, light_z),
                nominal_height=light_thickness,
            )
