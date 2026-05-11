from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Mapping

from ...parametric.maintenance_primitives import (
    create_cable_loose,
    create_crane_hook,
    create_machine_part,
    create_pallet,
    create_pipe_loose,
    create_spare_part,
    create_tool_rack,
    create_toolbox,
    create_workbench,
)
from ...parametric.primitives import create_box
from .shared import AddObjectFn, BiomeRoom, symmetric_positions, to_bool, to_mapping, to_positive_float

Vector3 = tuple[float, float, float]
Aabb2D = tuple[float, float, float, float]

PRIMITIVES = (
    "maintenance_workbench",
    "maintenance_tool_rack",
    "maintenance_toolbox",
    "maintenance_machine_part",
    "maintenance_spare_part",
    "maintenance_pallet",
    "maintenance_crane_hook",
    "maintenance_cable_loose",
    "maintenance_pipe_loose",
    "box",
)
RULES: dict[str, object] = {
    "layout": "repair zones + storage zones + protected walkways",
    "patterns": [
        "clean_workshop",
        "active_repair",
        "heavy_maintenance",
        "storage_dominant",
        "chaotic_workshop",
    ],
    "steps": [
        "split room into repair/storage/walkway zones",
        "place workbench + machine_part + toolbox in repair zones",
        "place tool_rack along walls and near repair zones",
        "place pallet + spare_part sets in storage zones",
        "place crane_hook above repair zones",
        "apply seeded scatter while preserving walkability",
    ],
    "constraints": [
        "walkways must stay clear",
        "storage elements cannot overlap repair zones",
    ],
    "rules": [
        "WalkwayRule",
        "StabilityRule",
        "ToolPlacementRule",
        "PartPlacementRule",
        "CraneClearanceRule",
    ],
    "auto_fix": True,
}


@dataclass(frozen=True)
class _PatternProfile:
    density_multiplier: float
    scatter_multiplier: float
    default_order_level: float
    repair_multiplier: float
    storage_zone_bias: int
    rack_multiplier: float
    pallet_multiplier: float
    spare_multiplier: float
    machine_size_multiplier: float
    machine_complexity_multiplier: float
    large_parts_multiplier: float
    force_crane: bool
    jitter_limit: float
    yaw_jitter_deg: float
    near_rack_stride: int


PATTERN_ALIASES = {
    "clean": "clean_workshop",
    "clean_workshop": "clean_workshop",
    "clean workshop": "clean_workshop",
    "clean-workshop": "clean_workshop",
    "active": "active_repair",
    "active_repair": "active_repair",
    "active repair": "active_repair",
    "active-repair": "active_repair",
    "repair_active": "active_repair",
    "heavy": "heavy_maintenance",
    "heavy_maintenance": "heavy_maintenance",
    "heavy maintenance": "heavy_maintenance",
    "heavy-maintenance": "heavy_maintenance",
    "storage": "storage_dominant",
    "storage_dominant": "storage_dominant",
    "storage dominant": "storage_dominant",
    "storage-dominant": "storage_dominant",
    "chaotic": "chaotic_workshop",
    "chaotic_workshop": "chaotic_workshop",
    "chaotic workshop": "chaotic_workshop",
    "chaotic-workshop": "chaotic_workshop",
}

PATTERN_PROFILES: dict[str, _PatternProfile] = {
    "clean_workshop": _PatternProfile(
        density_multiplier=0.85,
        scatter_multiplier=0.35,
        default_order_level=0.9,
        repair_multiplier=0.85,
        storage_zone_bias=0,
        rack_multiplier=0.85,
        pallet_multiplier=0.8,
        spare_multiplier=0.6,
        machine_size_multiplier=0.85,
        machine_complexity_multiplier=0.8,
        large_parts_multiplier=0.5,
        force_crane=False,
        jitter_limit=0.05,
        yaw_jitter_deg=2.0,
        near_rack_stride=3,
    ),
    "active_repair": _PatternProfile(
        density_multiplier=1.15,
        scatter_multiplier=1.0,
        default_order_level=0.45,
        repair_multiplier=1.2,
        storage_zone_bias=0,
        rack_multiplier=1.25,
        pallet_multiplier=1.2,
        spare_multiplier=1.8,
        machine_size_multiplier=1.1,
        machine_complexity_multiplier=1.25,
        large_parts_multiplier=1.4,
        force_crane=False,
        jitter_limit=0.18,
        yaw_jitter_deg=6.0,
        near_rack_stride=2,
    ),
    "heavy_maintenance": _PatternProfile(
        density_multiplier=1.2,
        scatter_multiplier=0.75,
        default_order_level=0.4,
        repair_multiplier=1.1,
        storage_zone_bias=0,
        rack_multiplier=1.0,
        pallet_multiplier=1.1,
        spare_multiplier=1.2,
        machine_size_multiplier=1.55,
        machine_complexity_multiplier=1.45,
        large_parts_multiplier=2.2,
        force_crane=True,
        jitter_limit=0.14,
        yaw_jitter_deg=4.0,
        near_rack_stride=2,
    ),
    "storage_dominant": _PatternProfile(
        density_multiplier=1.0,
        scatter_multiplier=0.65,
        default_order_level=0.62,
        repair_multiplier=0.7,
        storage_zone_bias=1,
        rack_multiplier=1.85,
        pallet_multiplier=1.9,
        spare_multiplier=1.5,
        machine_size_multiplier=0.9,
        machine_complexity_multiplier=0.9,
        large_parts_multiplier=0.75,
        force_crane=False,
        jitter_limit=0.1,
        yaw_jitter_deg=3.0,
        near_rack_stride=3,
    ),
    "chaotic_workshop": _PatternProfile(
        density_multiplier=1.3,
        scatter_multiplier=1.75,
        default_order_level=0.1,
        repair_multiplier=1.35,
        storage_zone_bias=0,
        rack_multiplier=1.2,
        pallet_multiplier=1.35,
        spare_multiplier=2.4,
        machine_size_multiplier=1.25,
        machine_complexity_multiplier=1.6,
        large_parts_multiplier=1.8,
        force_crane=False,
        jitter_limit=0.24,
        yaw_jitter_deg=18.0,
        near_rack_stride=1,
    ),
}


@dataclass(frozen=True)
class MinSpacingRule:
    min_spacing: float

    def conflicts(self, first: Aabb2D, second: Aabb2D) -> bool:
        return (
            first[0] < second[1] + self.min_spacing
            and first[1] > second[0] - self.min_spacing
            and first[2] < second[3] + self.min_spacing
            and first[3] > second[2] - self.min_spacing
        )


@dataclass(frozen=True)
class WalkwayRule:
    width: float

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    def intersects(self, bounds: Aabb2D) -> bool:
        y_min = -self.half_width
        y_max = self.half_width
        return bounds[2] < y_max and bounds[3] > y_min

    def auto_fix_y(self, center_y: float, depth: float, margin: float = 0.05) -> float:
        bounds = _aabb2d(0.0, center_y, 0.0, depth)
        if not self.intersects(bounds):
            return center_y
        sign = 1.0 if center_y >= 0.0 else -1.0
        return sign * (self.half_width + depth / 2.0 + margin)


@dataclass(frozen=True)
class StabilityRule:
    floor_z: float = 0.0
    epsilon: float = 1e-4

    def stable_center(self, center_z: float, nominal_height: float, support_top: float = 0.0) -> float:
        half_h = max(nominal_height / 2.0, self.epsilon)
        minimum = max(self.floor_z, support_top) + half_h
        return max(center_z, minimum)

    def is_stable(self, center_z: float, nominal_height: float, support_top: float = 0.0) -> bool:
        half_h = max(nominal_height / 2.0, self.epsilon)
        return center_z + self.epsilon >= max(self.floor_z, support_top) + half_h


@dataclass(frozen=True)
class ToolPlacementRule:
    max_distance: float

    def is_near_repair_zone(self, x: float, y: float, repair_zone_bounds: list[Aabb2D]) -> bool:
        if not repair_zone_bounds:
            return True
        max_dist_sq = self.max_distance * self.max_distance
        for bounds in repair_zone_bounds:
            cx = (bounds[0] + bounds[1]) / 2.0
            cy = (bounds[2] + bounds[3]) / 2.0
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy <= max_dist_sq:
                return True
        return False

    def auto_fix_near_repair_zone(self, x: float, y: float, repair_zone_bounds: list[Aabb2D]) -> tuple[float, float]:
        if self.is_near_repair_zone(x, y, repair_zone_bounds):
            return (x, y)
        if not repair_zone_bounds:
            return (x, y)
        target = min(
            repair_zone_bounds,
            key=lambda bounds: (x - (bounds[0] + bounds[1]) / 2.0) ** 2 + (y - (bounds[2] + bounds[3]) / 2.0) ** 2,
        )
        cx = (target[0] + target[1]) / 2.0
        cy = (target[2] + target[3]) / 2.0
        dx = x - cx
        dy = y - cy
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return (cx + self.max_distance * 0.7, cy)
        scale = self.max_distance / max((dx * dx + dy * dy) ** 0.5, 1e-9)
        return (cx + dx * scale, cy + dy * scale)


@dataclass(frozen=True)
class PartPlacementRule:
    floor_tolerance: float = 0.08
    pallet_attach_tolerance: float = 0.12

    def auto_fix_large_part_support(
        self,
        center_z: float,
        nominal_height: float,
        x: float,
        y: float,
        pallet_positions: list[tuple[float, float]],
        pallet_top_z: float,
        pallet_radius_x: float,
        pallet_radius_y: float,
        stability_rule: StabilityRule,
    ) -> tuple[float, float]:
        half_h = max(nominal_height / 2.0, 0.02)
        if center_z - half_h <= self.floor_tolerance:
            fixed = stability_rule.stable_center(center_z, nominal_height, support_top=0.0)
            return (fixed, 0.0)

        for px, py in pallet_positions:
            if abs(x - px) <= pallet_radius_x and abs(y - py) <= pallet_radius_y:
                fixed = stability_rule.stable_center(center_z, nominal_height, support_top=pallet_top_z)
                return (fixed, pallet_top_z)

        fixed = stability_rule.stable_center(center_z, nominal_height, support_top=0.0)
        return (fixed, 0.0)


@dataclass(frozen=True)
class CraneClearanceRule:
    clearance_radius: float

    def blocked(self, x: float, y: float, blocked_bounds: list[Aabb2D]) -> bool:
        for bounds in blocked_bounds:
            nearest_x = _clamp(x, bounds[0], bounds[1])
            nearest_y = _clamp(y, bounds[2], bounds[3])
            dx = x - nearest_x
            dy = y - nearest_y
            if dx * dx + dy * dy <= self.clearance_radius * self.clearance_radius:
                return True
        return False

    def auto_fix_xy(
        self,
        base_x: float,
        base_y: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        blocked_bounds: list[Aabb2D],
    ) -> tuple[float, float]:
        if not self.blocked(base_x, base_y, blocked_bounds):
            return (base_x, base_y)
        x_candidates = [
            _clamp(base_x, x_min, x_max),
            _clamp(base_x - self.clearance_radius * 1.2, x_min, x_max),
            _clamp(base_x + self.clearance_radius * 1.2, x_min, x_max),
        ]
        y_candidates = [
            _clamp(base_y, y_min, y_max),
            _clamp(0.0, y_min, y_max),
            _clamp(self.clearance_radius * 1.6, y_min, y_max),
            _clamp(-self.clearance_radius * 1.6, y_min, y_max),
        ]
        best_x = x_candidates[0]
        best_y = y_candidates[0]
        best_score = float("inf")
        for cx in x_candidates:
            for cy in y_candidates:
                score = (cx - base_x) * (cx - base_x) + (cy - base_y) * (cy - base_y)
                if self.blocked(cx, cy, blocked_bounds):
                    score += 1e6
                if score < best_score:
                    best_x = cx
                    best_y = cy
                    best_score = score
        return (best_x, best_y)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _stable_hash(text: str) -> int:
    value = 0
    for index, char in enumerate(text):
        value = (value * 131 + (index + 1) * ord(char)) & 0xFFFFFFFF
    return value


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
        low = float(value["min"])
        high = float(value["max"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        low = float(value[0])
        high = float(value[1])
    else:
        low = float(value)
        high = float(value)
    if low < 0.0 or high < 0.0:
        raise ValueError(f"{label} values must be >= 0.")
    if low > high:
        raise ValueError(f"{label} min cannot be greater than max.")
    return (low, high)


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
        low = int(value["min"])
        high = int(value["max"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        low = int(value[0])
        high = int(value[1])
    else:
        low = int(value)
        high = int(value)
    if low < 0 or high < 0:
        raise ValueError(f"{label} values must be >= 0.")
    if low > high:
        raise ValueError(f"{label} min cannot be greater than max.")
    return (low, high)


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


def _sample_bool(value: object, rng: Random, enabled: bool, default: bool, label: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        options = [to_bool(item, default=default) for item in value]
        if not enabled or len(options) == 1:
            return options[0]
        return options[rng.randint(0, len(options) - 1)]
    if isinstance(value, Mapping):
        if "enabled" in value:
            return to_bool(value.get("enabled"), default=default)
        raise ValueError(f"{label} mapping must contain 'enabled' key.")
    return to_bool(value, default=default)


def _scale_int(value: int, multiplier: float, minimum: int = 0, maximum: int | None = None) -> int:
    scaled = int(round(value * multiplier))
    scaled = max(minimum, scaled)
    if maximum is not None:
        scaled = min(maximum, scaled)
    return scaled


def _normalize_pattern(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return "active_repair"
    return PATTERN_ALIASES.get(normalized, "active_repair")


def _aabb2d(center_x: float, center_y: float, width: float, depth: float) -> Aabb2D:
    return (
        center_x - width / 2.0,
        center_x + width / 2.0,
        center_y - depth / 2.0,
        center_y + depth / 2.0,
    )


def _intersects(a: Aabb2D, b: Aabb2D) -> bool:
    return a[0] < b[1] and a[1] > b[0] and a[2] < b[3] and a[3] > b[2]


def populate(room: BiomeRoom, add_object: AddObjectFn, settings: Mapping[str, object]) -> None:
    cfg = to_mapping(settings.get("maintenance"))
    zones_cfg = to_mapping(cfg.get("zones"))
    tools_cfg = to_mapping(cfg.get("tools"))
    parts_cfg = to_mapping(cfg.get("parts"))
    organization_cfg = to_mapping(cfg.get("organization"))
    workbench_cfg = to_mapping(cfg.get("workbench"))
    rack_cfg = to_mapping(cfg.get("tool_rack"))
    pallet_cfg = to_mapping(cfg.get("pallet"))
    spare_cfg = to_mapping(cfg.get("spare_part"))
    machine_cfg = to_mapping(cfg.get("machine_part"))
    crane_cfg = to_mapping(cfg.get("crane_hook"))
    crane_global_cfg = to_mapping(cfg.get("crane"))
    rules_cfg = to_mapping(cfg.get("rules"))
    repair_scenario_cfg = to_mapping(cfg.get("repair_scenario"))

    scenario_value = cfg.get("scenario")
    scenario_name = ""
    scenario_flag: bool | None = None
    if isinstance(scenario_value, Mapping):
        scenario_mapping = to_mapping(scenario_value)
        if not repair_scenario_cfg:
            repair_scenario_cfg = scenario_mapping
        scenario_name = str(
            scenario_mapping.get("name", scenario_mapping.get("type", scenario_mapping.get("mode", "")))
        ).strip().lower()
        if "enabled" in scenario_mapping:
            scenario_flag = to_bool(scenario_mapping.get("enabled"), default=False)
    elif isinstance(scenario_value, str):
        scenario_name = scenario_value.strip().lower()
    elif scenario_value is not None:
        scenario_flag = to_bool(scenario_value, default=False)

    repair_scenario_name = str(
        repair_scenario_cfg.get("name", repair_scenario_cfg.get("type", scenario_name))
    ).strip().lower()
    repair_scenario_enabled = to_bool(
        repair_scenario_cfg.get("enabled"),
        default=bool(scenario_flag) or scenario_name == "repair" or repair_scenario_name == "repair",
    )

    seed = int(cfg.get("seed", 0)) + _stable_hash(room.room_id)
    random_variation = to_bool(cfg.get("random_variation", True), default=True)
    rng = Random(seed)
    pattern = _normalize_pattern(str(cfg.get("pattern", "active_repair")))
    pattern_profile = PATTERN_PROFILES[pattern]

    density = _sample_float(
        _parse_float_range(cfg.get("density"), 0.9, 1.35, "machinery.maintenance.density"),
        rng,
        random_variation,
    )
    density = _clamp(density * pattern_profile.density_multiplier, 0.35, 3.0)
    scatter_strength = _sample_float(
        _parse_float_range(
            cfg.get("scatter", cfg.get("clutter")),
            0.25,
            0.9,
            "machinery.maintenance.scatter",
        ),
        rng,
        random_variation,
    )
    order_range = _parse_float_range(
        organization_cfg.get("order_level"),
        max(pattern_profile.default_order_level - 0.2, 0.0),
        min(pattern_profile.default_order_level + 0.2, 1.0),
        "machinery.maintenance.organization.order_level",
    )
    order_level = _clamp(_sample_float(order_range, rng, random_variation), 0.0, 1.0)
    chaos_factor = 1.0 - order_level
    scatter_strength = _clamp(
        scatter_strength * pattern_profile.scatter_multiplier * (0.45 + chaos_factor * 1.35),
        0.04,
        3.0,
    )

    tool_density = _sample_float(
        _parse_float_range(
            tools_cfg.get("density"),
            0.5,
            1.5,
            "machinery.maintenance.tools.density",
        ),
        rng,
        random_variation,
    )
    tool_density = _clamp(
        tool_density * (0.65 + pattern_profile.rack_multiplier * 0.35),
        0.2,
        4.0,
    )

    edge_margin = to_positive_float(cfg.get("edge_margin"), 0.8, "machinery.maintenance.edge_margin")
    walkway_width_cfg = to_positive_float(
        cfg.get("min_walkway_width"),
        1.2,
        "machinery.maintenance.min_walkway_width",
    )
    min_clearance = to_positive_float(
        rules_cfg.get("min_clearance"),
        0.6,
        "machinery.maintenance.rules.min_clearance",
    )
    access_depth = to_positive_float(
        rules_cfg.get("access_depth"),
        1.0,
        "machinery.maintenance.rules.access_depth",
    )

    workbench_width = to_positive_float(workbench_cfg.get("width"), 1.8, "machinery.maintenance.workbench.width")
    workbench_depth = to_positive_float(workbench_cfg.get("depth"), 0.9, "machinery.maintenance.workbench.depth")
    workbench_height = to_positive_float(workbench_cfg.get("height"), 0.95, "machinery.maintenance.workbench.height")

    rack_width = to_positive_float(rack_cfg.get("width"), 1.0, "machinery.maintenance.tool_rack.width")
    rack_height = to_positive_float(rack_cfg.get("height"), 2.1, "machinery.maintenance.tool_rack.height")
    rack_depth_est = max(rack_width * 0.45, 0.2)

    toolbox_size_range = _parse_float_range(
        cfg.get("toolbox_size"),
        0.28,
        0.45,
        "machinery.maintenance.toolbox_size",
    )
    machine_size_range = _parse_float_range(
        machine_cfg.get("size"),
        0.32,
        0.58,
        "machinery.maintenance.machine_part.size",
    )
    machine_complexity_range = _parse_int_range(
        machine_cfg.get("complexity"),
        2,
        5,
        "machinery.maintenance.machine_part.complexity",
    )
    hook_height_range = _parse_float_range(
        crane_cfg.get("height"),
        1.0,
        1.8,
        "machinery.maintenance.crane_hook.height",
    )
    pallet_width = to_positive_float(pallet_cfg.get("width"), 1.1, "machinery.maintenance.pallet.width")
    pallet_depth = to_positive_float(pallet_cfg.get("depth"), 0.9, "machinery.maintenance.pallet.depth")
    pallet_count_range = _parse_int_range(
        pallet_cfg.get("count"),
        2,
        6,
        "machinery.maintenance.pallet.count",
    )
    spare_size_range = _parse_float_range(
        spare_cfg.get("size"),
        0.12,
        0.28,
        "machinery.maintenance.spare_part.size",
    )
    spare_per_pallet_range = _parse_int_range(
        spare_cfg.get("per_pallet"),
        2,
        5,
        "machinery.maintenance.spare_part.per_pallet",
    )
    spare_type_values = spare_cfg.get("types", ["gear", "shaft", "plate", "bearing", "valve"])
    if isinstance(spare_type_values, (list, tuple)):
        spare_types = [str(item).strip().lower() for item in spare_type_values if str(item).strip()]
    else:
        spare_types = [str(spare_type_values).strip().lower()]
    if not spare_types:
        spare_types = ["gear", "shaft", "plate"]

    repair_count_range = _parse_int_range(
        repair_scenario_cfg.get("count"),
        1,
        2,
        "machinery.maintenance.repair_scenario.count",
    )
    repair_subcomponents_range = _parse_int_range(
        repair_scenario_cfg.get("subcomponents"),
        3,
        6,
        "machinery.maintenance.repair_scenario.subcomponents",
    )
    repair_tools_per_target_range = _parse_int_range(
        repair_scenario_cfg.get("tools_per_target"),
        1,
        2,
        "machinery.maintenance.repair_scenario.tools_per_target",
    )
    repair_tool_size_scale_range = _parse_float_range(
        repair_scenario_cfg.get("tool_size_scale"),
        0.85,
        1.15,
        "machinery.maintenance.repair_scenario.tool_size_scale",
    )
    repair_disconnected_lines_enabled = _sample_bool(
        repair_scenario_cfg.get("disconnected_lines"),
        rng,
        random_variation,
        True,
        "machinery.maintenance.repair_scenario.disconnected_lines",
    )
    repair_cable_probability_range = _parse_float_range(
        repair_scenario_cfg.get("cable_probability"),
        0.75,
        1.0,
        "machinery.maintenance.repair_scenario.cable_probability",
    )
    repair_pipe_probability_range = _parse_float_range(
        repair_scenario_cfg.get("pipe_probability"),
        0.55,
        0.95,
        "machinery.maintenance.repair_scenario.pipe_probability",
    )
    repair_cable_length_range = _parse_float_range(
        repair_scenario_cfg.get("cable_length"),
        0.6,
        1.6,
        "machinery.maintenance.repair_scenario.cable_length",
    )
    repair_pipe_length_range = _parse_float_range(
        repair_scenario_cfg.get("pipe_length"),
        0.55,
        1.4,
        "machinery.maintenance.repair_scenario.pipe_length",
    )
    repair_cable_curvature_range = _parse_float_range(
        repair_scenario_cfg.get("cable_curvature"),
        0.15,
        0.72,
        "machinery.maintenance.repair_scenario.cable_curvature",
    )
    repair_pipe_curvature_range = _parse_float_range(
        repair_scenario_cfg.get("pipe_curvature"),
        0.08,
        0.58,
        "machinery.maintenance.repair_scenario.pipe_curvature",
    )

    explicit_repair_range = _parse_int_range(
        zones_cfg.get("repair", cfg.get("repair_zone_count")),
        2,
        10,
        "machinery.maintenance.zones.repair",
    )
    explicit_workbench_range = _parse_int_range(
        workbench_cfg.get("count", explicit_repair_range),
        explicit_repair_range[0],
        explicit_repair_range[1],
        "machinery.maintenance.workbench.count",
    )
    storage_zone_count_range = _parse_int_range(
        zones_cfg.get("storage"),
        2,
        2,
        "machinery.maintenance.zones.storage",
    )
    storage_zone_count = _sample_int(storage_zone_count_range, rng, random_variation)
    storage_zone_count = max(1, min(3, storage_zone_count + pattern_profile.storage_zone_bias))

    crane_enabled = _sample_bool(
        crane_global_cfg.get("enabled", crane_cfg.get("enabled")),
        rng,
        random_variation,
        True,
        "machinery.maintenance.crane.enabled",
    )
    if pattern_profile.force_crane:
        crane_enabled = True

    half_w = room.width / 2.0
    half_d = room.depth / 2.0
    usable_width = max(room.width - edge_margin * 2.0, 0.0)
    usable_depth = max(room.depth - edge_margin * 2.0, 0.0)
    if usable_width <= 0.2 or usable_depth <= 0.2:
        return

    max_walkway_width = max(usable_depth - 2.0 * (workbench_depth + access_depth + 0.35), 0.8)
    walkway_width = _clamp(walkway_width_cfg, 0.8, max_walkway_width)

    spacing_rule = MinSpacingRule(min_clearance)
    walkway_rule = WalkwayRule(walkway_width)
    stability_rule = StabilityRule(
        floor_z=0.0,
        epsilon=1e-4,
    )
    tool_placement_rule = ToolPlacementRule(
        max_distance=to_positive_float(
            rules_cfg.get("tool_max_distance"),
            max(1.6, workbench_depth + 1.1),
            "machinery.maintenance.rules.tool_max_distance",
        )
    )
    part_placement_rule = PartPlacementRule(
        floor_tolerance=to_positive_float(
            rules_cfg.get("floor_tolerance"),
            0.08,
            "machinery.maintenance.rules.floor_tolerance",
        ),
        pallet_attach_tolerance=to_positive_float(
            rules_cfg.get("pallet_attach_tolerance"),
            0.12,
            "machinery.maintenance.rules.pallet_attach_tolerance",
        ),
    )
    crane_clearance_rule = CraneClearanceRule(
        clearance_radius=to_positive_float(
            rules_cfg.get("crane_clearance_radius"),
            max(min_clearance * 0.85, 0.45),
            "machinery.maintenance.rules.crane_clearance_radius",
        )
    )

    counters: dict[str, int] = {}

    def emit(
        object_type: str,
        mesh: object,
        position: Vector3,
        rotation: Vector3 = (0.0, 0.0, 0.0),
    ) -> None:
        counters[object_type] = counters.get(object_type, 0) + 1
        add_object(object_type, counters[object_type], mesh, position, rotation)  # type: ignore[arg-type]

    def support_aligned_z(mesh: object, support_top: float, clearance: float = 0.003) -> float:
        vertices = getattr(mesh, "vertices", [])
        if not vertices:
            return support_top + clearance
        min_local_z = min(vertex[2] for vertex in vertices)
        return support_top - min_local_z + clearance

    # 1) Split into main zones.
    emit(
        "maintenance_walkway",
        create_box(width=max(usable_width, 0.6), height=0.02, depth=walkway_width),
        (0.0, 0.0, 0.01),
    )

    storage_width_scale = 0.8 if storage_zone_count == 1 else (1.0 if storage_zone_count == 2 else 1.3)
    storage_width_scale *= _clamp(pattern_profile.pallet_multiplier * 0.85, 0.75, 1.75)
    storage_zone_width = max((rack_depth_est + pallet_depth + 0.9) * storage_width_scale, 1.2)
    storage_zone_depth = max(usable_depth, 0.6)
    left_storage_x = -half_w + edge_margin + storage_zone_width / 2.0
    right_storage_x = half_w - edge_margin - storage_zone_width / 2.0
    storage_zone_specs: list[tuple[float, float, float, float]] = []
    if storage_zone_count == 1:
        emit(
            "maintenance_storage_zone",
            create_box(width=storage_zone_width, height=0.02, depth=storage_zone_depth),
            (right_storage_x, 0.0, 0.01),
        )
        storage_zone_specs.append((right_storage_x, 0.0, storage_zone_width, storage_zone_depth))
    else:
        emit(
            "maintenance_storage_zone",
            create_box(width=storage_zone_width, height=0.02, depth=storage_zone_depth),
            (left_storage_x, 0.0, 0.01),
        )
        emit(
            "maintenance_storage_zone",
            create_box(width=storage_zone_width, height=0.02, depth=storage_zone_depth),
            (right_storage_x, 0.0, 0.01),
        )
        storage_zone_specs.append((left_storage_x, 0.0, storage_zone_width, storage_zone_depth))
        storage_zone_specs.append((right_storage_x, 0.0, storage_zone_width, storage_zone_depth))

    if storage_zone_count >= 3:
        rear_zone_depth = _clamp(max(usable_depth * 0.26, 1.2), 1.2, max(usable_depth * 0.46, 1.2))
        rear_zone_width = max(usable_width - storage_zone_width * 1.15, workbench_width + 0.8)
        rear_zone_y = half_d - edge_margin - rear_zone_depth / 2.0
        emit(
            "maintenance_storage_zone",
            create_box(width=rear_zone_width, height=0.02, depth=rear_zone_depth),
            (0.0, rear_zone_y, 0.01),
        )
        storage_zone_specs.append((0.0, rear_zone_y, rear_zone_width, rear_zone_depth))

    # Candidate repair rows.
    row_base = walkway_rule.half_width + access_depth + workbench_depth / 2.0 + 0.18
    row_step = max(workbench_depth + access_depth + 0.75, 2.1)
    max_row_offset = half_d - edge_margin - workbench_depth / 2.0
    if max_row_offset <= row_base:
        row_offsets = [0.0]
    else:
        rows_per_side = int((max_row_offset - row_base) // row_step) + 1
        rows_per_side = max(1, rows_per_side)
        rows_per_side = min(rows_per_side, 1 if density < 1.1 else 2)
        row_offsets: list[float] = []
        for side in (1.0, -1.0):
            for row_index in range(rows_per_side):
                y = side * (row_base + row_index * row_step)
                if abs(y) + workbench_depth / 2.0 <= half_d - edge_margin + 1e-6:
                    row_offsets.append(y)
        row_offsets = sorted(row_offsets, key=lambda value: (abs(value), -value))

    left_reserved = 0.0
    right_reserved = 0.0
    for zone_x, _zone_y, zone_w, _zone_d in storage_zone_specs:
        if zone_x < -0.05:
            left_reserved = max(left_reserved, zone_w)
        elif zone_x > 0.05:
            right_reserved = max(right_reserved, zone_w)
    x_min = -half_w + edge_margin + left_reserved + workbench_width / 2.0
    x_max = half_w - edge_margin - right_reserved - workbench_width / 2.0
    if x_max < x_min:
        center_x = (x_min + x_max) / 2.0
        x_min = center_x
        x_max = center_x
    if x_max <= x_min + 0.2:
        x_positions = [(x_min + x_max) / 2.0]
    else:
        x_span = x_max - x_min
        x_slots = max(2, int((x_span + 1.0) // max(workbench_width + 0.8, 2.0)))
        x_slots = min(x_slots, 8)
        x_positions = [x_min + x_span * idx / max(x_slots - 1, 1) for idx in range(x_slots)]

    area = max(room.width * room.depth, 1.0)
    default_repair_count = max(2, min(14, int(area / 32.0 * max(density, 0.45))))
    repair_count_range = _parse_int_range(
        explicit_workbench_range,
        default_repair_count,
        default_repair_count,
        "machinery.maintenance.repair_zone_count",
    )
    target_repair_count = _sample_int(repair_count_range, rng, random_variation)
    target_repair_count = _scale_int(target_repair_count, pattern_profile.repair_multiplier, minimum=1, maximum=20)
    target_repair_count = max(2, target_repair_count)

    repair_activity = max(target_repair_count * density * (0.65 + 0.45 * tool_density), 1.0)
    loose_parts_range = _parse_int_range(
        parts_cfg.get("loose_parts"),
        0,
        max(8, int(round(repair_activity * 2.4))),
        "machinery.maintenance.parts.loose_parts",
    )
    large_parts_range = _parse_int_range(
        parts_cfg.get("large_parts"),
        0,
        max(2, int(round(repair_activity * 0.7))),
        "machinery.maintenance.parts.large_parts",
    )
    loose_parts_target = _sample_int(loose_parts_range, rng, random_variation)
    loose_parts_target = _scale_int(
        loose_parts_target,
        pattern_profile.spare_multiplier * (0.7 + chaos_factor * 0.7),
        minimum=0,
        maximum=120,
    )
    extra_large_parts_target = _sample_int(large_parts_range, rng, random_variation)
    extra_large_parts_target = _scale_int(
        extra_large_parts_target,
        pattern_profile.large_parts_multiplier,
        minimum=0,
        maximum=40,
    )
    if pattern_profile.force_crane:
        extra_large_parts_target = max(extra_large_parts_target, max(1, target_repair_count // 2))

    # 2) Place repair zones and core objects.
    candidate_positions = [(x, y) for y in row_offsets for x in x_positions]
    if not candidate_positions:
        candidate_positions = [(0.0, row_base if row_base <= max_row_offset else 0.0)]

    occupied_floor: list[Aabb2D] = []
    repair_zone_bounds: list[Aabb2D] = []
    accepted_workstations: list[tuple[float, float]] = []
    storage_zone_bounds = [_aabb2d(x, y, w, d) for x, y, w, d in storage_zone_specs]
    jitter_xy = (
        0.0
        if not random_variation
        else min(scatter_strength * (0.07 + chaos_factor * 0.22), pattern_profile.jitter_limit)
    )

    for x, y in candidate_positions:
        px = _clamp(
            x + rng.uniform(-jitter_xy, jitter_xy),
            x_min,
            x_max,
        )
        py = _clamp(
            y + rng.uniform(-jitter_xy, jitter_xy),
            -half_d + edge_margin + workbench_depth / 2.0,
            half_d - edge_margin - workbench_depth / 2.0,
        )
        py = _clamp(
            walkway_rule.auto_fix_y(py, workbench_depth, margin=0.08),
            -half_d + edge_margin + workbench_depth / 2.0,
            half_d - edge_margin - workbench_depth / 2.0,
        )
        bench_bounds = _aabb2d(px, py, workbench_width, workbench_depth)
        if walkway_rule.intersects(bench_bounds):
            continue
        if any(spacing_rule.conflicts(bench_bounds, other) for other in occupied_floor):
            continue
        accepted_workstations.append((px, py))
        occupied_floor.append(bench_bounds)
        repair_zone = _aabb2d(px, py, workbench_width + 0.6, workbench_depth + access_depth * 1.2)
        repair_zone_bounds.append(repair_zone)
        if len(accepted_workstations) >= target_repair_count:
            break

    if not accepted_workstations:
        center_x = (x_min + x_max) / 2.0
        span_x = max(x_max - x_min, 0.0)
        fallback_offset = max(0.6, span_x * 0.22)
        accepted_workstations = [
            (
                _clamp(center_x - fallback_offset, x_min, x_max),
                walkway_rule.auto_fix_y(row_base, workbench_depth, margin=0.08),
            ),
            (
                _clamp(center_x + fallback_offset, x_min, x_max),
                walkway_rule.auto_fix_y(row_base, workbench_depth, margin=0.08),
            ),
        ]
        for px, py in accepted_workstations:
            occupied_floor.append(_aabb2d(px, py, workbench_width, workbench_depth))
            repair_zone_bounds.append(_aabb2d(px, py, workbench_width + 0.6, workbench_depth + access_depth * 1.2))

    spare_parts_emitted = 0
    extra_large_parts_remaining = extra_large_parts_target
    near_rack_stride = max(
        1,
        pattern_profile.near_rack_stride + (1 if tool_density < 0.75 and pattern_profile.near_rack_stride > 1 else 0),
    )

    for idx, (x, y) in enumerate(accepted_workstations, start=1):
        yaw = 180.0 if y > 0.0 else 0.0
        if random_variation and pattern_profile.yaw_jitter_deg > 0.0:
            yaw += rng.uniform(-pattern_profile.yaw_jitter_deg, pattern_profile.yaw_jitter_deg)
        toward_center = -1.0 if y > 0.0 else 1.0

        emit(
            "maintenance_repair_zone",
            create_box(
                width=workbench_width + 0.6,
                height=0.02,
                depth=workbench_depth + access_depth * 1.2,
            ),
            (x, y, 0.01),
        )

        workbench_z = stability_rule.stable_center(
            workbench_height / 2.0,
            nominal_height=workbench_height,
            support_top=0.0,
        )
        emit(
            "workbench",
            create_workbench(
                width=workbench_width,
                depth=workbench_depth,
                height=workbench_height,
                variation=_clamp(0.2 + density * 0.2 + chaos_factor * 0.25, 0.0, 1.0),
                scatter=scatter_strength * (0.07 + chaos_factor * 0.08),
                seed=seed + idx * 13,
            ),
            (x, y, workbench_z),
            (0.0, 0.0, yaw),
        )

        machine_size = _sample_float(machine_size_range, rng, random_variation)
        machine_size *= pattern_profile.machine_size_multiplier * (0.92 + density * 0.08)
        machine_size = _clamp(machine_size, 0.12, max(workbench_width * 0.92, machine_size))
        machine_complexity = _sample_int(machine_complexity_range, rng, random_variation)
        machine_complexity = max(
            1,
            int(round(machine_complexity * pattern_profile.machine_complexity_multiplier)),
        )
        machine_x = x + rng.uniform(-workbench_width * 0.18, workbench_width * 0.18) * (1.0 if random_variation else 0.0)
        machine_y = y + toward_center * workbench_depth * 0.1
        machine_z = stability_rule.stable_center(
            workbench_height + max(machine_size * 0.24, 0.05),
            nominal_height=max(machine_size * 0.5, 0.08),
            support_top=workbench_height,
        )
        emit(
            "machine_part",
            create_machine_part(
                size=machine_size,
                complexity=machine_complexity,
                variation=_clamp(0.3 + density * 0.16 + chaos_factor * 0.25, 0.0, 1.0),
                scatter=scatter_strength * (0.06 + chaos_factor * 0.09),
                seed=seed + idx * 19,
            ),
            (machine_x, machine_y, machine_z),
            (0.0, 0.0, yaw),
        )

        if extra_large_parts_remaining > 0:
            large_size = _sample_float(machine_size_range, rng, random_variation)
            large_size *= pattern_profile.machine_size_multiplier * 1.18
            large_size = _clamp(large_size, 0.22, max(workbench_width * 1.35, 0.3))
            large_complexity = max(
                machine_complexity,
                int(round(_sample_int(machine_complexity_range, rng, random_variation) * pattern_profile.machine_complexity_multiplier * 1.15)),
            )
            large_offsets = (
                (0.0, toward_center * (workbench_depth + large_size * 0.7)),
                (workbench_width * 0.45, toward_center * (workbench_depth * 0.75 + large_size * 0.5)),
                (-workbench_width * 0.45, toward_center * (workbench_depth * 0.75 + large_size * 0.5)),
            )
            placed_large_part = False
            for off_x, off_y in large_offsets:
                large_x = _clamp(
                    x + off_x,
                    -half_w + edge_margin + large_size / 2.0,
                    half_w - edge_margin - large_size / 2.0,
                )
                large_y = _clamp(
                    y + off_y,
                    -half_d + edge_margin + large_size / 2.0,
                    half_d - edge_margin - large_size / 2.0,
                )
                large_bounds = _aabb2d(large_x, large_y, large_size * 1.05, large_size * 1.05)
                if walkway_rule.intersects(large_bounds):
                    continue
                if any(_intersects(large_bounds, zone) for zone in storage_zone_bounds):
                    continue
                if any(spacing_rule.conflicts(large_bounds, other) for other in occupied_floor):
                    continue
                large_nominal_h = max(large_size * 0.5, 0.1)
                large_z = stability_rule.stable_center(
                    max(large_size * 0.24, 0.06),
                    nominal_height=large_nominal_h,
                    support_top=0.0,
                )
                large_z, _ = part_placement_rule.auto_fix_large_part_support(
                    center_z=large_z,
                    nominal_height=large_nominal_h,
                    x=large_x,
                    y=large_y,
                    pallet_positions=[],
                    pallet_top_z=0.0,
                    pallet_radius_x=pallet_width * 0.5 + part_placement_rule.pallet_attach_tolerance,
                    pallet_radius_y=pallet_depth * 0.5 + part_placement_rule.pallet_attach_tolerance,
                    stability_rule=stability_rule,
                )
                emit(
                    "machine_part",
                    create_machine_part(
                        size=large_size,
                        complexity=large_complexity,
                        variation=_clamp(0.42 + chaos_factor * 0.45, 0.0, 1.0),
                        scatter=scatter_strength * (0.08 + chaos_factor * 0.1),
                        seed=seed + idx * 211 + extra_large_parts_remaining * 7,
                    ),
                    (large_x, large_y, large_z),
                    (0.0, 0.0, yaw),
                )
                occupied_floor.append(large_bounds)
                extra_large_parts_remaining -= 1
                placed_large_part = True
                break
            if not placed_large_part:
                fallback_large_size = _clamp(large_size * 0.92, 0.2, max(workbench_width * 0.95, 0.24))
                fallback_x = x + (-0.22 if idx % 2 == 0 else 0.22) * workbench_width
                fallback_y = y + toward_center * workbench_depth * 0.22
                fallback_nominal_h = max(fallback_large_size * 0.5, 0.08)
                fallback_z = stability_rule.stable_center(
                    max(fallback_large_size * 0.24, 0.05),
                    nominal_height=fallback_nominal_h,
                    support_top=0.0,
                )
                emit(
                    "machine_part",
                    create_machine_part(
                        size=fallback_large_size,
                        complexity=max(2, large_complexity),
                        variation=_clamp(0.35 + chaos_factor * 0.3, 0.0, 1.0),
                        scatter=scatter_strength * (0.05 + chaos_factor * 0.08),
                        seed=seed + idx * 239 + extra_large_parts_remaining * 9,
                    ),
                    (fallback_x, fallback_y, fallback_z),
                    (0.0, 0.0, yaw),
                )
                extra_large_parts_remaining -= 1

        toolbox_size = _sample_float(toolbox_size_range, rng, random_variation)
        toolbox_x = x + rng.uniform(-workbench_width * 0.22, workbench_width * 0.22) * (1.0 if random_variation else 0.0)
        toolbox_y = y - toward_center * workbench_depth * 0.2
        toolbox_x, toolbox_y = tool_placement_rule.auto_fix_near_repair_zone(
            toolbox_x,
            toolbox_y,
            repair_zone_bounds,
        )
        toolbox_y = _clamp(
            walkway_rule.auto_fix_y(toolbox_y, max(toolbox_size * 0.55, 0.22), margin=0.04),
            -half_d + edge_margin + max(toolbox_size * 0.55, 0.22) / 2.0,
            half_d - edge_margin - max(toolbox_size * 0.55, 0.22) / 2.0,
        )
        toolbox_z = stability_rule.stable_center(
            workbench_height + max(toolbox_size * 0.26, 0.04),
            nominal_height=max(toolbox_size * 0.42, 0.06),
            support_top=workbench_height,
        )
        emit(
            "toolbox",
            create_toolbox(
                size=toolbox_size,
                variation=_clamp(0.22 + density * 0.14 + chaos_factor * 0.18, 0.0, 1.0),
                scatter=scatter_strength * (0.1 + chaos_factor * 0.12),
                seed=seed + idx * 23,
            ),
            (toolbox_x, toolbox_y, toolbox_z),
            (0.0, 0.0, yaw),
        )

        station_spares = _scale_int(
            1 if order_level < 0.55 else 0,
            pattern_profile.spare_multiplier * (0.5 + chaos_factor * 1.1),
            minimum=0,
            maximum=4,
        )
        if pattern == "active_repair":
            station_spares = max(station_spares, 1)
        for spare_idx in range(station_spares):
            spare_type = spare_types[spare_idx % len(spare_types)] if not random_variation else spare_types[rng.randint(0, len(spare_types) - 1)]
            spare_size = _sample_float(spare_size_range, rng, random_variation)
            spare_x = x + rng.uniform(-workbench_width * 0.28, workbench_width * 0.28) * (1.0 if random_variation else 0.0)
            spare_y = y + rng.uniform(-workbench_depth * 0.22, workbench_depth * 0.22) * (1.0 if random_variation else 0.0)
            spare_z = stability_rule.stable_center(
                workbench_height + max(spare_size * 0.3, 0.02),
                nominal_height=max(spare_size * 0.4, 0.04),
                support_top=workbench_height,
            )
            emit(
                "spare_part",
                create_spare_part(
                    type=spare_type,
                    size=spare_size,
                    variation=_clamp(0.2 + density * 0.1 + chaos_factor * 0.16, 0.0, 1.0),
                    scatter=scatter_strength * (0.06 + chaos_factor * 0.08),
                    seed=seed + idx * 227 + spare_idx * 3,
                ),
                (spare_x, spare_y, spare_z),
            )
            spare_parts_emitted += 1

        # Tool rack near some workstations.
        if idx % near_rack_stride == 0:
            near_rx = _clamp(
                x + (-1.0 if idx % 4 == 0 else 1.0) * (workbench_width * 0.5 + rack_depth_est * 0.6),
                -half_w + edge_margin + rack_depth_est / 2.0,
                half_w - edge_margin - rack_depth_est / 2.0,
            )
            near_ry = _clamp(
                y + toward_center * (workbench_depth / 2.0 + rack_width / 2.0 + 0.25),
                -half_d + edge_margin + rack_width / 2.0,
                half_d - edge_margin - rack_width / 2.0,
            )
            near_rx, near_ry = tool_placement_rule.auto_fix_near_repair_zone(
                near_rx,
                near_ry,
                repair_zone_bounds,
            )
            near_ry = _clamp(
                walkway_rule.auto_fix_y(near_ry, rack_width, margin=0.05),
                -half_d + edge_margin + rack_width / 2.0,
                half_d - edge_margin - rack_width / 2.0,
            )
            near_bounds = _aabb2d(near_rx, near_ry, rack_depth_est, rack_width)
            if not walkway_rule.intersects(near_bounds) and not any(
                spacing_rule.conflicts(near_bounds, other) for other in occupied_floor
            ):
                emit(
                    "tool_rack",
                    create_tool_rack(
                        width=rack_width,
                        height=rack_height,
                        variation=_clamp(0.2 + density * 0.12 + tool_density * 0.08, 0.0, 1.0),
                        scatter=scatter_strength * (0.05 + chaos_factor * 0.07),
                        seed=seed + idx * 29,
                    ),
                    (
                        near_rx,
                        near_ry,
                        stability_rule.stable_center(rack_height / 2.0, nominal_height=rack_height, support_top=0.0),
                    ),
                    (0.0, 0.0, 0.0),
                )
                occupied_floor.append(near_bounds)

        if crane_enabled:
            crane_x, crane_y = crane_clearance_rule.auto_fix_xy(
                base_x=x,
                base_y=toward_center * (walkway_rule.half_width + crane_clearance_rule.clearance_radius * 0.4),
                x_min=-half_w + edge_margin + crane_clearance_rule.clearance_radius,
                x_max=half_w - edge_margin - crane_clearance_rule.clearance_radius,
                y_min=-half_d + edge_margin + crane_clearance_rule.clearance_radius,
                y_max=half_d - edge_margin - crane_clearance_rule.clearance_radius,
                blocked_bounds=occupied_floor,
            )
            hook_height = _sample_float(hook_height_range, rng, random_variation)
            hook_z = _clamp(
                room.height - hook_height / 2.0 - 0.08,
                hook_height / 2.0 + 0.05,
                room.height - 0.05,
            )
            emit(
                "crane_hook",
                create_crane_hook(
                    height=hook_height,
                    variation=_clamp(0.2 + density * 0.1 + chaos_factor * 0.12, 0.0, 1.0),
                    scatter=scatter_strength * (0.03 + chaos_factor * 0.05),
                    seed=seed + idx * 31,
                ),
                (crane_x, crane_y, hook_z),
                (0.0, 0.0, 0.0),
            )

    # 3) Add wall tool racks + 4) storage pallets with spare parts.
    rack_count_range = _parse_int_range(
        cfg.get("storage_rack_count"),
        max(2, int(round(len(accepted_workstations) * 0.45 * max(tool_density, 0.6)))),
        max(2, int(round(len(accepted_workstations) * 0.45 * max(tool_density, 0.6)))),
        "machinery.maintenance.storage_rack_count",
    )
    wall_rack_count = _sample_int(rack_count_range, rng, random_variation)
    wall_rack_count = _scale_int(
        wall_rack_count,
        pattern_profile.rack_multiplier * (0.7 + tool_density * 0.45),
        minimum=1,
        maximum=24,
    )
    storage_sides = sorted(
        {
            -1.0 if zone_x < 0.0 else 1.0
            for zone_x, _zone_y, _zone_w, _zone_d in storage_zone_specs
            if abs(zone_x) > 0.05
        }
    )
    if not storage_sides:
        storage_sides = [-1.0, 1.0]
    racks_per_side = max(1, (wall_rack_count + len(storage_sides) - 1) // len(storage_sides))
    rack_y_positions = symmetric_positions(
        racks_per_side,
        max(half_d - edge_margin - rack_width / 2.0, 0.0),
    )
    for side in storage_sides:
        wall_x = side * max(half_w - edge_margin - rack_depth_est / 2.0, rack_depth_est / 2.0)
        for y in rack_y_positions:
            rack_x, rack_y = tool_placement_rule.auto_fix_near_repair_zone(
                wall_x,
                y,
                repair_zone_bounds,
            )
            rack_x = _clamp(
                rack_x,
                -half_w + edge_margin + rack_depth_est / 2.0,
                half_w - edge_margin - rack_depth_est / 2.0,
            )
            rack_y = _clamp(
                walkway_rule.auto_fix_y(rack_y, rack_width, margin=0.05),
                -half_d + edge_margin + rack_width / 2.0,
                half_d - edge_margin - rack_width / 2.0,
            )
            bounds = _aabb2d(rack_x, rack_y, rack_depth_est, rack_width)
            if walkway_rule.intersects(bounds):
                continue
            if any(spacing_rule.conflicts(bounds, other) for other in occupied_floor):
                continue
            if any(_intersects(bounds, zone) for zone in repair_zone_bounds):
                continue
            emit(
                "tool_rack",
                create_tool_rack(
                    width=rack_width,
                    height=rack_height,
                    variation=_clamp(0.25 + density * 0.1 + tool_density * 0.08, 0.0, 1.0),
                    scatter=scatter_strength * (0.05 + chaos_factor * 0.07),
                    seed=seed + int(abs(rack_x) * 10) + int(abs(rack_y) * 10),
                ),
                (
                    rack_x,
                    rack_y,
                    stability_rule.stable_center(rack_height / 2.0, nominal_height=rack_height, support_top=0.0),
                ),
                (0.0, 0.0, 90.0 if side < 0.0 else -90.0),
            )
            occupied_floor.append(bounds)

    pallet_count = _sample_int(pallet_count_range, rng, random_variation)
    pallet_count = _scale_int(
        pallet_count,
        pattern_profile.pallet_multiplier * (0.65 + tool_density * 0.4),
        minimum=1,
        maximum=24,
    )
    pallet_height_est = max(min(pallet_width, pallet_depth) * 0.14, 0.05)

    storage_candidates: list[tuple[float, float]] = []
    for zone_x, zone_y, zone_w, zone_d in storage_zone_specs:
        zone_is_vertical = zone_d >= zone_w * 0.95
        if zone_is_vertical:
            count_per_zone = max(2, min(8, pallet_count))
            y_limit = max(zone_d / 2.0 - pallet_depth / 2.0 - 0.1, 0.0)
            for py in symmetric_positions(count_per_zone, y_limit):
                storage_candidates.append((zone_x, zone_y + py))
        else:
            count_per_zone = max(2, min(8, pallet_count))
            x_limit_local = max(zone_w / 2.0 - pallet_width / 2.0 - 0.1, 0.0)
            for px in symmetric_positions(count_per_zone, x_limit_local):
                storage_candidates.append((zone_x + px, zone_y))
    if not storage_candidates:
        storage_candidates = [
            (-half_w + edge_margin + storage_zone_width * 0.58, half_d - edge_margin - pallet_depth / 2.0),
            (half_w - edge_margin - storage_zone_width * 0.58, half_d - edge_margin - pallet_depth / 2.0),
            (-half_w + edge_margin + storage_zone_width * 0.58, -half_d + edge_margin + pallet_depth / 2.0),
            (half_w - edge_margin - storage_zone_width * 0.58, -half_d + edge_margin + pallet_depth / 2.0),
        ]

    if random_variation:
        storage_candidates = sorted(storage_candidates, key=lambda _: rng.random())

    placed_pallets = 0
    placed_pallet_positions: list[tuple[float, float]] = []
    for px, py in storage_candidates:
        if placed_pallets >= pallet_count:
            break
        jitter = min(scatter_strength * 0.14, 0.15) if random_variation else 0.0
        x = _clamp(
            px + rng.uniform(-jitter, jitter),
            -half_w + edge_margin + pallet_width / 2.0,
            half_w - edge_margin - pallet_width / 2.0,
        )
        y = _clamp(
            py + rng.uniform(-jitter, jitter),
            -half_d + edge_margin + pallet_depth / 2.0,
            half_d - edge_margin - pallet_depth / 2.0,
        )
        y = _clamp(
            walkway_rule.auto_fix_y(y, pallet_depth, margin=0.05),
            -half_d + edge_margin + pallet_depth / 2.0,
            half_d - edge_margin - pallet_depth / 2.0,
        )
        bounds = _aabb2d(x, y, pallet_width, pallet_depth)
        if walkway_rule.intersects(bounds):
            continue
        if any(spacing_rule.conflicts(bounds, other) for other in occupied_floor):
            continue
        if any(_intersects(bounds, zone) for zone in repair_zone_bounds):
            continue

        placed_pallets += 1
        placed_pallet_positions.append((x, y))
        emit(
            "pallet",
            create_pallet(
                width=pallet_width,
                depth=pallet_depth,
                variation=_clamp(0.2 + density * 0.14 + chaos_factor * 0.18, 0.0, 1.0),
                scatter=scatter_strength * (0.05 + chaos_factor * 0.06),
                seed=seed + placed_pallets * 37,
            ),
            (
                x,
                y,
                stability_rule.stable_center(pallet_height_est / 2.0, nominal_height=pallet_height_est, support_top=0.0),
            ),
        )
        occupied_floor.append(bounds)

        spare_count = _sample_int(spare_per_pallet_range, rng, random_variation)
        spare_count = _scale_int(
            spare_count,
            pattern_profile.spare_multiplier * (0.65 + chaos_factor * 0.85),
            minimum=1,
            maximum=18,
        )
        for spare_index in range(spare_count):
            part_type = spare_types[spare_index % len(spare_types)] if not random_variation else spare_types[rng.randint(0, len(spare_types) - 1)]
            part_size = _sample_float(spare_size_range, rng, random_variation)
            offset_x = rng.uniform(-pallet_width * 0.32, pallet_width * 0.32) if random_variation else 0.0
            offset_y = rng.uniform(-pallet_depth * 0.28, pallet_depth * 0.28) if random_variation else 0.0
            part_z = stability_rule.stable_center(
                pallet_height_est + max(part_size * 0.35, 0.03),
                nominal_height=max(part_size * 0.4, 0.04),
                support_top=pallet_height_est,
            )
            emit(
                "spare_part",
                create_spare_part(
                    type=part_type,
                    size=part_size,
                    variation=_clamp(0.2 + density * 0.1 + chaos_factor * 0.2, 0.0, 1.0),
                    scatter=scatter_strength * (0.04 + chaos_factor * 0.08),
                    seed=seed + placed_pallets * 41 + spare_index,
                ),
                (x + offset_x, y + offset_y, part_z),
            )
            spare_parts_emitted += 1

    if placed_pallets <= 0:
        fallback_candidates = [
            (right_storage_x, half_d - edge_margin - pallet_depth / 2.0),
            (right_storage_x, -half_d + edge_margin + pallet_depth / 2.0),
            (left_storage_x, half_d - edge_margin - pallet_depth / 2.0),
            (left_storage_x, -half_d + edge_margin + pallet_depth / 2.0),
        ]
        for x, y in fallback_candidates:
            bounds = _aabb2d(x, y, pallet_width, pallet_depth)
            if walkway_rule.intersects(bounds):
                continue
            if any(_intersects(bounds, zone) for zone in repair_zone_bounds):
                continue

            emit(
                "pallet",
                create_pallet(
                    width=pallet_width,
                    depth=pallet_depth,
                    variation=_clamp(0.2 + density * 0.14 + chaos_factor * 0.18, 0.0, 1.0),
                    scatter=scatter_strength * (0.05 + chaos_factor * 0.06),
                    seed=seed + 999,
                ),
                (
                    x,
                    y,
                    stability_rule.stable_center(pallet_height_est / 2.0, nominal_height=pallet_height_est, support_top=0.0),
                ),
            )
            placed_pallet_positions.append((x, y))
            fallback_spare_size = _sample_float(spare_size_range, rng, random_variation)
            emit(
                "spare_part",
                create_spare_part(
                    type=spare_types[0],
                    size=fallback_spare_size,
                    variation=_clamp(0.2 + density * 0.1 + chaos_factor * 0.2, 0.0, 1.0),
                    scatter=scatter_strength * (0.04 + chaos_factor * 0.08),
                    seed=seed + 1001,
                ),
                (
                    x,
                    y,
                    stability_rule.stable_center(
                        pallet_height_est + 0.08,
                        nominal_height=max(fallback_spare_size * 0.4, 0.04),
                        support_top=pallet_height_est,
                    ),
                ),
            )
            spare_parts_emitted += 1
            break

    if spare_parts_emitted < loose_parts_target:
        anchors: list[tuple[float, float, float, float, float]] = []
        for wx, wy in accepted_workstations:
            anchors.append((wx, wy, workbench_height, workbench_width, workbench_depth))
        for px, py in placed_pallet_positions:
            anchors.append((px, py, pallet_height_est, pallet_width, pallet_depth))
        if not anchors:
            anchors.append((0.0, 0.0, 0.05, pallet_width, pallet_depth))

        extra_needed = min(loose_parts_target - spare_parts_emitted, 180)
        for extra_index in range(extra_needed):
            if random_variation:
                anchor = anchors[rng.randint(0, len(anchors) - 1)]
            else:
                anchor = anchors[extra_index % len(anchors)]
            ax, ay, az, aw, ad = anchor
            part_type = spare_types[extra_index % len(spare_types)] if not random_variation else spare_types[rng.randint(0, len(spare_types) - 1)]
            part_size = _sample_float(spare_size_range, rng, random_variation)
            offset_x = rng.uniform(-aw * 0.34, aw * 0.34) if random_variation else 0.0
            offset_y = rng.uniform(-ad * 0.34, ad * 0.34) if random_variation else 0.0
            part_z = stability_rule.stable_center(
                az + max(part_size * 0.3, 0.02),
                nominal_height=max(part_size * 0.4, 0.04),
                support_top=az,
            )
            emit(
                "spare_part",
                create_spare_part(
                    type=part_type,
                    size=part_size,
                    variation=_clamp(0.25 + chaos_factor * 0.35, 0.0, 1.0),
                    scatter=scatter_strength * (0.05 + chaos_factor * 0.09),
                    seed=seed + 1300 + extra_index * 5,
                ),
                (ax + offset_x, ay + offset_y, part_z),
            )
            spare_parts_emitted += 1

    if repair_scenario_enabled and accepted_workstations:
        if not placed_pallet_positions:
            emergency_candidates = [
                (right_storage_x, half_d - edge_margin - pallet_depth / 2.0),
                (right_storage_x, -half_d + edge_margin + pallet_depth / 2.0),
                (left_storage_x, half_d - edge_margin - pallet_depth / 2.0),
                (left_storage_x, -half_d + edge_margin + pallet_depth / 2.0),
            ]
            for x, y in emergency_candidates:
                bounds = _aabb2d(x, y, pallet_width, pallet_depth)
                if walkway_rule.intersects(bounds):
                    continue
                if any(spacing_rule.conflicts(bounds, other) for other in occupied_floor):
                    continue
                if any(_intersects(bounds, zone) for zone in repair_zone_bounds):
                    continue
                emit(
                    "pallet",
                    create_pallet(
                        width=pallet_width,
                        depth=pallet_depth,
                        variation=_clamp(0.2 + density * 0.12 + chaos_factor * 0.14, 0.0, 1.0),
                        scatter=scatter_strength * (0.04 + chaos_factor * 0.05),
                        seed=seed + 1600,
                    ),
                    (
                        x,
                        y,
                        stability_rule.stable_center(
                            pallet_height_est / 2.0,
                            nominal_height=pallet_height_est,
                            support_top=0.0,
                        ),
                    ),
                )
                placed_pallet_positions.append((x, y))
                occupied_floor.append(bounds)
                break

        repair_targets = _sample_int(repair_count_range, rng, random_variation)
        repair_targets = max(1, min(repair_targets, len(accepted_workstations)))
        repair_subcomponents_default = _sample_int(repair_subcomponents_range, rng, random_variation)
        repair_cable_probability = _clamp(
            _sample_float(repair_cable_probability_range, rng, random_variation),
            0.0,
            1.0,
        )
        repair_pipe_probability = _clamp(
            _sample_float(repair_pipe_probability_range, rng, random_variation),
            0.0,
            1.0,
        )

        workstation_order = list(range(len(accepted_workstations)))
        if random_variation:
            for order_idx in range(len(workstation_order) - 1, 0, -1):
                swap_idx = rng.randint(0, order_idx)
                workstation_order[order_idx], workstation_order[swap_idx] = (
                    workstation_order[swap_idx],
                    workstation_order[order_idx],
                )
        selected_workstations = workstation_order[:repair_targets]

        for scenario_index, workstation_index in enumerate(selected_workstations, start=1):
            wx, wy = accepted_workstations[workstation_index]
            toward_center = -1.0 if wy > 0.0 else 1.0
            station_yaw = 180.0 if wy > 0.0 else 0.0

            surface_modes = ["bench", "floor"]
            if placed_pallet_positions:
                surface_modes.append("pallet")
            component_count = max(3, repair_subcomponents_default)
            if random_variation:
                component_count = max(3, _sample_int(repair_subcomponents_range, rng, random_variation))

            component_anchors: list[tuple[float, float, float]] = []
            tool_anchor_points: list[tuple[float, float]] = [(wx, wy)]

            for component_index in range(component_count):
                if component_index == 0:
                    target_surface = "bench"
                elif component_index == 1:
                    target_surface = "floor"
                elif component_index == 2 and "pallet" in surface_modes:
                    target_surface = "pallet"
                elif random_variation:
                    target_surface = surface_modes[rng.randint(0, len(surface_modes) - 1)]
                else:
                    target_surface = surface_modes[(component_index + scenario_index) % len(surface_modes)]

                component_size = _sample_float(machine_size_range, rng, random_variation)
                component_size *= 0.4 + (0.5 * (component_index + 1) / max(component_count, 1))
                component_size *= _clamp(0.88 + chaos_factor * 0.35, 0.7, 1.35)
                component_size = _clamp(component_size, 0.12, max(workbench_width * 0.65, component_size))

                component_complexity = _sample_int(machine_complexity_range, rng, random_variation)
                component_complexity = max(
                    1,
                    int(round(component_complexity * (0.55 + 0.22 * ((component_index % 3) + 1)))),
                )

                anchor_x = wx
                anchor_y = wy
                support_top = workbench_height

                if target_surface == "pallet" and placed_pallet_positions:
                    pallet_x, pallet_y = min(
                        placed_pallet_positions,
                        key=lambda point: (point[0] - wx) * (point[0] - wx) + (point[1] - wy) * (point[1] - wy),
                    )
                    jitter_x = pallet_width * (0.18 if random_variation else 0.1)
                    jitter_y = pallet_depth * (0.18 if random_variation else 0.1)
                    anchor_x = _clamp(
                        pallet_x + rng.uniform(-jitter_x, jitter_x),
                        -half_w + edge_margin + pallet_width / 2.0,
                        half_w - edge_margin - pallet_width / 2.0,
                    )
                    anchor_y = _clamp(
                        pallet_y + rng.uniform(-jitter_y, jitter_y),
                        -half_d + edge_margin + pallet_depth / 2.0,
                        half_d - edge_margin - pallet_depth / 2.0,
                    )
                    support_top = pallet_height_est
                    tool_anchor_points.append((anchor_x, anchor_y))
                elif target_surface == "floor":
                    footprint = max(component_size * 0.95, 0.2)
                    floor_offsets = [
                        (0.0, toward_center * (workbench_depth * 0.9 + footprint * 0.72 + 0.2)),
                        (workbench_width * 0.34, toward_center * (workbench_depth * 0.7 + footprint * 0.6 + 0.14)),
                        (-workbench_width * 0.34, toward_center * (workbench_depth * 0.7 + footprint * 0.6 + 0.14)),
                        (0.0, toward_center * (workbench_depth * 1.1 + footprint * 0.8 + 0.24)),
                    ]
                    floor_found = False
                    for off_x, off_y in floor_offsets:
                        jitter = min(scatter_strength * 0.09, footprint * 0.28) if random_variation else 0.0
                        candidate_x = _clamp(
                            wx + off_x + rng.uniform(-jitter, jitter),
                            -half_w + edge_margin + footprint / 2.0,
                            half_w - edge_margin - footprint / 2.0,
                        )
                        candidate_y = _clamp(
                            wy + off_y + rng.uniform(-jitter, jitter),
                            -half_d + edge_margin + footprint / 2.0,
                            half_d - edge_margin - footprint / 2.0,
                        )
                        candidate_y = _clamp(
                            walkway_rule.auto_fix_y(candidate_y, footprint, margin=0.06),
                            -half_d + edge_margin + footprint / 2.0,
                            half_d - edge_margin - footprint / 2.0,
                        )
                        candidate_bounds = _aabb2d(candidate_x, candidate_y, footprint, footprint)
                        if walkway_rule.intersects(candidate_bounds):
                            continue
                        if any(spacing_rule.conflicts(candidate_bounds, other) for other in occupied_floor):
                            continue
                        anchor_x = candidate_x
                        anchor_y = candidate_y
                        support_top = 0.0
                        occupied_floor.append(candidate_bounds)
                        floor_found = True
                        tool_anchor_points.append((anchor_x, anchor_y))
                        break
                    if not floor_found:
                        target_surface = "bench"

                if target_surface == "bench":
                    bench_jitter_x = workbench_width * (0.25 if random_variation else 0.12)
                    bench_jitter_y = workbench_depth * (0.18 if random_variation else 0.1)
                    anchor_x = _clamp(
                        wx + rng.uniform(-bench_jitter_x, bench_jitter_x),
                        x_min,
                        x_max,
                    )
                    anchor_y = _clamp(
                        wy
                        + toward_center * workbench_depth * 0.08
                        + rng.uniform(-bench_jitter_y, bench_jitter_y),
                        -half_d + edge_margin + workbench_depth / 2.0,
                        half_d - edge_margin - workbench_depth / 2.0,
                    )
                    support_top = workbench_height

                nominal_height = max(component_size * 0.48, 0.06)
                component_z = stability_rule.stable_center(
                    support_top + max(component_size * 0.24, 0.04),
                    nominal_height=nominal_height,
                    support_top=support_top,
                )
                if target_surface == "floor":
                    component_z, support_top = part_placement_rule.auto_fix_large_part_support(
                        center_z=component_z,
                        nominal_height=nominal_height,
                        x=anchor_x,
                        y=anchor_y,
                        pallet_positions=placed_pallet_positions,
                        pallet_top_z=pallet_height_est,
                        pallet_radius_x=pallet_width * 0.5 + part_placement_rule.pallet_attach_tolerance,
                        pallet_radius_y=pallet_depth * 0.5 + part_placement_rule.pallet_attach_tolerance,
                        stability_rule=stability_rule,
                    )

                yaw_offset = rng.uniform(-12.0, 12.0) if random_variation else float((component_index % 3) * 6 - 6)
                emit(
                    "repair_component",
                    create_machine_part(
                        size=component_size,
                        complexity=component_complexity,
                        variation=_clamp(0.35 + density * 0.15 + chaos_factor * 0.26, 0.0, 1.0),
                        scatter=scatter_strength * (0.07 + chaos_factor * 0.11),
                        seed=seed + 1700 + scenario_index * 97 + component_index * 11,
                    ),
                    (anchor_x, anchor_y, component_z),
                    (0.0, 0.0, station_yaw + yaw_offset),
                )
                component_anchors.append((anchor_x, anchor_y, support_top))

            tool_count = _sample_int(repair_tools_per_target_range, rng, random_variation)
            tool_count = max(1, min(tool_count, 8))
            tool_scale = _sample_float(repair_tool_size_scale_range, rng, random_variation)
            tools_placed = 0
            for tool_index in range(tool_count):
                if random_variation:
                    base_x, base_y = tool_anchor_points[rng.randint(0, len(tool_anchor_points) - 1)]
                else:
                    base_x, base_y = tool_anchor_points[tool_index % len(tool_anchor_points)]

                toolbox_size = _sample_float(toolbox_size_range, rng, random_variation) * tool_scale
                footprint_x = max(toolbox_size * 0.9, 0.14)
                footprint_y = max(toolbox_size * 0.62, 0.12)
                jitter = min(scatter_strength * 0.12, max(footprint_x, footprint_y) * 0.42) if random_variation else 0.0
                tool_x = _clamp(
                    base_x + rng.uniform(-jitter, jitter),
                    -half_w + edge_margin + footprint_x / 2.0,
                    half_w - edge_margin - footprint_x / 2.0,
                )
                tool_y = _clamp(
                    base_y + rng.uniform(-jitter, jitter),
                    -half_d + edge_margin + footprint_y / 2.0,
                    half_d - edge_margin - footprint_y / 2.0,
                )
                tool_x, tool_y = tool_placement_rule.auto_fix_near_repair_zone(tool_x, tool_y, repair_zone_bounds)
                tool_y = _clamp(
                    walkway_rule.auto_fix_y(tool_y, footprint_y, margin=0.04),
                    -half_d + edge_margin + footprint_y / 2.0,
                    half_d - edge_margin - footprint_y / 2.0,
                )
                tool_bounds = _aabb2d(tool_x, tool_y, footprint_x, footprint_y)
                if walkway_rule.intersects(tool_bounds):
                    continue
                if any(spacing_rule.conflicts(tool_bounds, other) for other in occupied_floor):
                    continue

                tool_z = stability_rule.stable_center(
                    max(toolbox_size * 0.22, 0.03),
                    nominal_height=max(toolbox_size * 0.42, 0.06),
                    support_top=0.0,
                )
                emit(
                    "repair_toolbox",
                    create_toolbox(
                        size=toolbox_size,
                        variation=_clamp(0.24 + density * 0.12 + chaos_factor * 0.18, 0.0, 1.0),
                        scatter=scatter_strength * (0.08 + chaos_factor * 0.1),
                        seed=seed + 1900 + scenario_index * 73 + tool_index * 7,
                    ),
                    (tool_x, tool_y, tool_z),
                )
                occupied_floor.append(tool_bounds)
                tools_placed += 1

            if tools_placed <= 0:
                fallback_tool_size = _sample_float(toolbox_size_range, rng, random_variation) * tool_scale
                fallback_x = wx
                fallback_y = _clamp(
                    wy - toward_center * workbench_depth * 0.18,
                    -half_d + edge_margin + workbench_depth / 2.0,
                    half_d - edge_margin - workbench_depth / 2.0,
                )
                fallback_z = stability_rule.stable_center(
                    workbench_height + max(fallback_tool_size * 0.24, 0.04),
                    nominal_height=max(fallback_tool_size * 0.42, 0.06),
                    support_top=workbench_height,
                )
                emit(
                    "repair_toolbox",
                    create_toolbox(
                        size=fallback_tool_size,
                        variation=_clamp(0.22 + density * 0.12 + chaos_factor * 0.16, 0.0, 1.0),
                        scatter=scatter_strength * (0.05 + chaos_factor * 0.08),
                        seed=seed + 2050 + scenario_index * 79,
                    ),
                    (fallback_x, fallback_y, fallback_z),
                    (0.0, 0.0, station_yaw),
                )

            if repair_disconnected_lines_enabled:
                line_specs = (
                    (
                        "maintenance_cable_loose",
                        repair_cable_probability,
                        repair_cable_length_range,
                        repair_cable_curvature_range,
                        True,
                    ),
                    (
                        "maintenance_pipe_loose",
                        repair_pipe_probability,
                        repair_pipe_length_range,
                        repair_pipe_curvature_range,
                        False,
                    ),
                )
                for line_type, probability, length_range, curvature_range, is_cable in line_specs:
                    if random_variation:
                        if rng.random() > probability:
                            continue
                    elif probability < 0.5:
                        continue

                    length = _sample_float(length_range, rng, random_variation)
                    curvature = _sample_float(curvature_range, rng, random_variation)
                    if random_variation:
                        if rng.random() < 0.5:
                            curvature = -curvature
                    elif not is_cable:
                        curvature = -curvature

                    if is_cable:
                        line_mesh = create_cable_loose(
                            length=length,
                            curvature=curvature,
                            variation=_clamp(0.25 + chaos_factor * 0.25, 0.0, 1.0),
                            scatter=scatter_strength * (0.06 + chaos_factor * 0.08),
                            seed=seed + 2200 + scenario_index * 59,
                        )
                    else:
                        line_mesh = create_pipe_loose(
                            length=length,
                            curvature=curvature,
                            variation=_clamp(0.2 + chaos_factor * 0.2, 0.0, 1.0),
                            scatter=scatter_strength * (0.05 + chaos_factor * 0.07),
                            seed=seed + 2400 + scenario_index * 61,
                        )

                    if random_variation and component_anchors:
                        anchor_idx = rng.randint(0, len(component_anchors) - 1)
                    elif component_anchors:
                        anchor_idx = (scenario_index - 1) % len(component_anchors)
                    else:
                        anchor_idx = -1

                    if anchor_idx >= 0:
                        line_x, line_y, line_support_top = component_anchors[anchor_idx]
                    else:
                        line_x, line_y, line_support_top = (wx, wy, 0.0)

                    line_span = max(length * 0.42, 0.2)
                    jitter_x = line_span * (0.22 if random_variation else 0.08)
                    jitter_y = line_span * (0.18 if random_variation else 0.06)
                    line_x = _clamp(
                        line_x + rng.uniform(-jitter_x, jitter_x),
                        -half_w + edge_margin + line_span / 2.0,
                        half_w - edge_margin - line_span / 2.0,
                    )
                    line_y = _clamp(
                        line_y + rng.uniform(-jitter_y, jitter_y),
                        -half_d + edge_margin + line_span / 2.0,
                        half_d - edge_margin - line_span / 2.0,
                    )
                    if line_support_top <= 1e-6:
                        line_y = _clamp(
                            walkway_rule.auto_fix_y(line_y, line_span, margin=0.03),
                            -half_d + edge_margin + line_span / 2.0,
                            half_d - edge_margin - line_span / 2.0,
                        )

                    line_z = support_aligned_z(line_mesh, line_support_top, clearance=0.003)
                    line_yaw = rng.uniform(-30.0, 30.0) if random_variation else (12.0 if is_cable else -12.0)
                    emit(
                        line_type,
                        line_mesh,
                        (line_x, line_y, line_z),
                        (0.0, 0.0, line_yaw),
                    )
