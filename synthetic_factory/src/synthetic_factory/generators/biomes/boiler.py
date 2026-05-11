from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable, Mapping

from ...parametric.boiler_primitives import (
    create_bunker,
    create_boiler_unit,
    create_chimney,
    create_control_box,
    create_heat_exchanger,
    create_ladder,
    create_pipe,
    create_pipe_support,
    create_pump,
    create_platform,
    create_support_beam,
    create_tank,
    create_valve_cluster,
    create_valve,
)
from ...parametric.primitives import create_box, create_column
from .shared import (
    AddObjectFn,
    BiomeRoom,
    room_area_scale,
    symmetric_positions,
    to_bool,
    to_mapping,
    to_positive_float,
)

PRIMITIVES = (
    "boiler_unit",
    "pipe",
    "valve",
    "valve_cluster",
    "pump",
    "bunker",
    "heat_exchanger",
    "pipe_support",
    "control_box",
    "chimney",
    "platform",
    "ladder",
    "support_beam",
    "tank",
)
RULES: dict[str, object] = {
    "layout": "central axis boilers + connected pipes + level platforms",
    "patterns": [
        "clustered_boilers",
        "linear_boilers",
        "central_tower",
        "dense_industrial",
    ],
    "rules": [
        "MinClearanceRule",
        "AccessibilityRule",
        "HeightConstraint",
        "PipeConnectivityRule",
        "PlatformSupportRule",
    ],
    "auto_fix": True,
    "boiler_count": "1..3",
    "pipe_count": "proportional to boiler_count and pipe density",
    "platform_levels": "1..3 levels tied to boiler height",
    "tanks": "0..5 on perimeter",
    "beams": "density-controlled near ceiling for pipe support",
    "collision_policy": "2D floor footprint checks",
    "access_policy": "preserve central walkway lane",
}


@dataclass(frozen=True)
class _Footprint:
    x: float
    y: float
    half_w: float
    half_d: float


Vector3 = tuple[float, float, float]
AxisToXYFn = Callable[[float, float], tuple[float, float]]
EmitFn = Callable[[str, object, Vector3, Vector3, float | None], None]


@dataclass(frozen=True)
class MinClearanceRule:
    """Guarantees minimal center-to-center spacing for axis-aligned objects."""

    minimum_distance: float

    def enforce_axis_spacing(
        self,
        axis_positions: list[float],
        axis_limit: float,
        object_radius: float,
    ) -> tuple[list[float], bool]:
        if len(axis_positions) <= 1:
            return (list(axis_positions), False)

        limit = max(0.0, float(axis_limit))
        required = max(0.0, 2.0 * float(object_radius) + self.minimum_distance)
        if limit <= 0.0 or required <= 1e-9:
            return (sorted(float(value) for value in axis_positions), False)

        values = sorted(float(value) for value in axis_positions)
        changed = False

        span = 2.0 * limit
        required_span = required * (len(values) - 1)
        if required_span > span + 1e-9:
            # Keep all objects within room bounds even when hard constraint is impossible.
            step = span / (len(values) - 1)
            start = -limit
            return ([start + step * idx for idx in range(len(values))], True)

        for idx in range(1, len(values)):
            minimum_value = values[idx - 1] + required
            if values[idx] < minimum_value:
                values[idx] = minimum_value
                changed = True

        overflow = values[-1] - limit
        if overflow > 1e-9:
            values = [value - overflow for value in values]
            changed = True

        underflow = -limit - values[0]
        if underflow > 1e-9:
            values = [value + underflow for value in values]
            changed = True

        clamped = [max(-limit, min(limit, value)) for value in values]
        if any(abs(left - right) > 1e-9 for left, right in zip(values, clamped)):
            changed = True
        return (clamped, changed)


@dataclass(frozen=True)
class AccessibilityRule:
    """Ensures boiler access path and service areas are present."""

    min_walkway: float
    edge_offset: float
    corridor_thickness: float = 0.04

    def corridor_dims(
        self,
        axis_along_x: bool,
        axis_length: float,
    ) -> tuple[float, float, Vector3]:
        corridor_length = max(axis_length - 2.0 * self.edge_offset, self.min_walkway)
        if axis_along_x:
            return (
                corridor_length,
                self.min_walkway,
                (0.0, 0.0, 0.0),
            )
        return (
            corridor_length,
            self.min_walkway,
            (0.0, 0.0, 90.0),
        )

    def ensure_service_zones(
        self,
        boiler_axes: list[float],
        existing_service_zone_count: int,
        zone_size: float,
        axis_to_xy: AxisToXYFn,
        emit: EmitFn,
    ) -> int:
        """Auto-fix: add missing service zones if some boilers have none."""
        if existing_service_zone_count >= len(boiler_axes):
            return 0

        fixed = 0
        missing_axes = boiler_axes[existing_service_zone_count:]
        for axis_value in missing_axes:
            x, y = axis_to_xy(axis_value, 0.0)
            emit(
                "service_zone",
                create_box(width=zone_size, height=self.corridor_thickness, depth=zone_size),
                (x, y, self.corridor_thickness / 2.0),
                (0.0, 0.0, 0.0),
                self.corridor_thickness,
            )
            fixed += 1
        return fixed


@dataclass(frozen=True)
class HeightConstraint:
    """Clamps object size/position to keep geometry inside room height."""

    max_height: float
    floor_height: float = 0.0
    margin: float = 0.02

    def clamp_size(self, size: float, minimum: float = 0.02) -> float:
        upper = max(minimum, self.max_height - self.floor_height - 2.0 * self.margin)
        return max(minimum, min(float(size), upper))

    def clamp_center(self, center_z: float, object_height: float) -> float:
        half = max(object_height / 2.0, 0.0)
        lower = self.floor_height + self.margin + half
        upper = self.max_height - self.margin - half
        if lower > upper:
            return (self.floor_height + self.max_height) / 2.0
        return max(lower, min(float(center_z), upper))

    def clamp_pose(
        self,
        center_z: float,
        object_height: float,
        minimum_height: float = 0.02,
    ) -> tuple[float, float, bool]:
        clamped_height = self.clamp_size(object_height, minimum=minimum_height)
        clamped_center = self.clamp_center(center_z, clamped_height)
        changed = (
            abs(clamped_height - object_height) > 1e-9
            or abs(clamped_center - center_z) > 1e-9
        )
        return (clamped_center, clamped_height, changed)


@dataclass
class _PipeConnectivityState:
    riser_axes: set[int]
    has_main_manifold: bool
    link_count: int


@dataclass(frozen=True)
class PipeConnectivityRule:
    """Guarantees that boilers are connected by a coherent pipe network."""

    def ensure_connectivity(
        self,
        state: _PipeConnectivityState,
        boiler_axes: list[float],
        axis_to_xy: AxisToXYFn,
        axis_rotation: Vector3,
        emit: EmitFn,
        boiler_height: float,
        pipe_z: float,
        pipe_radius: float,
    ) -> int:
        fixes = 0
        if not boiler_axes:
            return fixes

        for axis_value in boiler_axes:
            key = _axis_key(axis_value)
            if key in state.riser_axes:
                continue
            x, y = axis_to_xy(axis_value, 0.0)
            riser_len = max(pipe_z - boiler_height, 0.35)
            emit(
                "riser_pipe",
                create_pipe(
                    radius=max(pipe_radius * 0.95, 0.04),
                    length=riser_len,
                    bend_angle=0.0,
                ),
                (x, y, boiler_height + riser_len / 2.0),
                (0.0, 0.0, 0.0),
                riser_len,
            )
            state.riser_axes.add(key)
            fixes += 1

        if not state.has_main_manifold:
            if len(boiler_axes) == 1:
                manifold_len = max(2.0 * pipe_radius * 10.0, 1.2)
                center_axis = boiler_axes[0]
            else:
                manifold_len = max(boiler_axes[-1] - boiler_axes[0] + 2.0, 1.2)
                center_axis = (boiler_axes[0] + boiler_axes[-1]) / 2.0
            mx, my = axis_to_xy(center_axis, 0.0)
            emit(
                "pipe_manifold_main",
                create_pipe(radius=pipe_radius, length=manifold_len, bend_angle=0.0),
                (mx, my, pipe_z),
                axis_rotation,
                max(pipe_radius * 2.0, 0.08),
            )
            state.has_main_manifold = True
            fixes += 1

        if len(boiler_axes) > 1 and state.link_count == 0:
            for idx in range(len(boiler_axes) - 1):
                left = boiler_axes[idx]
                right = boiler_axes[idx + 1]
                seg_len = abs(right - left)
                if seg_len <= 0.2:
                    continue
                cx, cy = axis_to_xy((left + right) / 2.0, 0.0)
                emit(
                    "pipe_link",
                    create_pipe(
                        radius=max(pipe_radius * 0.95, 0.04),
                        length=seg_len,
                        bend_angle=0.0,
                    ),
                    (cx, cy, pipe_z * 0.98),
                    axis_rotation,
                    max(pipe_radius * 2.0, 0.08),
                )
                state.link_count += 1
                fixes += 1
        return fixes


@dataclass(frozen=True)
class PlatformSupportRule:
    """Ensures every service platform is structurally supported."""

    support_radius: float
    corner_inset: float = 0.2
    min_support_height: float = 0.35

    def ensure_supports(
        self,
        platform_center: Vector3,
        platform_width: float,
        platform_depth: float,
        platform_thickness: float,
        height_constraint: HeightConstraint,
        emit: EmitFn,
    ) -> int:
        support_height = max(platform_center[2] - platform_thickness / 2.0, self.min_support_height)
        support_height = height_constraint.clamp_size(
            support_height,
            minimum=self.min_support_height,
        )
        if support_height <= 0.0:
            return 0

        support_z = height_constraint.clamp_center(support_height / 2.0, support_height)
        inset_x = max(self.corner_inset, self.support_radius * 2.0)
        inset_y = max(self.corner_inset, self.support_radius * 2.0)
        inset_x = min(inset_x, max(platform_width / 2.0 - self.support_radius * 1.2, 0.0))
        inset_y = min(inset_y, max(platform_depth / 2.0 - self.support_radius * 1.2, 0.0))

        offsets = (
            (-inset_x, -inset_y),
            (-inset_x, inset_y),
            (inset_x, -inset_y),
            (inset_x, inset_y),
        )

        fixed = 0
        for dx, dy in offsets:
            emit(
                "platform_support",
                create_column(radius=self.support_radius, height=support_height, segments=12),
                (platform_center[0] + dx, platform_center[1] + dy, support_z),
                (0.0, 0.0, 0.0),
                support_height,
            )
            fixed += 1
        return fixed


PATTERN_ALIASES = {
    "clustered": "clustered_boilers",
    "clustered_boilers": "clustered_boilers",
    "cluster": "clustered_boilers",
    "linear": "linear_boilers",
    "line": "linear_boilers",
    "linear_boilers": "linear_boilers",
    "central": "central_tower",
    "tower": "central_tower",
    "central_tower": "central_tower",
    "dense": "dense_industrial",
    "dense_industrial": "dense_industrial",
    "industrial_dense": "dense_industrial",
}


def populate(room: BiomeRoom, add_object: AddObjectFn, settings: Mapping[str, object]) -> None:
    cfg = to_mapping(settings.get("boiler"))
    rules_cfg = to_mapping(cfg.get("rules"))
    boiler_params = to_mapping(cfg.get("boiler"))
    pipe_params = to_mapping(cfg.get("pipes"))
    platform_params = to_mapping(cfg.get("platforms"))
    tank_params = to_mapping(cfg.get("tanks"))
    structure_params = to_mapping(cfg.get("structure"))

    seed = int(cfg.get("seed", 0)) + _stable_hash(room.room_id)
    rng = Random(seed)
    random_variation = to_bool(cfg.get("random_variation", True), default=True)
    pattern = _normalize_pattern(str(cfg.get("pattern", "linear_boilers")))
    auto_fix = to_bool(rules_cfg.get("auto_fix", True), default=True)

    min_walkway = to_positive_float(
        cfg.get("min_walkway"),
        1.2,
        "machinery.boiler.min_walkway",
    )
    min_clearance = to_positive_float(
        rules_cfg.get("min_clearance", min_walkway),
        min_walkway,
        "machinery.boiler.rules.min_clearance",
    )
    service_clearance = to_positive_float(
        cfg.get("service_clearance"),
        0.8,
        "machinery.boiler.service_clearance",
    )
    edge_offset = to_positive_float(
        cfg.get("edge_offset"),
        1.0,
        "machinery.boiler.edge_offset",
    )
    clearance_rule = MinClearanceRule(minimum_distance=min_clearance)
    accessibility_rule = AccessibilityRule(
        min_walkway=min_walkway,
        edge_offset=edge_offset,
    )
    height_constraint = HeightConstraint(
        max_height=room.height,
        floor_height=0.0,
        margin=0.02,
    )
    pipe_connectivity_rule = PipeConnectivityRule()

    # New parameter schema with backward-compatible fallback.
    boiler_count_range = _parse_int_range(
        boiler_params.get("count", cfg.get("boiler_count_range", [1, 3])),
        default_min=1,
        default_max=3,
        label="machinery.boiler.boiler.count",
    )
    boiler_height_range = _parse_float_range(
        boiler_params.get("height", cfg.get("boiler_height_range", [8.0, 20.0])),
        default_min=8.0,
        default_max=20.0,
        label="machinery.boiler.boiler.height",
    )
    boiler_radius_range = _parse_float_range(
        boiler_params.get("radius", cfg.get("boiler_radius_range", [1.0, 3.0])),
        default_min=1.0,
        default_max=3.0,
        label="machinery.boiler.boiler.radius",
    )

    pipe_density_range = _parse_float_range(
        pipe_params.get("density", cfg.get("pipe_density_range", [0.5, 2.0])),
        default_min=0.5,
        default_max=2.0,
        label="machinery.boiler.pipes.density",
    )
    pipe_radius_range = _parse_float_range(
        pipe_params.get("radius", cfg.get("pipe_radius_range", [0.1, 0.5])),
        default_min=0.1,
        default_max=0.5,
        label="machinery.boiler.pipes.radius",
    )

    platform_levels_range = _parse_int_range(
        platform_params.get("levels", [1, 3]),
        default_min=1,
        default_max=3,
        label="machinery.boiler.platforms.levels",
    )
    platform_width_factor_range = _parse_float_range(
        platform_params.get("width_factor", [1.2, 2.0]),
        default_min=1.2,
        default_max=2.0,
        label="machinery.boiler.platforms.width_factor",
    )

    tank_count_range = _parse_int_range(
        tank_params.get("count", cfg.get("tank_count_range", [0, 5])),
        default_min=0,
        default_max=5,
        label="machinery.boiler.tanks.count",
    )

    beam_density_range = _parse_float_range(
        structure_params.get("beam_density", [0.2, 1.0]),
        default_min=0.2,
        default_max=1.0,
        label="machinery.boiler.structure.beam_density",
    )

    boiler_count_target = _sample_int(boiler_count_range, rng, random_variation)
    sampled_boiler_height = _sample_float(boiler_height_range, rng, random_variation)
    sampled_boiler_radius = _sample_float(boiler_radius_range, rng, random_variation)
    sampled_pipe_density = _sample_float(pipe_density_range, rng, random_variation)
    sampled_pipe_radius = _sample_float(pipe_radius_range, rng, random_variation)
    sampled_levels_count = _sample_int(platform_levels_range, rng, random_variation)
    sampled_width_factor = _sample_float(platform_width_factor_range, rng, random_variation)
    sampled_tank_count = _sample_int(tank_count_range, rng, random_variation)
    sampled_beam_density = _sample_float(beam_density_range, rng, random_variation)

    # Large boiler rooms should receive denser content automatically.
    area_scale = room_area_scale(room, reference_area=260.0, min_scale=0.75, max_scale=4.0)
    count_scale = max(1.0, area_scale)
    boiler_count_target = max(1, int(round(boiler_count_target * count_scale)))
    sampled_tank_count = max(0, int(round(sampled_tank_count * (0.8 + 0.45 * count_scale))))
    sampled_levels_count = max(1, int(round(sampled_levels_count * min(count_scale, 2.0))))
    sampled_pipe_density = sampled_pipe_density * (0.9 + 0.25 * count_scale)
    sampled_beam_density = sampled_beam_density * (0.9 + 0.2 * count_scale)

    if pattern == "central_tower":
        boiler_count_target = 1
        sampled_boiler_radius *= 1.35
        sampled_boiler_height *= 1.2
        sampled_pipe_density = max(sampled_pipe_density, 1.0)
        sampled_levels_count = max(2, sampled_levels_count)
        sampled_tank_count = max(sampled_tank_count, 2)
        sampled_beam_density = max(sampled_beam_density, 0.55)
    elif pattern == "clustered_boilers":
        boiler_count_target = max(2, boiler_count_target)
        sampled_pipe_density = max(sampled_pipe_density, 1.0)
        sampled_levels_count = max(2, sampled_levels_count)
    elif pattern == "dense_industrial":
        boiler_count_target = max(2, boiler_count_target)
        sampled_pipe_density = max(sampled_pipe_density * 1.5, 1.6)
        sampled_levels_count = max(2, sampled_levels_count)
        sampled_tank_count = max(sampled_tank_count, 2)
        sampled_beam_density = max(sampled_beam_density, 0.9)

    dynamic_count_max = max(
        boiler_count_range[1],
        int(round(boiler_count_range[1] * max(1.0, count_scale))),
    )
    dynamic_count_max = max(1, min(dynamic_count_max, 24))
    dynamic_count_min = max(1, boiler_count_range[0])

    # Keep sampled values feasible for the current room shell.
    max_feasible_radius = max(min(room.width, room.depth) * 0.22, 0.25)
    boiler_radius = min(sampled_boiler_radius, max_feasible_radius)
    boiler_radius = max(0.2, boiler_radius)

    boiler_height = min(sampled_boiler_height, room.height * 0.92)
    boiler_height = max(1.2, boiler_height)
    boiler_height = height_constraint.clamp_size(boiler_height, minimum=1.2)

    structure_ceiling = structure_params.get("ceiling_height", "auto")
    if isinstance(structure_ceiling, (int, float)):
        target_ceiling = float(structure_ceiling)
    else:
        target_ceiling = boiler_height_range[1] * 1.2
    ceiling_height = min(room.height, max(target_ceiling, boiler_height * 1.05))

    beam_z = min(room.height - 0.12, ceiling_height - 0.2)
    if beam_z <= boiler_height + 0.2:
        beam_z = min(room.height - 0.1, boiler_height + 0.35)

    pipe_z = min(beam_z - 0.15, boiler_height + max(0.8, 0.4 * boiler_height))
    pipe_z = max(pipe_z, boiler_height + 0.25)

    tank_radius = max(
        0.35,
        min(boiler_radius * 0.8, min(room.width, room.depth) * 0.2),
    )
    tank_height = min(room.height * 0.78, max(boiler_height * 0.8, 2.0))
    tank_height = height_constraint.clamp_size(tank_height, minimum=0.8)

    platform_thickness = to_positive_float(
        platform_params.get("thickness", cfg.get("platform_thickness", 0.18)),
        0.18,
        "machinery.boiler.platforms.thickness",
    )
    ladder_width = to_positive_float(
        platform_params.get("ladder_width", cfg.get("ladder_width", 0.75)),
        0.75,
        "machinery.boiler.platforms.ladder_width",
    )

    beam_offset = to_positive_float(
        structure_params.get("beam_offset", cfg.get("beam_offset", 1.6)),
        1.6,
        "machinery.boiler.structure.beam_offset",
    )
    beam_profile = to_mapping(
        structure_params.get("support_beam_profile", cfg.get("support_beam_profile"))
    )
    if not beam_profile:
        beam_profile = {
            "type": "i",
            "width": 0.22,
            "height": 0.32,
            "web_thickness": 0.016,
            "flange_thickness": 0.022,
        }
    try:
        beam_nominal_height = max(float(beam_profile.get("height", 0.32)), 0.05)
    except (TypeError, ValueError):
        beam_nominal_height = 0.32
    pipe_nominal_height = max(sampled_pipe_radius * 2.0, 0.06)
    beam_z = height_constraint.clamp_center(beam_z, beam_nominal_height)
    pipe_z = height_constraint.clamp_center(pipe_z, pipe_nominal_height)

    axis_along_x = room.width >= room.depth
    axis_length = room.width if axis_along_x else room.depth
    cross_length = room.depth if axis_along_x else room.width
    axis_rotation = (0.0, 0.0, 0.0) if axis_along_x else (0.0, 0.0, 90.0)
    cross_rotation = (0.0, 0.0, 90.0) if axis_along_x else (0.0, 0.0, 0.0)

    half_w = room.width / 2.0
    half_d = room.depth / 2.0

    def axis_to_xy(axis_value: float, lateral: float = 0.0) -> tuple[float, float]:
        if axis_along_x:
            return (axis_value, lateral)
        return (lateral, axis_value)

    counters: dict[str, int] = {}

    def emit(
        object_type: str,
        mesh: object,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        nominal_height: float | None = None,
    ) -> None:
        final_position = position
        if nominal_height is not None:
            clamped_z = height_constraint.clamp_center(position[2], nominal_height)
            final_position = (position[0], position[1], clamped_z)
        counters[object_type] = counters.get(object_type, 0) + 1
        add_object(
            object_type,
            counters[object_type],
            mesh,
            final_position,
            rotation,
        )

    footprints: list[_Footprint] = []

    def try_place_footprint(
        x: float,
        y: float,
        width: float,
        depth: float,
        clearance: float,
    ) -> bool:
        candidate = _Footprint(
            x=x,
            y=y,
            half_w=width / 2.0 + clearance,
            half_d=depth / 2.0 + clearance,
        )
        if abs(candidate.x) + candidate.half_w > half_w - 0.02:
            return False
        if abs(candidate.y) + candidate.half_d > half_d - 0.02:
            return False
        for existing in footprints:
            if _intersects(candidate, existing):
                return False
        footprints.append(candidate)
        return True

    # 1) Central axis + 2) Boiler placement.
    axis_margin = edge_offset + boiler_radius + service_clearance
    axis_limit = max(axis_length / 2.0 - axis_margin, 0.0)
    boiler_count = _fit_boiler_count(
        axis_limit=axis_limit,
        boiler_radius=boiler_radius,
        min_walkway=min_walkway,
        preferred_count=boiler_count_target,
        count_min=dynamic_count_min,
        count_max=dynamic_count_max,
    )

    if pattern == "central_tower":
        boiler_axis_positions = [0.0]
    elif pattern == "clustered_boilers":
        cluster_span = max(axis_limit * 0.42, boiler_radius * 2.0)
        boiler_axis_positions = (
            symmetric_positions(boiler_count, cluster_span)
            if boiler_count > 1
            else [0.0]
        )
        if random_variation and axis_limit > 0.2:
            shift = rng.uniform(-axis_limit * 0.12, axis_limit * 0.12)
            boiler_axis_positions = [
                max(-axis_limit, min(axis_limit, pos + shift))
                for pos in boiler_axis_positions
            ]
    elif pattern == "dense_industrial":
        dense_span = max(axis_limit * 0.95, boiler_radius * 2.5)
        boiler_axis_positions = (
            symmetric_positions(boiler_count, dense_span)
            if boiler_count > 1
            else [0.0]
        )
    else:
        boiler_axis_positions = (
            symmetric_positions(boiler_count, axis_limit)
            if boiler_count > 1
            else [0.0]
        )
    if auto_fix:
        boiler_axis_positions, _ = clearance_rule.enforce_axis_spacing(
            axis_positions=boiler_axis_positions,
            axis_limit=axis_limit,
            object_radius=boiler_radius,
        )
    placed_boiler_axis: list[float] = []

    zone_size = max(2.0 * (boiler_radius + service_clearance), 1.2)
    for axis_value in boiler_axis_positions:
        x, y = axis_to_xy(axis_value, 0.0)
        if not try_place_footprint(
            x=x,
            y=y,
            width=2.0 * boiler_radius,
            depth=2.0 * boiler_radius,
            clearance=0.05,
        ):
            continue

        placed_boiler_axis.append(axis_value)
        emit(
            "boiler_unit",
            create_boiler_unit(radius=boiler_radius, height=boiler_height, segment_count=24),
            (x, y, boiler_height / 2.0),
            nominal_height=boiler_height,
        )
        emit(
            "service_zone",
            create_box(width=zone_size, height=0.04, depth=zone_size),
            (x, y, 0.02),
            nominal_height=0.04,
        )

    if not placed_boiler_axis:
        placed_boiler_axis = [0.0]
        x, y = axis_to_xy(0.0, 0.0)
        footprints.append(
            _Footprint(
                x=x,
                y=y,
                half_w=boiler_radius + 0.05,
                half_d=boiler_radius + 0.05,
            )
        )
        emit(
            "boiler_unit",
            create_boiler_unit(radius=boiler_radius, height=boiler_height, segment_count=24),
            (x, y, boiler_height / 2.0),
            nominal_height=boiler_height,
        )
        emit(
            "service_zone",
            create_box(width=zone_size, height=0.04, depth=zone_size),
            (x, y, 0.02),
            nominal_height=0.04,
        )

    placed_boiler_axis.sort()

    if auto_fix:
        accessibility_rule.ensure_service_zones(
            boiler_axes=placed_boiler_axis,
            existing_service_zone_count=counters.get("service_zone", 0),
            zone_size=zone_size,
            axis_to_xy=axis_to_xy,
            emit=emit,
        )

        corridor_width, corridor_depth, corridor_rotation = accessibility_rule.corridor_dims(
            axis_along_x=axis_along_x,
            axis_length=axis_length,
        )
        emit(
            "access_corridor",
            create_box(
                width=corridor_width,
                height=accessibility_rule.corridor_thickness,
                depth=corridor_depth,
            ),
            (0.0, 0.0, accessibility_rule.corridor_thickness / 2.0),
            corridor_rotation,
            nominal_height=accessibility_rule.corridor_thickness,
        )

    if pattern == "central_tower":
        aux_ring = max(boiler_radius + min_walkway + 0.35, 0.8)
        for idx, lateral in enumerate(symmetric_positions(2, aux_ring)):
            ax, ay = axis_to_xy(0.0, lateral)
            if try_place_footprint(
                x=ax,
                y=ay,
                width=0.8,
                depth=0.8,
                clearance=0.05,
            ):
                emit(
                    "aux_module",
                    create_box(width=0.8, height=1.0, depth=0.8),
                    (ax, ay, 0.5),
                    nominal_height=1.0,
                )
                emit(
                    "pipe_valve",
                    create_valve(
                        radius=max(sampled_pipe_radius * 1.05, 0.05),
                        handle_size=max(sampled_pipe_radius * 2.8, 0.2),
                    ),
                    (ax, ay, min(pipe_z * 0.72, boiler_height * 0.85)),
                    cross_rotation if idx % 2 == 0 else axis_rotation,
                    nominal_height=max(sampled_pipe_radius * 3.1, 0.25),
                )

    # 3) Pipe system: risers + horizontal links + seeded random branches.
    pipe_state = _PipeConnectivityState(
        riser_axes=set(),
        has_main_manifold=False,
        link_count=0,
    )
    for axis_value in placed_boiler_axis:
        x, y = axis_to_xy(axis_value, 0.0)
        riser_len = max(pipe_z - boiler_height, 0.35)
        emit(
            "riser_pipe",
            create_pipe(radius=max(sampled_pipe_radius * 0.95, 0.04), length=riser_len, bend_angle=0.0),
            (x, y, boiler_height + riser_len / 2.0),
            nominal_height=riser_len,
        )
        pipe_state.riser_axes.add(_axis_key(axis_value))

    if len(placed_boiler_axis) == 1:
        manifold_len = max(2.0 * boiler_radius * 1.8, 1.2)
        center_axis = placed_boiler_axis[0]
    else:
        manifold_len = max(
            placed_boiler_axis[-1] - placed_boiler_axis[0] + 2.0 * boiler_radius,
            1.2,
        )
        center_axis = (placed_boiler_axis[0] + placed_boiler_axis[-1]) / 2.0

    mx, my = axis_to_xy(center_axis, 0.0)
    emit(
        "pipe_manifold_main",
        create_pipe(radius=sampled_pipe_radius, length=manifold_len, bend_angle=0.0),
        (mx, my, pipe_z),
        axis_rotation,
        nominal_height=pipe_nominal_height,
    )
    pipe_state.has_main_manifold = True

    if pattern == "clustered_boilers" and len(placed_boiler_axis) >= 2:
        cluster_cross_len = max(
            min(cross_length - 2.0 * edge_offset, 2.0 * (boiler_radius + min_walkway)),
            1.0,
        )
        emit(
            "pipe_manifold_cluster",
            create_pipe(radius=max(sampled_pipe_radius * 0.92, 0.05), length=cluster_cross_len, bend_angle=0.0),
            (mx, my, pipe_z * 0.96),
            cross_rotation,
            nominal_height=pipe_nominal_height,
        )
        pipe_state.link_count += 1

    if pattern == "dense_industrial":
        side_offset = min(max(min_walkway + boiler_radius, 0.8), cross_length / 2.0 - edge_offset)
        if side_offset > 0.25:
            emit(
                "pipe_manifold_side",
                create_pipe(radius=max(sampled_pipe_radius * 0.85, 0.05), length=manifold_len * 0.95, bend_angle=0.0),
                axis_to_xy(center_axis, side_offset) + (pipe_z * 0.93,),
                axis_rotation,
                nominal_height=pipe_nominal_height,
            )
            emit(
                "pipe_manifold_side",
                create_pipe(radius=max(sampled_pipe_radius * 0.85, 0.05), length=manifold_len * 0.95, bend_angle=0.0),
                axis_to_xy(center_axis, -side_offset) + (pipe_z * 0.93,),
                axis_rotation,
                nominal_height=pipe_nominal_height,
            )
            pipe_state.link_count += 2

    for idx in range(len(placed_boiler_axis) - 1):
        left = placed_boiler_axis[idx]
        right = placed_boiler_axis[idx + 1]
        seg_len = abs(right - left)
        if seg_len <= 0.2:
            continue
        cx, cy = axis_to_xy((left + right) / 2.0, 0.0)
        emit(
            "pipe_link",
            create_pipe(radius=max(sampled_pipe_radius * 0.95, 0.04), length=seg_len, bend_angle=0.0),
            (cx, cy, pipe_z * 0.98),
            axis_rotation,
            nominal_height=pipe_nominal_height,
        )
        pipe_state.link_count += 1

    branch_count = max(1, int(round(len(placed_boiler_axis) * sampled_pipe_density)))
    lateral_limit = max(cross_length / 2.0 - edge_offset - 0.35, min_walkway + 0.4)
    for _ in range(branch_count):
        source_axis = rng.choice(placed_boiler_axis)
        side = -1.0 if rng.random() < 0.5 else 1.0
        lateral_target = side * (
            min_walkway + (lateral_limit - min_walkway) * (0.45 + 0.5 * rng.random())
        )
        branch_len = abs(lateral_target)
        if branch_len <= 0.25:
            continue

        bend_angle = rng.uniform(20.0, 60.0) if random_variation else 35.0
        bx, by = axis_to_xy(source_axis, lateral_target / 2.0)
        emit(
            "pipe_branch",
            create_pipe(
                radius=max(sampled_pipe_radius * 0.8, 0.05),
                length=branch_len,
                bend_angle=bend_angle,
                segment_count=18,
            ),
            (bx, by, pipe_z * 0.92),
            cross_rotation,
            nominal_height=pipe_nominal_height,
        )

        vx, vy = axis_to_xy(source_axis, side * min(branch_len * 0.22, 0.48))
        emit(
            "pipe_valve",
            create_valve(
                radius=max(sampled_pipe_radius * 1.2, 0.05),
                handle_size=max(sampled_pipe_radius * 3.0, 0.22),
            ),
            (vx, vy, pipe_z * 0.92),
            cross_rotation,
            nominal_height=max(sampled_pipe_radius * 3.1, 0.25),
        )

    if auto_fix:
        pipe_connectivity_rule.ensure_connectivity(
            state=pipe_state,
            boiler_axes=placed_boiler_axis,
            axis_to_xy=axis_to_xy,
            axis_rotation=axis_rotation,
            emit=emit,
            boiler_height=boiler_height,
            pipe_z=pipe_z,
            pipe_radius=sampled_pipe_radius,
        )

    # 4) Platforms tied to boiler height.
    axis_span = max(
        placed_boiler_axis[-1] - placed_boiler_axis[0] + 2.0 * boiler_radius * sampled_width_factor,
        2.0,
    )
    cross_span = max(
        2.0 * (boiler_radius + min_walkway * 0.35) * sampled_width_factor,
        1.6,
    )
    axis_span = min(axis_span, axis_length - 2.0 * edge_offset)
    cross_span = min(cross_span, cross_length - 2.0 * edge_offset)

    platform_width = axis_span if axis_along_x else cross_span
    platform_depth = cross_span if axis_along_x else axis_span
    platform_center_axis = (placed_boiler_axis[0] + placed_boiler_axis[-1]) / 2.0
    px, py = axis_to_xy(platform_center_axis, 0.0)
    platform_support_rule = PlatformSupportRule(
        support_radius=max(0.05, min(0.12, platform_thickness * 0.4))
    )

    level_count = max(1, min(3, sampled_levels_count))
    platform_z_levels: list[float] = []
    for idx in range(1, level_count + 1):
        ratio = idx / (level_count + 1)
        level_z = min(boiler_height * ratio, beam_z - 0.25)
        if level_z <= platform_thickness * 1.6:
            continue
        platform_z_levels.append(level_z)
        emit(
            "service_platform",
            create_platform(width=platform_width, depth=platform_depth, height=platform_thickness),
            (px, py, level_z),
            nominal_height=platform_thickness,
        )
        if auto_fix:
            platform_support_rule.ensure_supports(
                platform_center=(px, py, level_z),
                platform_width=platform_width,
                platform_depth=platform_depth,
                platform_thickness=platform_thickness,
                height_constraint=height_constraint,
                emit=emit,
            )

    # 5) Ladders between platforms.
    if len(platform_z_levels) >= 2:
        lateral = max(min(platform_depth / 2.0 - 0.3, cross_length / 2.0 - edge_offset), 0.55)
        ladder_anchors = [placed_boiler_axis[0], placed_boiler_axis[-1]]
        for idx in range(len(platform_z_levels) - 1):
            z_low = platform_z_levels[idx]
            z_high = platform_z_levels[idx + 1]
            ladder_height = z_high - z_low
            steps = max(4, int(ladder_height / 0.28))
            for anchor_idx, axis_value in enumerate(ladder_anchors):
                side = -1.0 if anchor_idx % 2 == 0 else 1.0
                lx, ly = axis_to_xy(axis_value, side * lateral)
                if not try_place_footprint(
                    x=lx,
                    y=ly,
                    width=ladder_width,
                    depth=0.35,
                    clearance=max(min_walkway * 0.1, 0.08),
                ):
                    continue
                emit(
                    "ladder",
                    create_ladder(height=ladder_height, step_count=steps),
                    (lx, ly, (z_low + z_high) / 2.0),
                    cross_rotation,
                    nominal_height=ladder_height,
                )

    # 6) Tanks on perimeter.
    tank_count = sampled_tank_count
    axis_perimeter = max(axis_length / 2.0 - edge_offset - tank_radius, 0.45)
    cross_perimeter = max(cross_length / 2.0 - edge_offset - tank_radius, 0.45)
    perimeter_candidates: list[tuple[float, float]] = []
    for side in (-1.0, 1.0):
        for a in symmetric_positions(max(2, tank_count if tank_count > 0 else 2), axis_perimeter * 0.85):
            perimeter_candidates.append((a, side * cross_perimeter))
    for side in (-1.0, 1.0):
        for l in symmetric_positions(max(2, tank_count if tank_count > 0 else 2), cross_perimeter * 0.75):
            perimeter_candidates.append((side * axis_perimeter, l))
    rng.shuffle(perimeter_candidates)

    access_lane = min(
        boiler_radius + min_walkway + service_clearance,
        max(cross_perimeter * 0.45, min_walkway * 0.5),
    )
    placed_tanks = 0
    for axis_value, lateral_value in perimeter_candidates:
        if placed_tanks >= tank_count:
            break
        if abs(lateral_value) < access_lane:
            continue
        tx, ty = axis_to_xy(axis_value, lateral_value)
        if not try_place_footprint(
            x=tx,
            y=ty,
            width=2.0 * tank_radius,
            depth=2.0 * tank_radius,
            clearance=0.04,
        ):
            continue
        emit(
            "tank",
            create_tank(radius=tank_radius, height=tank_height),
            (tx, ty, tank_height / 2.0),
            nominal_height=tank_height,
        )
        placed_tanks += 1

    if tank_count > 0 and placed_tanks == 0:
        fallback_radius = max(0.25, tank_radius * 0.7)
        fallback_height = height_constraint.clamp_size(tank_height, minimum=0.8)
        fallback_lateral = max(cross_perimeter, access_lane)
        fx, fy = axis_to_xy(0.0, fallback_lateral)
        emit(
            "tank",
            create_tank(radius=fallback_radius, height=fallback_height),
            (fx, fy, fallback_height / 2.0),
            nominal_height=fallback_height,
        )
        placed_tanks += 1

    # 7) Ceiling beams supporting pipes.
    beam_count = max(1, int(round(max(1, len(placed_boiler_axis)) * sampled_beam_density * 2.0)))
    beam_count = min(beam_count, 6)
    beam_lateral_max = max(cross_length / 2.0 - edge_offset, 0.4)
    lateral_positions = symmetric_positions(beam_count, min(beam_offset, beam_lateral_max))
    beam_length = max(axis_length - 2.0 * edge_offset, axis_span)
    for lateral in lateral_positions:
        sx, sy = axis_to_xy(platform_center_axis, lateral)
        emit(
            "support_beam",
            create_support_beam(length=beam_length, profile_type=beam_profile),
            (sx, sy, beam_z),
            axis_rotation,
            nominal_height=beam_nominal_height,
        )

    cross_beam_count = max(1, int(round(sampled_beam_density * len(placed_boiler_axis))))
    cross_beam_count = min(cross_beam_count, len(placed_boiler_axis))
    if cross_beam_count > 0:
        if random_variation:
            selected_axes = rng.sample(placed_boiler_axis, cross_beam_count)
        else:
            selected_axes = placed_boiler_axis[:cross_beam_count]
        cross_beam_length = max(min(cross_length - 2.0 * edge_offset, 2.5 * boiler_radius), 1.0)
        for axis_value in selected_axes:
            cx, cy = axis_to_xy(axis_value, 0.0)
            emit(
                "support_beam",
                create_support_beam(length=cross_beam_length, profile_type=beam_profile),
                (cx, cy, beam_z),
                cross_rotation,
                nominal_height=beam_nominal_height,
            )

    hanger_height = max(beam_z - pipe_z, 0.2)
    for axis_value in placed_boiler_axis:
        hx, hy = axis_to_xy(axis_value, 0.0)
        emit(
            "pipe_hanger",
            create_box(width=0.08, height=hanger_height, depth=0.08),
            (hx, hy, pipe_z + hanger_height / 2.0),
            nominal_height=hanger_height,
        )

    # 8) Secondary equipment tied to the boiler system.
    pump_orientation = "x" if axis_along_x else "y"
    pump_size = max(boiler_radius * 0.75, 0.65)
    valve_cluster_count = max(2, min(5, int(round(2.0 + sampled_pipe_density))))
    valve_cluster_spacing = max(sampled_pipe_radius * 4.2, 0.26)
    chimney_radius = max(sampled_pipe_radius * 1.7, boiler_radius * 0.26, 0.15)
    chimney_height_available = room.height - boiler_height - 0.08
    if chimney_height_available > 0.45:
        chimney_height = min(max(boiler_height * 0.52, 0.9), chimney_height_available)
        chimney_height = height_constraint.clamp_size(chimney_height, minimum=0.45)
    else:
        chimney_height = 0.0

    for idx, axis_value in enumerate(placed_boiler_axis):
        side_sign = -1.0 if idx % 2 == 0 else 1.0

        # 8.1) Pump at each boiler base, with footprint collision checks.
        pump_lateral_limit = max(cross_length / 2.0 - edge_offset - pump_size * 0.55, 0.0)
        pump_offset = boiler_radius + pump_size * 0.65 + service_clearance * 0.25
        pump_offset = min(pump_offset, pump_lateral_limit)
        pump_offset = max(pump_offset, min(min_walkway * 0.55, pump_lateral_limit))
        pump_placed = False
        for candidate_side in (side_sign, -side_sign):
            if pump_offset <= 0.0:
                break
            pump_lateral = candidate_side * pump_offset
            px, py = axis_to_xy(axis_value, pump_lateral)
            if not try_place_footprint(
                x=px,
                y=py,
                width=pump_size * 1.35,
                depth=pump_size * 0.95,
                clearance=0.04,
            ):
                continue
            emit(
                "pump",
                create_pump(size=pump_size, orientation=pump_orientation),
                (px, py, pump_size * 0.42),
                nominal_height=max(pump_size * 0.95, 0.6),
            )
            pump_placed = True
            break
        if not pump_placed:
            fallback_lateral = min(
                max(boiler_radius * 0.55, min_walkway * 0.35),
                max(cross_length / 2.0 - edge_offset - pump_size * 0.25, 0.0),
            )
            px, py = axis_to_xy(axis_value, side_sign * fallback_lateral)
            emit(
                "pump",
                create_pump(size=pump_size, orientation=pump_orientation),
                (px, py, pump_size * 0.42),
                nominal_height=max(pump_size * 0.95, 0.6),
            )

        # 8.2) Valve clusters attached to local pipe lanes.
        valve_lateral_limit = max(cross_length / 2.0 - edge_offset - 0.2, 0.0)
        valve_lateral = min(max(boiler_radius * 0.35, sampled_pipe_radius * 2.4), valve_lateral_limit)
        vx, vy = axis_to_xy(axis_value, side_sign * valve_lateral)
        emit(
            "valve_cluster",
            create_valve_cluster(count=valve_cluster_count, spacing=valve_cluster_spacing),
            (vx, vy, pipe_z * 0.96),
            axis_rotation,
            nominal_height=max(sampled_pipe_radius * 3.2, 0.25),
        )

        # 8.3) Chimney stacks from each boiler top.
        if chimney_height > 0.0:
            cx, cy = axis_to_xy(axis_value, 0.0)
            emit(
                "chimney",
                create_chimney(height=chimney_height, radius=chimney_radius),
                (cx, cy, boiler_height + chimney_height / 2.0 + 0.03),
                nominal_height=chimney_height + chimney_radius * 0.8,
            )

    # 8.4) Side/overhead bunker linked with a pipe to the boiler axis.
    bunker_width = max(boiler_radius * 1.55, 1.1)
    bunker_height = min(max(boiler_height * 0.58, 2.3), room.height * 0.7)
    bunker_height = height_constraint.clamp_size(bunker_height, minimum=1.0)
    bunker_depth = bunker_width * 0.82
    bunker_outlet_radius = max(sampled_pipe_radius * 1.55, 0.12)
    bunker_axis = placed_boiler_axis[0]
    if random_variation and len(placed_boiler_axis) > 1:
        bunker_axis = rng.choice((placed_boiler_axis[0], placed_boiler_axis[-1]))

    bunker_side_sign = -1.0 if not random_variation else (-1.0 if rng.random() < 0.5 else 1.0)
    bunker_side_placed = False
    bunker_position: tuple[float, float, float] | None = None
    bunker_lateral_value = 0.0

    bunker_lateral_limit = cross_length / 2.0 - edge_offset - bunker_depth / 2.0 - 0.08
    bunker_lateral_min = min_walkway * 0.65 + bunker_depth * 0.35
    if bunker_lateral_limit >= bunker_lateral_min:
        bunker_lateral_value = bunker_side_sign * bunker_lateral_limit
        bx, by = axis_to_xy(bunker_axis, bunker_lateral_value)
        if try_place_footprint(
            x=bx,
            y=by,
            width=bunker_width,
            depth=bunker_depth,
            clearance=0.05,
        ):
            bunker_side_placed = True
            bunker_position = (bx, by, bunker_height / 2.0)
            emit(
                "bunker",
                create_bunker(
                    width=bunker_width,
                    height=bunker_height,
                    outlet_radius=bunker_outlet_radius,
                ),
                bunker_position,
                nominal_height=bunker_height,
            )

    if not bunker_side_placed:
        bunker_axis = (placed_boiler_axis[0] + placed_boiler_axis[-1]) / 2.0
        bx, by = axis_to_xy(bunker_axis, 0.0)
        bunker_top_z = min(
            room.height - bunker_height / 2.0 - 0.08,
            boiler_height + bunker_height * 0.62,
        )
        if bunker_top_z > bunker_height / 2.0 + 0.03:
            bunker_position = (bx, by, bunker_top_z)
            emit(
                "bunker",
                create_bunker(
                    width=bunker_width,
                    height=bunker_height,
                    outlet_radius=bunker_outlet_radius,
                ),
                bunker_position,
                nominal_height=bunker_height,
            )

    if bunker_position is not None:
        link_radius = max(sampled_pipe_radius * 0.82, 0.05)
        if bunker_side_placed:
            link_len = max(abs(bunker_lateral_value), 0.45)
            lx, ly = axis_to_xy(bunker_axis, bunker_lateral_value / 2.0)
            emit(
                "bunker_pipe",
                create_pipe(
                    radius=link_radius,
                    length=link_len,
                    bend_angle=25.0 if random_variation else 0.0,
                    segment_count=20,
                ),
                (lx, ly, min(pipe_z * 0.9, boiler_height * 0.8)),
                cross_rotation,
                nominal_height=pipe_nominal_height,
            )
        else:
            drop_len = max(bunker_position[2] - boiler_height, 0.45)
            emit(
                "bunker_pipe",
                create_pipe(radius=link_radius, length=drop_len, bend_angle=0.0, segment_count=20),
                (bunker_position[0], bunker_position[1], boiler_height + drop_len / 2.0),
                (0.0, -90.0, 0.0),
                nominal_height=drop_len,
            )

    # 8.5) Heat exchanger near pipe manifold with explicit connection.
    heat_radius = max(sampled_pipe_radius * 2.4, 0.18)
    heat_length = max(min(manifold_len * 0.42, axis_length * 0.55), 1.4)
    heat_side = -bunker_side_sign if bunker_side_placed else 1.0
    heat_lateral_limit = max(cross_length / 2.0 - edge_offset - heat_radius * 1.5, 0.0)
    heat_lateral = min(max(min_walkway * 0.8, boiler_radius * 0.9), heat_lateral_limit)
    if heat_lateral > 0.0:
        hx, hy = axis_to_xy(center_axis, heat_side * heat_lateral)
        heat_z = min(pipe_z - heat_radius * 1.1, room.height - heat_radius * 1.25)
        heat_z = max(heat_z, boiler_height * 0.58)
        emit(
            "heat_exchanger",
            create_heat_exchanger(length=heat_length, radius=heat_radius),
            (hx, hy, heat_z),
            axis_rotation,
            nominal_height=heat_radius * 2.6,
        )

        exchanger_link_len = max(abs(heat_lateral), 0.35)
        lx, ly = axis_to_xy(center_axis, heat_side * heat_lateral / 2.0)
        emit(
            "heat_exchanger_pipe",
            create_pipe(
                radius=max(sampled_pipe_radius * 0.78, 0.05),
                length=exchanger_link_len,
                bend_angle=15.0 if random_variation else 0.0,
                segment_count=18,
            ),
            (lx, ly, min(pipe_z * 0.96, heat_z + heat_radius * 0.2)),
            cross_rotation,
            nominal_height=pipe_nominal_height,
        )

    # 8.6) Pipe supports under long manifolds.
    support_threshold = max(2.2, boiler_radius * 2.8)
    if manifold_len > support_threshold:
        support_height = min(pipe_z - 0.08, room.height * 0.9)
        support_height = height_constraint.clamp_size(support_height, minimum=0.8)
        support_spacing = max(sampled_pipe_radius * 8.0, 0.65)
        support_count = max(1, int(round(manifold_len / max(2.6, support_spacing * 2.5))))
        support_span = max(manifold_len / 2.0 - support_spacing * 0.6, 0.0)
        support_offsets = symmetric_positions(support_count, support_span)
        support_lateral_limit = cross_length / 2.0 - edge_offset - support_spacing / 2.0 - 0.05
        support_lateral = min(
            max(min_walkway * 0.65 + sampled_pipe_radius * 2.0, min_walkway * 0.55),
            max(support_lateral_limit, 0.0),
        )
        support_added = 0

        if support_lateral > 0.0:
            for idx, offset in enumerate(support_offsets):
                side_sign = -1.0 if idx % 2 == 0 else 1.0
                sx, sy = axis_to_xy(center_axis + offset, side_sign * support_lateral)
                if not try_place_footprint(
                    x=sx,
                    y=sy,
                    width=max(support_spacing * 0.3, 0.18),
                    depth=support_spacing,
                    clearance=0.03,
                ):
                    continue
                emit(
                    "pipe_support",
                    create_pipe_support(height=support_height, spacing=support_spacing),
                    (sx, sy, support_height / 2.0),
                    axis_rotation,
                    nominal_height=support_height,
                )
                support_added += 1

                support_link_len = max(abs(support_lateral), 0.35)
                lx, ly = axis_to_xy(center_axis + offset, side_sign * support_lateral / 2.0)
                emit(
                    "pipe_support_link",
                    create_pipe(
                        radius=max(sampled_pipe_radius * 0.62, 0.04),
                        length=support_link_len,
                        bend_angle=0.0,
                        segment_count=16,
                    ),
                    (lx, ly, min(pipe_z * 0.95, support_height * 0.9)),
                    cross_rotation,
                    nominal_height=pipe_nominal_height,
                )
        if support_added == 0:
            sx, sy = axis_to_xy(center_axis, 0.0)
            emit(
                "pipe_support",
                create_pipe_support(height=support_height, spacing=support_spacing),
                (sx, sy, support_height / 2.0),
                axis_rotation,
                nominal_height=support_height,
            )

    # 8.7) Control boxes near service areas with lightweight connection lines.
    control_size = max(0.45, min(boiler_radius * 0.55, 1.15))
    control_axes = [placed_boiler_axis[0]]
    if len(placed_boiler_axis) > 1:
        control_axes.append(placed_boiler_axis[-1])
    control_lateral_limit = max(cross_length / 2.0 - edge_offset - control_size * 0.35, 0.0)
    control_lateral = min(
        max(boiler_radius + service_clearance * 0.5, min_walkway * 0.6),
        control_lateral_limit,
    )
    control_added = 0
    if control_lateral > 0.0:
        for idx, axis_value in enumerate(control_axes):
            side_sign = -1.0 if idx % 2 == 0 else 1.0
            cbx, cby = axis_to_xy(axis_value, side_sign * control_lateral)
            if not try_place_footprint(
                x=cbx,
                y=cby,
                width=control_size,
                depth=control_size * 0.5,
                clearance=0.03,
            ):
                continue
            control_z = control_size * 0.36
            emit(
                "control_box",
                create_control_box(size=control_size),
                (cbx, cby, control_z),
                nominal_height=control_size * 0.75,
            )
            control_added += 1

            control_link_len = max(abs(control_lateral), 0.3)
            lx, ly = axis_to_xy(axis_value, side_sign * control_lateral / 2.0)
            emit(
                "control_link_pipe",
                create_pipe(
                    radius=max(sampled_pipe_radius * 0.46, 0.04),
                    length=control_link_len,
                    bend_angle=10.0 if random_variation else 0.0,
                    segment_count=14,
                ),
                (lx, ly, min(pipe_z * 0.58, boiler_height * 0.46)),
                cross_rotation,
                nominal_height=pipe_nominal_height,
            )
    if control_added == 0 and placed_boiler_axis:
        axis_value = placed_boiler_axis[0]
        cbx, cby = axis_to_xy(axis_value, 0.0)
        emit(
            "control_box",
            create_control_box(size=control_size),
            (cbx, cby, control_size * 0.36),
            nominal_height=control_size * 0.75,
        )


def _stable_hash(text: str) -> int:
    value = 0
    for idx, char in enumerate(text):
        value = (value * 131 + (idx + 1) * ord(char)) & 0xFFFFFFFF
    return value


def _axis_key(value: float) -> int:
    return int(round(value * 1000.0))


def _intersects(a: _Footprint, b: _Footprint) -> bool:
    return (
        abs(a.x - b.x) < (a.half_w + b.half_w)
        and abs(a.y - b.y) < (a.half_d + b.half_d)
    )


def _fit_boiler_count(
    axis_limit: float,
    boiler_radius: float,
    min_walkway: float,
    preferred_count: int,
    count_min: int,
    count_max: int,
) -> int:
    minimum = max(1, count_min)
    maximum = max(minimum, min(24, count_max))
    preferred = max(minimum, min(maximum, preferred_count))
    required_spacing = 2.0 * boiler_radius + min_walkway

    for count in range(preferred, minimum - 1, -1):
        if count == 1:
            return 1
        positions = symmetric_positions(count, axis_limit)
        if len(positions) < 2:
            continue
        spacing = positions[1] - positions[0]
        if spacing >= required_spacing:
            return count

    return minimum


def _normalize_pattern(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return "linear_boilers"
    return PATTERN_ALIASES.get(normalized, "linear_boilers")


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
