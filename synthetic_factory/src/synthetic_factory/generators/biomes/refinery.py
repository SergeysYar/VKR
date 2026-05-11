from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, pi, sin, sqrt
from random import Random
from typing import Mapping, Sequence

from ...parametric.primitives import create_box
from ...parametric.refinery_primitives import (
    create_distillation_column,
    create_junction_node,
    create_ladder,
    create_pipe_from_path,
    create_process_vessel,
    create_pump,
    create_refinery_pipe,
    create_refinery_platform,
    create_refinery_support,
    create_storage_tank,
    create_valve,
)
from .refinery_rules import (
    AccessRule,
    ConnectivityRule,
    HeightRule,
    MinClearanceRule,
    NoPipeCollisionRule,
    SupportRule,
)
from .refinery_utils import (
    create_sensor_unit_mesh as _create_sensor_unit_mesh,
    create_small_control_box_mesh as _create_small_control_box_mesh,
    create_valve_cluster_mesh as _create_valve_cluster_mesh,
    dedupe_path as _dedupe_path,
    distance as _distance,
    largest_horizontal_segment as _largest_horizontal_segment,
    lerp as _lerp,
    polyline_length as _polyline_length,
    route_corners as _route_corners,
    smooth_path as _smooth_path,
    compose_edge_route as _compose_edge_route,
)
from .refinery_process_graph import ProcessGraph, generate_process_graph
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

PRIMITIVES = (
    "refinery_main_column",
    "refinery_secondary_vessel",
    "refinery_platform",
    "refinery_platform_support",
    "refinery_pipe_backbone",
    "refinery_pipe_riser",
    "refinery_pipe_link",
    "refinery_pipe_vertical_link",
    "refinery_pipe_node_link",
    "refinery_pipe_support",
    "refinery_junction_node",
    "refinery_access_zone",
    "refinery_storage_tank",
    "refinery_process_pipe",
    "refinery_process_valve",
    "refinery_process_pump",
    "refinery_process_pipe_support",
    "refinery_service_walkway",
    "refinery_ladder",
    "refinery_guardrail",
    "refinery_valve_cluster",
    "refinery_secondary_pipe_support",
    "refinery_small_control_box",
    "refinery_sensor_unit",
)
RULES: dict[str, object] = {
    "layout": "vertical towers + dense connected pipe graph + multi-level service platforms",
    "dominance": "vertical process columns (height >> radius)",
    "patterns": [
        "linear_refinery",
        "tower_cluster",
        "dense_refinery",
    ],
    "rules": [
        "MinClearanceRule",
        "ConnectivityRule",
        "NoPipeCollisionRule",
        "SupportRule",
        "AccessRule",
        "HeightRule",
    ],
    "auto_fix": True,
}

PATTERN_ALIASES = {
    "linear_refinery": "linear_refinery",
    "linear": "linear_refinery",
    "line": "linear_refinery",
    "tower_cluster": "tower_cluster",
    "cluster": "tower_cluster",
    "clustered": "tower_cluster",
    "dense_refinery": "dense_refinery",
    "dense": "dense_refinery",
}


@dataclass(frozen=True)
class _Footprint:
    x: float
    y: float
    half_w: float
    half_d: float


def populate(room: BiomeRoom, add_object: AddObjectFn, settings: Mapping[str, object]) -> None:
    cfg = to_mapping(settings.get("refinery"))
    main_cfg = to_mapping(cfg.get("main_columns"))
    secondary_cfg = to_mapping(cfg.get("secondary"))
    pipe_cfg = to_mapping(cfg.get("pipes"))
    platform_cfg = to_mapping(cfg.get("platforms"))
    structure_cfg = to_mapping(cfg.get("structure"))
    rules_cfg = to_mapping(cfg.get("rules"))
    process_cfg = to_mapping(cfg.get("process_graph"))
    routing_cfg = to_mapping(cfg.get("routing"))
    secondary_elements_cfg = to_mapping(cfg.get("secondary_elements"))

    seed = int(cfg.get("seed", 0)) + _stable_hash(room.room_id)
    rng = Random(seed)
    random_variation = to_bool(cfg.get("random_variation", True), default=True)
    pattern = _normalize_pattern(str(cfg.get("pattern", "tower_cluster")))

    edge_offset = to_positive_float(
        structure_cfg.get("edge_offset"),
        1.0,
        "machinery.refinery.structure.edge_offset",
    )
    min_clearance = to_positive_float(
        rules_cfg.get("min_clearance", structure_cfg.get("min_clearance", 0.9)),
        0.9,
        "machinery.refinery.rules.min_clearance",
    )
    walkway_width = to_positive_float(
        cfg.get("walkway_width"),
        1.4,
        "machinery.refinery.walkway_width",
    )
    clearance_rule = MinClearanceRule(min_distance=min_clearance)
    access_rule = AccessRule(walkway_width=walkway_width)
    height_rule = HeightRule(room_height=room.height)
    connectivity_rule = ConnectivityRule()
    rules_auto_fix = to_bool(rules_cfg.get("auto_fix", True), default=True)

    count_range = _parse_int_range(
        main_cfg.get("count", [2, 4]),
        default_min=2,
        default_max=4,
        label="machinery.refinery.main_columns.count",
    )
    radius_range = _parse_float_range(
        main_cfg.get("radius", [0.8, 1.8]),
        default_min=0.8,
        default_max=1.8,
        label="machinery.refinery.main_columns.radius",
    )
    height_range = _parse_float_range(
        main_cfg.get("height", [10.0, 24.0]),
        default_min=10.0,
        default_max=24.0,
        label="machinery.refinery.main_columns.height",
    )

    secondary_count_range = _parse_int_range(
        secondary_cfg.get("count", [3, 10]),
        default_min=3,
        default_max=10,
        label="machinery.refinery.secondary.count",
    )
    secondary_radius_range = _parse_float_range(
        secondary_cfg.get("radius", [0.35, 0.95]),
        default_min=0.35,
        default_max=0.95,
        label="machinery.refinery.secondary.radius",
    )
    secondary_length_range = _parse_float_range(
        secondary_cfg.get("length", [1.8, 4.8]),
        default_min=1.8,
        default_max=4.8,
        label="machinery.refinery.secondary.length",
    )

    pipe_density_range = _parse_float_range(
        pipe_cfg.get("density", [1.0, 2.4]),
        default_min=1.0,
        default_max=2.4,
        label="machinery.refinery.pipes.density",
    )
    pipe_radius_range = _parse_float_range(
        pipe_cfg.get("radius", [0.12, 0.42]),
        default_min=0.12,
        default_max=0.42,
        label="machinery.refinery.pipes.radius",
    )
    level_count_range = _parse_int_range(
        pipe_cfg.get("levels", [2, 4]),
        default_min=2,
        default_max=4,
        label="machinery.refinery.pipes.levels",
    )
    support_spacing_range = _parse_float_range(
        pipe_cfg.get("support_spacing", [2.5, 4.2]),
        default_min=2.5,
        default_max=4.2,
        label="machinery.refinery.pipes.support_spacing",
    )

    platform_width_factor_range = _parse_float_range(
        platform_cfg.get("width_factor", [1.4, 2.2]),
        default_min=1.4,
        default_max=2.2,
        label="machinery.refinery.platforms.width_factor",
    )
    platform_levels_range = _parse_int_range(
        platform_cfg.get("levels", [2, 3]),
        default_min=2,
        default_max=3,
        label="machinery.refinery.platforms.levels",
    )
    raw_guardrails = platform_cfg.get("guardrails")
    guardrails_cfg = to_mapping(raw_guardrails) if isinstance(raw_guardrails, Mapping) else {}
    guardrails_enabled = to_bool(
        guardrails_cfg.get("enabled", raw_guardrails if raw_guardrails is not None else False),
        default=False,
    )
    guardrail_height = to_positive_float(
        guardrails_cfg.get("height"),
        1.05,
        "machinery.refinery.platforms.guardrails.height",
    )
    guardrail_thickness = to_positive_float(
        guardrails_cfg.get("thickness"),
        0.08,
        "machinery.refinery.platforms.guardrails.thickness",
    )
    process_smooth_bends = to_bool(routing_cfg.get("smooth_bends", True), default=True)
    corner_chamfer = max(0.02, float(routing_cfg.get("corner_chamfer", 0.35)))
    pump_length_threshold = to_positive_float(
        routing_cfg.get("pump_length_threshold"),
        7.0,
        "machinery.refinery.routing.pump_length_threshold",
    )
    group_lane_spacing = to_positive_float(
        routing_cfg.get("group_lane_spacing"),
        0.75,
        "machinery.refinery.routing.group_lane_spacing",
    )
    secondary_enabled = to_bool(secondary_elements_cfg.get("enabled", True), default=True)
    secondary_density = to_positive_float(
        secondary_elements_cfg.get("density"),
        1.0,
        "machinery.refinery.secondary_elements.density",
    )
    sensor_density = to_positive_float(
        secondary_elements_cfg.get("sensor_density", secondary_density),
        1.0,
        "machinery.refinery.secondary_elements.sensor_density",
    )

    area_scale = room_area_scale(room, reference_area=300.0, min_scale=0.7, max_scale=3.2)
    count_scale = max(1.0, area_scale)

    main_count = _sample_int(count_range, rng, random_variation)
    secondary_count = _sample_int(secondary_count_range, rng, random_variation)
    main_radius = _sample_float(radius_range, rng, random_variation)
    main_height = _sample_float(height_range, rng, random_variation)
    secondary_radius = _sample_float(secondary_radius_range, rng, random_variation)
    secondary_length = _sample_float(secondary_length_range, rng, random_variation)
    pipe_density = _sample_float(pipe_density_range, rng, random_variation)
    pipe_radius = _sample_float(pipe_radius_range, rng, random_variation)
    level_count = _sample_int(level_count_range, rng, random_variation)
    support_spacing = _sample_float(support_spacing_range, rng, random_variation)
    platform_width_factor = _sample_float(platform_width_factor_range, rng, random_variation)
    platform_levels = _sample_int(platform_levels_range, rng, random_variation)
    support_spacing = to_positive_float(
        rules_cfg.get("support_spacing", support_spacing),
        support_spacing,
        "machinery.refinery.rules.support_spacing",
    )
    no_pipe_collision_rule = NoPipeCollisionRule(
        margin=to_positive_float(
            rules_cfg.get("pipe_collision_margin", 0.12),
            0.12,
            "machinery.refinery.rules.pipe_collision_margin",
        ),
        vertical_step=to_positive_float(
            rules_cfg.get("pipe_vertical_step", 0.28),
            0.28,
            "machinery.refinery.rules.pipe_vertical_step",
        ),
    )
    support_rule = SupportRule(spacing=support_spacing, min_support_height=0.42)

    main_count = max(1, min(10, int(round(main_count * count_scale))))
    secondary_count = max(1, min(40, int(round(secondary_count * (0.7 + count_scale * 0.65)))))
    pipe_density = pipe_density * (0.8 + count_scale * 0.25)
    corner_chamfer = max(corner_chamfer, pipe_radius * 1.85)

    if pattern == "linear_refinery":
        main_count = max(2, main_count)
        secondary_count = max(2, int(round(secondary_count * 0.8)))
        level_count = max(2, level_count)
    elif pattern == "dense_refinery":
        main_count = max(3, main_count)
        secondary_count = max(5, int(round(secondary_count * 1.45)))
        pipe_density = max(pipe_density, 1.8)
        level_count = max(3, level_count)
        platform_levels = max(2, platform_levels)

    main_radius = min(main_radius, max(min(room.width, room.depth) * 0.16, 0.35))
    main_radius = max(main_radius, 0.35)
    main_height = max(main_height, main_radius * 5.2)
    main_height = height_rule.clamp_size(main_height, minimum=3.0)

    axis_along_x = room.width >= room.depth
    axis_length = room.width if axis_along_x else room.depth
    cross_length = room.depth if axis_along_x else room.width
    axis_limit = max(axis_length / 2.0 - edge_offset - main_radius * 1.1, 0.0)
    cross_limit = max(cross_length / 2.0 - edge_offset, 0.0)
    axis_rotation = (0.0, 0.0, 0.0) if axis_along_x else (0.0, 0.0, 90.0)
    cross_rotation = (0.0, 0.0, 90.0) if axis_along_x else (0.0, 0.0, 0.0)

    def axis_to_xy(axis_value: float, lateral_value: float) -> tuple[float, float]:
        if axis_along_x:
            return (axis_value, lateral_value)
        return (lateral_value, axis_value)

    def planar_rotation(from_xy: tuple[float, float], to_xy: tuple[float, float]) -> Vector3:
        dx = to_xy[0] - from_xy[0]
        dy = to_xy[1] - from_xy[1]
        return (0.0, 0.0, 0.0) if abs(dx) >= abs(dy) else (0.0, 0.0, 90.0)

    while main_count > 1:
        required_span = (2.0 * main_radius + min_clearance) * (main_count - 1)
        if required_span <= axis_limit * 2.0 + 1e-6:
            break
        main_count -= 1

    if main_count <= 1:
        main_axis_positions = [0.0]
    else:
        proposed = symmetric_positions(main_count, axis_limit)
        fixed, _ = clearance_rule.enforce_axis_positions(
            proposed,
            axis_limit=axis_limit,
            object_radius=main_radius,
        )
        main_axis_positions = fixed

    occupied: list[_Footprint] = []

    def intersects_any(x: float, y: float, width: float, depth: float, margin: float) -> bool:
        candidate = _Footprint(
            x=x,
            y=y,
            half_w=width / 2.0 + margin,
            half_d=depth / 2.0 + margin,
        )
        for current in occupied:
            if (
                abs(candidate.x - current.x) < (candidate.half_w + current.half_w)
                and abs(candidate.y - current.y) < (candidate.half_d + current.half_d)
            ):
                return True
        return False

    def push_occupied(x: float, y: float, width: float, depth: float) -> None:
        occupied.append(_Footprint(x=x, y=y, half_w=width / 2.0, half_d=depth / 2.0))

    counters: dict[str, int] = {}

    def emit(object_type: str, mesh: object, position: Vector3, rotation: Vector3 = (0.0, 0.0, 0.0)) -> None:
        counters[object_type] = counters.get(object_type, 0) + 1
        add_object(object_type, counters[object_type], mesh, position, rotation)

    walkway_length = max(axis_length - 2.0 * edge_offset, walkway_width)
    walkway_mesh = create_box(
        width=walkway_length if axis_along_x else walkway_width,
        height=0.03,
        depth=walkway_width if axis_along_x else walkway_length,
    )
    emit("refinery_access_zone", walkway_mesh, (0.0, 0.0, 0.015))

    column_top_z = main_height / 2.0
    for axis_value in main_axis_positions:
        x, y = axis_to_xy(axis_value, 0.0)
        emit(
            "refinery_main_column",
            create_distillation_column(
                radius=main_radius,
                height=main_height,
                segment_count=32,
                tray_count=max(4, level_count + 1),
            ),
            (x, y, column_top_z),
            (0.0, 0.0, 0.0),
        )
        push_occupied(x, y, main_radius * 2.0, main_radius * 2.0)

    platform_thickness = max(main_radius * 0.14, 0.08)
    platform_levels = max(1, platform_levels)
    platform_records: list[tuple[int, float, float, float, float, float, float]] = []
    service_walkway_thickness = max(platform_thickness * 0.62, 0.06)
    service_walkway_width = max(min(walkway_width * 0.62, main_radius * 1.7), 0.6)

    def emit_guardrail_span(
        *,
        axis_center: float,
        lateral_center: float,
        span_axis: float,
        span_lateral: float,
        z_center: float,
    ) -> None:
        gx, gy = axis_to_xy(axis_center, lateral_center)
        if axis_along_x:
            width = span_axis
            depth = span_lateral
        else:
            width = span_lateral
            depth = span_axis
        emit(
            "refinery_guardrail",
            create_box(width=max(width, 0.04), height=guardrail_height, depth=max(depth, 0.04)),
            (gx, gy, z_center),
            (0.0, 0.0, 0.0),
        )

    for axis_value in main_axis_positions:
        x, y = axis_to_xy(axis_value, 0.0)
        for level_idx in range(platform_levels):
            t = (level_idx + 1) / (platform_levels + 1)
            level_z = t * (main_height * 0.92)
            level_z = height_rule.clamp_center(level_z, platform_thickness)
            p_width = max(main_radius * 2.4 * platform_width_factor, 1.2)
            p_depth = max(main_radius * 2.0 * platform_width_factor, 1.0)
            emit(
                "refinery_platform",
                create_refinery_platform(width=p_width, depth=p_depth, thickness=platform_thickness),
                (x, y, level_z),
                axis_rotation,
            )
            support_height = max(level_z - platform_thickness / 2.0, 0.4)
            emit(
                "refinery_platform_support",
                create_refinery_support(height=support_height, spacing=max(p_depth * 0.88, 0.9)),
                (x, y, support_height / 2.0),
                axis_rotation,
            )
            platform_records.append((level_idx, axis_value, 0.0, level_z, p_width, p_depth, platform_thickness))

            if guardrails_enabled:
                safe_thickness = min(guardrail_thickness, min(p_width, p_depth) * 0.3)
                rail_z = level_z + platform_thickness / 2.0 + guardrail_height / 2.0
                emit_guardrail_span(
                    axis_center=axis_value,
                    lateral_center=p_depth / 2.0 - safe_thickness / 2.0,
                    span_axis=p_width,
                    span_lateral=safe_thickness,
                    z_center=rail_z,
                )
                emit_guardrail_span(
                    axis_center=axis_value,
                    lateral_center=-p_depth / 2.0 + safe_thickness / 2.0,
                    span_axis=p_width,
                    span_lateral=safe_thickness,
                    z_center=rail_z,
                )
                emit_guardrail_span(
                    axis_center=axis_value + p_width / 2.0 - safe_thickness / 2.0,
                    lateral_center=0.0,
                    span_axis=safe_thickness,
                    span_lateral=p_depth,
                    z_center=rail_z,
                )
                emit_guardrail_span(
                    axis_center=axis_value - p_width / 2.0 + safe_thickness / 2.0,
                    lateral_center=0.0,
                    span_axis=safe_thickness,
                    span_lateral=p_depth,
                    z_center=rail_z,
                )

    platforms_by_level: dict[int, list[tuple[float, float, float, float, float]]] = {}
    for level_idx, axis_value, lateral_value, level_z, p_width, p_depth, _ in platform_records:
        platforms_by_level.setdefault(level_idx, []).append((axis_value, lateral_value, level_z, p_width, p_depth))

    for level_platforms in platforms_by_level.values():
        if len(level_platforms) <= 1:
            continue
        level_platforms.sort(key=lambda item: item[0])
        for left, right in zip(level_platforms, level_platforms[1:]):
            left_axis, _, left_z, left_w, _ = left
            right_axis, _, right_z, right_w, _ = right
            clear_gap = abs(right_axis - left_axis) - (left_w + right_w) / 2.0
            if clear_gap <= 0.45:
                continue
            walkway_axis = (left_axis + right_axis) / 2.0
            walkway_lateral = 0.0
            wx, wy = axis_to_xy(walkway_axis, walkway_lateral)
            walkway_z = (left_z + right_z) / 2.0
            emit(
                "refinery_service_walkway",
                create_box(
                    width=clear_gap if axis_along_x else service_walkway_width,
                    height=service_walkway_thickness,
                    depth=service_walkway_width if axis_along_x else clear_gap,
                ),
                (wx, wy, walkway_z),
                (0.0, 0.0, 0.0),
            )

    ladder_offset = min(cross_limit - 0.15, max(main_radius * 1.25, service_walkway_width * 0.75))
    for idx, axis_value in enumerate(main_axis_positions):
        column_platforms = [record for record in platform_records if abs(record[1] - axis_value) <= 1e-6]
        if column_platforms:
            top_platform_z = max(item[3] + item[6] / 2.0 for item in column_platforms)
        else:
            top_platform_z = main_height * 0.5
        ladder_height = max(1.2, top_platform_z - 0.02)
        ladder_lateral_sign = -1.0 if idx % 2 == 0 else 1.0
        ladder_lateral = ladder_lateral_sign * ladder_offset
        ladder_lateral = max(-cross_limit + 0.15, min(cross_limit - 0.15, ladder_lateral))
        lx, ly = axis_to_xy(axis_value, ladder_lateral)
        emit(
            "refinery_ladder",
            create_ladder(height=ladder_height, step_count=max(6, int(round(ladder_height / 0.28)))),
            (lx, ly, ladder_height / 2.0),
            cross_rotation,
        )

    secondary_positions: list[tuple[float, float, float]] = []
    side_radius = max(secondary_radius * 1.65, 0.35)
    side_span = max(
        access_rule.blocked_lane(secondary_radius * 2.0) + secondary_radius * 0.65,
        min(cross_limit * 0.75, main_radius + secondary_radius + min_clearance),
    )
    side_span = min(side_span, max(cross_limit - secondary_radius * 1.2, 0.0))
    if side_span <= 0.0:
        side_span = max(cross_limit * 0.5, secondary_radius * 1.5)

    for idx in range(secondary_count):
        axis_seed = main_axis_positions[idx % len(main_axis_positions)]
        jitter = 0.0
        if random_variation:
            jitter = rng.uniform(-main_radius * 1.1, main_radius * 1.1)
        axis_value = max(-axis_limit, min(axis_limit, axis_seed + jitter))
        sign = -1.0 if idx % 2 == 0 else 1.0
        lateral = sign * access_rule.ensure_lateral_access(
            side_span,
            secondary_radius * 2.0,
            cross_limit,
        )
        if random_variation:
            lateral = lateral + rng.uniform(-secondary_radius * 0.4, secondary_radius * 0.4)
        lateral = max(-cross_limit, min(cross_limit, lateral))
        sx, sy = axis_to_xy(axis_value, lateral)

        footprint_w = secondary_length
        footprint_d = secondary_radius * 2.2
        if intersects_any(sx, sy, footprint_w, footprint_d, margin=0.06):
            fallback_lateral = -lateral
            fx, fy = axis_to_xy(axis_value, fallback_lateral)
            if intersects_any(fx, fy, footprint_w, footprint_d, margin=0.06):
                continue
            sx, sy = fx, fy
            lateral = fallback_lateral

        vessel_height = max(secondary_radius * 2.0, 0.5)
        vessel_center_z = vessel_height / 2.0 + 0.08
        yaw = axis_rotation[2]
        emit(
            "refinery_secondary_vessel",
            create_process_vessel(radius=secondary_radius, length=secondary_length),
            (sx, sy, vessel_center_z),
            (0.0, 0.0, yaw),
        )
        push_occupied(sx, sy, footprint_w, footprint_d)
        secondary_positions.append((sx, sy, vessel_center_z + secondary_radius))

    if secondary_enabled:
        control_box_count = max(1, int(round(len(main_axis_positions) * max(secondary_density, 0.4))))
        control_box_size = max(main_radius * 0.38, 0.34)
        control_lateral_base = max(
            access_rule.blocked_lane(control_box_size),
            min(cross_limit * 0.45, main_radius * 1.25),
        )
        for idx in range(control_box_count):
            axis_value = main_axis_positions[idx % len(main_axis_positions)]
            sign = -1.0 if idx % 2 == 0 else 1.0
            lateral = sign * control_lateral_base
            bx, by = axis_to_xy(axis_value, lateral)
            if intersects_any(
                bx,
                by,
                control_box_size * 1.08,
                control_box_size * 0.7,
                margin=min_clearance * 0.12,
            ):
                alt_lateral = -lateral
                ax, ay = axis_to_xy(axis_value, alt_lateral)
                if intersects_any(
                    ax,
                    ay,
                    control_box_size * 1.08,
                    control_box_size * 0.7,
                    margin=min_clearance * 0.12,
                ):
                    continue
                bx, by = ax, ay
                lateral = alt_lateral

            emit(
                "refinery_small_control_box",
                _create_small_control_box_mesh(control_box_size),
                (bx, by, control_box_size * 0.32),
                axis_rotation if abs(lateral) < 1e-9 else cross_rotation,
            )
            push_occupied(
                bx,
                by,
                control_box_size * 1.08,
                control_box_size * 0.7,
            )

    floor_connector_thickness = max(service_walkway_thickness * 0.5, 0.04)
    floor_connector_width = max(service_walkway_width * 0.8, 0.55)
    for sx, sy, _ in secondary_positions:
        axis_value, lateral_value = (sx, sy) if axis_along_x else (sy, sx)
        free_run = abs(lateral_value) - secondary_radius * 0.75
        if free_run <= 0.35:
            continue
        corridor_center_lateral = lateral_value / 2.0
        cx, cy = axis_to_xy(axis_value, corridor_center_lateral)
        emit(
            "refinery_service_walkway",
            create_box(
                width=floor_connector_width if axis_along_x else free_run,
                height=floor_connector_thickness,
                depth=free_run if axis_along_x else floor_connector_width,
            ),
            (cx, cy, floor_connector_thickness / 2.0),
            (0.0, 0.0, 0.0),
        )

    node_count = max(2, int(round(main_count * 1.2)))
    node_count = max(node_count, int(round(pipe_density * 2.0)))
    node_count = min(node_count, max(4, secondary_count + 2))
    node_size = max(pipe_radius * 2.8, 0.2)
    node_positions: list[tuple[float, float, float]] = []
    for idx in range(node_count):
        axis_value = main_axis_positions[idx % len(main_axis_positions)]
        sign = -1.0 if idx % 2 == 0 else 1.0
        lateral = sign * min(max(side_span * 0.62, walkway_width), cross_limit - node_size)
        nx, ny = axis_to_xy(axis_value, lateral)
        node_z = min(main_height * 0.55 + (idx % max(level_count, 1)) * main_height * 0.08, room.height - 0.2)
        node_z = max(node_z, node_size / 2.0 + 0.08)
        emit("refinery_junction_node", create_junction_node(size=node_size), (nx, ny, node_z))
        node_positions.append((nx, ny, node_z))

    level_count = max(2, level_count)
    min_level_z = max(main_height * 0.38, 1.4)
    max_level_z = min(room.height - 0.35, main_height * 0.88)
    if min_level_z >= max_level_z:
        min_level_z = max(room.height * 0.55, 1.0)
        max_level_z = min(room.height - 0.25, min_level_z + 0.25)

    level_z_values: list[float] = []
    for idx in range(level_count):
        if level_count == 1:
            t = 0.5
        else:
            t = idx / (level_count - 1)
        level_z_values.append(min_level_z + (max_level_z - min_level_z) * t)

    axis_min = min(main_axis_positions)
    axis_max = max(main_axis_positions)
    axis_span = max(axis_max - axis_min + main_radius * 2.0, main_radius * 3.0)
    pipe_count = 0

    for level_idx, level_z in enumerate(level_z_values):
        lane_lateral = 0.0
        if level_idx % 2 == 1:
            offset = min(max(main_radius * 0.3, 0.25), max(cross_limit - walkway_width, 0.0))
            lane_lateral = offset
        px, py = axis_to_xy((axis_min + axis_max) / 2.0, lane_lateral)
        emit(
            "refinery_pipe_backbone",
            create_refinery_pipe(radius=pipe_radius, length=axis_span, bend_angle=0.0),
            (px, py, level_z),
            axis_rotation,
        )
        pipe_count += 1

        if axis_span >= support_spacing * 0.9:
            support_count = max(1, int(round(axis_span / max(support_spacing, 0.5))))
            support_offsets = symmetric_positions(
                support_count,
                max(axis_span / 2.0 - support_spacing * 0.5, 0.0),
            )
            for offset in support_offsets:
                sx, sy = axis_to_xy((axis_min + axis_max) / 2.0 + offset, lane_lateral)
                support_height = max(level_z - 0.08, 0.45)
                emit(
                    "refinery_pipe_support",
                    create_refinery_support(height=support_height, spacing=max(pipe_radius * 6.5, 0.8)),
                    (sx, sy, support_height / 2.0),
                    axis_rotation,
                )

    top_level = max(level_z_values)
    for axis_value in main_axis_positions:
        x, y = axis_to_xy(axis_value, 0.0)
        riser_len = max(top_level - main_height, 0.35)
        emit(
            "refinery_pipe_riser",
            create_refinery_pipe(radius=max(pipe_radius * 0.94, 0.06), length=riser_len, bend_angle=0.0),
            (x, y, main_height + riser_len / 2.0),
            (0.0, -90.0, 0.0),
        )
        pipe_count += 1

    for sx, sy, sz in secondary_positions:
        nearest_axis = min(main_axis_positions, key=lambda axis_value: abs(axis_to_xy(axis_value, 0.0)[0] - sx) + abs(axis_to_xy(axis_value, 0.0)[1] - sy))
        tx, ty = axis_to_xy(nearest_axis, 0.0)
        link_len = ((sx - tx) ** 2 + (sy - ty) ** 2) ** 0.5
        if link_len <= 0.12:
            continue
        target_level = min(level_z_values, key=lambda z: abs(z - (sz + 0.6)))
        emit(
            "refinery_pipe_link",
            create_refinery_pipe(radius=max(pipe_radius * 0.82, 0.05), length=link_len, bend_angle=0.0),
            ((sx + tx) / 2.0, (sy + ty) / 2.0, target_level),
            planar_rotation((sx, sy), (tx, ty)),
        )
        pipe_count += 1

        vertical_len = abs(target_level - sz)
        if vertical_len > 0.15:
            emit(
                "refinery_pipe_vertical_link",
                create_refinery_pipe(radius=max(pipe_radius * 0.72, 0.045), length=vertical_len, bend_angle=0.0),
                (sx, sy, (target_level + sz) / 2.0),
                (0.0, -90.0, 0.0),
            )
            pipe_count += 1

    for nx, ny, nz in node_positions:
        if secondary_positions:
            tx, ty, tz = min(
                secondary_positions,
                key=lambda item: (item[0] - nx) ** 2 + (item[1] - ny) ** 2 + (item[2] - nz) ** 2,
            )
        else:
            tx, ty = axis_to_xy(main_axis_positions[0], 0.0)
            tz = min(level_z_values, key=lambda z: abs(z - nz))

        segment_len = ((tx - nx) ** 2 + (ty - ny) ** 2) ** 0.5
        if segment_len > 0.1:
            emit(
                "refinery_pipe_node_link",
                create_refinery_pipe(radius=max(pipe_radius * 0.66, 0.04), length=segment_len, bend_angle=0.0),
                ((tx + nx) / 2.0, (ty + ny) / 2.0, (tz + nz) / 2.0),
                planar_rotation((nx, ny), (tx, ty)),
            )
            pipe_count += 1

    process_graph_settings = to_mapping(process_cfg.get("settings"))
    if not process_graph_settings:
        target_reactors = max(1, min(max(len(secondary_positions) // 2, 1), 4))
        target_exchangers = max(1, min(max(len(secondary_positions) - target_reactors, 1), 5))
        process_graph_settings = {
            "nodes": {
                "columns": {"count": [max(1, min(main_count, 3)), max(1, min(main_count, 3))]},
                "reactors": {"count": [target_reactors, target_reactors]},
                "exchangers": {"count": [target_exchangers, target_exchangers]},
                "tanks": {"count": [max(1, min(main_count, 4)), max(1, min(main_count, 4))]},
                "auxiliary": {"pump_per_link": [0, 0], "valve_per_link": [0, 0]},
            },
            "edges": {
                "include_feedback_loops": pattern != "linear_refinery",
                "feedback_loops": [1, 2],
            },
        }

    process_graph_seed = int(process_cfg.get("seed", seed + 173))
    process_graph = generate_process_graph(settings=process_graph_settings, seed=process_graph_seed)

    column_ids = [node.id for node in process_graph.nodes if node.type == "column"]
    reactor_ids = [node.id for node in process_graph.nodes if node.type == "reactor"]
    exchanger_ids = [node.id for node in process_graph.nodes if node.type == "exchanger"]
    tank_ids = [node.id for node in process_graph.nodes if node.type == "tank"]

    column_flow_points: list[Vector3] = []
    for idx, axis_value in enumerate(main_axis_positions):
        side = main_radius * (1.06 if idx % 2 == 0 else -1.06)
        cx, cy = axis_to_xy(axis_value, side)
        cz = min(room.height - 0.22, main_height * 0.78)
        column_flow_points.append((cx, cy, cz))

    tank_positions: list[Vector3] = []
    if tank_ids:
        tank_radius = max(secondary_radius * 1.25, 0.55)
        tank_height = height_rule.clamp_size(max(main_height * 0.44, 2.8), minimum=2.4)
        x_limit = max(room.width / 2.0 - edge_offset - tank_radius * 1.2, tank_radius + 0.2)
        y_limit = max(room.depth / 2.0 - edge_offset - tank_radius * 1.2, tank_radius + 0.2)
        for idx, _ in enumerate(tank_ids):
            tx = 0.0
            ty = 0.0
            fallback_theta = 2.0 * pi * (idx / max(len(tank_ids), 1))
            for attempt in range(8):
                theta = fallback_theta + attempt * (pi / 4.0)
                candidate_x = cos(theta) * x_limit * 0.92
                candidate_y = sin(theta) * y_limit * 0.92
                if not intersects_any(
                    candidate_x,
                    candidate_y,
                    tank_radius * 2.2,
                    tank_radius * 2.2,
                    margin=min_clearance * 0.22,
                ):
                    tx = candidate_x
                    ty = candidate_y
                    break
                tx = candidate_x
                ty = candidate_y

            emit(
                "refinery_storage_tank",
                create_storage_tank(radius=tank_radius, height=tank_height),
                (tx, ty, tank_height / 2.0),
                (0.0, 0.0, 0.0),
            )
            push_occupied(tx, ty, tank_radius * 2.2, tank_radius * 2.2)
            tank_positions.append((tx, ty, min(room.height - 0.2, tank_height * 0.78)))

    reactor_pool = secondary_positions[::2] if secondary_positions else []
    exchanger_pool = secondary_positions[1::2] if len(secondary_positions) > 1 else []
    if not reactor_pool:
        reactor_pool = list(secondary_positions)
    if not exchanger_pool:
        exchanger_pool = list(secondary_positions)

    fallback_reactors: list[Vector3] = []
    fallback_exchangers: list[Vector3] = []
    if not secondary_positions:
        for idx, axis_value in enumerate(main_axis_positions):
            lateral = access_rule.ensure_lateral_access(
                side_span * (1.0 if idx % 2 else -1.0),
                secondary_radius * 2.0,
                cross_limit,
            )
            rx, ry = axis_to_xy(axis_value, lateral)
            ex, ey = axis_to_xy(axis_value, -lateral)
            rz = min(room.height - 0.25, main_height * 0.52)
            fallback_reactors.append((rx, ry, rz))
            fallback_exchangers.append((ex, ey, rz + 0.24))

    node_points: dict[str, Vector3] = {}
    _assign_node_points(node_points, column_ids, column_flow_points)
    _assign_node_points(node_points, reactor_ids, reactor_pool or fallback_reactors or column_flow_points)
    _assign_node_points(node_points, exchanger_ids, exchanger_pool or fallback_exchangers or column_flow_points)
    _assign_node_points(node_points, tank_ids, tank_positions or column_flow_points)
    _assign_auxiliary_points(node_points, process_graph)

    routing_edges: list[tuple[str, str, dict[str, object]]] = [
        (edge.source_id, edge.target_id, dict(edge.pipe_parameters))
        for edge in process_graph.edges
    ]
    if rules_auto_fix:
        bridge_edges = connectivity_rule.ensure_all_nodes_connected(
            node_points=node_points,
            edges=[(source_id, target_id) for source_id, target_id, _ in routing_edges],
        )
        for source_id, target_id in bridge_edges:
            routing_edges.append(
                (
                    source_id,
                    target_id,
                    {
                        "radius": max(pipe_radius * 0.8, 0.08),
                        "duty": "auto_connectivity",
                    },
                )
            )

    lane_offsets: dict[str, float] = {}
    duty_counts: dict[str, int] = {}
    support_height_margin = 0.07
    generated_routes: list[list[Vector3]] = []
    for source_id, target_id, edge_parameters in routing_edges:
        if source_id not in node_points or target_id not in node_points:
            continue
        start = node_points[source_id]
        end = node_points[target_id]
        if _distance(start, end) <= 0.18:
            continue

        duty = str(edge_parameters.get("duty", "process")).strip().lower() or "process"
        edge_radius = _coerce_radius(edge_parameters.get("radius"), default=max(pipe_radius * 0.82, 0.06))
        lane_offset = lane_offsets.get(duty)
        if lane_offset is None:
            lane_index = len(lane_offsets)
            lane_sign = 1.0 if lane_index % 2 == 0 else -1.0
            lane_step = (lane_index // 2 + 1) * max(group_lane_spacing, edge_radius * 4.2)
            lane_offset = lane_sign * lane_step
            if cross_limit > edge_radius * 2.5:
                lane_offset = max(-cross_limit + edge_radius * 1.2, min(cross_limit - edge_radius * 1.2, lane_offset))
            lane_offsets[duty] = lane_offset

        duty_index = duty_counts.get(duty, 0)
        duty_counts[duty] = duty_index + 1

        lane_base_z = max(top_level + 0.55, start[2] + 0.55, end[2] + 0.55)
        lane_z = lane_base_z + duty_index * max(edge_radius * 1.8, 0.22)
        lane_z = min(lane_z, room.height - 0.2)

        raw_path = _compose_edge_route(
            start=start,
            end=end,
            axis_along_x=axis_along_x,
            lane_offset=lane_offset,
            lane_z=lane_z,
            axis_limit=axis_limit,
            cross_limit=cross_limit,
        )
        lane_raise_step = max(edge_radius * 1.7, 0.2)
        while (
            _path_hits_footprints(raw_path, occupied, margin=min_clearance * 0.2)
            and lane_z + lane_raise_step < room.height - 0.12
        ):
            lane_z += lane_raise_step
            raw_path = _compose_edge_route(
                start=start,
                end=end,
                axis_along_x=axis_along_x,
                lane_offset=lane_offset,
                lane_z=lane_z,
                axis_limit=axis_limit,
                cross_limit=cross_limit,
            )

        route = _smooth_path(raw_path, chamfer=corner_chamfer) if process_smooth_bends else _dedupe_path(raw_path)
        route = height_rule.clamp_path(route)
        if rules_auto_fix:
            route = no_pipe_collision_rule.resolve(route, generated_routes, height_rule)
            route = _dedupe_path(route)
        if len(route) < 2:
            continue

        emit(
            "refinery_process_pipe",
            create_pipe_from_path(points=route, radius=edge_radius, segments=18),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        pipe_count += 1
        generated_routes.append(route)

        horizontal_segment = _largest_horizontal_segment(route)
        if horizontal_segment is not None:
            seg_a, seg_b, seg_len = horizontal_segment
            yaw = degrees(atan2(seg_b[1] - seg_a[1], seg_b[0] - seg_a[0]))
            valve_pos = _lerp(seg_a, seg_b, 0.42)
            emit(
                "refinery_process_valve",
                create_valve(size=max(edge_radius * 4.2, 0.28)),
                valve_pos,
                (0.0, 0.0, yaw),
            )

            if secondary_enabled:
                corner_points = _route_corners(route)
                cluster_budget = max(1, min(2, int(round(secondary_density))))
                for corner_idx, (corner_position, corner_yaw) in enumerate(corner_points[:cluster_budget]):
                    cluster_size = max(edge_radius * 4.9, 0.32)
                    emit(
                        "refinery_valve_cluster",
                        _create_valve_cluster_mesh(cluster_size),
                        corner_position,
                        (0.0, 0.0, corner_yaw),
                    )
                    support_height = max(corner_position[2] - 0.06, 0.42)
                    emit(
                        "refinery_secondary_pipe_support",
                        create_refinery_support(
                            height=support_height,
                            spacing=max(edge_radius * 4.9, 0.7),
                        ),
                        (corner_position[0], corner_position[1], support_height / 2.0),
                        (0.0, 0.0, corner_yaw),
                    )

            if rules_auto_fix:
                for (support_point, support_yaw) in support_rule.support_points(route):
                    sx, sy, sz = support_point
                    support_height = max(sz - support_height_margin, 0.42)
                    emit(
                        "refinery_process_pipe_support",
                        create_refinery_support(height=support_height, spacing=max(edge_radius * 5.8, 0.72)),
                        (sx, sy, support_height / 2.0),
                        (0.0, 0.0, support_yaw),
                    )
            elif seg_len >= support_spacing * 0.95:
                support_count = max(1, int(seg_len // max(support_spacing, 0.5)))
                for support_idx in range(1, support_count + 1):
                    t = support_idx / (support_count + 1)
                    sx, sy, sz = _lerp(seg_a, seg_b, t)
                    support_height = max(sz - support_height_margin, 0.42)
                    emit(
                        "refinery_process_pipe_support",
                        create_refinery_support(height=support_height, spacing=max(edge_radius * 5.8, 0.72)),
                        (sx, sy, support_height / 2.0),
                        (0.0, 0.0, yaw),
                    )

            if _polyline_length(route) >= pump_length_threshold:
                pump_pos = _lerp(seg_a, seg_b, 0.72)
                emit(
                    "refinery_process_pump",
                    create_pump(size=max(edge_radius * 5.4, 0.48)),
                    pump_pos,
                    (0.0, 0.0, yaw),
                )

    if secondary_enabled:
        sensor_keypoints = [*column_flow_points, *node_positions, *tank_positions]
        if not sensor_keypoints:
            sensor_keypoints = [
                (0.0, 0.0, max(1.2, min(room.height - 0.25, top_level))),
            ]

        unique_sensor_points: list[Vector3] = []
        seen_sensor: set[tuple[int, int, int]] = set()
        for point in sensor_keypoints:
            key = (int(round(point[0] * 10.0)), int(round(point[1] * 10.0)), int(round(point[2] * 10.0)))
            if key in seen_sensor:
                continue
            seen_sensor.add(key)
            unique_sensor_points.append(point)

        sensor_limit = max(3, int(round(len(unique_sensor_points) * min(max(sensor_density, 0.4), 2.5))))
        for sensor_idx, (sx, sy, sz) in enumerate(unique_sensor_points[:sensor_limit]):
            sensor_size = max(pipe_radius * 1.9, 0.1)
            sensor_z = min(room.height - sensor_size * 0.25, sz + sensor_size * 0.45)
            emit(
                "refinery_sensor_unit",
                _create_sensor_unit_mesh(sensor_size),
                (sx, sy, sensor_z),
                (0.0, 0.0, 0.0),
            )

    connectivity_rule.ensure_backbone(pipe_count=pipe_count, emit=emit, room=room)


def _assign_node_points(
    node_points: dict[str, Vector3],
    node_ids: list[str],
    points: Sequence[Vector3],
) -> None:
    if not node_ids or not points:
        return
    for idx, node_id in enumerate(node_ids):
        node_points[node_id] = points[idx % len(points)]


def _assign_auxiliary_points(node_points: dict[str, Vector3], graph: ProcessGraph) -> None:
    incoming: dict[str, list[str]] = {}
    outgoing: dict[str, list[str]] = {}
    for edge in graph.edges:
        outgoing.setdefault(edge.source_id, []).append(edge.target_id)
        incoming.setdefault(edge.target_id, []).append(edge.source_id)

    for _ in range(max(1, len(graph.nodes))):
        progress = False
        for node in graph.nodes:
            if node.id in node_points:
                continue
            sources = incoming.get(node.id, [])
            targets = outgoing.get(node.id, [])
            source_point = next((node_points[source_id] for source_id in sources if source_id in node_points), None)
            target_point = next((node_points[target_id] for target_id in targets if target_id in node_points), None)

            if source_point is not None and target_point is not None:
                node_points[node.id] = (
                    (source_point[0] + target_point[0]) / 2.0,
                    (source_point[1] + target_point[1]) / 2.0,
                    (source_point[2] + target_point[2]) / 2.0,
                )
                progress = True
            elif source_point is not None:
                node_points[node.id] = (source_point[0], source_point[1], source_point[2] + 0.2)
                progress = True
            elif target_point is not None:
                node_points[node.id] = (target_point[0], target_point[1], target_point[2] + 0.2)
                progress = True
        if not progress:
            break


def _coerce_radius(value: object, default: float) -> float:
    if value is None:
        return max(default, 0.04)
    return max(float(value), 0.04)

def _path_hits_footprints(path: Sequence[Vector3], footprints: Sequence[_Footprint], margin: float) -> bool:
    if len(path) < 2 or not footprints:
        return False
    margin = max(0.0, float(margin))
    for idx in range(len(path) - 1):
        a = path[idx]
        b = path[idx + 1]
        if abs(a[2] - b[2]) > 1e-9:
            continue
        min_x = min(a[0], b[0])
        max_x = max(a[0], b[0])
        min_y = min(a[1], b[1])
        max_y = max(a[1], b[1])
        for footprint in footprints:
            left = footprint.x - footprint.half_w - margin
            right = footprint.x + footprint.half_w + margin
            bottom = footprint.y - footprint.half_d - margin
            top = footprint.y + footprint.half_d + margin
            if max_x < left or min_x > right:
                continue
            if max_y < bottom or min_y > top:
                continue
            return True
    return False


def _stable_hash(text: str) -> int:
    value = 0
    for idx, char in enumerate(text):
        value = (value * 131 + (idx + 1) * ord(char)) & 0xFFFFFFFF
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


def _normalize_pattern(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return "tower_cluster"
    return PATTERN_ALIASES.get(normalized, "tower_cluster")
