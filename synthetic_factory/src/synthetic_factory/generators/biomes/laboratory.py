from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, sqrt
from random import Random
from typing import Mapping

from ...parametric.laboratory_primitives import (
    create_bottle_cluster,
    create_cabinet,
    create_cable,
    create_equipment_unit,
    create_fume_hood,
    create_lab_bench,
    create_light_fixture,
    create_pipe,
    create_shelf,
    create_small_instrument,
    create_sink,
    create_tube_connection,
    create_wall_mounted_unit,
    create_waste_container,
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

PRIMITIVES = (
    "box",
    "lab_bench",
    "equipment_unit",
    "fume_hood",
    "shelf",
    "cabinet",
    "sink",
    "pipe",
    "cable",
    "light_fixture",
    "bottle_cluster",
    "small_instrument",
    "tube_connection",
    "waste_container",
    "wall_mounted_unit",
)
RULES: dict[str, object] = {
    "layout": "station zones + walkways + equipment + storage + infrastructure + lighting",
    "dependencies": [
        "stations depend on area",
        "equipment proportional to station count",
        "lighting proportional to station count",
    ],
    "patterns": [
        "wet_lab",
        "dry_lab",
        "analytical_lab",
        "teaching_lab",
        "research_lab",
    ],
    "rules": [
        "AccessRule",
        "MinSpacingRule",
        "EquipmentPlacementRule",
        "InfrastructureRule",
        "SafetyRule",
    ],
    "auto_fix": True,
}

STATION_TYPES = ("chemistry", "analysis", "preparation")
DEFAULT_EQUIPMENT_TYPES = ("analyzer", "mixer", "centrifuge", "pump", "controller", "heater")
EQUIPMENT_TYPE_BY_STATION: dict[str, tuple[str, ...]] = {
    "chemistry": ("reactor", "mixer", "heater", "pump"),
    "analysis": ("analyzer", "spectrometer", "chromatograph", "sensor"),
    "preparation": ("balance", "dispenser", "stirrer", "controller"),
}
PATTERN_ALIASES = {
    "wet_lab": "wet_lab",
    "wet lab": "wet_lab",
    "wet-lab": "wet_lab",
    "dry_lab": "dry_lab",
    "dry lab": "dry_lab",
    "dry-lab": "dry_lab",
    "analytical_lab": "analytical_lab",
    "analytical lab": "analytical_lab",
    "analytical-lab": "analytical_lab",
    "teaching_lab": "teaching_lab",
    "teaching lab": "teaching_lab",
    "teaching-lab": "teaching_lab",
    "research_lab": "research_lab",
    "research lab": "research_lab",
    "research-lab": "research_lab",
}


def _normalize_pattern(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        return "research_lab"
    return PATTERN_ALIASES.get(normalized, "research_lab")


def _stable_hash(text: str) -> int:
    value = 0
    for idx, char in enumerate(text):
        value = (value * 131 + (idx + 1) * ord(char)) & 0xFFFFFFFF
    return value


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class AccessRule:
    edge_margin: float = 0.04

    def enforce_station_positions(
        self,
        positions: list[tuple[float, float]],
        half_d: float,
        perimeter_walkway: float,
        module_d: float,
    ) -> tuple[list[tuple[float, float]], bool]:
        limit = max(0.0, half_d - perimeter_walkway - module_d / 2.0 - self.edge_margin)
        fixed: list[tuple[float, float]] = []
        changed = False
        for x, y in positions:
            ny = _clamp(y, -limit, limit)
            if abs(ny - y) > 1e-9:
                changed = True
            fixed.append((x, ny))
        return (fixed, changed)

    def compute_access_center(
        self,
        station_y: float,
        module_d: float,
        access_depth: float,
        half_d: float,
        perimeter_walkway: float,
    ) -> float:
        forward_y = -1.0 if station_y > 0.0 else 1.0
        center = station_y + forward_y * (module_d / 2.0 + access_depth / 2.0)
        low = -half_d + perimeter_walkway / 2.0 + self.edge_margin
        high = half_d - perimeter_walkway / 2.0 - self.edge_margin
        return _clamp(center, low, high)


@dataclass(frozen=True)
class MinSpacingRule:
    min_spacing: float

    def enforce_station_positions(
        self,
        positions: list[tuple[float, float]],
        module_w: float,
        module_d: float,
    ) -> tuple[list[tuple[float, float]], bool]:
        if len(positions) <= 1:
            return (list(positions), False)

        required_x = module_w + self.min_spacing
        required_y = module_d + self.min_spacing
        accepted: list[tuple[float, float]] = []
        changed = False
        for x, y in sorted(positions, key=lambda value: (value[1], value[0])):
            conflict = False
            for ox, oy in accepted:
                if abs(x - ox) < required_x and abs(y - oy) < required_y:
                    conflict = True
                    changed = True
                    break
            if not conflict:
                accepted.append((x, y))
        if not accepted:
            return ([positions[0]], True)
        return (accepted, changed)


@dataclass(frozen=True)
class EquipmentPlacementRule:
    near_table_distance: float

    def clamp_to_station(
        self,
        station_xy: tuple[float, float],
        bench_w: float,
        bench_d: float,
        equipment_xy: tuple[float, float],
    ) -> tuple[tuple[float, float], bool]:
        sx, sy = station_xy
        ex, ey = equipment_xy
        span_x = bench_w / 2.0 + self.near_table_distance
        span_y = bench_d / 2.0 + self.near_table_distance
        nx = _clamp(ex, sx - span_x, sx + span_x)
        ny = _clamp(ey, sy - span_y, sy + span_y)
        changed = abs(nx - ex) > 1e-9 or abs(ny - ey) > 1e-9
        return ((nx, ny), changed)


@dataclass(frozen=True)
class InfrastructureRule:
    def ensure_row_map(
        self,
        row_station_x: Mapping[float, list[float]],
        station_slots: list[tuple[float, float]],
    ) -> tuple[dict[float, list[float]], bool]:
        existing = {float(y): list(xs) for y, xs in row_station_x.items()}
        if existing:
            return (existing, False)
        fixed: dict[float, list[float]] = {}
        for sx, sy in station_slots:
            fixed.setdefault(float(sy), []).append(float(sx))
        return (fixed, bool(station_slots))


@dataclass(frozen=True)
class SafetyRule:
    walkway_clearance: float
    edge_margin: float = 0.04

    def enforce_fumehood_y(
        self,
        y: float,
        hood_depth: float,
        half_d: float,
        perimeter_walkway: float,
        walkway_centers: list[float],
        walkway_width: float,
    ) -> tuple[float, bool]:
        y_limit = max(0.0, half_d - perimeter_walkway - hood_depth / 2.0 - self.edge_margin)
        corrected = _clamp(y, -y_limit, y_limit)
        changed = abs(corrected - y) > 1e-9
        blocked_half = walkway_width / 2.0 + hood_depth / 2.0 + self.walkway_clearance
        for center in walkway_centers:
            delta = corrected - center
            if abs(delta) >= blocked_half:
                continue
            shift = blocked_half - abs(delta)
            corrected = corrected + shift if delta >= 0.0 else corrected - shift
            changed = True
        reclamped = _clamp(corrected, -y_limit, y_limit)
        if abs(reclamped - corrected) > 1e-9:
            changed = True
        return (reclamped, changed)


@dataclass(frozen=True)
class _EquipmentConnector:
    x: float
    y: float
    z: float
    direction: str


def _parse_string_options(value: object, default: tuple[str, ...], label: str) -> tuple[str, ...]:
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


def _sample_equipment_type(
    station_type: str,
    equipment_index: int,
    rng: Random,
    random_variation: bool,
    type_map: Mapping[str, tuple[str, ...]],
) -> str:
    options = type_map.get(station_type) or DEFAULT_EQUIPMENT_TYPES
    if not options:
        return "generic"
    if random_variation:
        return options[rng.randint(0, len(options) - 1)]
    return options[equipment_index % len(options)]


def _sample_cable_direction(
    mode: str,
    equipment_index: int,
    rng: Random,
    random_variation: bool,
) -> str:
    normalized = mode.strip().lower()
    if normalized not in {"up", "down", "mixed"}:
        normalized = "up"
    if normalized == "up":
        return "up"
    if normalized == "down":
        return "down"
    if random_variation:
        return "up" if rng.random() < 0.58 else "down"
    return "up" if equipment_index % 2 == 0 else "down"


def _generate_station_unit(
    *,
    station_index: int,
    station_xy: tuple[float, float],
    station_type: str,
    equipment_count: int,
    module_w: float,
    module_d: float,
    bench_height: float,
    access_depth: float,
    walkway_width: float,
    half_d: float,
    perimeter_walkway: float,
    room_h: float,
    fumehood_probability: float,
    water_enabled: bool,
    sink_probability: float,
    row_walkway_centers: list[float],
    auto_fix: bool,
    random_variation: bool,
    alignment_step: float,
    equipment_min_spacing: float,
    cable_mode: str,
    station_pipe_probability: float,
    yaw_jitter_max: float,
    equipment_type_map: Mapping[str, tuple[str, ...]],
    access_rule: AccessRule,
    equipment_rule: EquipmentPlacementRule,
    safety_rule: SafetyRule,
    rng: Random,
    emit: object,
) -> tuple[list[_EquipmentConnector], list[tuple[float, float, float]], list[tuple[float, float, float]], tuple[float, list[float]]]:
    sx, sy = station_xy
    row_x = [sx]
    equipment_connectors: list[_EquipmentConnector] = []
    sink_points: list[tuple[float, float, float]] = []
    service_pipe_points: list[tuple[float, float, float]] = []

    forward_y = -1.0 if sy > 0.0 else 1.0
    yaw = 180.0 if forward_y < 0.0 else 0.0
    if yaw_jitter_max > 0.0 and random_variation:
        yaw += rng.uniform(-yaw_jitter_max, yaw_jitter_max)
    rot = (0.0, 0.0, yaw)

    floor_h = 0.02
    emit("lab_station_module", create_box(module_w, floor_h, module_d), (sx, sy, floor_h / 2.0))
    bench_w = max(0.9, module_w * 0.72)
    bench_d = max(0.55, module_d * 0.48)
    emit("lab_bench", create_lab_bench(bench_w, bench_d, bench_height), (sx, sy, bench_height / 2.0), rot)

    if auto_fix:
        access_y = access_rule.compute_access_center(
            station_y=sy,
            module_d=module_d,
            access_depth=access_depth,
            half_d=half_d,
            perimeter_walkway=perimeter_walkway,
        )
    else:
        access_y = _clamp(
            sy + forward_y * (module_d / 2.0 + access_depth / 2.0),
            -half_d + perimeter_walkway / 2.0,
            half_d - perimeter_walkway / 2.0,
        )
    emit("lab_access_zone", create_box(bench_w, floor_h, min(access_depth, walkway_width)), (sx, access_y, floor_h / 2.0))

    has_sink = water_enabled and (station_type == "chemistry" or rng.random() < sink_probability)
    sink_xy: tuple[float, float] | None = None
    if has_sink:
        sink_xy = (sx + bench_w * 0.22, sy + forward_y * bench_d * 0.05)

    anchors = [(-0.28, 0.18), (0.0, 0.18), (0.28, 0.18), (-0.19, -0.12), (0.19, -0.12)]
    if random_variation:
        rng.shuffle(anchors)

    placed_local: list[tuple[float, float]] = []
    for j in range(equipment_count):
        chosen_xy: tuple[float, float] | None = None
        for offset in range(len(anchors)):
            ax, ay = anchors[(j + offset) % len(anchors)]
            ex = sx + ax * bench_w
            ey = sy + forward_y * ay * bench_d
            if auto_fix and alignment_step > 0.0:
                ex = sx + round((ex - sx) / alignment_step) * alignment_step
                ey = sy + round((ey - sy) / alignment_step) * alignment_step
            if auto_fix:
                (ex, ey), _ = equipment_rule.clamp_to_station(
                    station_xy=(sx, sy),
                    bench_w=bench_w,
                    bench_d=bench_d,
                    equipment_xy=(ex, ey),
                )
            overlap = any(abs(ex - px) < equipment_min_spacing and abs(ey - py) < equipment_min_spacing for px, py in placed_local)
            if overlap:
                continue
            if sink_xy is not None and abs(ex - sink_xy[0]) < equipment_min_spacing and abs(ey - sink_xy[1]) < equipment_min_spacing:
                continue
            chosen_xy = (ex, ey)
            break

        if chosen_xy is None:
            ex = sx + ((-1.0 if j % 2 == 0 else 1.0) * bench_w * 0.16)
            ey = sy + forward_y * bench_d * (0.06 + 0.08 * (j % 3))
            if auto_fix:
                (ex, ey), _ = equipment_rule.clamp_to_station(
                    station_xy=(sx, sy),
                    bench_w=bench_w,
                    bench_d=bench_d,
                    equipment_xy=(ex, ey),
                )
            chosen_xy = (ex, ey)

        ex, ey = chosen_xy
        placed_local.append((ex, ey))
        equipment_type = _sample_equipment_type(
            station_type=station_type,
            equipment_index=j + station_index * 7,
            rng=rng,
            random_variation=random_variation,
            type_map=equipment_type_map,
        )
        eh = 0.36 if station_type == "analysis" else 0.3
        ez = bench_height + eh / 2.0 + 0.01
        emit(
            "lab_equipment_unit",
            create_equipment_unit(width=max(bench_w * 0.2, 0.12), height=eh, type=equipment_type),
            (ex, ey, ez),
            rot,
        )
        equipment_connectors.append(
            _EquipmentConnector(
                x=ex,
                y=ey,
                z=bench_height + eh + 0.01,
                direction=_sample_cable_direction(
                    mode=cable_mode,
                    equipment_index=j,
                    rng=rng,
                    random_variation=random_variation,
                ),
            )
        )

    if station_type == "chemistry" or rng.random() < fumehood_probability:
        hh = _clamp(max(1.8, room_h * 0.48), 1.8, room_h - 0.15)
        hood_depth = _clamp(module_d * 0.58, 0.55, 1.25)
        hood_y = sy - forward_y * module_d * 0.2
        if auto_fix:
            hood_y, _ = safety_rule.enforce_fumehood_y(
                y=hood_y,
                hood_depth=hood_depth,
                half_d=half_d,
                perimeter_walkway=perimeter_walkway,
                walkway_centers=row_walkway_centers,
                walkway_width=walkway_width,
            )
        emit(
            "lab_fume_hood",
            create_fume_hood(
                width=_clamp(module_w * 0.9, 0.85, 2.4),
                height=hh,
                depth=hood_depth,
            ),
            (sx, hood_y, hh / 2.0),
            rot,
        )

    if has_sink and sink_xy is not None:
        sxk, syk = sink_xy
        emit(
            "lab_sink",
            create_sink(
                width=_clamp(bench_w * 0.34, 0.26, 0.5),
                depth=_clamp(bench_d * 0.32, 0.24, 0.45),
            ),
            (sxk, syk, bench_height + 0.01),
            rot,
        )
        sink_points.append((sxk, syk, bench_height + 0.05))
    elif rng.random() < station_pipe_probability:
        service_pipe_points.append((sx - bench_w * 0.18, sy + forward_y * bench_d * 0.22, bench_height + 0.04))

    return (equipment_connectors, sink_points, service_pipe_points, (sy, row_x))


def _parse_float_range(
    value: object,
    default_min: float,
    default_max: float,
    label: str,
    *,
    allow_zero: bool = False,
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
        minimum = float(value)
        maximum = float(value)
    if allow_zero:
        if minimum < 0.0 or maximum < 0.0:
            raise ValueError(f"{label} values must be >= 0.")
    else:
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
    *,
    allow_zero: bool = False,
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
        minimum = int(value)
        maximum = int(value)
    if allow_zero:
        if minimum < 0 or maximum < 0:
            raise ValueError(f"{label} values must be >= 0.")
    else:
        if minimum <= 0 or maximum <= 0:
            raise ValueError(f"{label} values must be > 0.")
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


def _weighted_station_type(rng: Random, weights: Mapping[str, float]) -> str:
    total = sum(max(float(weights.get(name, 0.0)), 0.0) for name in STATION_TYPES)
    if total <= 1e-12:
        return STATION_TYPES[0]
    roll = rng.uniform(0.0, total)
    cursor = 0.0
    for name in STATION_TYPES:
        cursor += max(float(weights.get(name, 0.0)), 0.0)
        if roll <= cursor:
            return name
    return STATION_TYPES[-1]


def _distribute_equipment(station_count: int, density: float) -> list[int]:
    total = max(station_count, min(int(round(station_count * density)), station_count * 5))
    counts = [1] * station_count
    remaining = total - station_count
    cursor = 0
    while remaining > 0 and station_count > 0:
        idx = cursor % station_count
        cursor += 1
        if counts[idx] >= 5:
            continue
        counts[idx] += 1
        remaining -= 1
    return counts


def _primary_station_type(weights: Mapping[str, float]) -> str:
    best_name = STATION_TYPES[0]
    best_weight = float("-inf")
    for name in STATION_TYPES:
        weight = float(weights.get(name, 0.0))
        if weight > best_weight:
            best_weight = weight
            best_name = name
    return best_name


def populate(room: BiomeRoom, add_object: AddObjectFn, settings: Mapping[str, object]) -> None:
    cfg = to_mapping(settings.get("laboratory"))
    rules_cfg = to_mapping(cfg.get("rules"))
    room_cfg = to_mapping(cfg.get("room"))
    stations_cfg = to_mapping(cfg.get("stations"))
    equipment_cfg = to_mapping(cfg.get("equipment"))
    fumehood_cfg = to_mapping(cfg.get("fumehood"))
    storage_cfg = to_mapping(cfg.get("storage"))
    infra_cfg = to_mapping(cfg.get("infrastructure"))
    light_cfg = to_mapping(cfg.get("lighting"))
    secondary_cfg = to_mapping(cfg.get("secondary"))

    seed = int(cfg.get("seed", 0)) + _stable_hash(room.room_id)
    rng = Random(seed)
    random_variation = to_bool(cfg.get("random_variation", True), default=True)
    pattern = _normalize_pattern(str(cfg.get("pattern", "research_lab")))
    auto_fix = to_bool(cfg.get("auto_fix", rules_cfg.get("auto_fix", True)), default=True)

    room_w = min(room.width, _sample_float(_parse_float_range(room_cfg.get("width"), 8.0, 30.0, "room.width"), rng, random_variation))
    room_d = min(room.depth, _sample_float(_parse_float_range(room_cfg.get("depth"), 8.0, 30.0, "room.depth"), rng, random_variation))
    room_h = min(room.height, _sample_float(_parse_float_range(room_cfg.get("height"), 2.5, 4.5, "room.height"), rng, random_variation))

    station_count_target = _sample_int(_parse_int_range(stations_cfg.get("count"), 2, 12, "stations.count"), rng, random_variation)
    station_spacing = _sample_float(_parse_float_range(stations_cfg.get("spacing"), 1.5, 3.0, "stations.spacing"), rng, random_variation)
    type_variability = _sample_float(
        _parse_float_range(stations_cfg.get("type_variability", [0.15, 0.65]), 0.15, 0.65, "stations.type_variability", allow_zero=True),
        rng,
        random_variation,
    )
    equipment_density = _sample_float(_parse_float_range(equipment_cfg.get("density"), 1.0, 5.0, "equipment.density"), rng, random_variation)
    fumehood_probability = _sample_float(
        _parse_float_range(fumehood_cfg.get("probability"), 0.0, 0.5, "fumehood.probability", allow_zero=True),
        rng,
        random_variation,
    )
    shelves_count = _sample_int(_parse_int_range(storage_cfg.get("shelves"), 0, 10, "storage.shelves", allow_zero=True), rng, random_variation)
    cabinets_count = _sample_int(_parse_int_range(storage_cfg.get("cabinets"), 0, 8, "storage.cabinets", allow_zero=True), rng, random_variation)
    cable_density = _sample_float(_parse_float_range(infra_cfg.get("cable_density"), 0.5, 2.0, "infrastructure.cable_density"), rng, random_variation)
    pipe_density = _sample_float(_parse_float_range(infra_cfg.get("pipe_density"), 0.2, 1.0, "infrastructure.pipe_density"), rng, random_variation)
    light_intensity = _sample_float(_parse_float_range(light_cfg.get("intensity"), 0.5, 2.0, "lighting.intensity"), rng, random_variation)

    # Scale station/equipment density with room area.
    area_scale = room_area_scale(room, reference_area=160.0, min_scale=0.75, max_scale=3.2)
    station_count_target = max(1, int(round(station_count_target * area_scale)))
    shelves_count = max(0, int(round(shelves_count * (0.75 + 0.25 * area_scale))))
    cabinets_count = max(0, int(round(cabinets_count * (0.75 + 0.25 * area_scale))))
    equipment_density = equipment_density * (0.85 + 0.22 * area_scale)
    cable_density = cable_density * (0.85 + 0.2 * area_scale)
    pipe_density = pipe_density * (0.85 + 0.18 * area_scale)
    light_intensity = light_intensity * (0.9 + 0.15 * area_scale)

    perimeter_walkway = to_positive_float(cfg.get("perimeter_walkway"), max(0.8, station_spacing * 0.5), "laboratory.perimeter_walkway")
    walkway_width = to_positive_float(cfg.get("walkway_width"), max(1.0, station_spacing * 0.65), "laboratory.walkway_width")
    access_depth = to_positive_float(cfg.get("access_depth"), 0.85, "laboratory.access_depth")
    bench_height = to_positive_float(cfg.get("bench_height"), 0.9, "laboratory.bench_height")
    module_w = _sample_float(_parse_float_range(cfg.get("module_width"), 1.9, 2.8, "module_width"), rng, random_variation)
    module_d = _sample_float(_parse_float_range(cfg.get("module_depth"), 1.4, 2.2, "module_depth"), rng, random_variation)
    water_enabled = to_bool(cfg.get("water_enabled", False), default=False)
    sink_probability = _clamp(float(cfg.get("sink_probability", 0.35)), 0.0, 1.0)
    cable_radius = to_positive_float(cfg.get("cable_radius"), 0.008, "laboratory.cable_radius")
    pipe_radius = to_positive_float(cfg.get("pipe_radius"), 0.01, "laboratory.pipe_radius")
    near_storage_prob = _clamp(float(cfg.get("near_station_storage_probability", 0.34)), 0.0, 1.0)
    station_alignment_step = to_positive_float(
        cfg.get("station_alignment_step"),
        0.04,
        "laboratory.station_alignment_step",
    )
    equipment_min_spacing = to_positive_float(
        rules_cfg.get("equipment_min_spacing"),
        0.16,
        "laboratory.rules.equipment_min_spacing",
    )
    cable_mode = str(cfg.get("equipment_cable_mode", "mixed")).strip().lower()
    station_pipe_probability = _clamp(float(cfg.get("station_pipe_probability", 0.45)), 0.0, 1.0)
    generic_equipment_types = _parse_string_options(
        cfg.get("equipment_types"),
        DEFAULT_EQUIPMENT_TYPES,
        "laboratory.equipment_types",
    )
    type_map_cfg = to_mapping(cfg.get("equipment_type_map"))
    equipment_type_map: dict[str, tuple[str, ...]] = {}
    for station_name in STATION_TYPES:
        equipment_type_map[station_name] = _parse_string_options(
            type_map_cfg.get(station_name),
            EQUIPMENT_TYPE_BY_STATION.get(station_name, generic_equipment_types),
            f"laboratory.equipment_type_map.{station_name}",
        )
    min_station_spacing = to_positive_float(
        rules_cfg.get("min_spacing", cfg.get("min_station_spacing", 0.95)),
        0.95,
        "laboratory.rules.min_spacing",
    )
    safety_walkway_clearance = to_positive_float(
        rules_cfg.get("safety_walkway_clearance"),
        0.06,
        "laboratory.rules.safety_walkway_clearance",
    )
    equipment_near_distance = to_positive_float(
        rules_cfg.get("equipment_near_distance"),
        0.24,
        "laboratory.rules.equipment_near_distance",
    )
    rule_edge_margin = to_positive_float(
        rules_cfg.get("edge_margin"),
        0.04,
        "laboratory.rules.edge_margin",
    )
    detail_density = _clamp(float(secondary_cfg.get("detail_density", 1.0)), 0.0, 2.0)
    tube_curvature_max = _clamp(float(secondary_cfg.get("tube_curvature_max", 0.35)), 0.0, 1.0)
    wall_units_enabled = to_bool(secondary_cfg.get("wall_units_enabled", True), default=True)

    access_rule = AccessRule(edge_margin=rule_edge_margin)
    spacing_rule = MinSpacingRule(min_spacing=min_station_spacing)
    equipment_rule = EquipmentPlacementRule(near_table_distance=equipment_near_distance)
    infrastructure_rule = InfrastructureRule()
    safety_rule = SafetyRule(walkway_clearance=safety_walkway_clearance, edge_margin=rule_edge_margin)
    if auto_fix:
        walkway_width = max(walkway_width, access_depth)

    strict_grid = False
    chaotic_layout = False
    station_type_override: str | None = None
    station_type_weights_override: dict[str, float] | None = None
    yaw_jitter_max = 0.0
    emit_extra_pipe_rows = False

    if pattern == "wet_lab":
        water_enabled = True
        sink_probability = max(sink_probability, 0.75)
        pipe_density = min(1.0, max(pipe_density * 1.45, 0.72))
        cable_density = min(2.0, max(cable_density, 0.85))
        station_pipe_probability = max(station_pipe_probability, 0.65)
        fumehood_probability = max(fumehood_probability, 0.3)
        station_type_weights_override = {"chemistry": 0.6, "analysis": 0.25, "preparation": 0.15}
        emit_extra_pipe_rows = True
    elif pattern == "dry_lab":
        water_enabled = False
        sink_probability = 0.0
        station_pipe_probability = min(station_pipe_probability, 0.2)
        pipe_density = max(0.2, min(pipe_density, 0.35))
        cable_density = min(2.0, max(cable_density * 1.35, 1.1))
        equipment_density = min(5.0, max(equipment_density * 1.2, 1.0))
        fumehood_probability = min(fumehood_probability, 0.2)
        station_type_weights_override = {"chemistry": 0.12, "analysis": 0.58, "preparation": 0.3}
    elif pattern == "analytical_lab":
        station_count_target = max(1, int(round(station_count_target * 0.62)))
        equipment_density = min(5.0, max(equipment_density * 1.7, 2.2))
        fumehood_probability = min(fumehood_probability, 0.35)
        station_type_weights_override = {"chemistry": 0.15, "analysis": 0.7, "preparation": 0.15}
    elif pattern == "teaching_lab":
        strict_grid = True
        station_spacing = max(station_spacing, 1.9)
        walkway_width = max(walkway_width, 1.4)
        type_variability = 0.0
        equipment_density = _clamp(round(equipment_density), 1.0, 5.0)
        fumehood_probability = min(fumehood_probability, 0.15)
        near_storage_prob = min(near_storage_prob, 0.2)
        candidate_type = str(cfg.get("teaching_station_type", "analysis")).strip().lower()
        station_type_override = candidate_type if candidate_type in STATION_TYPES else "analysis"
        station_type_weights_override = {name: (1.0 if name == station_type_override else 0.01) for name in STATION_TYPES}
    else:  # research_lab
        chaotic_layout = True
        type_variability = max(type_variability, 0.65)
        equipment_density = min(5.0, max(equipment_density, 2.1))
        cable_density = min(2.0, max(cable_density * 1.2, 1.0))
        pipe_density = min(1.0, max(pipe_density * 1.1, 0.45))
        yaw_jitter_max = 24.0

    half_w = room_w / 2.0
    half_d = room_d / 2.0
    inner_half_w = max(half_w - perimeter_walkway, 0.7)
    inner_half_d = max(half_d - perimeter_walkway, 0.7)
    pitch_x = max(module_w + station_spacing, module_w * 1.2)
    pitch_y = max(module_d + walkway_width, module_d * 1.3)
    cols = max(1, int((max(room_w - 2.0 * perimeter_walkway, module_w) + station_spacing) // max(pitch_x, 0.1)))
    rows = max(1, int((max(room_d - 2.0 * perimeter_walkway, module_d) + walkway_width) // max(pitch_y, 0.1)))
    x_positions = symmetric_positions(cols, min(max(inner_half_w - module_w / 2.0, 0.0), (cols - 1) * pitch_x / 2.0)) or [0.0]
    row_positions = symmetric_positions(rows, min(max(inner_half_d - module_d / 2.0, 0.0), (rows - 1) * pitch_y / 2.0)) or [0.0]
    slots = [(x, y) for y in row_positions for x in x_positions] or [(0.0, 0.0)]

    # Dependency: stations depend on area and requested range.
    area_capacity = max(1, int((room_w * room_d * 0.62) // max(module_w * module_d, 0.6)))
    station_count = min(len(slots), area_capacity, max(1, station_count_target))
    selected_slots = list(slots[:station_count])
    if chaotic_layout:
        rng.shuffle(selected_slots)
    elif random_variation and not strict_grid:
        rng.shuffle(selected_slots)
    if strict_grid or not chaotic_layout:
        selected_slots.sort(key=lambda p: (p[1], p[0]))
    if chaotic_layout:
        jitter_x = max(0.06, min((pitch_x - module_w) * 0.42, module_w * 0.28))
        jitter_y = max(0.06, min((pitch_y - module_d) * 0.42, module_d * 0.28))
        x_min = -inner_half_w + module_w / 2.0
        x_max = inner_half_w - module_w / 2.0
        y_min = -inner_half_d + module_d / 2.0
        y_max = inner_half_d - module_d / 2.0
        jittered_slots: list[tuple[float, float]] = []
        for sx, sy in selected_slots:
            jx = _clamp(sx + rng.uniform(-jitter_x, jitter_x), x_min, x_max)
            jy = _clamp(sy + rng.uniform(-jitter_y, jitter_y), y_min, y_max)
            jittered_slots.append((jx, jy))
        selected_slots = jittered_slots

    if auto_fix:
        selected_slots, _ = access_rule.enforce_station_positions(
            positions=selected_slots,
            half_d=half_d,
            perimeter_walkway=perimeter_walkway,
            module_d=module_d,
        )
        selected_slots, _ = spacing_rule.enforce_station_positions(
            positions=selected_slots,
            module_w=module_w,
            module_d=module_d,
        )
        selected_slots, _ = access_rule.enforce_station_positions(
            positions=selected_slots,
            half_d=half_d,
            perimeter_walkway=perimeter_walkway,
            module_d=module_d,
        )
        if not selected_slots:
            selected_slots = [(0.0, 0.0)]

    # Dependency: equipment proportional to station count.
    equipment_counts = _distribute_equipment(station_count=len(selected_slots), density=equipment_density)
    if pattern == "analytical_lab":
        equipment_counts = [min(5, max(2, value + 1)) for value in equipment_counts]
    elif pattern == "teaching_lab":
        uniform_count = max(1, min(5, int(round(equipment_density))))
        equipment_counts = [uniform_count] * len(selected_slots)
    elif pattern == "research_lab":
        equipment_counts = [
            max(1, min(5, value + (rng.randint(-1, 2) if random_variation else 0)))
            for value in equipment_counts
        ]
    elif pattern == "dry_lab":
        equipment_counts = [min(5, value + (1 if idx % 2 == 0 else 0)) for idx, value in enumerate(equipment_counts)]

    station_weights = {"chemistry": 0.34, "analysis": 0.38, "preparation": 0.28}
    mix_cfg = to_mapping(stations_cfg.get("type_mix", cfg.get("station_type_weights")))
    for name in STATION_TYPES:
        if name in mix_cfg:
            station_weights[name] = max(float(mix_cfg[name]), 0.01)
    if station_type_weights_override:
        for name in STATION_TYPES:
            station_weights[name] = max(float(station_type_weights_override.get(name, station_weights[name])), 0.01)
    dominant_station_type = _primary_station_type(station_weights)
    variability_enabled = random_variation and not strict_grid
    for name in STATION_TYPES:
        if random_variation and type_variability > 0.0:
            station_weights[name] = max(station_weights[name] * (1.0 + rng.uniform(-type_variability, type_variability)), 0.01)

    counters: dict[str, int] = {}

    def emit(object_type: str, mesh: object, pos: tuple[float, float, float], rot: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        counters[object_type] = counters.get(object_type, 0) + 1
        add_object(object_type, counters[object_type], mesh, pos, rot)  # type: ignore[arg-type]

    # Zones and walkways.
    floor_h = 0.02
    emit("lab_perimeter_walkway", create_box(room_w, floor_h, perimeter_walkway), (0.0, -half_d + perimeter_walkway / 2.0, floor_h / 2.0))
    emit("lab_perimeter_walkway", create_box(room_w, floor_h, perimeter_walkway), (0.0, half_d - perimeter_walkway / 2.0, floor_h / 2.0))
    emit("lab_perimeter_walkway", create_box(perimeter_walkway, floor_h, max(room_d - 2.0 * perimeter_walkway, perimeter_walkway)), (-half_w + perimeter_walkway / 2.0, 0.0, floor_h / 2.0))
    emit("lab_perimeter_walkway", create_box(perimeter_walkway, floor_h, max(room_d - 2.0 * perimeter_walkway, perimeter_walkway)), (half_w - perimeter_walkway / 2.0, 0.0, floor_h / 2.0))
    row_walkway_centers: list[float] = []
    for i in range(max(len(row_positions) - 1, 0)):
        y = (row_positions[i] + row_positions[i + 1]) / 2.0
        row_walkway_centers.append(y)
        emit("lab_walkway", create_box(max(room_w - 2.0 * perimeter_walkway, module_w), floor_h, walkway_width), (0.0, y, floor_h / 2.0))

    equipment_connectors: list[_EquipmentConnector] = []
    sink_points: list[tuple[float, float, float]] = []
    service_pipe_points: list[tuple[float, float, float]] = []
    row_station_x: dict[float, list[float]] = {}
    shelf_slots: list[tuple[float, float, float, float]] = []

    for idx, (sx, sy) in enumerate(selected_slots):
        if station_type_override is not None:
            station_type = station_type_override
        elif variability_enabled or chaotic_layout:
            station_type = _weighted_station_type(rng, station_weights)
        else:
            station_type = dominant_station_type if strict_grid else STATION_TYPES[idx % len(STATION_TYPES)]
        station_connectors, station_sinks, station_service_pipes, row_data = _generate_station_unit(
            station_index=idx,
            station_xy=(sx, sy),
            station_type=station_type,
            equipment_count=equipment_counts[idx],
            module_w=module_w,
            module_d=module_d,
            bench_height=bench_height,
            access_depth=access_depth,
            walkway_width=walkway_width,
            half_d=half_d,
            perimeter_walkway=perimeter_walkway,
            room_h=room_h,
            fumehood_probability=fumehood_probability,
            water_enabled=water_enabled,
            sink_probability=sink_probability,
            row_walkway_centers=row_walkway_centers,
            auto_fix=auto_fix,
            random_variation=random_variation,
            alignment_step=station_alignment_step,
            equipment_min_spacing=equipment_min_spacing,
            cable_mode=cable_mode,
            station_pipe_probability=station_pipe_probability,
            yaw_jitter_max=yaw_jitter_max,
            equipment_type_map=equipment_type_map,
            access_rule=access_rule,
            equipment_rule=equipment_rule,
            safety_rule=safety_rule,
            rng=rng,
            emit=emit,
        )
        equipment_connectors.extend(station_connectors)
        sink_points.extend(station_sinks)
        service_pipe_points.extend(station_service_pipes)
        row_y, row_x = row_data
        row_station_x.setdefault(row_y, []).extend(row_x)

        if detail_density <= 0.0:
            continue

        forward_y = -1.0 if sy > 0.0 else 1.0
        station_yaw = 180.0 if forward_y < 0.0 else 0.0

        bottle_count = max(2, min(7, int(round(equipment_counts[idx] * (0.8 + 0.6 * detail_density)))))
        bottle_x = sx - module_w * 0.16
        bottle_y = sy + forward_y * module_d * 0.08
        emit(
            "lab_bottle_cluster",
            create_bottle_cluster(
                bottle_radius=max(0.015, module_w * 0.0075),
                bottle_height=max(0.09, bench_height * 0.18),
                count=bottle_count,
            ),
            (bottle_x, bottle_y, bench_height + 0.01),
            (0.0, 0.0, station_yaw),
        )

        instrument_limit = max(1, min(4, int(round(1.5 * detail_density))))
        instrument_width = _clamp(module_w * 0.12, 0.09, 0.22)
        instrument_depth = _clamp(module_d * 0.12, 0.08, 0.2)
        instrument_height = _clamp(bench_height * 0.14, 0.07, 0.18)
        for j, connector in enumerate(station_connectors[:instrument_limit]):
            offset_x = (-1.0 if j % 2 == 0 else 1.0) * instrument_width * 0.58
            offset_y = -forward_y * instrument_depth * 0.25
            emit(
                "lab_small_instrument",
                create_small_instrument(
                    width=instrument_width,
                    depth=instrument_depth,
                    height=instrument_height,
                ),
                (
                    connector.x + offset_x,
                    connector.y + offset_y,
                    bench_height + 0.01,
                ),
                (0.0, 0.0, station_yaw),
            )

        tubes_emitted = 0
        sorted_connectors = sorted(station_connectors, key=lambda c: (c.x, c.y))
        for left, right in zip(sorted_connectors, sorted_connectors[1:]):
            if tubes_emitted >= max(1, int(round(2.0 * detail_density))):
                break
            dx = right.x - left.x
            dy = right.y - left.y
            run = sqrt(dx * dx + dy * dy)
            if run <= 0.08:
                continue
            curvature = 0.0
            if random_variation and tube_curvature_max > 0.0:
                curvature = rng.uniform(-tube_curvature_max, tube_curvature_max)
            emit(
                "lab_tube_connection",
                create_tube_connection(
                    radius=max(pipe_radius * 0.36, 0.0025),
                    length=max(run * 0.92, 0.08),
                    curvature=curvature,
                ),
                (
                    (left.x + right.x) / 2.0,
                    (left.y + right.y) / 2.0,
                    bench_height + max(instrument_height * 0.42, 0.035),
                ),
                (0.0, 0.0, degrees(atan2(dy, dx))),
            )
            tubes_emitted += 1

        if station_sinks and station_connectors:
            sink_x, sink_y, _ = station_sinks[0]
            nearest = min(station_connectors, key=lambda conn: (conn.x - sink_x) ** 2 + (conn.y - sink_y) ** 2)
            dx = nearest.x - sink_x
            dy = nearest.y - sink_y
            run = sqrt(dx * dx + dy * dy)
            if run > 0.07:
                emit(
                    "lab_tube_connection",
                    create_tube_connection(
                        radius=max(pipe_radius * 0.34, 0.0022),
                        length=max(run * 0.95, 0.07),
                        curvature=0.12 if random_variation else 0.0,
                    ),
                    (
                        (nearest.x + sink_x) / 2.0,
                        (nearest.y + sink_y) / 2.0,
                        bench_height + 0.03,
                    ),
                    (0.0, 0.0, degrees(atan2(dy, dx))),
                )

        waste_x = _clamp(
            sx + (-1.0 if sx >= 0.0 else 1.0) * module_w * 0.34,
            -half_w + perimeter_walkway + 0.2,
            half_w - perimeter_walkway - 0.2,
        )
        waste_y = _clamp(
            sy + forward_y * (module_d * 0.5 + min(access_depth, walkway_width) * 0.2),
            -half_d + perimeter_walkway + 0.2,
            half_d - perimeter_walkway - 0.2,
        )
        emit(
            "lab_waste_container",
            create_waste_container(
                width=_clamp(module_w * 0.16, 0.2, 0.38),
                depth=_clamp(module_d * 0.14, 0.18, 0.32),
                height=_clamp(bench_height * 0.5, 0.32, 0.65),
            ),
            (waste_x, waste_y, 0.0),
        )

    # Storage.
    wall_y = max(half_d - perimeter_walkway - 0.45, 0.0)
    wall_x_positions = symmetric_positions(max(2, (shelves_count + cabinets_count + 1) // 2), max(half_w - perimeter_walkway - 0.95, 0.0)) or [0.0]
    for side in (-1.0, 1.0):
        for x in wall_x_positions:
            if shelves_count <= 0 and cabinets_count <= 0:
                break
            y = side * wall_y
            yaw = 0.0 if side < 0.0 else 180.0
            if shelves_count > 0:
                shelf_height = min(room_h * 0.6, 1.9)
                emit("lab_shelf", create_shelf(width=0.95, height=shelf_height, levels=3), (x, y, 0.0), (0.0, 0.0, yaw))
                shelf_slots.append((x, y, shelf_height, yaw))
                shelves_count -= 1
            elif cabinets_count > 0:
                emit("lab_cabinet", create_cabinet(width=0.85, height=min(room_h * 0.56, 1.75), depth=0.52), (x, y, 0.0), (0.0, 0.0, yaw))
                cabinets_count -= 1
    for sx, sy in selected_slots:
        if shelves_count <= 0 and cabinets_count <= 0:
            break
        if rng.random() > near_storage_prob:
            continue
        nx = _clamp(sx + (-1.0 if sx >= 0.0 else 1.0) * module_w * 0.42, -half_w + perimeter_walkway + 0.5, half_w - perimeter_walkway - 0.5)
        if shelves_count > 0:
            shelf_height = min(room_h * 0.5, 1.55)
            emit("lab_shelf", create_shelf(width=0.8, height=shelf_height, levels=2), (nx, sy, 0.0))
            shelf_slots.append((nx, sy, shelf_height, 0.0))
            shelves_count -= 1
        elif cabinets_count > 0:
            emit("lab_cabinet", create_cabinet(width=0.72, height=min(room_h * 0.46, 1.4), depth=0.44), (nx, sy, 0.0))
            cabinets_count -= 1

    if detail_density > 0.0:
        for sx, sy, shelf_height, yaw in shelf_slots:
            cluster_count = 1
            if random_variation and rng.random() < _clamp(0.25 + 0.35 * detail_density, 0.0, 0.85):
                cluster_count = 2
            for idx in range(cluster_count):
                offset_x = (-1.0 if idx % 2 == 0 else 1.0) * 0.18
                if random_variation:
                    offset_x += rng.uniform(-0.05, 0.05)
                offset_y = (-1.0 if sy > 0.0 else 1.0) * 0.06
                emit(
                    "lab_bottle_cluster",
                    create_bottle_cluster(
                        bottle_radius=0.012,
                        bottle_height=0.075,
                        count=max(2, int(round(2 + detail_density))),
                    ),
                    (
                        sx + offset_x,
                        sy + offset_y,
                        max(shelf_height * 0.56, 0.45),
                    ),
                    (0.0, 0.0, yaw),
                )

    if wall_units_enabled and detail_density > 0.0:
        units_per_side = max(1, min(6, int(round((len(selected_slots) / 2.0) * max(detail_density, 0.45)))))
        wall_unit_span = max(half_w - perimeter_walkway - 0.75, 0.0)
        unit_y = max(half_d - perimeter_walkway - 0.08, 0.0)
        unit_z = _clamp(room_h * 0.58, 1.1, max(room_h - 0.32, 1.1))
        for side in (-1.0, 1.0):
            yaw = 0.0 if side < 0.0 else 180.0
            for x in symmetric_positions(units_per_side, wall_unit_span):
                emit(
                    "lab_wall_mounted_unit",
                    create_wall_mounted_unit(
                        width=_clamp(module_w * 0.22, 0.35, 0.72),
                        height=_clamp(room_h * 0.14, 0.26, 0.62),
                        depth=_clamp(module_d * 0.1, 0.12, 0.28),
                    ),
                    (x, side * unit_y, unit_z),
                    (0.0, 0.0, yaw),
                )

    # Infrastructure.
    cable_norm = _clamp((cable_density - 0.5) / 1.5, 0.0, 1.0)
    pipe_norm = _clamp((pipe_density - 0.2) / 0.8, 0.0, 1.0)
    cable_z = min(room_h - 0.1, max(2.05, room_h * 0.82))
    pipe_z = min(room_h - 0.2, max(1.8, room_h * 0.72))
    if auto_fix:
        row_station_x, _ = infrastructure_rule.ensure_row_map(
            row_station_x=row_station_x,
            station_slots=selected_slots,
        )
        if equipment_connectors and not row_station_x:
            avg_y = sum(conn.y for conn in equipment_connectors) / len(equipment_connectors)
            row_station_x = {avg_y: [conn.x for conn in equipment_connectors]}

    for row_y, xs in row_station_x.items():
        if not xs:
            continue
        xmin = min(xs) - module_w * 0.45
        xmax = max(xs) + module_w * 0.45
        emit("lab_cable_trunk", create_cable(radius=max(cable_radius * (1.05 + 0.55 * cable_norm), 0.0015), length=max(xmax - xmin, module_w * 0.45)), ((xmin + xmax) / 2.0, row_y, cable_z))
        if emit_extra_pipe_rows:
            emit(
                "lab_pipe_row",
                create_pipe(radius=max(pipe_radius * 0.92, 0.002), length=max(xmax - xmin, module_w * 0.45)),
                ((xmin + xmax) / 2.0, row_y, max(pipe_z - 0.16, 0.12)),
            )
    for connector in equipment_connectors:
        ex = connector.x
        ey = connector.y
        ez = connector.z
        if connector.direction == "down":
            dn_len = max(ez - 0.04, 0.05)
            emit("lab_cable_drop_down", create_cable(radius=max(cable_radius * 0.82, 0.0012), length=dn_len), (ex, ey, dn_len / 2.0), (0.0, -90.0, 0.0))
            continue
        up_len = max(cable_z - ez, 0.06)
        emit("lab_cable_drop_up", create_cable(radius=cable_radius, length=up_len), (ex, ey, ez + up_len / 2.0), (0.0, -90.0, 0.0))
        if random_variation and rng.random() < (0.15 + 0.75 * cable_norm):
            dn_len = max(ez - 0.04, 0.05)
            emit("lab_cable_drop_down", create_cable(radius=max(cable_radius * 0.82, 0.0012), length=dn_len), (ex, ey, dn_len / 2.0), (0.0, -90.0, 0.0))

    wall_pipe_y = max(half_d - perimeter_walkway - 0.26, 0.0)
    wall_len = max(room_w - 2.0 * perimeter_walkway - 0.1, 0.5)
    pr = max(pipe_radius * (1.0 + 0.35 * pipe_norm), 0.002)
    emit("lab_pipe_main", create_pipe(radius=pr, length=wall_len), (0.0, wall_pipe_y, pipe_z))
    emit("lab_pipe_main", create_pipe(radius=pr, length=wall_len), (0.0, -wall_pipe_y, pipe_z))
    emit("lab_pipe_ceiling", create_pipe(radius=max(pipe_radius * (0.95 + 0.2 * pipe_norm), 0.002), length=wall_len * 0.95), (0.0, 0.0, cable_z - 0.16))
    for sxk, syk, szk in sink_points:
        src_y = wall_pipe_y if syk >= 0.0 else -wall_pipe_y
        lat_len = abs(src_y - syk)
        if lat_len > 0.05:
            emit("lab_pipe_branch", create_pipe(radius=pipe_radius, length=lat_len), (sxk, (src_y + syk) / 2.0, pipe_z), (0.0, 0.0, 90.0))
        drop_len = max(pipe_z - szk, 0.08)
        emit("lab_pipe_drop", create_pipe(radius=max(pipe_radius * 0.85, 0.002), length=drop_len), (sxk, syk, szk + drop_len / 2.0), (0.0, -90.0, 0.0))
    dedup_service: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    for spx, spy, spz in service_pipe_points:
        key = (int(round(spx * 1000.0)), int(round(spy * 1000.0)), int(round(spz * 1000.0)))
        dedup_service[key] = (spx, spy, spz)
    for spx, spy, spz in dedup_service.values():
        src_y = wall_pipe_y if spy >= 0.0 else -wall_pipe_y
        lat_len = abs(src_y - spy)
        if lat_len > 0.05:
            emit("lab_pipe_service_branch", create_pipe(radius=max(pipe_radius * 0.92, 0.002), length=lat_len), (spx, (src_y + spy) / 2.0, pipe_z), (0.0, 0.0, 90.0))
        drop_len = max(pipe_z - spz, 0.06)
        emit("lab_pipe_service_drop", create_pipe(radius=max(pipe_radius * 0.82, 0.002), length=drop_len), (spx, spy, spz + drop_len / 2.0), (0.0, -90.0, 0.0))

    # Lighting proportional to station count and intensity.
    light_norm = _clamp((light_intensity - 0.5) / 1.5, 0.0, 1.0)
    fixture_size = _clamp(0.52 * (0.85 + 0.32 * light_intensity), 0.18, 1.45)
    station_scale = _clamp(1.2 * (0.9 + 0.25 * light_intensity), 1.0, 2.3)
    grid_density = 1.0 * (0.75 + 1.2 * light_norm)
    spacing = max(2.4 / max(grid_density, 0.1), 1.0)
    lcols = max(1, int((room_w - 2.0 * perimeter_walkway) // spacing) + 1)
    lrows = max(1, int((room_d - 2.0 * perimeter_walkway) // spacing) + 1)
    lx_span = max(inner_half_w - 0.35, 0.0)
    ly_span = max(inner_half_d - 0.35, 0.0)
    lz = room_h - 0.09
    for lx in symmetric_positions(lcols, lx_span):
        for ly in symmetric_positions(lrows, ly_span):
            emit("lab_light_fixture", create_light_fixture(size=fixture_size), (lx, ly, lz))
    for sx, sy in selected_slots:
        emit("lab_light_station", create_light_fixture(size=fixture_size * station_scale), (sx, sy, lz - 0.02))
