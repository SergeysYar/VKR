from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, sqrt
from typing import Mapping

from ..parametric.primitives import create_beam, create_box, create_column, create_floor, create_wall
from ..procedural.perlin_noise import PerlinNoise2D
from ..scene.scene_graph import Scene, SceneObject, Transform
from .auxiliary_generator import AuxiliaryGenerator
from .biomes import (
    BIOME_REGISTRY,
    DEFAULT_BIOME_ORDER,
    DEFAULT_BIOME_PROFILES,
    BiomeRoom,
    get_biome_definition,
)
from .infrastructure_generator import InfrastructureGenerator
from .exterior_generator import ExteriorGenerator
from .room_generator import RoomGenerator, RoomParams
from .site_generator import SiteGenerator


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


def _to_mapping(value: object, label: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return {str(k): v for k, v in value.items()}


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


def _to_non_negative_int(value: object, default: int, label: str) -> int:
    if value is None:
        return default
    number = int(value)
    if number < 0:
        raise ValueError(f"{label} must be >= 0.")
    return number


def _range_max(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if "max" in value:
            return float(value["max"])
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return max(float(value[0]), float(value[1]))
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(frozen=True)
class FactoryParams:
    factory_width: float
    factory_depth: float
    number_of_rooms: int
    room_size_range: tuple[float, float]
    corridor_width: float
    room_height: float | None = None
    layout_strategy: str = "grid"
    noise: dict[str, object] = field(default_factory=dict)
    biomes: dict[str, object] = field(default_factory=dict)
    columns: dict[str, object] = field(default_factory=dict)
    beams: dict[str, object] = field(default_factory=dict)
    machinery: dict[str, object] = field(default_factory=dict)
    exterior: dict[str, object] = field(default_factory=dict)
    site: dict[str, object] = field(default_factory=dict)
    seed: int | None = 0


@dataclass(frozen=True)
class RoomLayout:
    room_id: str
    row: int
    col: int
    center_x: float
    center_y: float
    width: float
    depth: float
    biome: str = "workshop"


class FactoryGenerator:
    """
    Deterministic factory generator.

    No randomness is used, so output is fully controlled by input parameters.
    """

    def __init__(self, params: FactoryParams | Mapping[str, object]) -> None:
        self.params = self._coerce_params(params)
        self.scene = Scene()
        self.room_generator = RoomGenerator()
        self.layout: list[RoomLayout] = []

        self._grid_rows = 0
        self._grid_cols = 0
        self._room_slot_width = 0.0
        self._room_slot_depth = 0.0
        self._occupied_cells: set[tuple[int, int]] = set()

        self._column_x_positions: list[float] = []
        self._column_y_positions: list[float] = []
        self._column_height: float = 0.0

    def generate_layout(self) -> list[RoomLayout]:
        strategy = self.params.layout_strategy.strip().lower()
        if strategy in {"perlin", "noise", "minecraft"}:
            return self._generate_layout_perlin()
        return self._generate_layout_grid()

    def _generate_layout_grid(self) -> list[RoomLayout]:
        width = self.params.factory_width
        depth = self.params.factory_depth
        room_count = self.params.number_of_rooms
        min_room_size, max_room_size = self.params.room_size_range
        corridor_width = self.params.corridor_width

        rows, cols = self._compute_grid(room_count, width, depth)

        total_corridor_width = corridor_width * (cols + 1)
        total_corridor_depth = corridor_width * (rows + 1)
        usable_width = width - total_corridor_width
        usable_depth = depth - total_corridor_depth

        if usable_width <= 0.0 or usable_depth <= 0.0:
            raise ValueError(
                "Factory dimensions are too small for corridor_width and number_of_rooms."
            )

        room_slot_width = usable_width / cols
        room_slot_depth = usable_depth / rows
        min_slot_size = min(room_slot_width, room_slot_depth)
        if min_room_size > min_slot_size:
            raise ValueError(
                "room_size_range minimum is too large for computed room slots."
            )

        layout: list[RoomLayout] = []
        occupied_cells: set[tuple[int, int]] = set()

        for index in range(room_count):
            row = index // cols
            col = index % cols
            t = 0.5 if room_count == 1 else index / (room_count - 1)
            target_size = min_room_size + (max_room_size - min_room_size) * t

            room_width = min(target_size, room_slot_width)
            room_depth = min(target_size, room_slot_depth)
            if room_width <= 0.0 or room_depth <= 0.0:
                raise ValueError("Computed room size must be > 0.")

            center_x = (
                -width / 2.0
                + corridor_width
                + col * (room_slot_width + corridor_width)
                + room_slot_width / 2.0
            )
            center_y = (
                -depth / 2.0
                + corridor_width
                + row * (room_slot_depth + corridor_width)
                + room_slot_depth / 2.0
            )

            layout.append(
                RoomLayout(
                    room_id=f"factory_room_{index + 1}",
                    row=row,
                    col=col,
                    center_x=center_x,
                    center_y=center_y,
                    width=room_width,
                    depth=room_depth,
                    biome="unassigned",
                )
            )
            occupied_cells.add((row, col))

        layout = self._assign_biomes(
            layout,
            room_slot_width=room_slot_width,
            room_slot_depth=room_slot_depth,
        )
        self.layout = layout
        self._occupied_cells = occupied_cells
        self._grid_rows = rows
        self._grid_cols = cols
        self._room_slot_width = room_slot_width
        self._room_slot_depth = room_slot_depth
        return layout

    def _generate_layout_perlin(self) -> list[RoomLayout]:
        width = self.params.factory_width
        depth = self.params.factory_depth
        room_count = self.params.number_of_rooms
        min_room_size, max_room_size = self.params.room_size_range
        corridor_width = self.params.corridor_width
        noise_cfg = self.params.noise

        full_occupancy = _to_bool(noise_cfg.get("full_occupancy", True), default=True)
        slot_multiplier = _to_positive_float(
            noise_cfg.get("slot_multiplier"),
            2.0,
            "noise.slot_multiplier",
        )
        candidate_slots = (
            room_count
            if full_occupancy
            else max(room_count, int(ceil(room_count * slot_multiplier)))
        )

        rows, cols = self._compute_grid(candidate_slots, width, depth)
        room_slot_width = 0.0
        room_slot_depth = 0.0

        while True:
            total_corridor_width = corridor_width * (cols + 1)
            total_corridor_depth = corridor_width * (rows + 1)
            usable_width = width - total_corridor_width
            usable_depth = depth - total_corridor_depth

            if usable_width > 0.0 and usable_depth > 0.0:
                room_slot_width = usable_width / cols
                room_slot_depth = usable_depth / rows
                min_slot_size = min(room_slot_width, room_slot_depth)
                if min_room_size <= min_slot_size:
                    break

            if candidate_slots <= room_count:
                raise ValueError(
                    "Factory dimensions are too small for Perlin layout with current room sizes."
                )

            candidate_slots -= 1
            rows, cols = self._compute_grid(candidate_slots, width, depth)

        threshold = float(noise_cfg.get("threshold", 0.35))
        margin_ratio = float(noise_cfg.get("room_margin_ratio", 0.92))
        margin_ratio = max(0.55, min(0.98, margin_ratio))
        min_distance = _to_positive_float(
            noise_cfg.get("min_distance"),
            min(room_slot_width, room_slot_depth) * 0.75,
            "noise.min_distance",
        )
        sampling_scale = _to_positive_float(
            noise_cfg.get("sampling_scale"),
            max(0.02, 2.0 / max(width, depth)),
            "noise.sampling_scale",
        )
        aspect_strength = _to_positive_float(
            noise_cfg.get("aspect_strength"),
            0.35,
            "noise.aspect_strength",
        )
        aspect_strength = min(aspect_strength, 0.8)
        connectivity_bias = float(noise_cfg.get("connectivity_bias", 0.0))
        connectivity_bias = max(0.0, min(1.0, connectivity_bias))

        perlin = self._build_perlin_noise()

        candidates: list[tuple[int, int, float, float, float, float]] = []
        for row in range(rows):
            for col in range(cols):
                center_x = (
                    -width / 2.0
                    + corridor_width
                    + col * (room_slot_width + corridor_width)
                    + room_slot_width / 2.0
                )
                center_y = (
                    -depth / 2.0
                    + corridor_width
                    + row * (room_slot_depth + corridor_width)
                    + room_slot_depth / 2.0
                )

                nx = center_x * sampling_scale
                ny = center_y * sampling_scale
                primary = perlin.sample01(nx, ny)
                secondary = perlin.sample01(nx + 37.19, ny - 11.73)
                score = 0.78 * primary + 0.22 * secondary
                candidates.append((row, col, center_x, center_y, score, secondary))

        ranked = sorted(candidates, key=lambda item: (item[4], item[5]), reverse=True)

        selected: list[tuple[int, int, float, float, float, float]]
        selected_cells: set[tuple[int, int]]
        if full_occupancy and room_count == len(candidates):
            selected = list(candidates)
            selected_cells = {(row, col) for row, col, _, _, _, _ in selected}
        elif connectivity_bias >= 0.5:
            selected = self._select_connected_perlin_cells(
                ranked=ranked,
                room_count=room_count,
                threshold=threshold,
            )
            selected_cells = {(row, col) for row, col, _, _, _, _ in selected}
        else:
            selected = []
            selected_cells = set()
            for item in ranked:
                if len(selected) >= room_count:
                    break
                row, col, center_x, center_y, score, _ = item
                if score < threshold and len(selected) >= max(1, room_count // 2):
                    continue
                if not self._is_far_enough(center_x, center_y, selected, min_distance):
                    continue
                selected.append(item)
                selected_cells.add((row, col))

            if len(selected) < room_count:
                for item in ranked:
                    if len(selected) >= room_count:
                        break
                    row, col, _, _, _, _ = item
                    if (row, col) in selected_cells:
                        continue
                    selected.append(item)
                    selected_cells.add((row, col))

        selected.sort(key=lambda item: (item[0], item[1]))

        layout: list[RoomLayout] = []
        occupied_cells: set[tuple[int, int]] = set()
        for index, (row, col, center_x, center_y, score, shape_noise) in enumerate(
            selected,
            start=1,
        ):
            target_size = min_room_size + (max_room_size - min_room_size) * score
            aspect = 1.0 + (shape_noise * 2.0 - 1.0) * aspect_strength
            aspect = max(0.45, aspect)

            room_width = min(target_size * aspect, room_slot_width * margin_ratio)
            room_depth = min(target_size / aspect, room_slot_depth * margin_ratio)
            room_width = max(min_room_size * 0.55, room_width)
            room_depth = max(min_room_size * 0.55, room_depth)

            layout.append(
                RoomLayout(
                    room_id=f"factory_room_{index}",
                    row=row,
                    col=col,
                    center_x=center_x,
                    center_y=center_y,
                    width=room_width,
                    depth=room_depth,
                    biome="unassigned",
                )
            )
            occupied_cells.add((row, col))

        layout = self._assign_biomes(
            layout,
            room_slot_width=room_slot_width,
            room_slot_depth=room_slot_depth,
        )
        self.layout = layout
        self._occupied_cells = occupied_cells
        self._grid_rows = rows
        self._grid_cols = cols
        self._room_slot_width = room_slot_width
        self._room_slot_depth = room_slot_depth
        return layout

    def _assign_biomes(
        self,
        layout: list[RoomLayout],
        room_slot_width: float | None = None,
        room_slot_depth: float | None = None,
    ) -> list[RoomLayout]:
        if not layout:
            return layout

        settings = self.params.biomes
        if not _to_bool(settings.get("enabled", True), default=True):
            return [
                RoomLayout(
                    room_id=spec.room_id,
                    row=spec.row,
                    col=spec.col,
                    center_x=spec.center_x,
                    center_y=spec.center_y,
                    width=spec.width,
                    depth=spec.depth,
                    biome="room",
                )
                for spec in layout
            ]

        workshop_biome = str(settings.get("workshop_biome", "workshop")).strip().lower() or "workshop"
        raw_order = settings.get("cycle_order", DEFAULT_BIOME_ORDER)
        order: list[str] = []
        if isinstance(raw_order, (list, tuple)):
            for item in raw_order:
                biome = str(item).strip().lower()
                if biome:
                    order.append(biome)
        if not order:
            order = list(DEFAULT_BIOME_ORDER)

        ensure_all_types = _to_bool(settings.get("ensure_all_types", False), default=False)
        if ensure_all_types:
            for biome_name in BIOME_REGISTRY:
                if biome_name not in order:
                    order.append(biome_name)

        non_workshop = [name for name in order if name != workshop_biome]
        if not non_workshop:
            non_workshop = [name for name in DEFAULT_BIOME_ORDER if name != workshop_biome]
        if not non_workshop:
            non_workshop = ["office"]

        by_area = sorted(layout, key=lambda spec: spec.width * spec.depth, reverse=True)
        workshop_policy = str(settings.get("workshop_policy", "largest")).strip().lower()
        workshop_room_id = by_area[0].room_id
        if workshop_policy == "central_largest" and by_area:
            candidate_count = _to_non_negative_int(
                settings.get("workshop_candidate_count"),
                max(1, min(len(by_area), max(2, len(by_area) // 2))),
                "biomes.workshop_candidate_count",
            )
            candidate_count = max(1, min(candidate_count, len(by_area)))
            candidates = by_area[:candidate_count]
            workshop_room_id = min(
                candidates,
                key=lambda spec: spec.center_x * spec.center_x + spec.center_y * spec.center_y,
            ).room_id

        workshop_room_ids = {workshop_room_id}
        workshop_area_ratio_raw = settings.get("workshop_area_ratio")
        if workshop_area_ratio_raw is not None:
            workshop_area_ratio = max(0.0, min(1.0, float(workshop_area_ratio_raw)))
            if workshop_area_ratio > 0.0:
                total_area = sum(spec.width * spec.depth for spec in layout)
                target_area = total_area * workshop_area_ratio
                clustered_assignment = _to_bool(
                    settings.get("clustered_assignment", True),
                    default=True,
                )
                workshop_room_ids = self._select_workshop_room_ids(
                    layout=layout,
                    seed_room_id=workshop_room_id,
                    target_area=target_area,
                    clustered_assignment=clustered_assignment,
                )

        mandatory_biomes: list[str] = []
        if ensure_all_types:
            mandatory_cycle = [workshop_biome, *[name for name in non_workshop if name != workshop_biome]]
            mandatory_count = min(len(layout), len(mandatory_cycle))
            mandatory_biomes = mandatory_cycle[:mandatory_count]
            required_non_workshop = [name for name in mandatory_biomes if name != workshop_biome]
            max_workshop_rooms = max(1, len(layout) - len(required_non_workshop))
            workshop_room_ids = self._trim_workshop_rooms_for_biome_coverage(
                layout=layout,
                workshop_room_ids=workshop_room_ids,
                seed_room_id=workshop_room_id,
                keep_count=max_workshop_rooms,
            )

        non_workshop_specs = [spec for spec in layout if spec.room_id not in workshop_room_ids]
        required_non_workshop_queue = [name for name in mandatory_biomes if name != workshop_biome]
        planned_non_workshop: list[str] = []
        cursor = 0
        for _ in non_workshop_specs:
            if required_non_workshop_queue:
                biome = required_non_workshop_queue.pop(0)
            else:
                biome = non_workshop[cursor % len(non_workshop)]
                cursor += 1
            planned_non_workshop.append(biome)

        height_grouping_enabled = _to_bool(settings.get("height_grouping", True), default=True)
        if height_grouping_enabled and len(non_workshop_specs) > 1:
            axis_mode = str(settings.get("height_group_axis", "auto")).strip().lower()
            high_first = _to_bool(settings.get("height_group_high_first", False), default=False)
            ordered_specs = self._sorted_specs_for_height_grouping(non_workshop_specs, axis_mode)
            planned_non_workshop.sort(
                key=lambda biome_name: self._expected_biome_height(biome_name),
                reverse=high_first,
            )
        else:
            ordered_specs = non_workshop_specs

        assigned_non_workshop = {
            spec.room_id: biome_name
            for spec, biome_name in zip(ordered_specs, planned_non_workshop)
        }

        assigned: list[RoomLayout] = []
        for spec in layout:
            if spec.room_id in workshop_room_ids:
                biome = workshop_biome
            else:
                biome = assigned_non_workshop.get(spec.room_id, non_workshop[0])

            profile = self._biome_profile(biome)
            adjusted_width, adjusted_depth = self._apply_biome_size_profile(
                spec=spec,
                biome=biome,
                profile=profile,
                room_slot_width=room_slot_width,
                room_slot_depth=room_slot_depth,
            )

            assigned.append(
                RoomLayout(
                    room_id=spec.room_id,
                    row=spec.row,
                    col=spec.col,
                    center_x=spec.center_x,
                    center_y=spec.center_y,
                    width=adjusted_width,
                    depth=adjusted_depth,
                    biome=biome,
                )
            )

        return assigned

    def _trim_workshop_rooms_for_biome_coverage(
        self,
        layout: list[RoomLayout],
        workshop_room_ids: set[str],
        seed_room_id: str,
        keep_count: int,
    ) -> set[str]:
        if keep_count <= 0:
            return set()
        if len(workshop_room_ids) <= keep_count:
            return set(workshop_room_ids)

        room_by_id = {spec.room_id: spec for spec in layout}
        seed = room_by_id.get(seed_room_id)
        seed_x = seed.center_x if seed is not None else 0.0
        seed_y = seed.center_y if seed is not None else 0.0

        candidates = [room_by_id[room_id] for room_id in workshop_room_ids if room_id in room_by_id]
        ranked = sorted(
            candidates,
            key=lambda spec: (
                0 if spec.room_id == seed_room_id else 1,
                -(spec.width * spec.depth),
                (spec.center_x - seed_x) ** 2 + (spec.center_y - seed_y) ** 2,
                spec.room_id,
            ),
        )
        return {spec.room_id for spec in ranked[:keep_count]}

    def _select_workshop_room_ids(
        self,
        layout: list[RoomLayout],
        seed_room_id: str,
        target_area: float,
        clustered_assignment: bool,
    ) -> set[str]:
        if not layout:
            return set()

        room_by_id = {spec.room_id: spec for spec in layout}
        seed = room_by_id.get(seed_room_id, layout[0])

        if target_area <= seed.width * seed.depth:
            return {seed.room_id}

        if not clustered_assignment:
            selected: set[str] = set()
            accumulated = 0.0
            for spec in sorted(layout, key=lambda item: item.width * item.depth, reverse=True):
                selected.add(spec.room_id)
                accumulated += spec.width * spec.depth
                if accumulated + 1e-6 >= target_area:
                    break
            return selected

        return self._select_workshop_cluster_room_ids(layout, seed, target_area)

    def _select_workshop_cluster_room_ids(
        self,
        layout: list[RoomLayout],
        seed: RoomLayout,
        target_area: float,
    ) -> set[str]:
        by_cell = {(spec.row, spec.col): spec for spec in layout}
        selected: set[str] = {seed.room_id}
        selected_cells: set[tuple[int, int]] = {(seed.row, seed.col)}
        accumulated = seed.width * seed.depth

        while accumulated + 1e-6 < target_area and len(selected) < len(layout):
            frontier: list[RoomLayout] = []
            for row, col in selected_cells:
                for neighbor in self._adjacent_specs(row, col, by_cell):
                    if neighbor.room_id in selected:
                        continue
                    frontier.append(neighbor)

            if not frontier:
                remaining = [spec for spec in layout if spec.room_id not in selected]
                if not remaining:
                    break
                next_spec = min(
                    remaining,
                    key=lambda spec: (spec.center_x - seed.center_x) ** 2 + (spec.center_y - seed.center_y) ** 2,
                )
            else:
                unique_frontier = {spec.room_id: spec for spec in frontier}.values()
                next_spec = max(unique_frontier, key=lambda spec: spec.width * spec.depth)

            selected.add(next_spec.room_id)
            selected_cells.add((next_spec.row, next_spec.col))
            accumulated += next_spec.width * next_spec.depth

        return selected

    def _adjacent_specs(
        self,
        row: int,
        col: int,
        by_cell: Mapping[tuple[int, int], RoomLayout],
    ) -> list[RoomLayout]:
        neighbors: list[RoomLayout] = []
        for cell in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            spec = by_cell.get(cell)
            if spec is not None:
                neighbors.append(spec)
        return neighbors

    def _industrial_biomes(self) -> set[str]:
        defaults = {
            "workshop",
            "refinery",
            "boiler",
            "storage",
            "electrical",
            "maintenance",
            "laboratory",
        }
        raw = self.params.biomes.get("industrial_biomes")
        if not isinstance(raw, (list, tuple)):
            return defaults

        parsed = {
            str(item).strip().lower()
            for item in raw
            if str(item).strip()
        }
        return parsed or defaults

    def _biome_profile(self, biome: str) -> dict[str, float]:
        name = biome.strip().lower()
        profile = dict(DEFAULT_BIOME_PROFILES.get(name, {}))

        raw_profiles = self.params.biomes.get("profiles")
        if isinstance(raw_profiles, Mapping):
            custom_profile = raw_profiles.get(name)
            if isinstance(custom_profile, Mapping):
                for key, value in custom_profile.items():
                    profile[str(key)] = value

        normalized: dict[str, float] = {}
        for key, value in profile.items():
            if isinstance(value, (int, float)):
                normalized[str(key)] = float(value)
        return normalized

    def _apply_biome_size_profile(
        self,
        spec: RoomLayout,
        biome: str,
        profile: Mapping[str, float],
        room_slot_width: float | None,
        room_slot_depth: float | None,
    ) -> tuple[float, float]:
        biome_settings = self.params.biomes
        biome_name = biome.strip().lower()
        size_bias_enabled = _to_bool(
            biome_settings.get("industrial_size_bias", False),
            default=False,
        )
        industrial_biomes = self._industrial_biomes()
        is_industrial = biome_name in industrial_biomes

        size_multiplier = max(0.35, float(profile.get("size_multiplier", 1.0)))
        if size_bias_enabled:
            if is_industrial:
                size_multiplier *= _to_positive_float(
                    biome_settings.get("industrial_size_multiplier"),
                    1.35,
                    "biomes.industrial_size_multiplier",
                )
            else:
                size_multiplier *= _to_positive_float(
                    biome_settings.get("non_industrial_size_multiplier"),
                    0.55,
                    "biomes.non_industrial_size_multiplier",
                )
        width_multiplier = max(0.35, float(profile.get("width_multiplier", size_multiplier)))
        depth_multiplier = max(0.35, float(profile.get("depth_multiplier", size_multiplier)))

        max_ratio = float(profile.get("max_size_ratio", 0.98))
        max_ratio = max(0.35, min(0.99, max_ratio))
        min_ratio = float(profile.get("min_size_ratio", 0.45))
        min_ratio = max(0.2, min(max_ratio, min_ratio))

        if size_bias_enabled:
            if is_industrial:
                industrial_min_ratio = max(
                    0.45,
                    min(
                        0.99,
                        float(biome_settings.get("industrial_min_size_ratio", 0.78)),
                    ),
                )
                industrial_max_ratio = max(
                    industrial_min_ratio,
                    min(
                        0.99,
                        float(biome_settings.get("industrial_max_size_ratio", 0.99)),
                    ),
                )
                min_ratio = max(min_ratio, industrial_min_ratio)
                max_ratio = max(max_ratio, industrial_max_ratio)
            else:
                non_industrial_max_ratio = max(
                    0.25,
                    min(
                        0.95,
                        float(biome_settings.get("non_industrial_max_size_ratio", 0.58)),
                    ),
                )
                non_industrial_min_ratio = max(
                    0.2,
                    min(
                        non_industrial_max_ratio,
                        float(biome_settings.get("non_industrial_min_size_ratio", 0.28)),
                    ),
                )
                min_ratio = min(min_ratio, non_industrial_min_ratio)
                max_ratio = min(max_ratio, non_industrial_max_ratio)
                min_ratio = min(min_ratio, max_ratio)

        slot_width = room_slot_width if room_slot_width is not None and room_slot_width > 0.0 else spec.width
        slot_depth = room_slot_depth if room_slot_depth is not None and room_slot_depth > 0.0 else spec.depth

        target_width = spec.width * width_multiplier
        target_depth = spec.depth * depth_multiplier

        min_room_size, _ = self.params.room_size_range
        absolute_min = max(1.0, min_room_size * 0.45)
        min_width = max(min(slot_width * min_ratio, slot_width), absolute_min)
        min_depth = max(min(slot_depth * min_ratio, slot_depth), absolute_min)
        max_width = max(min_width, slot_width * max_ratio)
        max_depth = max(min_depth, slot_depth * max_ratio)

        adjusted_width = min(max(target_width, min_width), max_width)
        adjusted_depth = min(max(target_depth, min_depth), max_depth)
        return (adjusted_width, adjusted_depth)

    def _sorted_specs_for_height_grouping(
        self,
        specs: list[RoomLayout],
        axis_mode: str,
    ) -> list[RoomLayout]:
        if axis_mode not in {"x", "y", "auto"}:
            axis_mode = "auto"

        if axis_mode == "auto":
            axis_mode = "x" if self.params.factory_width >= self.params.factory_depth else "y"

        if axis_mode == "y":
            return sorted(
                specs,
                key=lambda spec: (
                    spec.center_y,
                    spec.center_x,
                    spec.row,
                    spec.col,
                    spec.room_id,
                ),
            )

        return sorted(
            specs,
            key=lambda spec: (
                spec.center_x,
                spec.center_y,
                spec.row,
                spec.col,
                spec.room_id,
            ),
        )

    def _expected_biome_height(self, biome: str) -> float:
        profile = self._biome_profile(biome)
        multiplier = max(0.5, float(profile.get("height_multiplier", 1.0)))
        height = self._default_room_height() * multiplier

        biome_name = biome.strip().lower()
        if biome_name == "boiler":
            boiler_settings = self.params.machinery.get("boiler")
            if isinstance(boiler_settings, Mapping):
                boiler_block = boiler_settings.get("boiler")
                boiler_height_max = None
                if isinstance(boiler_block, Mapping):
                    boiler_height_max = _range_max(boiler_block.get("height"))
                if boiler_height_max is None:
                    boiler_height_max = _range_max(boiler_settings.get("boiler_height_range"))
                if boiler_height_max is None:
                    boiler_height_max = _range_max(boiler_settings.get("boiler_height"))
                if boiler_height_max is not None and boiler_height_max > 0.0:
                    height = max(height, boiler_height_max * 1.2)

        min_height = profile.get("min_height")
        if min_height is not None:
            height = max(height, max(1.0, float(min_height)))

        max_height = profile.get("max_height")
        if max_height is not None:
            height = min(height, max(1.0, float(max_height)))

        return max(1.0, height)

    def _effective_room_height(self, spec: RoomLayout) -> float:
        profile = self._biome_profile(spec.biome)
        multiplier = float(profile.get("height_multiplier", 1.0))
        multiplier = max(0.5, multiplier)
        height = self._room_height(spec) * multiplier

        if spec.biome.strip().lower() == "boiler":
            boiler_settings = self.params.machinery.get("boiler")
            if isinstance(boiler_settings, Mapping):
                boiler_block = boiler_settings.get("boiler")
                boiler_height_max = None
                if isinstance(boiler_block, Mapping):
                    boiler_height_max = _range_max(boiler_block.get("height"))
                if boiler_height_max is None:
                    boiler_height_max = _range_max(boiler_settings.get("boiler_height_range"))
                if boiler_height_max is None:
                    boiler_height_max = _range_max(boiler_settings.get("boiler_height"))

                if boiler_height_max is not None and boiler_height_max > 0.0:
                    height = max(height, boiler_height_max * 1.2)

        min_height = profile.get("min_height")
        if min_height is not None:
            min_value = max(1.0, float(min_height))
            height = max(height, min_value)

        max_height = profile.get("max_height")
        if max_height is not None:
            max_value = max(1.0, float(max_height))
            height = min(height, max_value)

        return height

    def instantiate_rooms(self) -> Scene:
        if not self.layout:
            self.generate_layout()

        self.scene = Scene()
        self.room_generator = RoomGenerator()
        inter_room_doors_enabled = _to_bool(
            self.params.biomes.get("inter_room_doors", True),
            default=True,
        )

        for spec in self.layout:
            profile = self._biome_profile(spec.biome)
            room_height = self._effective_room_height(spec)
            wall_thickness = self._wall_thickness(spec) * float(
                profile.get("wall_thickness_multiplier", 1.0)
            )
            wall_thickness = max(0.1, wall_thickness)
            max_wall_thickness = min(spec.width, spec.depth) * 0.4
            wall_thickness = min(wall_thickness, max_wall_thickness)

            door_count = int(round(profile.get("door_count", float(self._door_count(spec)))))
            window_count = int(round(profile.get("window_count", float(self._window_count(spec)))))
            door_sides = (
                self._door_sides_for_room(spec) if inter_room_doors_enabled else tuple()
            )
            door_count = max(1, door_count, len(door_sides))
            if not door_sides:
                door_sides = ("south",)
            window_count = max(0, window_count)
            door_height = float(profile.get("door_height", self.params.biomes.get("door_height", 2.3)))
            door_height = max(1.8, min(door_height, room_height * 0.82))

            generated_parts = self.room_generator.generate_room(
                RoomParams(
                    width=spec.width,
                    depth=spec.depth,
                    height=room_height,
                    wall_thickness=wall_thickness,
                    door_count=door_count,
                    window_count=window_count,
                    door_sides=door_sides,
                    door_height=door_height,
                )
            )

            room_root = SceneObject(
                id=spec.room_id,
                type=f"room_{spec.biome}",
                transform=Transform(position=(spec.center_x, spec.center_y, 0.0)),
            )
            for part in generated_parts:
                room_root.add_child(part)

            self.scene.add_object(room_root)

        return self.scene

    def place_corridors(self) -> Scene:
        if not self.layout:
            self.generate_layout()
        if not self.scene.objects:
            self.instantiate_rooms()

        self._replace_root_group("corridors")
        group = SceneObject(id="corridors", type="corridor_group")
        biome_settings = self.params.biomes
        include_unified_shell = _to_bool(biome_settings.get("unified_shell", True), default=True)
        shell_ceiling_mode = str(biome_settings.get("shell_ceiling_mode", "stepped")).strip().lower()
        corridor_ceiling_mode = str(
            biome_settings.get("corridor_ceiling_mode", "global_min_room")
        ).strip().lower()
        shell_wall_height_mode = str(
            biome_settings.get("shell_wall_height_mode", "ceiling"),
        ).strip().lower()
        include_connector_spurs = _to_bool(biome_settings.get("connector_spurs", True), default=True)
        allow_corridorless_links = _to_bool(
            biome_settings.get("allow_corridorless_links", True),
            default=True,
        )
        corridorless_link_ratio = float(biome_settings.get("corridorless_link_ratio", 0.18))
        corridorless_link_ratio = max(0.0, min(0.9, corridorless_link_ratio))
        corridorless_height_delta = _to_positive_float(
            biome_settings.get("corridorless_height_delta"),
            1.4,
            "biomes.corridorless_height_delta",
        )
        connector_span = _to_positive_float(
            biome_settings.get("connector_span"),
            max(1.2, self.params.corridor_width * 0.75),
            "biomes.connector_span",
        )
        shell_wall_thickness = _to_positive_float(
            biome_settings.get("shell_wall_thickness"),
            max(0.16, min(0.45, self.params.corridor_width * 0.18)),
            "biomes.shell_wall_thickness",
        )
        room_height_by_cell: dict[tuple[int, int], float] = {
            (spec.row, spec.col): self._effective_room_height(spec)
            for spec in self.layout
        }
        layout_by_cell = {(spec.row, spec.col): spec for spec in self.layout}
        default_room_height = self._default_room_height()
        global_min_room_height = min(room_height_by_cell.values(), default=default_room_height)
        shell_height = max(room_height_by_cell.values(), default=default_room_height)
        corridor_reference_ceiling_height = self._resolve_corridor_ceiling_height(
            mode=corridor_ceiling_mode,
            neighbor_heights=list(room_height_by_cell.values()),
            default_height=default_room_height,
            global_min_height=global_min_room_height,
        )
        shell_wall_height = shell_height
        if shell_wall_height_mode in {"ceiling", "corridor_ceiling", "room_min"}:
            shell_wall_height = max(
                0.2,
                min(shell_height, corridor_reference_ceiling_height),
            )

        if include_unified_shell:
            group.add_child(
                SceneObject(
                    id="shell_floor",
                    type="shell_floor",
                    transform=Transform(position=(0.0, 0.0, 0.0)),
                    mesh=create_floor(
                        width=self.params.factory_width,
                        depth=self.params.factory_depth,
                    ),
                )
            )

            if shell_ceiling_mode == "flat":
                group.add_child(
                    SceneObject(
                        id="shell_ceiling",
                        type="shell_ceiling",
                        transform=Transform(position=(0.0, 0.0, shell_height)),
                        mesh=create_floor(
                            width=self.params.factory_width,
                            depth=self.params.factory_depth,
                        ),
                    )
                )
            else:
                for spec in self.layout:
                    room_ceiling_height = room_height_by_cell.get(
                        (spec.row, spec.col),
                        default_room_height,
                    )
                    group.add_child(
                        SceneObject(
                            id=f"shell_ceiling_{spec.room_id}",
                            type="shell_ceiling",
                            transform=Transform(
                                position=(spec.center_x, spec.center_y, room_ceiling_height)
                            ),
                            mesh=create_floor(
                                width=self._room_slot_width,
                                depth=self._room_slot_depth,
                            ),
                        )
                    )

            half_width = self.params.factory_width / 2.0 - shell_wall_thickness / 2.0
            half_depth = self.params.factory_depth / 2.0 - shell_wall_thickness / 2.0
            group.add_child(
                SceneObject(
                    id="shell_wall_south",
                    type="shell_wall",
                    transform=Transform(position=(0.0, -half_depth, shell_wall_height / 2.0)),
                    mesh=create_wall(
                        length=self.params.factory_width,
                        height=shell_wall_height,
                        thickness=shell_wall_thickness,
                    ),
                )
            )
            group.add_child(
                SceneObject(
                    id="shell_wall_north",
                    type="shell_wall",
                    transform=Transform(position=(0.0, half_depth, shell_wall_height / 2.0)),
                    mesh=create_wall(
                        length=self.params.factory_width,
                        height=shell_wall_height,
                        thickness=shell_wall_thickness,
                    ),
                )
            )
            group.add_child(
                SceneObject(
                    id="shell_wall_west",
                    type="shell_wall",
                    transform=Transform(
                        position=(-half_width, 0.0, shell_wall_height / 2.0),
                        rotation=(0.0, 0.0, 90.0),
                    ),
                    mesh=create_wall(
                        length=self.params.factory_depth,
                        height=shell_wall_height,
                        thickness=shell_wall_thickness,
                    ),
                )
            )
            group.add_child(
                SceneObject(
                    id="shell_wall_east",
                    type="shell_wall",
                    transform=Transform(
                        position=(half_width, 0.0, shell_wall_height / 2.0),
                        rotation=(0.0, 0.0, 90.0),
                    ),
                    mesh=create_wall(
                        length=self.params.factory_depth,
                        height=shell_wall_height,
                        thickness=shell_wall_thickness,
                    ),
                )
            )

        include_corridor_ceilings = include_unified_shell and shell_ceiling_mode != "flat"
        corridor_width = self.params.corridor_width

        for row in range(self._grid_rows + 1):
            center_y = (
                -self.params.factory_depth / 2.0
                + corridor_width / 2.0
                + row * (self._room_slot_depth + corridor_width)
            )
            for col in range(self._grid_cols):
                center_x = (
                    -self.params.factory_width / 2.0
                    + corridor_width
                    + col * (self._room_slot_width + corridor_width)
                    + self._room_slot_width / 2.0
                )
                heights: list[float] = []
                top_spec = layout_by_cell.get((row, col))
                bottom_spec = layout_by_cell.get((row - 1, col))
                if top_spec is not None:
                    heights.append(room_height_by_cell.get((row, col), default_room_height))
                if bottom_spec is not None:
                    heights.append(room_height_by_cell.get((row - 1, col), default_room_height))
                corridor_z = self._resolve_corridor_ceiling_height(
                    mode=corridor_ceiling_mode,
                    neighbor_heights=heights,
                    default_height=default_room_height,
                    global_min_height=global_min_room_height,
                )

                use_direct_link = False
                if (
                    allow_corridorless_links
                    and top_spec is not None
                    and bottom_spec is not None
                    and abs(
                        room_height_by_cell.get((row, col), default_room_height)
                        - room_height_by_cell.get((row - 1, col), default_room_height)
                    )
                    <= corridorless_height_delta
                ):
                    probability = self._deterministic_unit_interval(row=row, col=col, salt=11)
                    use_direct_link = probability < corridorless_link_ratio

                if use_direct_link and top_spec is not None and bottom_spec is not None:
                    link_width = max(
                        1.0,
                        min(
                            self._room_slot_width * 0.78,
                            min(top_spec.width, bottom_spec.width) * 0.82,
                        ),
                    )
                    group.add_child(
                        SceneObject(
                            id=f"room_link_h_{row + 1}_{col + 1}",
                            type="room_link",
                            transform=Transform(position=(center_x, center_y, 0.0)),
                            mesh=create_floor(width=link_width, depth=corridor_width),
                        )
                    )
                    if include_corridor_ceilings:
                        group.add_child(
                            SceneObject(
                                id=f"shell_ceiling_room_link_h_{row + 1}_{col + 1}",
                                type="shell_ceiling",
                                transform=Transform(position=(center_x, center_y, corridor_z)),
                                mesh=create_floor(width=link_width, depth=corridor_width),
                            )
                        )
                    continue

                group.add_child(
                    SceneObject(
                        id=f"corridor_h_{row + 1}_{col + 1}",
                        type="corridor",
                        transform=Transform(position=(center_x, center_y, 0.0)),
                        mesh=create_floor(width=self._room_slot_width, depth=corridor_width),
                    )
                )
                if include_corridor_ceilings:
                    group.add_child(
                        SceneObject(
                            id=f"shell_ceiling_corridor_h_{row + 1}_{col + 1}",
                            type="shell_ceiling",
                            transform=Transform(position=(center_x, center_y, corridor_z)),
                            mesh=create_floor(width=self._room_slot_width, depth=corridor_width),
                        )
                    )

        for col in range(self._grid_cols + 1):
            center_x = (
                -self.params.factory_width / 2.0
                + corridor_width / 2.0
                + col * (self._room_slot_width + corridor_width)
            )
            for row in range(self._grid_rows):
                center_y = (
                    -self.params.factory_depth / 2.0
                    + corridor_width
                    + row * (self._room_slot_depth + corridor_width)
                    + self._room_slot_depth / 2.0
                )
                heights = []
                right_spec = layout_by_cell.get((row, col))
                left_spec = layout_by_cell.get((row, col - 1))
                if right_spec is not None:
                    heights.append(room_height_by_cell.get((row, col), default_room_height))
                if left_spec is not None:
                    heights.append(room_height_by_cell.get((row, col - 1), default_room_height))
                corridor_z = self._resolve_corridor_ceiling_height(
                    mode=corridor_ceiling_mode,
                    neighbor_heights=heights,
                    default_height=default_room_height,
                    global_min_height=global_min_room_height,
                )

                use_direct_link = False
                if (
                    allow_corridorless_links
                    and right_spec is not None
                    and left_spec is not None
                    and abs(
                        room_height_by_cell.get((row, col), default_room_height)
                        - room_height_by_cell.get((row, col - 1), default_room_height)
                    )
                    <= corridorless_height_delta
                ):
                    probability = self._deterministic_unit_interval(row=row, col=col, salt=29)
                    use_direct_link = probability < corridorless_link_ratio

                if use_direct_link and right_spec is not None and left_spec is not None:
                    link_depth = max(
                        1.0,
                        min(
                            self._room_slot_depth * 0.78,
                            min(right_spec.depth, left_spec.depth) * 0.82,
                        ),
                    )
                    group.add_child(
                        SceneObject(
                            id=f"room_link_v_{col + 1}_{row + 1}",
                            type="room_link",
                            transform=Transform(position=(center_x, center_y, 0.0)),
                            mesh=create_floor(width=corridor_width, depth=link_depth),
                        )
                    )
                    if include_corridor_ceilings:
                        group.add_child(
                            SceneObject(
                                id=f"shell_ceiling_room_link_v_{col + 1}_{row + 1}",
                                type="shell_ceiling",
                                transform=Transform(position=(center_x, center_y, corridor_z)),
                                mesh=create_floor(width=corridor_width, depth=link_depth),
                            )
                        )
                    continue

                group.add_child(
                    SceneObject(
                        id=f"corridor_v_{col + 1}_{row + 1}",
                        type="corridor",
                        transform=Transform(position=(center_x, center_y, 0.0)),
                        mesh=create_floor(width=corridor_width, depth=self._room_slot_depth),
                    )
                )
                if include_corridor_ceilings:
                    group.add_child(
                        SceneObject(
                            id=f"shell_ceiling_corridor_v_{col + 1}_{row + 1}",
                            type="shell_ceiling",
                            transform=Transform(position=(center_x, center_y, corridor_z)),
                            mesh=create_floor(width=corridor_width, depth=self._room_slot_depth),
                        )
                    )

        for row in range(self._grid_rows + 1):
            center_y = (
                -self.params.factory_depth / 2.0
                + corridor_width / 2.0
                + row * (self._room_slot_depth + corridor_width)
            )
            for col in range(self._grid_cols + 1):
                center_x = (
                    -self.params.factory_width / 2.0
                    + corridor_width / 2.0
                    + col * (self._room_slot_width + corridor_width)
                )
                heights: list[float] = []
                for dr in (-1, 0):
                    for dc in (-1, 0):
                        value = room_height_by_cell.get((row + dr, col + dc))
                        if value is not None:
                            heights.append(value)
                corridor_z = self._resolve_corridor_ceiling_height(
                    mode=corridor_ceiling_mode,
                    neighbor_heights=heights,
                    default_height=default_room_height,
                    global_min_height=global_min_room_height,
                )
                group.add_child(
                    SceneObject(
                        id=f"corridor_junction_{row + 1}_{col + 1}",
                        type="corridor",
                        transform=Transform(position=(center_x, center_y, 0.0)),
                        mesh=create_floor(width=corridor_width, depth=corridor_width),
                    )
                )
                if include_corridor_ceilings:
                    group.add_child(
                        SceneObject(
                            id=f"shell_ceiling_corridor_junction_{row + 1}_{col + 1}",
                            type="shell_ceiling",
                            transform=Transform(position=(center_x, center_y, corridor_z)),
                            mesh=create_floor(width=corridor_width, depth=corridor_width),
                        )
                    )

        if include_connector_spurs:
            for spec in self.layout:
                connector_ceiling_z = self._resolve_corridor_ceiling_height(
                    mode=corridor_ceiling_mode,
                    neighbor_heights=[room_height_by_cell.get((spec.row, spec.col), default_room_height)],
                    default_height=default_room_height,
                    global_min_height=global_min_room_height,
                )
                for connector in self._build_room_corridor_spurs(
                    spec=spec,
                    connector_span=connector_span,
                    include_ceiling=include_corridor_ceilings,
                    ceiling_height=connector_ceiling_z,
                ):
                    group.add_child(connector)

        self.scene.add_object(group)
        return self.scene

    def _build_room_corridor_spurs(
        self,
        spec: RoomLayout,
        connector_span: float,
        include_ceiling: bool = False,
        ceiling_height: float | None = None,
    ) -> list[SceneObject]:
        slot_min_x, slot_max_x, slot_min_y, slot_max_y = self._slot_bounds(spec)
        room_min_x = spec.center_x - spec.width / 2.0
        room_max_x = spec.center_x + spec.width / 2.0
        room_min_y = spec.center_y - spec.depth / 2.0
        room_max_y = spec.center_y + spec.depth / 2.0

        span_x = max(0.9, min(connector_span, spec.width * 0.75))
        span_y = max(0.9, min(connector_span, spec.depth * 0.75))
        sides = set(self._door_sides_for_room(spec))
        connectors: list[SceneObject] = []

        def add_connector(
            connector_id: str,
            center_x: float,
            center_y: float,
            width: float,
            depth: float,
        ) -> None:
            connectors.append(
                SceneObject(
                    id=connector_id,
                    type="corridor_connector",
                    transform=Transform(position=(center_x, center_y, 0.0)),
                    mesh=create_floor(width=width, depth=depth),
                )
            )
            if include_ceiling and ceiling_height is not None:
                connectors.append(
                    SceneObject(
                        id=f"{connector_id}_ceiling",
                        type="shell_ceiling",
                        transform=Transform(position=(center_x, center_y, ceiling_height)),
                        mesh=create_floor(width=width, depth=depth),
                    )
                )

        if "east" in sides:
            gap = slot_max_x - room_max_x
            if gap > 0.05:
                add_connector(
                    connector_id=f"{spec.room_id}_connector_east",
                    center_x=room_max_x + gap / 2.0,
                    center_y=spec.center_y,
                    width=gap,
                    depth=span_y,
                )

        if "west" in sides:
            gap = room_min_x - slot_min_x
            if gap > 0.05:
                add_connector(
                    connector_id=f"{spec.room_id}_connector_west",
                    center_x=room_min_x - gap / 2.0,
                    center_y=spec.center_y,
                    width=gap,
                    depth=span_y,
                )

        if "north" in sides:
            gap = slot_max_y - room_max_y
            if gap > 0.05:
                add_connector(
                    connector_id=f"{spec.room_id}_connector_north",
                    center_x=spec.center_x,
                    center_y=room_max_y + gap / 2.0,
                    width=span_x,
                    depth=gap,
                )

        if "south" in sides:
            gap = room_min_y - slot_min_y
            if gap > 0.05:
                add_connector(
                    connector_id=f"{spec.room_id}_connector_south",
                    center_x=spec.center_x,
                    center_y=room_min_y - gap / 2.0,
                    width=span_x,
                    depth=gap,
                )

        return connectors

    def _slot_bounds(self, spec: RoomLayout) -> tuple[float, float, float, float]:
        slot_min_x = (
            -self.params.factory_width / 2.0
            + self.params.corridor_width
            + spec.col * (self._room_slot_width + self.params.corridor_width)
        )
        slot_min_y = (
            -self.params.factory_depth / 2.0
            + self.params.corridor_width
            + spec.row * (self._room_slot_depth + self.params.corridor_width)
        )
        slot_max_x = slot_min_x + self._room_slot_width
        slot_max_y = slot_min_y + self._room_slot_depth
        return (slot_min_x, slot_max_x, slot_min_y, slot_max_y)

    def _resolve_corridor_ceiling_height(
        self,
        mode: str,
        neighbor_heights: list[float],
        default_height: float,
        global_min_height: float,
    ) -> float:
        normalized_mode = mode.strip().lower()
        valid_heights = [value for value in neighbor_heights if value > 0.0]

        if normalized_mode in {"global_min", "global_min_room", "lowest", "lowest_room"}:
            return global_min_height
        if not valid_heights:
            return default_height
        if normalized_mode in {"max", "max_neighbor", "highest_neighbor"}:
            return max(valid_heights)
        if normalized_mode in {"avg", "average", "mean", "avg_neighbor"}:
            return sum(valid_heights) / len(valid_heights)
        return min(valid_heights)

    def _deterministic_unit_interval(self, row: int, col: int, salt: int = 0) -> float:
        seed = int(self.params.seed or 0) & 0xFFFFFFFF
        value = seed
        value ^= (row + 1) * 73856093
        value ^= (col + 1) * 19349663
        value ^= (salt + 1) * 83492791
        value &= 0xFFFFFFFF
        return value / 4294967295.0 if value else 0.0

    def place_factory_flow(self) -> Scene:
        if not self.layout:
            self.generate_layout()
        if not self.scene.objects:
            self.instantiate_rooms()

        self._replace_root_group("factory_flow")
        flow_cfg = _to_mapping(self.params.machinery.get("factory_flow"), "machinery.factory_flow")
        if not _to_bool(flow_cfg.get("enabled", False), default=False):
            return self.scene

        workshop_rooms = [spec for spec in self.layout if spec.biome == "workshop"]
        if not workshop_rooms:
            return self.scene
        workshop = max(workshop_rooms, key=lambda spec: spec.width * spec.depth)

        connector_width = _to_positive_float(
            flow_cfg.get("connector_width"),
            max(1.2, self.params.corridor_width * 0.7),
            "machinery.factory_flow.connector_width",
        )
        spine_width = _to_positive_float(
            flow_cfg.get("spine_width"),
            max(connector_width, self.params.corridor_width * 0.95),
            "machinery.factory_flow.spine_width",
        )
        floor_thickness = _to_positive_float(
            flow_cfg.get("floor_thickness"),
            0.04,
            "machinery.factory_flow.floor_thickness",
        )
        edge_margin = _to_positive_float(
            flow_cfg.get("edge_margin"),
            max(0.6, self.params.corridor_width * 0.35),
            "machinery.factory_flow.edge_margin",
        )
        z = floor_thickness / 2.0

        max_spine_width = max(0.6, self.params.factory_depth - edge_margin * 2.0)
        spine_width = min(spine_width, max_spine_width)

        group = SceneObject(id="factory_flow", type="factory_flow_group")
        spine_length = max(0.6, self.params.factory_width - edge_margin * 2.0)
        group.add_child(
            SceneObject(
                id="factory_flow_spine",
                type="factory_spine",
                transform=Transform(position=(0.0, workshop.center_y, z)),
                mesh=create_box(width=spine_length, height=floor_thickness, depth=spine_width),
            )
        )

        for spec in self.layout:
            if spec.room_id == workshop.room_id:
                continue

            if abs(spec.center_y - workshop.center_y) > 1e-6:
                segment_depth = abs(spec.center_y - workshop.center_y)
                center_y = (spec.center_y + workshop.center_y) / 2.0
                group.add_child(
                    SceneObject(
                        id=f"factory_flow_link_y_{spec.room_id}",
                        type="factory_link",
                        transform=Transform(position=(spec.center_x, center_y, z)),
                        mesh=create_box(
                            width=connector_width,
                            height=floor_thickness,
                            depth=max(0.4, segment_depth),
                        ),
                    )
                )

            if abs(spec.center_x - workshop.center_x) > 1e-6:
                segment_width = abs(spec.center_x - workshop.center_x)
                center_x = (spec.center_x + workshop.center_x) / 2.0
                group.add_child(
                    SceneObject(
                        id=f"factory_flow_link_x_{spec.room_id}",
                        type="factory_link",
                        transform=Transform(position=(center_x, workshop.center_y, z)),
                        mesh=create_box(
                            width=max(0.4, segment_width),
                            height=floor_thickness,
                            depth=connector_width,
                        ),
                    )
                )

        if group.children:
            self.scene.add_object(group)

        return self.scene

    def place_columns(self) -> Scene:
        if not self.layout:
            self.generate_layout()
        if not self.scene.objects:
            self.instantiate_rooms()

        settings = self.params.columns
        if not _to_bool(settings.get("enabled", True), default=True):
            self._replace_root_group("structure_columns")
            self._column_x_positions = []
            self._column_y_positions = []
            return self.scene

        spacing = _to_positive_float(settings.get("spacing"), 6.0, "columns.spacing")
        radius = _to_positive_float(settings.get("radius"), 0.3, "columns.radius")
        height = _to_positive_float(
            settings.get("height"),
            self._default_room_height(),
            "columns.height",
        )
        edge_offset = _to_positive_float(
            settings.get("edge_offset"),
            spacing / 2.0,
            "columns.edge_offset",
        )

        half_w = self.params.factory_width / 2.0
        half_d = self.params.factory_depth / 2.0
        x_positions = self._axis_positions(-half_w + edge_offset, half_w - edge_offset, spacing)
        y_positions = self._axis_positions(-half_d + edge_offset, half_d - edge_offset, spacing)

        if not x_positions or not y_positions:
            center_x = 0.0
            center_y = 0.0
            x_positions = [center_x]
            y_positions = [center_y]

        self._replace_root_group("structure_columns")
        group = SceneObject(id="structure_columns", type="column_group")
        index = 1
        for x in x_positions:
            for y in y_positions:
                group.add_child(
                    SceneObject(
                        id=f"column_{index}",
                        type="column",
                        transform=Transform(position=(x, y, height / 2.0)),
                        mesh=create_column(radius=radius, height=height),
                    )
                )
                index += 1

        self.scene.add_object(group)
        self._column_x_positions = x_positions
        self._column_y_positions = y_positions
        self._column_height = height
        return self.scene

    def place_beams(self) -> Scene:
        if not self.layout:
            self.generate_layout()
        if not self.scene.objects:
            self.instantiate_rooms()

        settings = self.params.beams
        if not _to_bool(settings.get("enabled", True), default=True):
            self._replace_root_group("structure_beams")
            return self.scene

        if len(self._column_x_positions) < 2 or len(self._column_y_positions) < 2:
            self.place_columns()

        if len(self._column_x_positions) < 2 and len(self._column_y_positions) < 2:
            self._replace_root_group("structure_beams")
            return self.scene

        elevation = _to_positive_float(
            settings.get("elevation"),
            self._column_height,
            "beams.elevation",
        )
        profile = _to_mapping(settings.get("profile"), "beams.profile")
        if not profile:
            profile = {
                "type": "i",
                "width": 0.3,
                "height": 0.45,
                "web_thickness": 0.02,
                "flange_thickness": 0.03,
            }

        self._replace_root_group("structure_beams")
        group = SceneObject(id="structure_beams", type="beam_group")

        beam_id = 1
        if len(self._column_x_positions) >= 2:
            for y in self._column_y_positions:
                for idx in range(len(self._column_x_positions) - 1):
                    x1 = self._column_x_positions[idx]
                    x2 = self._column_x_positions[idx + 1]
                    length = abs(x2 - x1)
                    center_x = (x1 + x2) / 2.0

                    group.add_child(
                        SceneObject(
                            id=f"beam_x_{beam_id}",
                            type="beam",
                            transform=Transform(position=(center_x, y, elevation)),
                            mesh=create_beam(length=length, profile_type=profile),
                        )
                    )
                    beam_id += 1

        if len(self._column_y_positions) >= 2:
            for x in self._column_x_positions:
                for idx in range(len(self._column_y_positions) - 1):
                    y1 = self._column_y_positions[idx]
                    y2 = self._column_y_positions[idx + 1]
                    length = abs(y2 - y1)
                    center_y = (y1 + y2) / 2.0

                    group.add_child(
                        SceneObject(
                            id=f"beam_y_{beam_id}",
                            type="beam",
                            transform=Transform(
                                position=(x, center_y, elevation),
                                rotation=(0.0, 0.0, 90.0),
                            ),
                            mesh=create_beam(length=length, profile_type=profile),
                        )
                    )
                    beam_id += 1

        if group.children:
            self.scene.add_object(group)
        return self.scene

    def place_machinery(self) -> Scene:
        if not self.layout:
            self.generate_layout()
        if not self.scene.objects:
            self.instantiate_rooms()

        machinery_settings = self.params.machinery
        content_enabled = _to_bool(machinery_settings.get("enabled", True), default=True)

        for spec in self.layout:
            room_root = self.scene.get_object(spec.room_id)
            if room_root is None:
                continue

            group_id = f"{spec.room_id}_biome_content"
            room_root.remove_child(group_id)
            if not content_enabled:
                continue

            content_group = SceneObject(
                id=group_id,
                type=f"{spec.biome}_content_group",
            )

            biome_name = spec.biome.strip().lower()
            biome_definition = get_biome_definition(biome_name)
            room = BiomeRoom(
                room_id=spec.room_id,
                width=spec.width,
                depth=spec.depth,
                height=self._effective_room_height(spec),
                biome=biome_name,
            )

            def add_object(
                object_type: str,
                index: int,
                mesh: object,
                position: tuple[float, float, float],
                rotation: tuple[float, float, float],
            ) -> None:
                self._append_biome_object(
                    room_id=spec.room_id,
                    group=content_group,
                    object_type=object_type,
                    index=index,
                    mesh=mesh,
                    position=position,
                    rotation=rotation,
                )

            biome_definition.populate(room, add_object, machinery_settings)

            if content_group.children:
                room_root.add_child(content_group)

        return self.scene

    def place_auxiliary(self) -> Scene:
        if not self.layout:
            self.generate_layout()
        if not self.scene.objects:
            self.instantiate_rooms()

        machinery_settings = self.params.machinery
        auxiliary_settings = _to_mapping(machinery_settings.get("auxiliary"), "machinery.auxiliary")
        auxiliary_enabled = _to_bool(auxiliary_settings.get("enabled", False), default=False)
        if not auxiliary_enabled:
            return self.scene

        populated_biomes: set[str] = set()
        for spec in self.layout:
            room_root = self.scene.get_object(spec.room_id)
            if room_root is None:
                continue
            content_group = next(
                (child for child in room_root.children if child.id == f"{spec.room_id}_biome_content"),
                None,
            )
            if content_group is not None and content_group.children:
                populated_biomes.add(spec.biome.strip().lower())

        if not populated_biomes:
            return self.scene

        auxiliary_seed = int(auxiliary_settings.get("seed", 0)) + int(self.params.seed or 0)
        auxiliary_generator = AuxiliaryGenerator(auxiliary_settings)
        for biome_name in sorted(populated_biomes):
            auxiliary_generator.generate(
                self.scene,
                biome_name,
                {
                    "seed": auxiliary_seed,
                    "random_variation": _to_bool(auxiliary_settings.get("random_variation", True), default=True),
                },
            )
        return self.scene

    def place_infrastructure(self) -> Scene:
        if not self.layout:
            self.generate_layout()
        if not self.scene.objects:
            self.instantiate_rooms()

        infrastructure_settings = _to_mapping(
            self.params.machinery.get("infrastructure"),
            "machinery.infrastructure",
        )
        if "seed" not in infrastructure_settings:
            infrastructure_settings["seed"] = int(self.params.seed or 0)
        group_id = str(infrastructure_settings.get("group_id", "global_infrastructure")).strip() or "global_infrastructure"

        if not _to_bool(infrastructure_settings.get("enabled", False), default=False):
            self._replace_root_group(group_id)
            return self.scene

        infrastructure_generator = InfrastructureGenerator(infrastructure_settings)
        self.scene = infrastructure_generator.generate(self.scene)
        return self.scene

    def place_exterior(self) -> Scene:
        if not self.layout:
            self.generate_layout()
        if not self.scene.objects:
            self.instantiate_rooms()

        exterior_settings = _to_mapping(self.params.exterior, "exterior")
        if "seed" not in exterior_settings:
            exterior_settings["seed"] = int(self.params.seed or 0)
        group_id = str(exterior_settings.get("group_id", "exterior")).strip() or "exterior"
        if not _to_bool(exterior_settings.get("enabled", True), default=True):
            self._replace_root_group(group_id)
            return self.scene

        room_heights = [self._effective_room_height(spec) for spec in self.layout]
        interior_max_height = max(room_heights, default=self._default_room_height())
        exterior_generator = ExteriorGenerator(exterior_settings)
        self.scene = exterior_generator.generate(
            self.scene,
            interior_width=self.params.factory_width,
            interior_depth=self.params.factory_depth,
            interior_max_height=interior_max_height,
        )
        return self.scene

    def place_site(self) -> Scene:
        if not self.layout:
            self.generate_layout()
        if not self.scene.objects:
            self.instantiate_rooms()

        site_settings = _to_mapping(self.params.site, "site")
        if "seed" not in site_settings:
            site_settings["seed"] = int(self.params.seed or 0)
        exterior_settings = _to_mapping(self.params.exterior, "exterior")
        if "equipment_density" in exterior_settings and "equipment_density" not in site_settings:
            site_settings["equipment_density"] = exterior_settings["equipment_density"]
        group_id = str(site_settings.get("group_id", "site")).strip() or "site"
        if not _to_bool(site_settings.get("enabled", True), default=True):
            self._replace_root_group(group_id)
            return self.scene

        room_heights = [self._effective_room_height(spec) for spec in self.layout]
        interior_max_height = max(room_heights, default=self._default_room_height())
        exterior_generator = ExteriorGenerator(exterior_settings)
        exterior_width, exterior_depth, building_height = exterior_generator.resolve_dimensions(
            self.scene,
            interior_width=self.params.factory_width,
            interior_depth=self.params.factory_depth,
            interior_max_height=interior_max_height,
        )

        site_generator = SiteGenerator(site_settings)
        self.scene = site_generator.generate(
            self.scene,
            interior_width=self.params.factory_width,
            interior_depth=self.params.factory_depth,
            exterior_width=exterior_width,
            exterior_depth=exterior_depth,
            building_height=building_height,
        )
        return self.scene

    def _append_biome_object(
        self,
        room_id: str,
        group: SceneObject,
        object_type: str,
        index: int,
        mesh: object,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
    ) -> None:
        group.add_child(
            SceneObject(
                id=f"{room_id}_{object_type}_{index}",
                type=object_type,
                transform=Transform(position=position, rotation=rotation),
                mesh=mesh,  # type: ignore[arg-type]
            )
        )

    def generate_scene(self) -> Scene:
        self.generate_layout()
        self.instantiate_rooms()
        self.place_corridors()
        self.place_exterior()
        self.place_site()
        self.place_factory_flow()
        self.place_columns()
        self.place_beams()
        self.place_machinery()
        self.place_infrastructure()
        self.place_auxiliary()
        return self.scene

    def _coerce_params(self, params: FactoryParams | Mapping[str, object]) -> FactoryParams:
        if isinstance(params, FactoryParams):
            cfg = params
        elif isinstance(params, Mapping):
            cfg = FactoryParams(
                factory_width=float(params["factory_width"]),
                factory_depth=float(params["factory_depth"]),
                number_of_rooms=int(params["number_of_rooms"]),
                room_size_range=self._coerce_room_size_range(params["room_size_range"]),
                corridor_width=float(params["corridor_width"]),
                room_height=float(params["room_height"]) if "room_height" in params and params["room_height"] is not None else None,
                layout_strategy=str(params.get("layout_strategy", "grid")),
                noise=_to_mapping(params.get("noise"), "noise"),
                biomes=_to_mapping(params.get("biomes"), "biomes"),
                columns=_to_mapping(params.get("columns"), "columns"),
                beams=_to_mapping(params.get("beams"), "beams"),
                machinery=_to_mapping(params.get("machinery"), "machinery"),
                exterior=_to_mapping(params.get("exterior"), "exterior"),
                site=_to_mapping(params.get("site"), "site"),
                seed=int(params["seed"]) if "seed" in params and params["seed"] is not None else 0,
            )
        else:
            raise TypeError("params must be FactoryParams or mapping.")

        self._validate_params(cfg)
        return cfg

    def _validate_params(self, params: FactoryParams) -> None:
        if params.factory_width <= 0.0:
            raise ValueError("factory_width must be > 0.")
        if params.factory_depth <= 0.0:
            raise ValueError("factory_depth must be > 0.")
        if params.number_of_rooms <= 0:
            raise ValueError("number_of_rooms must be > 0.")
        if params.corridor_width <= 0.0:
            raise ValueError("corridor_width must be > 0.")
        if params.room_height is not None and params.room_height <= 0.0:
            raise ValueError("room_height must be > 0 when provided.")

        min_size, max_size = params.room_size_range
        if min_size <= 0.0 or max_size <= 0.0:
            raise ValueError("room_size_range values must be > 0.")
        if min_size > max_size:
            raise ValueError("room_size_range must be (min, max).")

        strategy = params.layout_strategy.strip().lower()
        if strategy not in {"grid", "perlin", "noise", "minecraft"}:
            raise ValueError("layout_strategy must be one of: grid, perlin, noise, minecraft.")

        cycle_order = params.biomes.get("cycle_order")
        if cycle_order is not None and not isinstance(cycle_order, (list, tuple)):
            raise TypeError("biomes.cycle_order must be a list or tuple.")
        if isinstance(cycle_order, (list, tuple)):
            unknown = [
                str(name)
                for name in cycle_order
                if str(name).strip().lower() not in BIOME_REGISTRY
            ]
            if unknown:
                raise ValueError(
                    f"Unknown biome names in biomes.cycle_order: {', '.join(unknown)}."
                )

        profiles = params.biomes.get("profiles")
        if profiles is not None and not isinstance(profiles, Mapping):
            raise TypeError("biomes.profiles must be a mapping.")

        workshop_area_ratio = params.biomes.get("workshop_area_ratio")
        if workshop_area_ratio is not None:
            ratio_value = float(workshop_area_ratio)
            if ratio_value < 0.0 or ratio_value > 1.0:
                raise ValueError("biomes.workshop_area_ratio must be in range [0.0, 1.0].")

        if "industrial_size_multiplier" in params.biomes:
            _to_positive_float(
                params.biomes.get("industrial_size_multiplier"),
                1.35,
                "biomes.industrial_size_multiplier",
            )
        if "non_industrial_size_multiplier" in params.biomes:
            _to_positive_float(
                params.biomes.get("non_industrial_size_multiplier"),
                0.55,
                "biomes.non_industrial_size_multiplier",
            )

        for key in (
            "industrial_min_size_ratio",
            "industrial_max_size_ratio",
            "non_industrial_min_size_ratio",
            "non_industrial_max_size_ratio",
        ):
            if key not in params.biomes:
                continue
            ratio_value = float(params.biomes.get(key))
            if ratio_value <= 0.0 or ratio_value > 1.0:
                raise ValueError(f"biomes.{key} must be in range (0.0, 1.0].")

    def _coerce_room_size_range(self, value: object) -> tuple[float, float]:
        if isinstance(value, Mapping):
            if "min" not in value or "max" not in value:
                raise ValueError("room_size_range mapping must contain 'min' and 'max'.")
            minimum = float(value["min"])
            maximum = float(value["max"])
            return (minimum, maximum)

        if isinstance(value, (list, tuple)) and len(value) == 2:
            minimum = float(value[0])
            maximum = float(value[1])
            return (minimum, maximum)

        raise TypeError("room_size_range must be (min, max) tuple/list or mapping.")

    def _compute_grid(
        self,
        number_of_rooms: int,
        factory_width: float,
        factory_depth: float,
    ) -> tuple[int, int]:
        aspect_ratio = factory_width / factory_depth
        best_rows = 1
        best_cols = number_of_rooms
        best_error = float("inf")

        for rows in range(1, int(sqrt(number_of_rooms)) + 1):
            if number_of_rooms % rows != 0:
                continue
            cols = number_of_rooms // rows

            for candidate_rows, candidate_cols in ((rows, cols), (cols, rows)):
                candidate_ratio = candidate_cols / candidate_rows
                error = abs(candidate_ratio - aspect_ratio)
                if error < best_error:
                    best_rows = candidate_rows
                    best_cols = candidate_cols
                    best_error = error

        return best_rows, best_cols

    def _room_height(self, spec: RoomLayout) -> float:
        min_size, max_size = self.params.room_size_range
        if self.params.room_height is not None and self.params.room_height > 0.0:
            base_height = self.params.room_height
        else:
            base_height = (min_size + max_size) / 2.0
        return max(base_height, min(spec.width, spec.depth) / 3.2)

    def _default_room_height(self) -> float:
        if self.params.room_height is not None and self.params.room_height > 0.0:
            return self.params.room_height
        min_size, max_size = self.params.room_size_range
        return max((min_size + max_size) / 2.0, min_size)

    def _wall_thickness(self, spec: RoomLayout) -> float:
        by_corridor = self.params.corridor_width / 4.0
        by_room = min(spec.width, spec.depth) / 8.0
        thickness = min(by_corridor, by_room)
        if thickness <= 0.0:
            raise ValueError("Computed wall thickness must be > 0.")
        return thickness

    def _neighbor_count(self, row: int, col: int) -> int:
        neighbors = (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        )
        return sum(1 for cell in neighbors if cell in self._occupied_cells)

    def _exterior_side_count(self, row: int, col: int) -> int:
        neighbors = (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        )
        return sum(1 for cell in neighbors if cell not in self._occupied_cells)

    def _door_count(self, spec: RoomLayout) -> int:
        return max(1, self._neighbor_count(spec.row, spec.col))

    def _door_sides_for_room(self, spec: RoomLayout) -> tuple[str, ...]:
        sides: list[str] = []
        if (spec.row + 1, spec.col) in self._occupied_cells:
            sides.append("north")
        if (spec.row - 1, spec.col) in self._occupied_cells:
            sides.append("south")
        if (spec.row, spec.col + 1) in self._occupied_cells:
            sides.append("east")
        if (spec.row, spec.col - 1) in self._occupied_cells:
            sides.append("west")
        return tuple(sides)

    def _window_count(self, spec: RoomLayout) -> int:
        return self._exterior_side_count(spec.row, spec.col)

    def _axis_positions(self, start: float, end: float, spacing: float) -> list[float]:
        if spacing <= 0.0:
            raise ValueError("spacing must be > 0.")
        if end < start:
            return []

        positions: list[float] = []
        current = start
        guard = 0
        while current <= end + 1e-9:
            positions.append(current)
            current += spacing
            guard += 1
            if guard > 10000:
                break

        if not positions:
            positions.append((start + end) / 2.0)
        return positions

    def _build_perlin_noise(self) -> PerlinNoise2D:
        noise_cfg = self.params.noise
        seed = int(noise_cfg.get("seed", self.params.seed or 0))
        octaves = _to_non_negative_int(noise_cfg.get("octaves"), 4, "noise.octaves")
        octaves = max(1, octaves)
        frequency = _to_positive_float(noise_cfg.get("frequency"), 1.0, "noise.frequency")
        persistence = _to_positive_float(
            noise_cfg.get("persistence"),
            0.5,
            "noise.persistence",
        )
        lacunarity = _to_positive_float(
            noise_cfg.get("lacunarity"),
            2.0,
            "noise.lacunarity",
        )
        return PerlinNoise2D(
            seed=seed,
            octaves=octaves,
            frequency=frequency,
            persistence=persistence,
            lacunarity=lacunarity,
        )

    def _is_far_enough(
        self,
        center_x: float,
        center_y: float,
        selected: list[tuple[int, int, float, float, float, float]],
        min_distance: float,
    ) -> bool:
        min_distance_sq = min_distance * min_distance
        for _, _, x, y, _, _ in selected:
            dx = center_x - x
            dy = center_y - y
            if dx * dx + dy * dy < min_distance_sq:
                return False
        return True

    def _is_adjacent_to_selected(
        self,
        row: int,
        col: int,
        selected_cells: set[tuple[int, int]],
    ) -> bool:
        neighbors = (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        )
        return any(cell in selected_cells for cell in neighbors)

    def _select_connected_perlin_cells(
        self,
        ranked: list[tuple[int, int, float, float, float, float]],
        room_count: int,
        threshold: float,
    ) -> list[tuple[int, int, float, float, float, float]]:
        if room_count <= 0 or not ranked:
            return []

        selected: list[tuple[int, int, float, float, float, float]] = [ranked[0]]
        selected_cells: set[tuple[int, int]] = {(ranked[0][0], ranked[0][1])}
        threshold_gate = max(1, room_count // 2)

        while len(selected) < room_count and len(selected_cells) < len(ranked):
            choice: tuple[int, int, float, float, float, float] | None = None

            for item in ranked:
                row, col, _, _, score, _ = item
                if (row, col) in selected_cells:
                    continue
                if not self._is_adjacent_to_selected(row, col, selected_cells):
                    continue
                if score < threshold and len(selected) >= threshold_gate:
                    continue
                choice = item
                break

            if choice is None:
                for item in ranked:
                    row, col, _, _, _, _ = item
                    if (row, col) in selected_cells:
                        continue
                    if not self._is_adjacent_to_selected(row, col, selected_cells):
                        continue
                    choice = item
                    break

            if choice is None:
                for item in ranked:
                    row, col, _, _, _, _ = item
                    if (row, col) in selected_cells:
                        continue
                    choice = item
                    break

            if choice is None:
                break

            selected.append(choice)
            selected_cells.add((choice[0], choice[1]))

        return selected

    def _replace_root_group(self, object_id: str) -> None:
        existing = self.scene.get_object(object_id)
        if existing is not None:
            self.scene.remove_object(object_id)
