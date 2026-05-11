from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Mapping

from ..parametric.primitives import create_box, create_column, create_wall
from ..scene.scene_graph import Scene, SceneObject, Transform


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
    return {str(key): item for key, item in value.items()}


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


def _symmetric_positions(count: int, max_offset: float) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    if max_offset <= 0.0:
        return [0.0 for _ in range(count)]
    step = (2.0 * max_offset) / (count - 1)
    return [(-max_offset + step * index) for index in range(count)]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _hash_unit(seed: int, salt: str) -> float:
    value = (seed + 1) & 0xFFFFFFFF
    for char in salt:
        value = (value * 1664525 + ord(char) + 1013904223) & 0xFFFFFFFF
    if value == 0:
        return 0.0
    return value / 4294967295.0


def _resolve_ranged_float(
    value: object,
    *,
    default: float,
    label: str,
    seed: int,
    salt: str,
) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or not _is_number(value[0]) or not _is_number(value[1]):
            raise TypeError(f"{label} range must contain two numeric values.")
        low = float(value[0])
        high = float(value[1])
        if high < low:
            low, high = high, low
        return low + (high - low) * _hash_unit(seed, salt)
    return float(value)


def _hash_noise_2d(ix: int, iy: int, seed: int) -> float:
    value = (ix + 1) * 73856093
    value ^= (iy + 1) * 19349663
    value ^= (seed + 1) * 83492791
    value &= 0xFFFFFFFF
    value ^= value >> 13
    value = (value * 1274126177) & 0xFFFFFFFF
    normalized = value / 4294967295.0 if value else 0.0
    return normalized * 2.0 - 1.0


@dataclass(frozen=True)
class SiteDimensions:
    interior_width: float
    interior_depth: float
    exterior_width: float
    exterior_depth: float
    building_height: float
    terrain_width: float
    terrain_depth: float


@dataclass(frozen=True)
class _AccessTarget:
    side: str
    x: float
    y: float


class SiteGenerator:
    """
    Generates surrounding territory and outdoor industrial environment.

    Designed to complement ExteriorGenerator:
    - envelopes interior with site coverage
    - adds roads/fences/logistics
    - adds external industrial equipment
    """

    def __init__(self, settings: Mapping[str, object] | None = None) -> None:
        self.settings = _to_mapping(settings, "site")
        self.enabled = _to_bool(self.settings.get("enabled", True), default=True)
        self.group_id = str(self.settings.get("group_id", "site")).strip() or "site"
        self.seed = int(self.settings.get("seed", 0))

    def generate(
        self,
        scene: Scene,
        *,
        interior_width: float,
        interior_depth: float,
        exterior_width: float,
        exterior_depth: float,
        building_height: float,
    ) -> Scene:
        existing = scene.get_object(self.group_id)
        if existing is not None:
            scene.remove_object(self.group_id)
        if not self.enabled:
            return scene

        terrain_margin = _to_non_negative_float(
            self.settings.get("terrain_margin"),
            22.0,
            "site.terrain_margin",
        )
        terrain_cfg = _to_mapping(self.settings.get("terrain"), "site.terrain")
        terrain_size = _resolve_ranged_float(
            terrain_cfg.get("size"),
            default=0.0,
            label="site.terrain.size",
            seed=self.seed,
            salt="terrain.size",
        )
        if terrain_size < 0.0:
            raise ValueError("site.terrain.size must be >= 0.")

        terrain_width = exterior_width + 2.0 * terrain_margin
        terrain_depth = exterior_depth + 2.0 * terrain_margin
        if terrain_size > 0.0:
            terrain_width = max(terrain_width, terrain_size)
            terrain_depth = max(terrain_depth, terrain_size)
        dims = SiteDimensions(
            interior_width=interior_width,
            interior_depth=interior_depth,
            exterior_width=exterior_width,
            exterior_depth=exterior_depth,
            building_height=building_height,
            terrain_width=terrain_width,
            terrain_depth=terrain_depth,
        )

        group = SceneObject(id=self.group_id, type="site_group")
        self._add_terrain(group, dims)
        self._add_road_ring(group, dims)
        self._add_access_roads(scene, group, dims)
        self._add_logistics_zone(group, dims)
        self._add_outdoor_industrial(group, dims)
        self._add_parking(group, dims)
        self._add_service_zones(group, dims)
        self._add_perimeter_fence(group, dims)

        scene.add_object(group)
        return scene

    def _add_terrain(self, group: SceneObject, dims: SiteDimensions) -> None:
        terrain_cfg = _to_mapping(self.settings.get("terrain"), "site.terrain")
        thickness = _to_positive_float(
            terrain_cfg.get("thickness"),
            0.24,
            "site.terrain.thickness",
        )
        seed = self.seed
        group.add_child(
            SceneObject(
                id="site_ground",
                type="site_ground",
                transform=Transform(position=(0.0, 0.0, -thickness / 2.0)),
                mesh=create_box(width=dims.terrain_width, height=thickness, depth=dims.terrain_depth),
            )
        )

        variation_cfg = _to_mapping(terrain_cfg.get("height_noise"), "site.terrain.height_noise")
        if not _to_bool(variation_cfg.get("enabled", False), default=False):
            return

        amplitude = _to_non_negative_float(
            variation_cfg.get("amplitude"),
            0.06,
            "site.terrain.height_noise.amplitude",
        )
        if amplitude <= 0.0:
            return

        cells_x = max(1, _to_non_negative_int(variation_cfg.get("cells_x"), 10, "site.terrain.height_noise.cells_x"))
        cells_y = max(1, _to_non_negative_int(variation_cfg.get("cells_y"), 8, "site.terrain.height_noise.cells_y"))
        patch_thickness = _to_positive_float(
            variation_cfg.get("patch_thickness"),
            max(0.04, thickness * 0.45),
            "site.terrain.height_noise.patch_thickness",
        )

        tile_w = dims.terrain_width / cells_x
        tile_d = dims.terrain_depth / cells_y
        origin_x = -dims.terrain_width / 2.0 + tile_w / 2.0
        origin_y = -dims.terrain_depth / 2.0 + tile_d / 2.0

        patch_index = 1
        for ix in range(cells_x):
            for iy in range(cells_y):
                noise = _hash_noise_2d(ix, iy, seed)
                top_z = noise * amplitude
                center_z = top_z - patch_thickness / 2.0
                group.add_child(
                    SceneObject(
                        id=f"site_ground_patch_{patch_index}",
                        type="site_ground_patch",
                        transform=Transform(
                            position=(origin_x + ix * tile_w, origin_y + iy * tile_d, center_z)
                        ),
                        mesh=create_box(
                            width=tile_w * 1.02,
                            height=patch_thickness,
                            depth=tile_d * 1.02,
                        ),
                    )
                )
                patch_index += 1

    def _add_road_ring(self, group: SceneObject, dims: SiteDimensions) -> None:
        road_cfg = _to_mapping(self.settings.get("road"), "site.road")
        if not _to_bool(road_cfg.get("enabled", True), default=True):
            return

        density = self._road_density()
        width = _to_positive_float(road_cfg.get("width"), 6.0, "site.road.width")
        thickness = _to_positive_float(
            road_cfg.get("thickness"),
            0.12,
            "site.road.thickness",
        )
        offset = _to_non_negative_float(
            road_cfg.get("offset_from_building"),
            5.5,
            "site.road.offset_from_building",
        )
        z = thickness / 2.0

        half_building_w = dims.exterior_width / 2.0
        half_building_d = dims.exterior_depth / 2.0
        half_inner_w = half_building_w + offset
        half_inner_d = half_building_d + offset
        outer_w = 2.0 * (half_inner_w + width)
        outer_d = 2.0 * (half_inner_d + width)

        group.add_child(
            SceneObject(
                id="site_road_north",
                type="site_road",
                transform=Transform(position=(0.0, half_inner_d + width / 2.0, z)),
                mesh=create_box(width=outer_w, height=thickness, depth=width),
            )
        )
        group.add_child(
            SceneObject(
                id="site_road_south",
                type="site_road",
                transform=Transform(position=(0.0, -half_inner_d - width / 2.0, z)),
                mesh=create_box(width=outer_w, height=thickness, depth=width),
            )
        )
        group.add_child(
            SceneObject(
                id="site_road_west",
                type="site_road",
                transform=Transform(position=(-half_inner_w - width / 2.0, 0.0, z)),
                mesh=create_box(width=width, height=thickness, depth=2.0 * half_inner_d),
            )
        )
        group.add_child(
            SceneObject(
                id="site_road_east",
                type="site_road",
                transform=Transform(position=(half_inner_w + width / 2.0, 0.0, z)),
                mesh=create_box(width=width, height=thickness, depth=2.0 * half_inner_d),
            )
        )

        if density > 1.1:
            secondary_count = max(1, int(round(density)) - 1)
            secondary_width = max(1.8, width * 0.65)
            y_offsets = _symmetric_positions(secondary_count, max(0.0, half_inner_d * 0.6))
            x_offsets = _symmetric_positions(secondary_count, max(0.0, half_inner_w * 0.6))
            for index, y in enumerate(y_offsets, start=1):
                group.add_child(
                    SceneObject(
                        id=f"site_road_secondary_h_{index}",
                        type="site_road_secondary",
                        transform=Transform(position=(0.0, y, z)),
                        mesh=create_box(width=2.0 * half_inner_w, height=thickness, depth=secondary_width),
                    )
                )
            for index, x in enumerate(x_offsets, start=1):
                group.add_child(
                    SceneObject(
                        id=f"site_road_secondary_v_{index}",
                        type="site_road_secondary",
                        transform=Transform(position=(x, 0.0, z)),
                        mesh=create_box(width=secondary_width, height=thickness, depth=2.0 * half_inner_d),
                    )
                )

    def _add_access_roads(self, scene: Scene, group: SceneObject, dims: SiteDimensions) -> None:
        road_cfg = _to_mapping(self.settings.get("road"), "site.road")
        if not _to_bool(road_cfg.get("enabled", True), default=True):
            return
        if not _to_bool(road_cfg.get("connect_to_building", True), default=True):
            return

        density = self._road_density()
        ring_width = _to_positive_float(road_cfg.get("width"), 6.0, "site.road.width")
        thickness = _to_positive_float(road_cfg.get("thickness"), 0.12, "site.road.thickness")
        offset = _to_non_negative_float(
            road_cfg.get("offset_from_building"),
            5.5,
            "site.road.offset_from_building",
        )
        connector_width = _to_positive_float(
            road_cfg.get("connector_width"),
            max(2.2, ring_width * 0.72),
            "site.road.connector_width",
        )
        z = thickness / 2.0

        include_doors = _to_bool(road_cfg.get("connect_doors", True), default=True)
        include_gates = _to_bool(road_cfg.get("connect_gates", True), default=True)
        min_spacing = _to_positive_float(
            road_cfg.get("min_connector_spacing"),
            max(2.2, connector_width * 0.8),
            "site.road.min_connector_spacing",
        )
        min_spacing = max(0.8, min_spacing / density)

        raw_targets = self._collect_access_targets(
            scene,
            dims,
            include_doors=include_doors,
            include_gates=include_gates,
        )
        targets = self._dedupe_access_targets(raw_targets, min_spacing=min_spacing)

        if not targets:
            targets = [_AccessTarget(side="south", x=0.0, y=-dims.exterior_depth / 2.0)]

        half_w = dims.exterior_width / 2.0
        half_d = dims.exterior_depth / 2.0
        ring_north = half_d + offset + ring_width / 2.0
        ring_south = -ring_north
        ring_east = half_w + offset + ring_width / 2.0
        ring_west = -ring_east

        for index, target in enumerate(targets, start=1):
            if target.side in {"south", "north"}:
                end_y = ring_south if target.side == "south" else ring_north
                center_y = (target.y + end_y) / 2.0
                length = max(connector_width * 0.6, abs(end_y - target.y) + ring_width * 0.45)
                group.add_child(
                    SceneObject(
                        id=f"site_access_road_{target.side}_{index}",
                        type="site_access_road",
                        transform=Transform(position=(target.x, center_y, z)),
                        mesh=create_box(width=connector_width, height=thickness, depth=length),
                    )
                )
            else:
                end_x = ring_west if target.side == "west" else ring_east
                center_x = (target.x + end_x) / 2.0
                length = max(connector_width * 0.6, abs(end_x - target.x) + ring_width * 0.45)
                group.add_child(
                    SceneObject(
                        id=f"site_access_road_{target.side}_{index}",
                        type="site_access_road",
                        transform=Transform(position=(center_x, target.y, z)),
                        mesh=create_box(width=length, height=thickness, depth=connector_width),
                    )
                )

    def _collect_access_targets(
        self,
        scene: Scene,
        dims: SiteDimensions,
        *,
        include_doors: bool,
        include_gates: bool,
    ) -> list[_AccessTarget]:
        wanted_types: set[str] = set()
        if include_gates:
            wanted_types.add("exterior_gate")
        if include_doors:
            wanted_types.add("exterior_door")
        if not wanted_types:
            return []

        targets: list[_AccessTarget] = []
        half_w = dims.exterior_width / 2.0
        half_d = dims.exterior_depth / 2.0
        for obj in scene.traverse():
            if obj.type not in wanted_types:
                continue
            x, y, _ = obj.transform.position
            side = self._nearest_building_side(x, y, half_w=half_w, half_d=half_d)
            targets.append(_AccessTarget(side=side, x=x, y=y))
        return targets

    def _nearest_building_side(self, x: float, y: float, *, half_w: float, half_d: float) -> str:
        distances = {
            "south": abs(y + half_d),
            "north": abs(y - half_d),
            "west": abs(x + half_w),
            "east": abs(x - half_w),
        }
        return min(distances, key=distances.get)

    def _dedupe_access_targets(
        self,
        targets: list[_AccessTarget],
        *,
        min_spacing: float,
    ) -> list[_AccessTarget]:
        if not targets:
            return []

        grouped: dict[str, list[_AccessTarget]] = {"south": [], "north": [], "west": [], "east": []}
        for target in targets:
            grouped[target.side].append(target)

        deduped: list[_AccessTarget] = []
        for side, side_targets in grouped.items():
            if side in {"south", "north"}:
                side_targets.sort(key=lambda item: item.x)
            else:
                side_targets.sort(key=lambda item: item.y)

            for candidate in side_targets:
                keep = True
                for existing in deduped:
                    if existing.side != side:
                        continue
                    distance = abs(candidate.x - existing.x) if side in {"south", "north"} else abs(candidate.y - existing.y)
                    if distance < min_spacing:
                        keep = False
                        break
                if keep:
                    deduped.append(candidate)
        return deduped

    def _add_logistics_zone(self, group: SceneObject, dims: SiteDimensions) -> None:
        logistics_cfg = _to_mapping(self.settings.get("logistics"), "site.logistics")
        if not _to_bool(logistics_cfg.get("enabled", True), default=True):
            return

        equipment_density = self._equipment_density()
        dock_count_raw = logistics_cfg.get("dock_count", "auto")
        dock_width = _to_positive_float(
            logistics_cfg.get("dock_width"),
            4.8,
            "site.logistics.dock_width",
        )
        dock_depth = _to_positive_float(
            logistics_cfg.get("dock_depth"),
            2.4,
            "site.logistics.dock_depth",
        )
        dock_height = _to_positive_float(
            logistics_cfg.get("dock_height"),
            1.2,
            "site.logistics.dock_height",
        )
        gap = _to_non_negative_float(
            logistics_cfg.get("dock_gap"),
            1.6,
            "site.logistics.dock_gap",
        )
        edge_margin = _to_non_negative_float(
            logistics_cfg.get("edge_margin"),
            2.8,
            "site.logistics.edge_margin",
        )

        span = max(0.0, dims.exterior_width - edge_margin * 2.0)
        if span <= dock_width:
            dock_count = 1
        elif isinstance(dock_count_raw, str) and dock_count_raw.strip().lower() == "auto":
            dock_count = max(1, int(round(dims.interior_width / 30.0)))
        else:
            dock_count = max(
                1,
                _to_non_negative_int(dock_count_raw, 1, "site.logistics.dock_count"),
            )
        dock_count = max(1, int(round(dock_count * equipment_density)))

        max_count = max(1, int((span + gap) // max(dock_width + gap, 0.1)))
        dock_count = min(dock_count, max_count)
        max_offset = max(0.0, span / 2.0 - dock_width / 2.0)
        x_positions = _symmetric_positions(dock_count, max_offset)
        y = -dims.exterior_depth / 2.0 - dock_depth / 2.0 - 1.0
        z = dock_height / 2.0

        for index, x in enumerate(x_positions, start=1):
            group.add_child(
                SceneObject(
                    id=f"site_loading_dock_{index}",
                    type="site_loading_dock",
                    transform=Transform(position=(x, y, z)),
                    mesh=create_box(width=dock_width, height=dock_height, depth=dock_depth),
                )
            )

    def _add_parking(self, group: SceneObject, dims: SiteDimensions) -> None:
        parking_cfg = _to_mapping(self.settings.get("parking"), "site.parking")
        if not _to_bool(parking_cfg.get("enabled", False), default=False):
            return

        road_cfg = _to_mapping(self.settings.get("road"), "site.road")
        road_width = _to_positive_float(road_cfg.get("width"), 6.0, "site.road.width")
        road_offset = _to_non_negative_float(
            road_cfg.get("offset_from_building"),
            5.5,
            "site.road.offset_from_building",
        )

        side = str(parking_cfg.get("side", "north")).strip().lower() or "north"
        if side not in {"north", "south", "east", "west"}:
            side = "north"

        lot_width = _to_positive_float(parking_cfg.get("width"), 26.0, "site.parking.width")
        lot_depth = _to_positive_float(parking_cfg.get("depth"), 15.0, "site.parking.depth")
        thickness = _to_positive_float(parking_cfg.get("thickness"), 0.08, "site.parking.thickness")
        clearance = _to_non_negative_float(
            parking_cfg.get("offset_from_road"),
            2.4,
            "site.parking.offset_from_road",
        )

        half_w = dims.exterior_width / 2.0
        half_d = dims.exterior_depth / 2.0
        ring_offset_w = half_w + road_offset + road_width
        ring_offset_d = half_d + road_offset + road_width

        if side == "north":
            x, y = 0.0, ring_offset_d + clearance + lot_depth / 2.0
        elif side == "south":
            x, y = 0.0, -ring_offset_d - clearance - lot_depth / 2.0
        elif side == "west":
            x, y = -ring_offset_w - clearance - lot_depth / 2.0, 0.0
            lot_width, lot_depth = lot_depth, lot_width
        else:
            x, y = ring_offset_w + clearance + lot_depth / 2.0, 0.0
            lot_width, lot_depth = lot_depth, lot_width

        z = thickness / 2.0
        group.add_child(
            SceneObject(
                id="site_parking",
                type="site_parking",
                transform=Transform(position=(x, y, z)),
                mesh=create_box(width=lot_width, height=thickness, depth=lot_depth),
            )
        )

        slots_cfg = _to_mapping(parking_cfg.get("slots"), "site.parking.slots")
        slot_width = _to_positive_float(slots_cfg.get("width"), 2.6, "site.parking.slots.width")
        slot_depth = _to_positive_float(slots_cfg.get("depth"), 5.2, "site.parking.slots.depth")
        aisle = _to_positive_float(slots_cfg.get("aisle"), 6.0, "site.parking.slots.aisle")
        line_width = _to_positive_float(slots_cfg.get("line_width"), 0.08, "site.parking.slots.line_width")
        line_height = _to_positive_float(slots_cfg.get("line_height"), 0.02, "site.parking.slots.line_height")

        rows = max(1, int((lot_depth - aisle) // max(0.4, slot_depth)))
        cols = max(1, int(lot_width // max(0.4, slot_width)))
        usable_width = cols * slot_width
        usable_depth = rows * slot_depth
        offset_x = -usable_width / 2.0
        offset_y = -usable_depth / 2.0

        mark_index = 1
        mark_z = z + thickness / 2.0 + line_height / 2.0
        for row in range(rows):
            for col in range(cols + 1):
                lx = x + offset_x + col * slot_width
                ly = y + offset_y + row * slot_depth + slot_depth / 2.0
                group.add_child(
                    SceneObject(
                        id=f"site_parking_line_{mark_index}",
                        type="site_parking_line",
                        transform=Transform(position=(lx, ly, mark_z)),
                        mesh=create_box(width=line_width, height=line_height, depth=slot_depth * 0.9),
                    )
                )
                mark_index += 1

    def _add_service_zones(self, group: SceneObject, dims: SiteDimensions) -> None:
        zones_cfg = _to_mapping(self.settings.get("zones"), "site.zones")
        if not _to_bool(zones_cfg.get("enabled", True), default=True):
            return

        thickness = _to_positive_float(zones_cfg.get("thickness"), 0.07, "site.zones.thickness")
        margin = _to_non_negative_float(zones_cfg.get("margin"), 1.1, "site.zones.margin")
        min_size = _to_positive_float(zones_cfg.get("min_size"), 2.2, "site.zones.min_size")
        z = thickness / 2.0
        zone_index = 1

        target_types = {
            "site_tank",
            "site_transformer",
            "site_loading_dock",
            "site_parking",
        }
        for child in list(group.children):
            if child.type not in target_types:
                continue
            footprint = self._object_footprint(child)
            if footprint is None:
                continue
            width, depth = footprint
            zone_width = max(min_size, width + margin * 2.0)
            zone_depth = max(min_size, depth + margin * 2.0)
            cx, cy, _ = child.transform.position
            group.add_child(
                SceneObject(
                    id=f"site_zone_{zone_index}",
                    type="site_zone",
                    transform=Transform(position=(cx, cy, z)),
                    mesh=create_box(width=zone_width, height=thickness, depth=zone_depth),
                )
            )
            zone_index += 1

    def _object_footprint(self, obj: SceneObject) -> tuple[float, float] | None:
        if obj.mesh is None or not obj.mesh.vertices:
            return None
        xs = [vertex[0] for vertex in obj.mesh.vertices]
        ys = [vertex[1] for vertex in obj.mesh.vertices]
        width = max(xs) - min(xs)
        depth = max(ys) - min(ys)
        sx, sy, _ = obj.transform.scale
        return (max(0.01, width * abs(sx)), max(0.01, depth * abs(sy)))

    def _road_density(self) -> float:
        terrain_cfg = _to_mapping(self.settings.get("terrain"), "site.terrain")
        road_cfg = _to_mapping(self.settings.get("road"), "site.road")
        value = _resolve_ranged_float(
            terrain_cfg.get("road_density", road_cfg.get("density")),
            default=1.0,
            label="site.terrain.road_density",
            seed=self.seed,
            salt="terrain.road_density",
        )
        return max(0.1, value)

    def _equipment_density(self) -> float:
        value = _resolve_ranged_float(
            self.settings.get("equipment_density"),
            default=1.0,
            label="site.equipment_density",
            seed=self.seed,
            salt="site.equipment_density",
        )
        return max(0.1, value)

    def _add_perimeter_fence(self, group: SceneObject, dims: SiteDimensions) -> None:
        fence_cfg = _to_mapping(self.settings.get("fence"), "site.fence")
        if not _to_bool(fence_cfg.get("enabled", True), default=True):
            return

        inset = _to_non_negative_float(
            fence_cfg.get("inset"),
            2.0,
            "site.fence.inset",
        )
        height = _to_positive_float(fence_cfg.get("height"), 2.6, "site.fence.height")
        thickness = _to_positive_float(
            fence_cfg.get("thickness"),
            0.16,
            "site.fence.thickness",
        )
        post_spacing = _to_positive_float(
            fence_cfg.get("post_spacing"),
            6.0,
            "site.fence.post_spacing",
        )
        post_size = _to_positive_float(
            fence_cfg.get("post_size"),
            0.18,
            "site.fence.post_size",
        )

        half_w = max(0.2, dims.terrain_width / 2.0 - inset)
        half_d = max(0.2, dims.terrain_depth / 2.0 - inset)
        z = height / 2.0

        group.add_child(
            SceneObject(
                id="site_fence_north",
                type="site_fence",
                transform=Transform(position=(0.0, half_d, z)),
                mesh=create_wall(length=2.0 * half_w, height=height, thickness=thickness),
            )
        )
        group.add_child(
            SceneObject(
                id="site_fence_south",
                type="site_fence",
                transform=Transform(position=(0.0, -half_d, z)),
                mesh=create_wall(length=2.0 * half_w, height=height, thickness=thickness),
            )
        )
        group.add_child(
            SceneObject(
                id="site_fence_west",
                type="site_fence",
                transform=Transform(position=(-half_w, 0.0, z), rotation=(0.0, 0.0, 90.0)),
                mesh=create_wall(length=2.0 * half_d, height=height, thickness=thickness),
            )
        )
        group.add_child(
            SceneObject(
                id="site_fence_east",
                type="site_fence",
                transform=Transform(position=(half_w, 0.0, z), rotation=(0.0, 0.0, 90.0)),
                mesh=create_wall(length=2.0 * half_d, height=height, thickness=thickness),
            )
        )

        x_count = max(2, int(ceil((2.0 * half_w) / post_spacing)) + 1)
        y_count = max(2, int(ceil((2.0 * half_d) / post_spacing)) + 1)
        x_positions = _symmetric_positions(x_count, half_w)
        y_positions = _symmetric_positions(y_count, half_d)
        post_index = 1

        for x in x_positions:
            for y in (-half_d, half_d):
                group.add_child(
                    SceneObject(
                        id=f"site_fence_post_{post_index}",
                        type="site_fence_post",
                        transform=Transform(position=(x, y, z)),
                        mesh=create_box(width=post_size, height=height, depth=post_size),
                    )
                )
                post_index += 1
        for y in y_positions:
            for x in (-half_w, half_w):
                group.add_child(
                    SceneObject(
                        id=f"site_fence_post_{post_index}",
                        type="site_fence_post",
                        transform=Transform(position=(x, y, z)),
                        mesh=create_box(width=post_size, height=height, depth=post_size),
                    )
                )
                post_index += 1

    def _add_outdoor_industrial(self, group: SceneObject, dims: SiteDimensions) -> None:
        industrial_cfg = _to_mapping(self.settings.get("industrial"), "site.industrial")
        if not _to_bool(industrial_cfg.get("enabled", True), default=True):
            return

        equipment_density = self._equipment_density()
        # Tank yard (east side)
        tank_cfg = _to_mapping(industrial_cfg.get("tanks"), "site.industrial.tanks")
        tank_count_raw = tank_cfg.get("count", "auto")
        tank_radius = _to_positive_float(tank_cfg.get("radius"), 1.15, "site.industrial.tanks.radius")
        tank_height = _to_positive_float(tank_cfg.get("height"), 3.4, "site.industrial.tanks.height")
        tank_side_offset = _to_non_negative_float(
            tank_cfg.get("side_offset"),
            8.0,
            "site.industrial.tanks.side_offset",
        )
        tank_span_margin = _to_non_negative_float(
            tank_cfg.get("span_margin"),
            4.0,
            "site.industrial.tanks.span_margin",
        )

        if isinstance(tank_count_raw, str) and tank_count_raw.strip().lower() == "auto":
            tank_count = max(2, int(round((dims.interior_width * dims.interior_depth) / 5000.0)))
        else:
            tank_count = max(0, _to_non_negative_int(tank_count_raw, 2, "site.industrial.tanks.count"))
        tank_count = max(0, int(round(tank_count * equipment_density)))

        usable_span = max(0.0, dims.exterior_depth - tank_span_margin * 2.0)
        if usable_span >= tank_radius * 2.0 and tank_count > 0:
            y_positions = _symmetric_positions(
                tank_count,
                max(0.0, usable_span / 2.0 - tank_radius),
            )
            tank_x = dims.exterior_width / 2.0 + tank_side_offset
            for index, y in enumerate(y_positions, start=1):
                group.add_child(
                    SceneObject(
                        id=f"site_tank_{index}",
                        type="site_tank",
                        transform=Transform(position=(tank_x, y, tank_height / 2.0)),
                        mesh=create_column(radius=tank_radius, height=tank_height, segments=24),
                    )
                )

            pipe_thickness = _to_positive_float(
                tank_cfg.get("pipe_thickness"),
                0.22,
                "site.industrial.tanks.pipe_thickness",
            )
            pipe_length = max(1.0, tank_side_offset - tank_radius)
            if pipe_length > 0.5:
                for index, y in enumerate(y_positions, start=1):
                    group.add_child(
                        SceneObject(
                            id=f"site_pipe_bridge_{index}",
                            type="site_pipe_bridge",
                            transform=Transform(
                                position=(
                                    dims.exterior_width / 2.0 + pipe_length / 2.0,
                                    y,
                                    tank_height * 0.6,
                                )
                            ),
                            mesh=create_box(width=pipe_length, height=pipe_thickness, depth=pipe_thickness),
                        )
                    )

        # Transformer row (west side)
        transformer_cfg = _to_mapping(industrial_cfg.get("transformers"), "site.industrial.transformers")
        transformer_count = max(
            0,
            _to_non_negative_int(transformer_cfg.get("count"), 2, "site.industrial.transformers.count"),
        )
        transformer_count = max(0, int(round(transformer_count * equipment_density)))
        if transformer_count > 0:
            transformer_w = _to_positive_float(
                transformer_cfg.get("width"),
                2.2,
                "site.industrial.transformers.width",
            )
            transformer_d = _to_positive_float(
                transformer_cfg.get("depth"),
                1.8,
                "site.industrial.transformers.depth",
            )
            transformer_h = _to_positive_float(
                transformer_cfg.get("height"),
                2.0,
                "site.industrial.transformers.height",
            )
            transformer_side_offset = _to_non_negative_float(
                transformer_cfg.get("side_offset"),
                8.5,
                "site.industrial.transformers.side_offset",
            )
            y_positions = _symmetric_positions(
                transformer_count,
                max(0.0, dims.exterior_depth / 2.0 - transformer_d - 3.5),
            )
            x = -dims.exterior_width / 2.0 - transformer_side_offset
            for index, y in enumerate(y_positions, start=1):
                group.add_child(
                    SceneObject(
                        id=f"site_transformer_{index}",
                        type="site_transformer",
                        transform=Transform(position=(x, y, transformer_h / 2.0)),
                        mesh=create_box(
                            width=transformer_w,
                            height=transformer_h,
                            depth=transformer_d,
                        ),
                    )
                )
