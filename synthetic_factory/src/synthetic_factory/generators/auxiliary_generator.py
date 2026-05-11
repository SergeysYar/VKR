from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Mapping

from ..geometry.mesh import Mesh
from ..parametric.primitives import create_beam, create_box, create_column
from ..scene.scene_graph import AnchorPoint, Scene, SceneObject, Transform

Vector3 = tuple[float, float, float]
Aabb = tuple[float, float, float, float, float, float]


BIOME_PRIMARY_TYPES: dict[str, set[str]] = {
    "laboratory": {"lab_bench", "lab_equipment_unit", "lab_fume_hood"},
    "control": {"console", "control_panel", "control_rack"},
    "boiler": {"boiler_unit", "tank", "service_platform"},
    "electrical": {"electrical_cabinet", "junction_box", "cooling_unit"},
    "workshop": {"machine", "conveyor", "cnc", "press", "robotic_cell"},
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


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


def _to_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Expected mapping for auxiliary settings block.")
    return {str(key): item for key, item in value.items()}


def _parse_float_range(
    value: object,
    default_min: float,
    default_max: float,
    *,
    min_limit: float,
    max_limit: float,
    label: str,
) -> tuple[float, float]:
    if value is None:
        low, high = default_min, default_max
    elif isinstance(value, Mapping):
        if "min" not in value or "max" not in value:
            raise ValueError(f"{label} mapping must have min/max.")
        low = float(value["min"])
        high = float(value["max"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        low = float(value[0])
        high = float(value[1])
    else:
        low = float(value)
        high = float(value)

    if low > high:
        raise ValueError(f"{label} min cannot be greater than max.")
    low = _clamp(low, min_limit, max_limit)
    high = _clamp(high, min_limit, max_limit)
    if low > high:
        low, high = high, low
    return (low, high)


def _parse_int_range(
    value: object,
    default_min: int,
    default_max: int,
    *,
    min_limit: int,
    max_limit: int,
    label: str,
) -> tuple[int, int]:
    if value is None:
        low, high = default_min, default_max
    elif isinstance(value, Mapping):
        if "min" not in value or "max" not in value:
            raise ValueError(f"{label} mapping must have min/max.")
        low = int(value["min"])
        high = int(value["max"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        low = int(value[0])
        high = int(value[1])
    else:
        low = int(value)
        high = int(value)

    if low > high:
        raise ValueError(f"{label} min cannot be greater than max.")
    low = max(min_limit, min(max_limit, low))
    high = max(min_limit, min(max_limit, high))
    if low > high:
        low, high = high, low
    return (low, high)


def _sample_float(value_range: tuple[float, float], rng: Random, random_variation: bool) -> float:
    low, high = value_range
    if not random_variation or abs(high - low) <= 1e-12:
        return (low + high) / 2.0
    return rng.uniform(low, high)


def _sample_int(value_range: tuple[int, int], rng: Random, random_variation: bool) -> int:
    low, high = value_range
    if not random_variation or low == high:
        return int(round((low + high) / 2.0))
    return rng.randint(low, high)


def _stable_hash(text: str) -> int:
    value = 0
    for idx, char in enumerate(text):
        value = (value * 131 + (idx + 1) * ord(char)) & 0xFFFFFFFF
    return value


def _mesh_aabb(mesh: Mesh) -> Aabb | None:
    if not mesh.vertices:
        return None
    xs = [vertex[0] for vertex in mesh.vertices]
    ys = [vertex[1] for vertex in mesh.vertices]
    zs = [vertex[2] for vertex in mesh.vertices]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def _aabb_from_mesh_and_position(mesh: Mesh, position: Vector3) -> Aabb | None:
    mesh_bounds = _mesh_aabb(mesh)
    if mesh_bounds is None:
        return None
    min_x, max_x, min_y, max_y, min_z, max_z = mesh_bounds
    px, py, pz = position
    return (
        min_x + px,
        max_x + px,
        min_y + py,
        max_y + py,
        min_z + pz,
        max_z + pz,
    )


def _intersects(a: Aabb, b: Aabb, clearance: float) -> bool:
    return (
        a[0] < b[1] - clearance
        and a[1] > b[0] + clearance
        and a[2] < b[3] - clearance
        and a[3] > b[2] + clearance
        and a[4] < b[5] - clearance
        and a[5] > b[4] + clearance
    )


def _is_pipe_type(object_type: str) -> bool:
    name = object_type.strip().lower()
    return "pipe" in name


@dataclass
class AuxiliaryContext:
    scene: Scene
    biome: str
    rng: Random
    density: float
    complexity: int
    max_children_per_object: int
    margin: float
    allow_above_ratio: float
    random_variation: bool
    boiler_bunker_probability: float
    boiler_pump_count: int
    electrical_transformer_probability: float
    laboratory_clutter_level: float
    generated_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _CandidateChild:
    child: SceneObject
    anchor_type: str | None = None
    anchor_offset: Vector3 = (0.0, 0.0, 0.0)
    allow_above: bool = False


class AuxiliaryGenerator:
    """Adds secondary detail objects attached to biome primary objects."""

    def __init__(self, settings: Mapping[str, object] | None = None) -> None:
        self.settings = {str(key): value for key, value in (settings or {}).items()}

    def generate(
        self,
        scene: Scene,
        biome: str,
        context: Mapping[str, object] | None = None,
    ) -> Scene:
        ctx = self._build_context(scene=scene, biome=biome, context=context)
        for obj in list(scene.traverse()):
            if self._is_auxiliary_type(obj.type):
                continue
            for child in self.generate_for_object(obj, ctx):
                self.attach_to_parent(child, obj)
        return scene

    def generate_for_object(self, obj: SceneObject, context: AuxiliaryContext) -> list[SceneObject]:
        if obj.mesh is None:
            return []
        if not self._matches_biome(context.biome, obj.type):
            return []
        if context.density <= 0.0:
            return []
        spawn_chance = _clamp(context.density, 0.0, 1.0)
        if context.random_variation and context.rng.random() > spawn_chance:
            return []

        parent_box = _mesh_aabb(obj.mesh)
        if parent_box is None:
            return []
        obj.ensure_anchor_points()

        existing_reserved = self._collect_existing_child_aabbs(obj)
        accepted: list[SceneObject] = []
        accepted_boxes: list[Aabb] = []
        used_anchor_keys: set[tuple[str, float, float, float]] = set()

        effective_max_children = self._effective_max_children(context)
        for candidate in self._build_candidates(obj, context):
            if len(accepted) >= effective_max_children:
                break

            child = candidate.child
            if self._has_child_with_type(obj, child.type):
                continue

            if candidate.anchor_type is not None:
                selected_anchor = self._select_anchor(
                    obj=obj,
                    anchor_type=candidate.anchor_type,
                    child=child,
                    used_anchor_keys=used_anchor_keys,
                )
                if selected_anchor is None:
                    continue
                if not self._place_child_on_anchor(
                    child=child,
                    parent_bounds=parent_box,
                    anchor=selected_anchor,
                    anchor_offset=candidate.anchor_offset,
                    context=context,
                    allow_above=candidate.allow_above,
                ):
                    continue
                used_anchor_keys.add(self._anchor_key(selected_anchor))
            else:
                if not self._fit_child_to_parent(
                    child=child,
                    parent_bounds=parent_box,
                    context=context,
                    allow_above=candidate.allow_above,
                ):
                    continue

            child_box = _aabb_from_mesh_and_position(child.mesh, child.transform.position) if child.mesh else None
            if child_box is None:
                continue

            if any(_intersects(child_box, sibling_box, context.margin * 0.5) for sibling_box in existing_reserved):
                continue
            if any(_intersects(child_box, sibling_box, context.margin * 0.5) for sibling_box in accepted_boxes):
                continue

            accepted.append(child)
            accepted_boxes.append(child_box)

        return accepted

    def attach_to_parent(self, child: SceneObject, parent: SceneObject) -> None:
        parent.add_child(child)

    def _build_context(
        self,
        scene: Scene,
        biome: str,
        context: Mapping[str, object] | None,
    ) -> AuxiliaryContext:
        merged: dict[str, object] = dict(self.settings)
        if context:
            merged.update({str(key): value for key, value in context.items()})

        normalized_biome = biome.strip().lower()
        seed = int(merged.get("seed", 0)) + _stable_hash(normalized_biome)
        random_variation = _to_bool(merged.get("random_variation", True), default=True)
        rng = Random(seed)

        density_range = _parse_float_range(
            merged.get("density"),
            0.75,
            0.75,
            min_limit=0.0,
            max_limit=3.0,
            label="auxiliary.density",
        )
        complexity_range = _parse_int_range(
            merged.get("complexity"),
            1,
            1,
            min_limit=1,
            max_limit=5,
            label="auxiliary.complexity",
        )
        density = _sample_float(density_range, rng, random_variation)
        complexity = _sample_int(complexity_range, rng, random_variation)

        max_children = max(1, int(merged.get("max_children_per_object", 2)))
        margin = max(0.0, float(merged.get("margin", 0.02)))
        allow_above_ratio = _clamp(float(merged.get("allow_above_ratio", 0.24)), 0.0, 1.0)
        boiler_cfg = _to_mapping(merged.get("boiler"))
        electrical_cfg = _to_mapping(merged.get("electrical"))
        laboratory_cfg = _to_mapping(merged.get("laboratory"))

        boiler_bunker_probability = _sample_float(
            _parse_float_range(
                boiler_cfg.get("bunker_probability"),
                0.45,
                0.45,
                min_limit=0.0,
                max_limit=1.0,
                label="auxiliary.boiler.bunker_probability",
            ),
            rng,
            random_variation,
        )
        boiler_pump_count = _sample_int(
            _parse_int_range(
                boiler_cfg.get("pump_count"),
                1,
                1,
                min_limit=1,
                max_limit=8,
                label="auxiliary.boiler.pump_count",
            ),
            rng,
            random_variation,
        )
        electrical_transformer_probability = _sample_float(
            _parse_float_range(
                electrical_cfg.get("transformer_probability"),
                0.5,
                0.5,
                min_limit=0.0,
                max_limit=1.0,
                label="auxiliary.electrical.transformer_probability",
            ),
            rng,
            random_variation,
        )
        laboratory_clutter_level = _sample_float(
            _parse_float_range(
                laboratory_cfg.get("clutter_level"),
                1.0,
                1.0,
                min_limit=0.0,
                max_limit=4.0,
                label="auxiliary.laboratory.clutter_level",
            ),
            rng,
            random_variation,
        )

        return AuxiliaryContext(
            scene=scene,
            biome=normalized_biome,
            rng=rng,
            density=density,
            complexity=complexity,
            max_children_per_object=max_children,
            margin=margin,
            allow_above_ratio=allow_above_ratio,
            random_variation=random_variation,
            boiler_bunker_probability=boiler_bunker_probability,
            boiler_pump_count=boiler_pump_count,
            electrical_transformer_probability=electrical_transformer_probability,
            laboratory_clutter_level=laboratory_clutter_level,
        )

    def _matches_biome(self, biome: str, object_type: str) -> bool:
        normalized_biome = biome.strip().lower()
        target_types = BIOME_PRIMARY_TYPES.get(normalized_biome)
        if target_types is None:
            return not self._is_auxiliary_type(object_type)
        if object_type in target_types:
            return True
        if normalized_biome == "boiler" and _is_pipe_type(object_type):
            return True
        return False

    def _is_auxiliary_type(self, object_type: str) -> bool:
        return object_type.startswith("aux_")

    def _has_child_with_type(self, obj: SceneObject, object_type: str) -> bool:
        return any(child.type == object_type for child in obj.children)

    def _collect_existing_child_aabbs(self, obj: SceneObject) -> list[Aabb]:
        boxes: list[Aabb] = []
        for child in obj.children:
            if child.mesh is None:
                continue
            child_box = _aabb_from_mesh_and_position(child.mesh, child.transform.position)
            if child_box is not None:
                boxes.append(child_box)
        return boxes

    def _effective_max_children(self, context: AuxiliaryContext) -> int:
        complexity_factor = 1.0 + max(context.complexity - 1, 0) * 0.35
        density_factor = max(context.density, 0.0)
        target = context.max_children_per_object * complexity_factor * max(density_factor, 0.35)
        return max(1, min(int(round(target)), 24))

    def _anchor_key(self, anchor: AnchorPoint) -> tuple[str, float, float, float]:
        px, py, pz = anchor.position
        return (
            anchor.name or anchor.type,
            round(px, 6),
            round(py, 6),
            round(pz, 6),
        )

    def _select_anchor(
        self,
        obj: SceneObject,
        anchor_type: str,
        child: SceneObject,
        used_anchor_keys: set[tuple[str, float, float, float]],
    ) -> AnchorPoint | None:
        candidates = obj.anchors_by_type(anchor_type)  # type: ignore[arg-type]
        if not candidates:
            return None

        available = [anchor for anchor in candidates if self._anchor_key(anchor) not in used_anchor_keys]
        if not available:
            available = list(candidates)

        cx, cy, cz = child.transform.position
        return min(
            available,
            key=lambda anchor: (
                (anchor.position[0] - cx) ** 2
                + (anchor.position[1] - cy) ** 2
                + (anchor.position[2] - cz) ** 2
            ),
        )

    def _place_child_on_anchor(
        self,
        child: SceneObject,
        parent_bounds: Aabb,
        anchor: AnchorPoint,
        anchor_offset: Vector3,
        context: AuxiliaryContext,
        allow_above: bool,
    ) -> bool:
        if child.mesh is None:
            return False
        child_bounds = _mesh_aabb(child.mesh)
        if child_bounds is None:
            return False

        pmin_x, pmax_x, pmin_y, pmax_y, pmin_z, pmax_z = parent_bounds
        cmin_x, cmax_x, cmin_y, cmax_y, cmin_z, cmax_z = child_bounds
        c_half_x = (cmax_x - cmin_x) / 2.0
        c_half_y = (cmax_y - cmin_y) / 2.0
        c_half_z = (cmax_z - cmin_z) / 2.0
        ax = anchor.position[0] + anchor_offset[0]
        ay = anchor.position[1] + anchor_offset[1]
        az = anchor.position[2] + anchor_offset[2]

        x_low = pmin_x + c_half_x + context.margin
        x_high = pmax_x - c_half_x - context.margin
        y_low = pmin_y + c_half_y + context.margin
        y_high = pmax_y - c_half_y - context.margin
        z_low = pmin_z + c_half_z + context.margin
        z_high_inside = pmax_z - c_half_z - context.margin
        z_high_above = pmax_z + max((pmax_z - pmin_z) * context.allow_above_ratio, c_half_z + context.margin)
        z_high = z_high_above if allow_above else z_high_inside

        if x_low > x_high or y_low > y_high:
            return False

        if anchor.type == "top":
            x_value = _clamp(ax, x_low, x_high)
            y_value = _clamp(ay, y_low, y_high)
            if allow_above:
                z_value = az + c_half_z + context.margin
            else:
                if z_low > z_high_inside:
                    return False
                z_value = az - c_half_z - context.margin
                z_value = _clamp(z_value, z_low, z_high_inside)
        elif anchor.type == "bottom":
            if z_low > z_high:
                return False
            x_value = _clamp(ax, x_low, x_high)
            y_value = _clamp(ay, y_low, y_high)
            z_value = az - c_half_z - context.margin
        else:  # side
            distances = [
                ("x", 1.0, abs(ax - pmax_x)),
                ("x", -1.0, abs(ax - pmin_x)),
                ("y", 1.0, abs(ay - pmax_y)),
                ("y", -1.0, abs(ay - pmin_y)),
            ]
            axis, direction, _ = min(distances, key=lambda item: item[2])
            side_sign = 1.0 if allow_above else -1.0

            if axis == "x":
                x_value = ax + direction * side_sign * (c_half_x + context.margin)
                y_value = _clamp(ay, y_low, y_high)
            else:
                x_value = _clamp(ax, x_low, x_high)
                y_value = ay + direction * side_sign * (c_half_y + context.margin)

            if z_low > z_high:
                return False
            z_value = _clamp(az, z_low, z_high)

        child.transform = Transform(
            position=(x_value, y_value, z_value),
            rotation=child.transform.rotation,
            scale=child.transform.scale,
        )
        return True

    def _next_aux_id(self, parent: SceneObject, suffix: str, context: AuxiliaryContext) -> str:
        base = f"{parent.id}_aux_{suffix}"
        index = 1
        while True:
            candidate = f"{base}_{index}"
            if candidate in context.generated_ids:
                index += 1
                continue
            if context.scene.get_object(candidate) is not None:
                index += 1
                continue
            context.generated_ids.add(candidate)
            return candidate

    def _probability_trigger(self, probability: float, context: AuxiliaryContext) -> bool:
        p = _clamp(probability, 0.0, 1.0)
        if context.random_variation:
            return context.rng.random() < p
        return p >= 0.5

    def _append_complexity_details(
        self,
        obj: SceneObject,
        context: AuxiliaryContext,
        candidates: list[_CandidateChild],
        *,
        width: float,
        depth: float,
        height: float,
        max_z: float,
    ) -> list[_CandidateChild]:
        extra_count = max(0, context.complexity - 2)
        if context.biome == "laboratory":
            extra_count = max(extra_count, int(round(max(context.laboratory_clutter_level - 1.0, 0.0))))
        extra_count = min(extra_count, 3)
        if extra_count <= 0:
            return candidates

        x_step = max(width * 0.18, 0.04)
        y_step = max(depth * 0.12, 0.03)
        detail_h = max(height * 0.07, 0.03)
        detail_w = max(min(width, depth) * 0.16, 0.04)
        detail_d = max(min(width, depth) * 0.12, 0.03)
        center = (extra_count - 1) / 2.0

        for idx in range(extra_count):
            detail = SceneObject(
                id=self._next_aux_id(obj, f"detail_{idx + 1}", context),
                type="aux_detail_module",
                transform=Transform(position=(0.0, 0.0, max_z)),
                mesh=create_box(width=detail_w, height=detail_h, depth=detail_d),
            )
            candidates.append(
                _CandidateChild(
                    detail,
                    anchor_type="top",
                    anchor_offset=((idx - center) * x_step, (1.0 if idx % 2 == 0 else -1.0) * y_step, 0.0),
                )
            )
        return candidates

    def _build_candidates(self, obj: SceneObject, context: AuxiliaryContext) -> list[_CandidateChild]:
        if obj.mesh is None:
            return []
        bounds = _mesh_aabb(obj.mesh)
        if bounds is None:
            return []
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        width = max(max_x - min_x, 0.01)
        depth = max(max_y - min_y, 0.01)
        height = max(max_z - min_z, 0.01)

        if obj.type == "lab_bench":
            tray_h = max(min(height * 0.08, 0.05), 0.01)
            tray = SceneObject(
                id=self._next_aux_id(obj, "tray", context),
                type="aux_tray",
                transform=Transform(position=(0.0, 0.0, max_z)),
                mesh=create_box(width=max(width * 0.34, 0.12), height=tray_h, depth=max(depth * 0.2, 0.08)),
            )
            rail = SceneObject(
                id=self._next_aux_id(obj, "rail", context),
                type="aux_tool_rail",
                transform=Transform(
                    position=(0.0, -depth * 0.4, max_z * 0.65),
                    rotation=(0.0, 0.0, 0.0),
                ),
                mesh=create_beam(
                    length=max(width * 0.42, 0.16),
                    profile_type={"type": "rect", "width": max(depth * 0.03, 0.01), "height": max(tray_h * 0.9, 0.01)},
                ),
            )
            candidates = [
                _CandidateChild(tray, anchor_type="top"),
                _CandidateChild(rail, anchor_type="side"),
            ]
            clutter_count = max(0, min(8, int(round(context.laboratory_clutter_level * context.complexity)) - 1))
            clutter_step = max(width * 0.14, 0.05)
            clutter_center = (clutter_count - 1) / 2.0
            for idx in range(clutter_count):
                clutter = SceneObject(
                    id=self._next_aux_id(obj, f"clutter_{idx + 1}", context),
                    type="aux_lab_clutter",
                    transform=Transform(position=(0.0, 0.0, max_z)),
                    mesh=create_box(
                        width=max(min(width, depth) * 0.1, 0.03),
                        height=max(height * 0.05, 0.02),
                        depth=max(min(width, depth) * 0.08, 0.02),
                    ),
                )
                candidates.append(
                    _CandidateChild(
                        clutter,
                        anchor_type="top",
                        anchor_offset=((idx - clutter_center) * clutter_step, (0.06 if idx % 2 == 0 else -0.06), 0.0),
                        allow_above=True,
                    )
                )
            return self._append_complexity_details(
                obj,
                context,
                candidates,
                width=width,
                depth=depth,
                height=height,
                max_z=max_z,
            )

        if obj.type == "lab_equipment_unit":
            lamp_radius = max(min(width, depth) * 0.08, 0.01)
            lamp_h = max(height * 0.12, 0.03)
            indicator = SceneObject(
                id=self._next_aux_id(obj, "indicator", context),
                type="aux_indicator",
                transform=Transform(position=(0.0, 0.0, max_z)),
                mesh=create_column(radius=lamp_radius, height=lamp_h, segments=10),
            )
            candidates = [_CandidateChild(indicator, anchor_type="top", allow_above=True)]
            return self._append_complexity_details(
                obj,
                context,
                candidates,
                width=width,
                depth=depth,
                height=height,
                max_z=max_z,
            )

        if obj.type == "lab_fume_hood":
            filter_h = max(height * 0.08, 0.03)
            filter_box = SceneObject(
                id=self._next_aux_id(obj, "filter", context),
                type="aux_filter_module",
                transform=Transform(position=(0.0, -depth * 0.18, max_z)),
                mesh=create_box(width=max(width * 0.35, 0.12), height=filter_h, depth=max(depth * 0.24, 0.08)),
            )
            candidates = [_CandidateChild(filter_box, anchor_type="top", allow_above=True)]
            return self._append_complexity_details(
                obj,
                context,
                candidates,
                width=width,
                depth=depth,
                height=height,
                max_z=max_z,
            )

        if obj.type == "console":
            keyboard_h = max(height * 0.05, 0.01)
            keyboard = SceneObject(
                id=self._next_aux_id(obj, "keyboard", context),
                type="aux_keyboard",
                transform=Transform(position=(0.0, depth * 0.12, max_z)),
                mesh=create_box(width=max(width * 0.35, 0.12), height=keyboard_h, depth=max(depth * 0.18, 0.07)),
            )
            candidates = [_CandidateChild(keyboard, anchor_type="top")]
            return self._append_complexity_details(
                obj,
                context,
                candidates,
                width=width,
                depth=depth,
                height=height,
                max_z=max_z,
            )

        if obj.type == "control_panel":
            diode_h = max(height * 0.12, 0.04)
            diode = SceneObject(
                id=self._next_aux_id(obj, "diode", context),
                type="aux_panel_diode",
                transform=Transform(position=(0.0, depth * 0.3, max_z)),
                mesh=create_column(radius=max(width * 0.03, 0.008), height=diode_h, segments=10),
            )
            candidates = [_CandidateChild(diode, anchor_type="top", allow_above=True)]
            return self._append_complexity_details(
                obj,
                context,
                candidates,
                width=width,
                depth=depth,
                height=height,
                max_z=max_z,
            )

        if obj.type == "electrical_cabinet":
            nameplate_h = max(height * 0.05, 0.03)
            nameplate = SceneObject(
                id=self._next_aux_id(obj, "nameplate", context),
                type="aux_nameplate",
                transform=Transform(position=(0.0, depth * 0.45, max_z * 0.55)),
                mesh=create_box(width=max(width * 0.42, 0.16), height=nameplate_h, depth=max(depth * 0.02, 0.01)),
            )
            gland = SceneObject(
                id=self._next_aux_id(obj, "gland", context),
                type="aux_cable_gland",
                transform=Transform(position=(0.0, 0.0, max_z)),
                mesh=create_column(radius=max(min(width, depth) * 0.06, 0.01), height=max(nameplate_h * 1.6, 0.04), segments=10),
            )
            candidates = [
                _CandidateChild(nameplate, anchor_type="side"),
                _CandidateChild(gland, anchor_type="top", allow_above=True),
            ]
            if self._probability_trigger(context.electrical_transformer_probability, context):
                transformer = SceneObject(
                    id=self._next_aux_id(obj, "transformer", context),
                    type="aux_transformer",
                    transform=Transform(position=(max_x, 0.0, min_z + height * 0.35)),
                    mesh=create_box(
                        width=max(width * 0.36, 0.2),
                        height=max(height * 0.24, 0.2),
                        depth=max(depth * 0.28, 0.14),
                    ),
                )
                candidates.append(_CandidateChild(transformer, anchor_type="side", allow_above=True))
            return self._append_complexity_details(
                obj,
                context,
                candidates,
                width=width,
                depth=depth,
                height=height,
                max_z=max_z,
            )

        if obj.type in {"boiler_unit", "tank"}:
            gauge = SceneObject(
                id=self._next_aux_id(obj, "gauge", context),
                type="aux_pressure_gauge",
                transform=Transform(position=(width * 0.25, 0.0, max_z * 0.72)),
                mesh=create_column(radius=max(min(width, depth) * 0.05, 0.02), height=max(height * 0.08, 0.05), segments=12),
            )
            candidates: list[_CandidateChild] = [
                _CandidateChild(gauge, anchor_type="side", allow_above=True),
            ]
            if obj.type == "boiler_unit":
                pump_size = max(min(width, depth) * 0.34, 0.2)
                pump_count = max(1, min(context.boiler_pump_count, 8))
                pump_radius = max(min(width, depth) * 0.18, 0.16)
                for idx in range(pump_count):
                    pump = SceneObject(
                        id=self._next_aux_id(obj, f"pump_{idx + 1}", context),
                        type="aux_pump",
                        transform=Transform(position=(0.0, 0.0, min_z)),
                        mesh=create_box(
                            width=max(pump_size, 0.2),
                            height=max(pump_size * 0.6, 0.14),
                            depth=max(pump_size * 0.45, 0.12),
                        ),
                    )
                    if pump_count == 1:
                        offset = (0.0, 0.0, 0.0)
                    else:
                        phase = idx % 4
                        if phase == 0:
                            offset = (pump_radius, 0.0, 0.0)
                        elif phase == 1:
                            offset = (-pump_radius, 0.0, 0.0)
                        elif phase == 2:
                            offset = (0.0, pump_radius, 0.0)
                        else:
                            offset = (0.0, -pump_radius, 0.0)
                    candidates.append(_CandidateChild(pump, anchor_type="bottom", anchor_offset=offset))

                if self._probability_trigger(context.boiler_bunker_probability, context):
                    bunker = SceneObject(
                        id=self._next_aux_id(obj, "bunker", context),
                        type="aux_bunker",
                        transform=Transform(position=(max_x, 0.0, min_z + height * 0.32)),
                        mesh=create_box(
                            width=max(width * 0.5, 0.3),
                            height=max(height * 0.26, 0.3),
                            depth=max(depth * 0.3, 0.2),
                        ),
                    )
                    candidates.append(_CandidateChild(bunker, anchor_type="side", allow_above=True))
            return self._append_complexity_details(
                obj,
                context,
                candidates,
                width=width,
                depth=depth,
                height=height,
                max_z=max_z,
            )

        if obj.type == "service_platform":
            guard_h = max(height * 0.2, 0.06)
            guard = SceneObject(
                id=self._next_aux_id(obj, "guardrail", context),
                type="aux_guardrail",
                transform=Transform(position=(0.0, -depth * 0.4, max_z)),
                mesh=create_box(width=max(width * 0.7, 0.2), height=guard_h, depth=max(depth * 0.04, 0.02)),
            )
            candidates = [_CandidateChild(guard, anchor_type="side", allow_above=True)]
            return self._append_complexity_details(
                obj,
                context,
                candidates,
                width=width,
                depth=depth,
                height=height,
                max_z=max_z,
            )

        if obj.type in {"machine", "conveyor", "cnc", "press", "robotic_cell"}:
            sign = SceneObject(
                id=self._next_aux_id(obj, "safety_sign", context),
                type="aux_safety_sign",
                transform=Transform(position=(0.0, depth * 0.3, max_z + max(height * 0.05, 0.03))),
                mesh=create_box(width=max(width * 0.2, 0.08), height=max(height * 0.06, 0.03), depth=max(depth * 0.02, 0.01)),
            )
            candidates = [_CandidateChild(sign, anchor_type="side", allow_above=True)]
            return self._append_complexity_details(
                obj,
                context,
                candidates,
                width=width,
                depth=depth,
                height=height,
                max_z=max_z,
            )

        if _is_pipe_type(obj.type):
            valve_h = max(min(width, depth) * 0.24, 0.08)
            valve_count = max(1, min(4, 1 + (context.complexity - 1) // 2 + (1 if context.density > 1.6 else 0)))
            spread = max(width * 0.16, 0.08)
            center = (valve_count - 1) / 2.0
            candidates: list[_CandidateChild] = []
            for idx in range(valve_count):
                valve = SceneObject(
                    id=self._next_aux_id(obj, f"valve_{idx + 1}", context),
                    type="aux_valve",
                    transform=Transform(position=(0.0, 0.0, max_z)),
                    mesh=create_column(radius=max(valve_h * 0.26, 0.02), height=valve_h, segments=12),
                )
                candidates.append(
                    _CandidateChild(
                        valve,
                        anchor_type="side",
                        allow_above=True,
                        anchor_offset=((idx - center) * spread, 0.0, 0.0),
                    )
                )
            return candidates

        return []

    def _fit_child_to_parent(
        self,
        child: SceneObject,
        parent_bounds: Aabb,
        context: AuxiliaryContext,
        allow_above: bool,
    ) -> bool:
        if child.mesh is None:
            return False
        child_bounds = _mesh_aabb(child.mesh)
        if child_bounds is None:
            return False

        pmin_x, pmax_x, pmin_y, pmax_y, pmin_z, pmax_z = parent_bounds
        cmin_x, cmax_x, cmin_y, cmax_y, cmin_z, cmax_z = child_bounds
        c_half_x = (cmax_x - cmin_x) / 2.0
        c_half_y = (cmax_y - cmin_y) / 2.0
        c_half_z = (cmax_z - cmin_z) / 2.0

        x_low = pmin_x + c_half_x + context.margin
        x_high = pmax_x - c_half_x - context.margin
        y_low = pmin_y + c_half_y + context.margin
        y_high = pmax_y - c_half_y - context.margin

        if x_low > x_high or y_low > y_high:
            return False

        if allow_above:
            z_low = pmin_z + c_half_z + context.margin
            z_high = pmax_z + max((pmax_z - pmin_z) * context.allow_above_ratio, c_half_z + context.margin)
        else:
            z_low = pmin_z + c_half_z + context.margin
            z_high = pmax_z - c_half_z - context.margin

        if z_low > z_high:
            z_value = (pmin_z + pmax_z) / 2.0
        else:
            z_value = _clamp(child.transform.position[2], z_low, z_high)

        x_value = _clamp(child.transform.position[0], x_low, x_high)
        y_value = _clamp(child.transform.position[1], y_low, y_high)
        child.transform = Transform(
            position=(x_value, y_value, z_value),
            rotation=child.transform.rotation,
            scale=child.transform.scale,
        )
        return True
