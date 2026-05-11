from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import ceil, cos, radians, sin, sqrt
from typing import Iterable, Mapping

from ..geometry.mesh import Mesh
from ..parametric.primitives import create_box, create_column, create_wall
from ..scene.scene_graph import Scene, SceneObject, Transform

Matrix4 = list[list[float]]

_STRUCTURAL_TYPES = {
    "floor",
    "ceiling",
    "wall",
    "corridor_floor",
    "corridor_ceiling",
    "corridor_wall",
    "shell_wall",
    "shell_ceiling",
    "room_link",
    "room_connector",
    "factory_spine",
    "factory_link",
}


def _to_mapping(value: object, label: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return {str(key): item for key, item in value.items()}


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


def _to_string_set(value: object) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        result: set[str] = set()
        for item in value:
            text = str(item).strip().lower()
            if text:
                result.add(text)
        return result
    return set()


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
    rx = radians(x)
    ry = radians(y)
    rz = radians(z)

    cx = cos(rx)
    sx = sin(rx)
    cy = cos(ry)
    sy = sin(ry)
    cz = cos(rz)
    sz = sin(rz)

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
    return _mat_mul(
        _translation_matrix(tx, ty, tz),
        _mat_mul(_rotation_matrix_euler_deg(rx, ry, rz), _scale_matrix(sx, sy, sz)),
    )


def _transform_point(matrix: Matrix4, point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    tx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3]
    ty = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3]
    tz = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3]
    tw = matrix[3][0] * x + matrix[3][1] * y + matrix[3][2] * z + matrix[3][3]
    if abs(tw) > 1e-9:
        return (tx / tw, ty / tw, tz / tw)
    return (tx, ty, tz)


@dataclass(frozen=True)
class _Bounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) / 2.0

    @property
    def center_y(self) -> float:
        return (self.min_y + self.max_y) / 2.0

    @property
    def width(self) -> float:
        return max(0.01, self.max_x - self.min_x)

    @property
    def depth(self) -> float:
        return max(0.01, self.max_y - self.min_y)

    @property
    def height(self) -> float:
        return max(0.01, self.max_z - self.min_z)


@dataclass(frozen=True)
class _ExteriorContext:
    center_x: float
    center_y: float
    base_z: float
    interior_width: float
    interior_depth: float
    interior_height: float
    exterior_width: float
    exterior_depth: float
    building_height: float
    wall_thickness: float
    wall_offset: float
    room_volumes: tuple["_RoomVolume", ...]


@dataclass(frozen=True)
class _RoomVolume:
    room_id: str
    biome: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float


@dataclass(frozen=True)
class _FacadeOpening:
    side: str
    offset: float
    width: float
    height: float
    kind: str


@dataclass(frozen=True)
class _FacadeSection:
    side: str
    start: float
    end: float
    biome: str
    kind: str


@dataclass(frozen=True)
class ShellContainmentRule:
    margin: float


@dataclass(frozen=True)
class NoCollisionRule:
    padding: float
    correction_step: float


@dataclass(frozen=True)
class AccessRule:
    min_entrances: int
    clearance: float


@dataclass(frozen=True)
class PipeContinuityRule:
    tolerance: float
    link_thickness: float


@dataclass(frozen=True)
class GroundContactRule:
    tolerance: float
    supported_types: frozenset[str]


class ExteriorGenerator:
    """
    Post-process exterior generation over existing interior scene.

    Public API requested by pipeline/task:
    - generate_building_shell(scene)
    - generate_roof(scene)
    - generate_facade(scene)
    - generate_external_equipment(scene)
    - generate_terrain(scene)
    """

    def __init__(self, settings: Mapping[str, object] | None = None) -> None:
        self.settings = _to_mapping(settings, "exterior")
        self.enabled = _to_bool(self.settings.get("enabled", True), default=True)
        self.group_id = str(self.settings.get("group_id", "exterior")).strip() or "exterior"
        self.seed = int(self.settings.get("seed", 0))
        self._context: _ExteriorContext | None = None
        self._openings_by_side: dict[str, list[_FacadeOpening]] = {
            "south": [],
            "north": [],
            "west": [],
            "east": [],
        }
        self._facade_sections_by_side: dict[str, list[_FacadeSection]] = {
            "south": [],
            "north": [],
            "west": [],
            "east": [],
        }

    def resolve_dimensions(
        self,
        scene: Scene,
        *,
        interior_width: float | None = None,
        interior_depth: float | None = None,
        interior_max_height: float | None = None,
    ) -> tuple[float, float, float]:
        """
        Resolve exterior footprint and height using the same logic as generation.

        Returns:
            (exterior_width, exterior_depth, building_height)
        """
        context = self._resolve_context(
            scene,
            interior_width=interior_width,
            interior_depth=interior_depth,
            interior_max_height=interior_max_height,
        )
        return (context.exterior_width, context.exterior_depth, context.building_height)

    def generate(
        self,
        scene: Scene,
        *,
        interior_width: float | None = None,
        interior_depth: float | None = None,
        interior_max_height: float | None = None,
    ) -> Scene:
        existing = scene.get_object(self.group_id)
        if existing is not None:
            scene.remove_object(self.group_id)

        if not self.enabled:
            return scene

        self._context = self._resolve_context(
            scene,
            interior_width=interior_width,
            interior_depth=interior_depth,
            interior_max_height=interior_max_height,
        )
        self._openings_by_side = {
            "south": [],
            "north": [],
            "west": [],
            "east": [],
        }
        self._facade_sections_by_side = {
            "south": [],
            "north": [],
            "west": [],
            "east": [],
        }
        scene.add_object(SceneObject(id=self.group_id, type="exterior_group"))
        self._generate_pass(scene)
        if self._apply_exterior_rules(scene, allow_rebuild=True):
            group = self._ensure_group(scene)
            group.children = []
            self._openings_by_side = {
                "south": [],
                "north": [],
                "west": [],
                "east": [],
            }
            self._facade_sections_by_side = {
                "south": [],
                "north": [],
                "west": [],
                "east": [],
            }
            self._generate_pass(scene)
            self._apply_exterior_rules(scene, allow_rebuild=False)
        return scene

    def _generate_pass(self, scene: Scene) -> None:
        self.generate_building_shell(scene)
        self.generate_roof(scene)
        self.generate_facade(scene)
        self.generate_external_equipment(scene)
        self.generate_terrain(scene)

    def generate_building_shell(self, scene: Scene) -> list[SceneObject]:
        group = self._ensure_group(scene)
        ctx = self._ensure_context(scene)

        foundation_thickness = _to_positive_float(
            self.settings.get("foundation_thickness"),
            0.45,
            "exterior.foundation_thickness",
        )
        foundation_border = _to_non_negative_float(
            self.settings.get("foundation_border"),
            1.0,
            "exterior.foundation_border",
        )
        opening_clearance = _to_non_negative_float(
            self.settings.get("opening_clearance"),
            0.18,
            "exterior.opening_clearance",
        )
        openings_by_side = self._collect_shell_openings(scene, ctx, opening_clearance=opening_clearance)
        self._openings_by_side = openings_by_side
        self._facade_sections_by_side = self._build_facade_sections(
            ctx=ctx,
            openings_by_side=openings_by_side,
        )

        wall_z = ctx.base_z + ctx.building_height / 2.0
        wall_x = ctx.exterior_width / 2.0 - ctx.wall_thickness / 2.0
        wall_y = ctx.exterior_depth / 2.0 - ctx.wall_thickness / 2.0

        generated: list[SceneObject] = []
        generated.append(
            SceneObject(
                id="exterior_foundation",
                type="exterior_foundation",
                transform=Transform(position=(ctx.center_x, ctx.center_y, ctx.base_z - foundation_thickness / 2.0)),
                mesh=create_box(
                    width=ctx.exterior_width + foundation_border * 2.0,
                    height=foundation_thickness,
                    depth=ctx.exterior_depth + foundation_border * 2.0,
                ),
            )
        )
        for side in ("south", "north", "west", "east"):
            if side in {"south", "north"}:
                position = (
                    ctx.center_x,
                    ctx.center_y - wall_y if side == "south" else ctx.center_y + wall_y,
                    wall_z,
                )
                rotation = (0.0, 0.0, 0.0)
            else:
                position = (
                    ctx.center_x - wall_x if side == "west" else ctx.center_x + wall_x,
                    ctx.center_y,
                    wall_z,
                )
                rotation = (0.0, 0.0, 90.0)

            generated.append(
                SceneObject(
                    id=f"exterior_wall_{side}",
                    type="exterior_wall",
                    transform=Transform(position=position, rotation=rotation),
                    mesh=self._build_side_wall_mesh(
                        side=side,
                        ctx=ctx,
                        openings=openings_by_side.get(side, []),
                    ),
                )
            )

        opening_index = 1
        for side in ("south", "north", "west", "east"):
            for opening in openings_by_side.get(side, []):
                if side == "south":
                    marker_position = (
                        ctx.center_x + opening.offset,
                        ctx.center_y - wall_y,
                        ctx.base_z + opening.height / 2.0,
                    )
                    marker_rotation = (0.0, 0.0, 0.0)
                elif side == "north":
                    marker_position = (
                        ctx.center_x + opening.offset,
                        ctx.center_y + wall_y,
                        ctx.base_z + opening.height / 2.0,
                    )
                    marker_rotation = (0.0, 0.0, 0.0)
                elif side == "west":
                    marker_position = (
                        ctx.center_x - wall_x,
                        ctx.center_y + opening.offset,
                        ctx.base_z + opening.height / 2.0,
                    )
                    marker_rotation = (0.0, 0.0, 90.0)
                else:
                    marker_position = (
                        ctx.center_x + wall_x,
                        ctx.center_y + opening.offset,
                        ctx.base_z + opening.height / 2.0,
                    )
                    marker_rotation = (0.0, 0.0, 90.0)

                generated.append(
                    SceneObject(
                        id=f"exterior_opening_{opening.kind}_{side}_{opening_index}",
                        type="exterior_opening",
                        transform=Transform(position=marker_position, rotation=marker_rotation),
                        mesh=None,
                    )
                )
                opening_index += 1

        for obj in generated:
            self._upsert_child(group, obj)
        return generated

    def generate_roof(self, scene: Scene) -> list[SceneObject]:
        group = self._ensure_group(scene)
        ctx = self._ensure_context(scene)

        roof_type, roof_cfg = self._roof_type_config()
        roof_thickness = _to_positive_float(
            self.settings.get("roof_thickness"),
            0.3,
            "exterior.roof_thickness",
        )
        roof_overhang = _to_non_negative_float(
            self.settings.get("roof_overhang"),
            0.55,
            "exterior.roof_overhang",
        )
        roof_lift = float(self.settings.get("roof_lift", 0.0))

        roof_width = ctx.exterior_width + roof_overhang * 2.0
        roof_depth = ctx.exterior_depth + roof_overhang * 2.0
        base_z = ctx.base_z + ctx.building_height + roof_lift

        generated: list[SceneObject] = []
        if roof_type == "flat":
            roof_obj = SceneObject(
                id="exterior_roof",
                type="exterior_roof",
                transform=Transform(position=(ctx.center_x, ctx.center_y, base_z + roof_thickness / 2.0)),
                mesh=create_box(width=roof_width, height=roof_thickness, depth=roof_depth),
            )
        elif roof_type == "gabled":
            rise = _to_positive_float(
                roof_cfg.get("gabled_rise", roof_cfg.get("rise")),
                max(1.2, min(roof_width, roof_depth) * 0.12),
                "exterior.roof.gabled_rise",
            )
            ridge_axis = str(roof_cfg.get("ridge_axis", "x")).strip().lower() or "x"
            if ridge_axis not in {"x", "y"}:
                ridge_axis = "x"
            roof_obj = SceneObject(
                id="exterior_roof",
                type="exterior_roof",
                transform=Transform(position=(ctx.center_x, ctx.center_y, base_z)),
                mesh=self._create_gabled_roof_mesh(
                    width=roof_width,
                    depth=roof_depth,
                    thickness=roof_thickness,
                    rise=rise,
                    ridge_axis=ridge_axis,
                ),
            )
        else:
            rise = _to_positive_float(
                roof_cfg.get("sawtooth_rise", roof_cfg.get("rise")),
                max(1.0, min(roof_width, roof_depth) * 0.1),
                "exterior.roof.sawtooth_rise",
            )
            saw_axis = str(roof_cfg.get("sawtooth_axis", roof_cfg.get("axis", "x"))).strip().lower() or "x"
            if saw_axis not in {"x", "y"}:
                saw_axis = "x"

            span = roof_width if saw_axis == "x" else roof_depth
            tooth_width = _to_positive_float(
                roof_cfg.get("tooth_width"),
                8.0,
                "exterior.roof.tooth_width",
            )
            raw_tooth_count = roof_cfg.get("tooth_count", "auto")
            if isinstance(raw_tooth_count, str) and raw_tooth_count.strip().lower() == "auto":
                tooth_count = max(2, int(round(span / max(2.0, tooth_width))))
            else:
                tooth_count = max(
                    2,
                    _to_non_negative_int(raw_tooth_count, 2, "exterior.roof.tooth_count"),
                )
            clerestory_ratio = float(roof_cfg.get("clerestory_ratio", 0.28))
            clerestory_ratio = max(0.08, min(0.85, clerestory_ratio))

            roof_obj = SceneObject(
                id="exterior_roof",
                type="exterior_roof",
                transform=Transform(position=(ctx.center_x, ctx.center_y, base_z)),
                mesh=self._create_sawtooth_roof_mesh(
                    width=roof_width,
                    depth=roof_depth,
                    thickness=roof_thickness,
                    rise=rise,
                    tooth_count=tooth_count,
                    axis=saw_axis,
                    clerestory_ratio=clerestory_ratio,
                ),
            )
        generated.append(roof_obj)

        parapet_cfg = _to_mapping(self.settings.get("parapet"), "exterior.parapet")
        parapet_on_sloped = _to_bool(parapet_cfg.get("on_sloped_roofs", False), default=False)
        if _to_bool(parapet_cfg.get("enabled", True), default=True) and (roof_type == "flat" or parapet_on_sloped):
            parapet_height = _to_positive_float(
                parapet_cfg.get("height"),
                0.9,
                "exterior.parapet.height",
            )
            parapet_thickness = _to_positive_float(
                parapet_cfg.get("thickness"),
                0.24,
                "exterior.parapet.thickness",
            )
            parapet_z = base_z + roof_thickness + parapet_height / 2.0
            half_w = roof_width / 2.0 - parapet_thickness / 2.0
            half_d = roof_depth / 2.0 - parapet_thickness / 2.0

            generated.extend(
                [
                    SceneObject(
                        id="exterior_parapet_north",
                        type="exterior_parapet",
                        transform=Transform(position=(ctx.center_x, ctx.center_y + half_d, parapet_z)),
                        mesh=create_wall(
                            length=roof_width,
                            height=parapet_height,
                            thickness=parapet_thickness,
                        ),
                    ),
                    SceneObject(
                        id="exterior_parapet_south",
                        type="exterior_parapet",
                        transform=Transform(position=(ctx.center_x, ctx.center_y - half_d, parapet_z)),
                        mesh=create_wall(
                            length=roof_width,
                            height=parapet_height,
                            thickness=parapet_thickness,
                        ),
                    ),
                    SceneObject(
                        id="exterior_parapet_west",
                        type="exterior_parapet",
                        transform=Transform(
                            position=(ctx.center_x - half_w, ctx.center_y, parapet_z),
                            rotation=(0.0, 0.0, 90.0),
                        ),
                        mesh=create_wall(
                            length=roof_depth,
                            height=parapet_height,
                            thickness=parapet_thickness,
                        ),
                    ),
                    SceneObject(
                        id="exterior_parapet_east",
                        type="exterior_parapet",
                        transform=Transform(
                            position=(ctx.center_x + half_w, ctx.center_y, parapet_z),
                            rotation=(0.0, 0.0, 90.0),
                        ),
                        mesh=create_wall(
                            length=roof_depth,
                            height=parapet_height,
                            thickness=parapet_thickness,
                        ),
                    ),
                ]
            )

        for obj in generated:
            self._upsert_child(group, obj)
        return generated

    def generate_facade(self, scene: Scene) -> list[SceneObject]:
        group = self._ensure_group(scene)
        ctx = self._ensure_context(scene)
        generated: list[SceneObject] = []
        facade_cfg = _to_mapping(self.settings.get("facade"), "exterior.facade")
        panel_thickness = _to_positive_float(
            facade_cfg.get("panel_thickness"),
            0.08,
            "exterior.facade.panel_thickness",
        )
        panel_reveal = _to_non_negative_float(
            facade_cfg.get("panel_reveal"),
            0.05,
            "exterior.facade.panel_reveal",
        )
        min_panel_height = _to_positive_float(
            facade_cfg.get("min_panel_height"),
            2.6,
            "exterior.facade.min_panel_height",
        )
        section_every = _to_non_negative_int(
            facade_cfg.get("section_every"),
            3,
            "exterior.facade.section_every",
        )
        seam_width = _to_positive_float(
            facade_cfg.get("seam_width"),
            0.12,
            "exterior.facade.seam_width",
        )
        openings_by_side = self._openings_by_side or self._collect_shell_openings(
            scene,
            ctx,
            opening_clearance=0.0,
        )
        sections_by_side = self._facade_sections_by_side or self._build_facade_sections(
            ctx=ctx,
            openings_by_side=openings_by_side,
        )

        half_wall_x = ctx.exterior_width / 2.0 - ctx.wall_thickness / 2.0
        half_wall_y = ctx.exterior_depth / 2.0 - ctx.wall_thickness / 2.0
        panel_index = 1
        seam_index = 1

        for side in ("south", "north", "west", "east"):
            side_sections = sections_by_side.get(side, [])
            for section_idx, section in enumerate(side_sections):
                section_length = max(0.08, section.end - section.start)
                offset = (section.start + section.end) / 2.0
                panel_height = self._segment_facade_height(ctx, side=side, segment_offset=offset)
                panel_height = max(min_panel_height, min(ctx.building_height, panel_height))

                if side in {"south", "north"}:
                    sign = -1.0 if side == "south" else 1.0
                    x = ctx.center_x + offset
                    y = ctx.center_y + sign * (
                        half_wall_y + ctx.wall_thickness / 2.0 + panel_reveal + panel_thickness / 2.0
                    )
                    z = ctx.base_z + panel_height / 2.0
                    panel_mesh = create_box(
                        width=max(0.08, section_length * 0.94),
                        height=panel_height,
                        depth=panel_thickness,
                    )
                else:
                    sign = -1.0 if side == "west" else 1.0
                    x = ctx.center_x + sign * (
                        half_wall_x + ctx.wall_thickness / 2.0 + panel_reveal + panel_thickness / 2.0
                    )
                    y = ctx.center_y + offset
                    z = ctx.base_z + panel_height / 2.0
                    panel_mesh = create_box(
                        width=panel_thickness,
                        height=panel_height,
                        depth=max(0.08, section_length * 0.94),
                    )

                panel_obj = SceneObject(
                    id=f"exterior_wall_panel_{side}_{panel_index}",
                    type="exterior_facade_panel",
                    transform=Transform(position=(x, y, z)),
                    mesh=panel_mesh,
                )
                generated.append(panel_obj)
                self._upsert_child(group, panel_obj)
                panel_index += 1

                if section.kind == "window":
                    for window_idx, window_offset in enumerate(
                        self._section_window_offsets(section.start, section.end, section.biome),
                        start=1,
                    ):
                        window = self._build_window_object(
                            ctx=ctx,
                            side=side,
                            offset=window_offset,
                            index=window_idx,
                            section_index=panel_index,
                            wall_x=half_wall_x,
                            wall_y=half_wall_y,
                        )
                        if window is None:
                            continue
                        generated.append(window)
                        self._upsert_child(group, window)

                if section_every > 0 and section_idx > 0 and (section_idx % section_every == 0):
                    if side in {"south", "north"}:
                        sign = -1.0 if side == "south" else 1.0
                        sx = ctx.center_x + section.start
                        sy = ctx.center_y + sign * (
                            half_wall_y + ctx.wall_thickness / 2.0 + panel_reveal + panel_thickness / 2.0
                        )
                        sz = ctx.base_z + panel_height / 2.0
                        seam_mesh = create_box(
                            width=seam_width,
                            height=panel_height,
                            depth=panel_thickness * 1.05,
                        )
                    else:
                        sign = -1.0 if side == "west" else 1.0
                        sx = ctx.center_x + sign * (
                            half_wall_x + ctx.wall_thickness / 2.0 + panel_reveal + panel_thickness / 2.0
                        )
                        sy = ctx.center_y + section.start
                        sz = ctx.base_z + panel_height / 2.0
                        seam_mesh = create_box(
                            width=panel_thickness * 1.05,
                            height=panel_height,
                            depth=seam_width,
                        )
                    seam_obj = SceneObject(
                        id=f"exterior_facade_section_{side}_{seam_index}",
                        type="exterior_facade_section",
                        transform=Transform(position=(sx, sy, sz)),
                        mesh=seam_mesh,
                    )
                    generated.append(seam_obj)
                    self._upsert_child(group, seam_obj)
                    seam_index += 1

        door_openings = [
            opening
            for side in ("south", "north", "west", "east")
            for opening in openings_by_side.get(side, [])
            if opening.kind == "door"
        ]
        gate_openings = [
            opening
            for side in ("south", "north", "west", "east")
            for opening in openings_by_side.get(side, [])
            if opening.kind == "gate"
        ]

        door_cfg = _to_mapping(facade_cfg.get("doors"), "exterior.facade.doors")
        door_leaf_depth = _to_positive_float(
            door_cfg.get("leaf_depth"),
            0.08,
            "exterior.facade.doors.leaf_depth",
        )
        door_index = 1
        for opening in door_openings:
            door = self._build_door_or_gate_leaf(
                ctx=ctx,
                opening=opening,
                wall_x=half_wall_x,
                wall_y=half_wall_y,
                depth=door_leaf_depth,
                kind="door",
                index=door_index,
            )
            if door is None:
                continue
            generated.append(door)
            self._upsert_child(group, door)
            door_index += 1

        gate_leaf_cfg = _to_mapping(facade_cfg.get("gates"), "exterior.facade.gates")
        gate_leaf_depth = _to_positive_float(
            gate_leaf_cfg.get("leaf_depth"),
            0.1,
            "exterior.facade.gates.leaf_depth",
        )
        gate_index = 1
        for opening in gate_openings:
            gate_leaf = self._build_door_or_gate_leaf(
                ctx=ctx,
                opening=opening,
                wall_x=half_wall_x,
                wall_y=half_wall_y,
                depth=gate_leaf_depth,
                kind="gate",
                index=gate_index,
            )
            if gate_leaf is None:
                continue
            generated.append(gate_leaf)
            self._upsert_child(group, gate_leaf)
            gate_index += 1

        if gate_openings:
            frame_cfg = _to_mapping(facade_cfg.get("gate_frame"), "exterior.facade.gate_frame")
            frame_thickness = _to_positive_float(
                frame_cfg.get("thickness"),
                0.16,
                "exterior.facade.gate_frame.thickness",
            )
            frame_depth = _to_positive_float(
                frame_cfg.get("depth"),
                0.2,
                "exterior.facade.gate_frame.depth",
            )
            frame_gap = _to_non_negative_float(
                frame_cfg.get("gap_from_wall"),
                0.04,
                "exterior.facade.gate_frame.gap_from_wall",
            )
            frame_index = 1
            for opening in gate_openings:
                for frame in self._build_opening_frame(
                    ctx=ctx,
                    opening=opening,
                    wall_x=half_wall_x,
                    wall_y=half_wall_y,
                    frame_thickness=frame_thickness,
                    frame_depth=frame_depth,
                    frame_gap=frame_gap,
                    frame_index=frame_index,
                ):
                    generated.append(frame)
                    self._upsert_child(group, frame)
                    frame_index += 1

            canopies_cfg = _to_mapping(self.settings.get("canopies"), "exterior.canopies")
            if _to_bool(canopies_cfg.get("enabled", True), default=True):
                canopy_depth = _to_positive_float(
                    canopies_cfg.get("depth"),
                    2.4,
                    "exterior.canopies.depth",
                )
                canopy_thickness = _to_positive_float(
                    canopies_cfg.get("thickness"),
                    0.14,
                    "exterior.canopies.thickness",
                )
                canopy_width_factor = _to_positive_float(
                    canopies_cfg.get("width_factor"),
                    1.25,
                    "exterior.canopies.width_factor",
                )
                z_offset = float(canopies_cfg.get("z_offset", 0.25))
                canopy_index = 1
                for opening in gate_openings:
                    canopy = self._build_gate_canopy(
                        ctx=ctx,
                        opening=opening,
                        wall_x=half_wall_x,
                        wall_y=half_wall_y,
                        canopy_depth=canopy_depth,
                        canopy_thickness=canopy_thickness,
                        canopy_width_factor=canopy_width_factor,
                        z_offset=z_offset,
                        canopy_index=canopy_index,
                    )
                    if canopy is None:
                        continue
                    generated.append(canopy)
                    self._upsert_child(group, canopy)
                    canopy_index += 1

        return generated

    def generate_external_equipment(self, scene: Scene) -> list[SceneObject]:
        group = self._ensure_group(scene)
        ctx = self._ensure_context(scene)
        equipment_density = self._equipment_density()
        roof_type, roof_cfg = self._roof_type_config()
        roof_thickness = _to_positive_float(
            self.settings.get("roof_thickness"),
            0.3,
            "exterior.roof_thickness",
        )
        roof_lift = float(self.settings.get("roof_lift", 0.0))
        roof_peak = self._roof_peak_height(
            ctx=ctx,
            roof_type=roof_type,
            roof_cfg=roof_cfg,
            thickness=roof_thickness,
            lift=roof_lift,
        )
        roof_width = ctx.exterior_width + _to_non_negative_float(
            self.settings.get("roof_overhang"),
            0.55,
            "exterior.roof_overhang",
        ) * 2.0
        roof_depth = ctx.exterior_depth + _to_non_negative_float(
            self.settings.get("roof_overhang"),
            0.55,
            "exterior.roof_overhang",
        ) * 2.0

        generated: list[SceneObject] = []

        tech_cfg = _to_mapping(self.settings.get("roof_units"), "exterior.roof_units")
        if _to_bool(tech_cfg.get("enabled", True), default=True):
            unit_width = _to_positive_float(tech_cfg.get("width"), 1.8, "exterior.roof_units.width")
            unit_depth = _to_positive_float(tech_cfg.get("depth"), 1.4, "exterior.roof_units.depth")
            unit_height = _to_positive_float(tech_cfg.get("height"), 1.0, "exterior.roof_units.height")
            edge_margin = _to_non_negative_float(
                tech_cfg.get("edge_margin"),
                2.8,
                "exterior.roof_units.edge_margin",
            )
            density = _to_non_negative_float(
                tech_cfg.get("count_per_1000m2"),
                2.0,
                "exterior.roof_units.count_per_1000m2",
            )

            raw_count = tech_cfg.get("count")
            if raw_count is None or (isinstance(raw_count, str) and raw_count.strip().lower() == "auto"):
                area = max(1.0, ctx.exterior_width * ctx.exterior_depth)
                target_count = max(1, int(round((area / 1000.0) * density)))
            else:
                target_count = max(0, _to_non_negative_int(raw_count, 0, "exterior.roof_units.count"))
            target_count = max(0, int(round(target_count * equipment_density)))

            max_x = roof_width / 2.0 - edge_margin - unit_width / 2.0
            max_y = roof_depth / 2.0 - edge_margin - unit_depth / 2.0
            if max_x > 0.0 and max_y > 0.0 and target_count > 0:
                cols = max(1, int(round(sqrt(target_count * (roof_width / max(roof_depth, 0.1))))))
                rows = max(1, int(ceil(target_count / cols)))
                x_positions = _symmetric_positions(cols, max_x)
                y_positions = _symmetric_positions(rows, max_y)
                z = roof_peak + unit_height / 2.0

                index = 1
                for y in y_positions:
                    for x in x_positions:
                        if index > target_count:
                            break
                        obj = SceneObject(
                            id=f"exterior_technical_block_{index}",
                            type="exterior_technical_block",
                            transform=Transform(position=(ctx.center_x + x, ctx.center_y + y, z)),
                            mesh=create_box(width=unit_width, height=unit_height, depth=unit_depth),
                        )
                        generated.append(obj)
                        self._upsert_child(group, obj)
                        index += 1
                    if index > target_count:
                        break

        ventilation_cfg = _to_mapping(self.settings.get("ventilation"), "exterior.ventilation")
        if _to_bool(ventilation_cfg.get("enabled", True), default=True):
            shaft_width = _to_positive_float(
                ventilation_cfg.get("shaft_width"),
                0.85,
                "exterior.ventilation.shaft_width",
            )
            shaft_depth = _to_positive_float(
                ventilation_cfg.get("shaft_depth"),
                0.85,
                "exterior.ventilation.shaft_depth",
            )
            shaft_height = _to_positive_float(
                ventilation_cfg.get("shaft_height"),
                1.6,
                "exterior.ventilation.shaft_height",
            )
            density = _to_non_negative_float(
                ventilation_cfg.get("count_per_1000m2"),
                3.0,
                "exterior.ventilation.count_per_1000m2",
            )
            margin = _to_non_negative_float(
                ventilation_cfg.get("edge_margin"),
                2.4,
                "exterior.ventilation.edge_margin",
            )
            raw_count = ventilation_cfg.get("count")
            if raw_count is None or (isinstance(raw_count, str) and raw_count.strip().lower() == "auto"):
                count = max(1, int(round((ctx.exterior_width * ctx.exterior_depth / 1000.0) * density)))
            else:
                count = max(0, _to_non_negative_int(raw_count, 0, "exterior.ventilation.count"))
            count = max(0, int(round(count * equipment_density)))

            max_x = roof_width / 2.0 - margin - shaft_width / 2.0
            max_y = roof_depth / 2.0 - margin - shaft_depth / 2.0
            if max_x > 0.0 and max_y > 0.0 and count > 0:
                cols = max(1, int(round(sqrt(count * (roof_width / max(roof_depth, 0.1))))))
                rows = max(1, int(ceil(count / cols)))
                x_positions = _symmetric_positions(cols, max_x)
                y_positions = _symmetric_positions(rows, max_y)
                z = roof_peak + shaft_height / 2.0
                vent_index = 1
                for y in y_positions:
                    for x in x_positions:
                        if vent_index > count:
                            break
                        obj = SceneObject(
                            id=f"exterior_vent_shaft_{vent_index}",
                            type="exterior_vent_shaft",
                            transform=Transform(position=(ctx.center_x + x, ctx.center_y + y, z)),
                            mesh=create_box(width=shaft_width, height=shaft_height, depth=shaft_depth),
                        )
                        generated.append(obj)
                        self._upsert_child(group, obj)
                        vent_index += 1
                    if vent_index > count:
                        break

        pipe_cfg = _to_mapping(self.settings.get("roof_pipes"), "exterior.roof_pipes")
        if _to_bool(pipe_cfg.get("enabled", True), default=True):
            radius = _to_positive_float(pipe_cfg.get("radius"), 0.24, "exterior.roof_pipes.radius")
            height = _to_positive_float(pipe_cfg.get("height"), 2.4, "exterior.roof_pipes.height")
            per_boiler = max(1, _to_non_negative_int(pipe_cfg.get("per_boiler_room"), 2, "exterior.roof_pipes.per_boiler_room"))
            pipe_index = 1
            for root in scene.objects:
                root_type = root.type.strip().lower()
                if not root_type.startswith("room_boiler"):
                    continue
                bounds = self._measure_subtree_bounds(root)
                if bounds is None:
                    continue
                width = max(0.5, bounds.max_x - bounds.min_x)
                depth = max(0.5, bounds.max_y - bounds.min_y)
                count = max(1, int(round((width * depth) / 120.0)))
                count = max(per_boiler, count)
                count = max(1, int(round(count * equipment_density)))
                max_offset = max(0.0, min(width, depth) * 0.24)
                offsets = _symmetric_positions(count, max_offset)
                for offset in offsets:
                    x = (bounds.min_x + bounds.max_x) / 2.0 + offset
                    y = (bounds.min_y + bounds.max_y) / 2.0
                    obj = SceneObject(
                        id=f"exterior_roof_pipe_{pipe_index}",
                        type="exterior_roof_pipe",
                        transform=Transform(position=(x, y, roof_peak + height / 2.0)),
                        mesh=create_column(radius=radius, height=height, segments=20),
                    )
                    generated.append(obj)
                    self._upsert_child(group, obj)
                    pipe_index += 1

        linked = self._generate_biome_linked_exterior(
            group=group,
            ctx=ctx,
            roof_peak=roof_peak,
            equipment_density=equipment_density,
        )
        generated.extend(linked)

        service_network = self._generate_exterior_service_network(
            group=group,
            ctx=ctx,
            equipment_density=equipment_density,
        )
        generated.extend(service_network)

        return generated

    def _generate_biome_linked_exterior(
        self,
        *,
        group: SceneObject,
        ctx: _ExteriorContext,
        roof_peak: float,
        equipment_density: float,
    ) -> list[SceneObject]:
        links_cfg = _to_mapping(self.settings.get("biome_links"), "exterior.biome_links")
        if not _to_bool(links_cfg.get("enabled", True), default=True):
            return []
        if not ctx.room_volumes:
            return []

        generated: list[SceneObject] = []
        boiler_cfg = _to_mapping(links_cfg.get("boiler"), "exterior.biome_links.boiler")
        electrical_cfg = _to_mapping(links_cfg.get("electrical"), "exterior.biome_links.electrical")
        maintenance_cfg = _to_mapping(links_cfg.get("maintenance"), "exterior.biome_links.maintenance")
        laboratory_cfg = _to_mapping(links_cfg.get("laboratory"), "exterior.biome_links.laboratory")

        boiler_pipe_index = 1
        chimney_index = 1
        transformer_index = 1
        cable_index = 1
        repair_pad_index = 1
        lab_vent_index = 1
        lab_block_index = 1

        for room in ctx.room_volumes:
            biome = room.biome.strip().lower()
            side, base_offset = self._room_exterior_anchor(ctx, room)

            if biome == "boiler":
                pipe_radius = _to_positive_float(
                    boiler_cfg.get("pipe_radius"),
                    0.24,
                    "exterior.biome_links.boiler.pipe_radius",
                )
                pipe_length = _to_positive_float(
                    boiler_cfg.get("pipe_length"),
                    2.8,
                    "exterior.biome_links.boiler.pipe_length",
                )
                chimney_radius = _to_positive_float(
                    boiler_cfg.get("chimney_radius"),
                    max(pipe_radius * 1.2, 0.28),
                    "exterior.biome_links.boiler.chimney_radius",
                )
                chimney_height = _to_positive_float(
                    boiler_cfg.get("chimney_height"),
                    7.0,
                    "exterior.biome_links.boiler.chimney_height",
                )
                raw_pipe_count = boiler_cfg.get("pipes_per_room", "auto")
                if isinstance(raw_pipe_count, str) and raw_pipe_count.strip().lower() == "auto":
                    room_area = max(1.0, (room.max_x - room.min_x) * (room.max_y - room.min_y))
                    pipe_count = max(1, int(round(room_area / 120.0)))
                else:
                    pipe_count = max(
                        1,
                        _to_non_negative_int(
                            raw_pipe_count,
                            1,
                            "exterior.biome_links.boiler.pipes_per_room",
                        ),
                    )
                pipe_count = max(1, int(round(pipe_count * equipment_density)))

                spread_span = (room.max_x - room.min_x) if side in {"south", "north"} else (room.max_y - room.min_y)
                spread = _symmetric_positions(pipe_count, max(0.0, spread_span * 0.22))
                pipe_z = min(
                    ctx.base_z + ctx.building_height - 0.9,
                    max(ctx.base_z + 2.2, room.max_z - 0.8),
                )
                for delta in spread:
                    along = base_offset + delta
                    pipe_position = self._outside_side_position(
                        ctx,
                        side=side,
                        side_offset=along,
                        outward=pipe_length / 2.0 + 0.06,
                        z=pipe_z,
                    )
                    pipe_mesh = self._outward_segment_mesh(
                        side=side,
                        length=pipe_length,
                        thickness=max(0.08, pipe_radius * 2.0),
                    )
                    pipe = SceneObject(
                        id=f"exterior_boiler_pipe_{boiler_pipe_index}",
                        type="exterior_boiler_pipe",
                        transform=Transform(position=pipe_position),
                        mesh=pipe_mesh,
                    )
                    generated.append(pipe)
                    self._upsert_child(group, pipe)
                    boiler_pipe_index += 1

                    chimney = SceneObject(
                        id=f"exterior_chimney_{chimney_index}",
                        type="exterior_chimney",
                        transform=Transform(
                            position=self._outside_side_position(
                                ctx,
                                side=side,
                                side_offset=along,
                                outward=pipe_length + chimney_radius + 0.12,
                                z=ctx.base_z + chimney_height / 2.0,
                            )
                        ),
                        mesh=create_column(radius=chimney_radius, height=chimney_height, segments=24),
                    )
                    generated.append(chimney)
                    self._upsert_child(group, chimney)
                    chimney_index += 1

            elif biome == "electrical":
                transformer_width = _to_positive_float(
                    electrical_cfg.get("transformer_width"),
                    2.4,
                    "exterior.biome_links.electrical.transformer_width",
                )
                transformer_depth = _to_positive_float(
                    electrical_cfg.get("transformer_depth"),
                    1.8,
                    "exterior.biome_links.electrical.transformer_depth",
                )
                transformer_height = _to_positive_float(
                    electrical_cfg.get("transformer_height"),
                    2.2,
                    "exterior.biome_links.electrical.transformer_height",
                )
                cable_height = _to_positive_float(
                    electrical_cfg.get("cable_height"),
                    2.4,
                    "exterior.biome_links.electrical.cable_height",
                )
                cable_thickness = _to_positive_float(
                    electrical_cfg.get("cable_thickness"),
                    0.14,
                    "exterior.biome_links.electrical.cable_thickness",
                )
                raw_count = electrical_cfg.get("transformers_per_room", "auto")
                if isinstance(raw_count, str) and raw_count.strip().lower() == "auto":
                    room_area = max(1.0, (room.max_x - room.min_x) * (room.max_y - room.min_y))
                    transformer_count = max(1, min(3, int(round(room_area / 140.0))))
                else:
                    transformer_count = max(
                        1,
                        _to_non_negative_int(
                            raw_count,
                            1,
                            "exterior.biome_links.electrical.transformers_per_room",
                        ),
                    )
                transformer_count = max(1, int(round(transformer_count * equipment_density)))

                spread_span = (room.max_x - room.min_x) if side in {"south", "north"} else (room.max_y - room.min_y)
                spread = _symmetric_positions(transformer_count, max(0.0, spread_span * 0.25))
                wall_face = self._side_outer_face(ctx, side)
                for delta in spread:
                    along = base_offset + delta
                    transformer_outward = transformer_depth / 2.0 + 1.6
                    transformer_pos = self._outside_side_position(
                        ctx,
                        side=side,
                        side_offset=along,
                        outward=transformer_outward,
                        z=ctx.base_z + transformer_height / 2.0,
                    )
                    transformer_mesh = (
                        create_box(
                            width=transformer_width,
                            height=transformer_height,
                            depth=transformer_depth,
                        )
                        if side in {"south", "north"}
                        else create_box(
                            width=transformer_depth,
                            height=transformer_height,
                            depth=transformer_width,
                        )
                    )
                    transformer = SceneObject(
                        id=f"exterior_transformer_{transformer_index}",
                        type="exterior_transformer",
                        transform=Transform(position=transformer_pos),
                        mesh=transformer_mesh,
                    )
                    generated.append(transformer)
                    self._upsert_child(group, transformer)
                    transformer_index += 1

                    cable_run = abs(self._side_normal_coordinate(side, transformer_pos[0], transformer_pos[1]) - wall_face)
                    if cable_run <= 0.05:
                        continue
                    cable_center = self._outside_side_position(
                        ctx,
                        side=side,
                        side_offset=along,
                        outward=cable_run / 2.0,
                        z=ctx.base_z + cable_height,
                    )
                    cable_mesh = self._outward_segment_mesh(
                        side=side,
                        length=cable_run,
                        thickness=cable_thickness,
                    )
                    cable = SceneObject(
                        id=f"exterior_power_cable_{cable_index}",
                        type="exterior_power_cable",
                        transform=Transform(position=cable_center),
                        mesh=cable_mesh,
                    )
                    generated.append(cable)
                    self._upsert_child(group, cable)
                    cable_index += 1

            elif biome == "maintenance":
                pad_thickness = _to_positive_float(
                    maintenance_cfg.get("pad_thickness"),
                    0.08,
                    "exterior.biome_links.maintenance.pad_thickness",
                )
                pad_depth = _to_positive_float(
                    maintenance_cfg.get("pad_depth"),
                    5.2,
                    "exterior.biome_links.maintenance.pad_depth",
                )
                raw_pad_count = maintenance_cfg.get("pads_per_room", "auto")
                if isinstance(raw_pad_count, str) and raw_pad_count.strip().lower() == "auto":
                    pad_count = max(1, int(round(equipment_density)))
                else:
                    pad_count = max(
                        1,
                        _to_non_negative_int(
                            raw_pad_count,
                            1,
                            "exterior.biome_links.maintenance.pads_per_room",
                        ),
                    )
                pad_count = max(1, int(round(pad_count * equipment_density)))
                room_span = (room.max_x - room.min_x) if side in {"south", "north"} else (room.max_y - room.min_y)
                pad_width = max(3.8, min(11.0, room_span * 0.82))
                span = (room.max_x - room.min_x) if side in {"south", "north"} else (room.max_y - room.min_y)
                pad_offsets = _symmetric_positions(pad_count, max(0.0, span * 0.2))
                for delta in pad_offsets:
                    pad_position = self._outside_side_position(
                        ctx,
                        side=side,
                        side_offset=base_offset + delta,
                        outward=pad_depth / 2.0 + 0.7,
                        z=ctx.base_z + pad_thickness / 2.0,
                    )
                    pad_mesh = (
                        create_box(width=pad_width, height=pad_thickness, depth=pad_depth)
                        if side in {"south", "north"}
                        else create_box(width=pad_depth, height=pad_thickness, depth=pad_width)
                    )
                    repair_pad = SceneObject(
                        id=f"exterior_repair_pad_{repair_pad_index}",
                        type="exterior_repair_pad",
                        transform=Transform(position=pad_position),
                        mesh=pad_mesh,
                    )
                    generated.append(repair_pad)
                    self._upsert_child(group, repair_pad)
                    repair_pad_index += 1

            elif biome == "laboratory":
                vent_radius = _to_positive_float(
                    laboratory_cfg.get("vent_radius"),
                    0.22,
                    "exterior.biome_links.laboratory.vent_radius",
                )
                vent_height = _to_positive_float(
                    laboratory_cfg.get("vent_height"),
                    1.6,
                    "exterior.biome_links.laboratory.vent_height",
                )
                block_width = _to_positive_float(
                    laboratory_cfg.get("block_width"),
                    1.1,
                    "exterior.biome_links.laboratory.block_width",
                )
                block_depth = _to_positive_float(
                    laboratory_cfg.get("block_depth"),
                    0.9,
                    "exterior.biome_links.laboratory.block_depth",
                )
                block_height = _to_positive_float(
                    laboratory_cfg.get("block_height"),
                    0.72,
                    "exterior.biome_links.laboratory.block_height",
                )
                room_area = max(1.0, (room.max_x - room.min_x) * (room.max_y - room.min_y))
                vent_count = max(1, min(3, int(round(room_area / 110.0))))
                vent_count = max(1, int(round(vent_count * equipment_density)))
                vent_offsets = _symmetric_positions(vent_count, max(0.0, min(room.max_x - room.min_x, room.max_y - room.min_y) * 0.2))
                room_center_x = (room.min_x + room.max_x) / 2.0
                room_center_y = (room.min_y + room.max_y) / 2.0
                roof_limit_x = ctx.exterior_width / 2.0 - 1.1
                roof_limit_y = ctx.exterior_depth / 2.0 - 1.1
                for delta in vent_offsets:
                    vent_x = max(
                        ctx.center_x - roof_limit_x,
                        min(ctx.center_x + roof_limit_x, room_center_x + (delta if side in {"south", "north"} else 0.0)),
                    )
                    vent_y = max(
                        ctx.center_y - roof_limit_y,
                        min(ctx.center_y + roof_limit_y, room_center_y + (delta if side in {"east", "west"} else 0.0)),
                    )
                    vent = SceneObject(
                        id=f"exterior_lab_vent_{lab_vent_index}",
                        type="exterior_lab_vent",
                        transform=Transform(position=(vent_x, vent_y, roof_peak + vent_height / 2.0)),
                        mesh=create_column(radius=vent_radius, height=vent_height, segments=16),
                    )
                    generated.append(vent)
                    self._upsert_child(group, vent)
                    lab_vent_index += 1

                    tech = SceneObject(
                        id=f"exterior_lab_technical_block_{lab_block_index}",
                        type="exterior_lab_technical_block",
                        transform=Transform(
                            position=(
                                max(ctx.center_x - roof_limit_x, min(ctx.center_x + roof_limit_x, vent_x + block_width * 0.78)),
                                max(ctx.center_y - roof_limit_y, min(ctx.center_y + roof_limit_y, vent_y - block_depth * 0.62)),
                                roof_peak + block_height / 2.0,
                            )
                        ),
                        mesh=create_box(width=block_width, height=block_height, depth=block_depth),
                    )
                    generated.append(tech)
                    self._upsert_child(group, tech)
                    lab_block_index += 1

            elif biome == "control":
                # Control rooms are intentionally clean on the exterior: windows are handled by facade generation.
                continue

        return generated

    def _generate_exterior_service_network(
        self,
        *,
        group: SceneObject,
        ctx: _ExteriorContext,
        equipment_density: float,
    ) -> list[SceneObject]:
        services_cfg = _to_mapping(self.settings.get("services"), "exterior.services")
        if not _to_bool(services_cfg.get("enabled", True), default=True):
            return []
        if not ctx.room_volumes:
            return []

        tray_height = _to_positive_float(
            services_cfg.get("tray_height"),
            3.2,
            "exterior.services.tray_height",
        )
        tray_width = _to_positive_float(
            services_cfg.get("tray_width"),
            0.45,
            "exterior.services.tray_width",
        )
        tray_thickness = _to_positive_float(
            services_cfg.get("tray_thickness"),
            0.12,
            "exterior.services.tray_thickness",
        )
        tray_offset = _to_non_negative_float(
            services_cfg.get("tray_offset"),
            1.0,
            "exterior.services.tray_offset",
        )
        support_spacing = _to_positive_float(
            services_cfg.get("support_spacing"),
            3.0,
            "exterior.services.support_spacing",
        )
        support_width = _to_positive_float(
            services_cfg.get("support_width"),
            0.12,
            "exterior.services.support_width",
        )
        support_depth = _to_positive_float(
            services_cfg.get("support_depth"),
            0.12,
            "exterior.services.support_depth",
        )
        corner_margin = _to_non_negative_float(
            services_cfg.get("corner_margin"),
            1.4,
            "exterior.services.corner_margin",
        )
        cable_height = _to_positive_float(
            services_cfg.get("cable_height"),
            max(2.2, tray_height + 0.06),
            "exterior.services.cable_height",
        )
        cable_thickness = _to_positive_float(
            services_cfg.get("cable_thickness"),
            0.09,
            "exterior.services.cable_thickness",
        )
        pipe_height = _to_positive_float(
            services_cfg.get("pipe_height"),
            max(2.4, tray_height + 0.35),
            "exterior.services.pipe_height",
        )
        pipe_radius = _to_positive_float(
            services_cfg.get("pipe_radius"),
            0.14,
            "exterior.services.pipe_radius",
        )
        pipe_casing_thickness = _to_positive_float(
            services_cfg.get("pipe_casing_thickness"),
            max(pipe_radius * 2.4, pipe_radius * 2.0 + 0.08),
            "exterior.services.pipe_casing_thickness",
        )
        connector_raw = services_cfg.get("connectors_per_room", "auto")

        default_pipe_biomes = {
            "boiler",
            "refinery",
            "workshop",
            "maintenance",
            "laboratory",
            "storage",
            "electrical",
        }
        default_wire_biomes = {
            "office",
            "control",
            "electrical",
            "laboratory",
            "maintenance",
            "workshop",
            "storage",
            "refinery",
            "boiler",
        }
        pipe_biomes = _to_string_set(services_cfg.get("pipe_biomes")) or default_pipe_biomes
        wire_biomes = _to_string_set(services_cfg.get("wire_biomes")) or default_wire_biomes

        tray_outward = tray_offset + tray_width / 2.0
        pipe_main_outward = tray_outward + tray_width * 0.34
        support_height = max(0.2, tray_height - ctx.base_z)

        generated: list[SceneObject] = []
        support_index = 1
        wire_main_index = 1
        pipe_main_index = 1
        wire_drop_index = 1
        pipe_drop_index = 1
        pipe_casing_index = 1

        for side in ("south", "north", "west", "east"):
            span = ctx.exterior_width if side in {"south", "north"} else ctx.exterior_depth
            run_length = max(0.6, span - corner_margin * 2.0)
            if run_length <= 0.0:
                continue

            tray_center = self._outside_side_position(
                ctx,
                side=side,
                side_offset=0.0,
                outward=tray_outward,
                z=ctx.base_z + tray_height,
            )
            tray_mesh = (
                create_box(width=run_length, height=tray_thickness, depth=tray_width)
                if side in {"south", "north"}
                else create_box(width=tray_width, height=tray_thickness, depth=run_length)
            )
            tray = SceneObject(
                id=f"exterior_cable_tray_{side}",
                type="exterior_cable_tray",
                transform=Transform(position=tray_center),
                mesh=tray_mesh,
            )
            generated.append(tray)
            self._upsert_child(group, tray)

            wire_main_mesh = (
                create_box(width=run_length, height=cable_thickness, depth=max(cable_thickness * 1.35, 0.06))
                if side in {"south", "north"}
                else create_box(width=max(cable_thickness * 1.35, 0.06), height=cable_thickness, depth=run_length)
            )
            wire_main = SceneObject(
                id=f"exterior_cable_bundle_{wire_main_index}",
                type="exterior_cable_bundle",
                transform=Transform(
                    position=self._outside_side_position(
                        ctx,
                        side=side,
                        side_offset=0.0,
                        outward=tray_outward,
                        z=ctx.base_z + cable_height,
                    )
                ),
                mesh=wire_main_mesh,
            )
            generated.append(wire_main)
            self._upsert_child(group, wire_main)
            wire_main_index += 1

            pipe_main_thickness = max(0.08, pipe_radius * 2.0)
            pipe_main_mesh = (
                create_box(width=run_length, height=pipe_main_thickness, depth=pipe_main_thickness)
                if side in {"south", "north"}
                else create_box(width=pipe_main_thickness, height=pipe_main_thickness, depth=run_length)
            )
            pipe_main = SceneObject(
                id=f"exterior_pipe_main_{pipe_main_index}",
                type="exterior_pipe_main",
                transform=Transform(
                    position=self._outside_side_position(
                        ctx,
                        side=side,
                        side_offset=0.0,
                        outward=pipe_main_outward,
                        z=ctx.base_z + pipe_height,
                    )
                ),
                mesh=pipe_main_mesh,
            )
            generated.append(pipe_main)
            self._upsert_child(group, pipe_main)
            pipe_main_index += 1

            support_count = max(2, int(ceil(run_length / max(0.25, support_spacing))) + 1)
            max_offset = max(0.0, run_length / 2.0)
            for offset in _symmetric_positions(support_count, max_offset):
                if self._is_side_offset_blocked_by_opening(side, offset, max(0.35, tray_width)):
                    continue
                support = SceneObject(
                    id=f"exterior_cable_tray_support_{support_index}",
                    type="exterior_cable_tray_support",
                    transform=Transform(
                        position=self._outside_side_position(
                            ctx,
                            side=side,
                            side_offset=offset,
                            outward=tray_outward,
                            z=ctx.base_z + support_height / 2.0,
                        )
                    ),
                    mesh=create_box(width=support_width, height=support_height, depth=support_depth),
                )
                generated.append(support)
                self._upsert_child(group, support)
                support_index += 1

        for room in ctx.room_volumes:
            biome = room.biome.strip().lower()
            side, base_offset = self._room_exterior_anchor(ctx, room)
            base_offset = self._shift_offset_away_from_openings(
                ctx=ctx,
                side=side,
                offset=base_offset,
                clearance=0.55,
            )
            room_span = (room.max_x - room.min_x) if side in {"south", "north"} else (room.max_y - room.min_y)
            room_area = max(1.0, (room.max_x - room.min_x) * (room.max_y - room.min_y))

            if isinstance(connector_raw, str) and connector_raw.strip().lower() == "auto":
                connector_count = max(1, min(3, int(round(room_area / 140.0))))
            else:
                connector_count = max(
                    1,
                    _to_non_negative_int(
                        connector_raw,
                        1,
                        "exterior.services.connectors_per_room",
                    ),
                )
            connector_count = max(1, int(round(connector_count * equipment_density)))

            pipe_factor_by_biome = {
                "boiler": 1.6,
                "refinery": 1.8,
                "workshop": 1.2,
                "maintenance": 1.15,
                "laboratory": 1.0,
                "storage": 0.8,
                "electrical": 0.6,
                "office": 0.35,
                "control": 0.35,
            }
            wire_factor_by_biome = {
                "office": 1.5,
                "control": 1.6,
                "electrical": 1.9,
                "laboratory": 1.25,
                "maintenance": 1.0,
                "workshop": 0.9,
                "storage": 0.75,
                "refinery": 0.8,
                "boiler": 0.75,
            }
            pipe_factor = pipe_factor_by_biome.get(biome, 1.0)
            wire_factor = wire_factor_by_biome.get(biome, 1.0)

            pipe_count = int(round(connector_count * pipe_factor))
            wire_count = int(round(connector_count * wire_factor))
            if biome not in pipe_biomes:
                pipe_count = 0
            if biome not in wire_biomes:
                wire_count = 0

            pipe_count = max(0, min(6, pipe_count))
            wire_count = max(0, min(8, wire_count))
            spread = _symmetric_positions(max(pipe_count, wire_count, 1), max(0.0, room_span * 0.22))

            for idx in range(pipe_count):
                along = base_offset + spread[idx]
                along = self._shift_offset_away_from_openings(
                    ctx=ctx,
                    side=side,
                    offset=along,
                    clearance=0.45,
                )
                outlet_length = max(0.2, pipe_main_outward)
                pipe_thickness = max(0.08, pipe_radius * 2.0)
                pipe_obj = SceneObject(
                    id=f"exterior_service_pipe_{pipe_drop_index}",
                    type="exterior_service_pipe",
                    transform=Transform(
                        position=self._outside_side_position(
                            ctx,
                            side=side,
                            side_offset=along,
                            outward=outlet_length / 2.0,
                            z=ctx.base_z + pipe_height,
                        )
                    ),
                    mesh=self._outward_segment_mesh(side=side, length=outlet_length, thickness=pipe_thickness),
                )
                generated.append(pipe_obj)
                self._upsert_child(group, pipe_obj)

                casing_obj = SceneObject(
                    id=f"exterior_pipe_casing_{pipe_casing_index}",
                    type="exterior_pipe_casing",
                    transform=Transform(position=pipe_obj.transform.position),
                    mesh=self._outward_segment_mesh(
                        side=side,
                        length=max(0.24, outlet_length * 0.82),
                        thickness=max(pipe_thickness + 0.02, pipe_casing_thickness),
                    ),
                )
                generated.append(casing_obj)
                self._upsert_child(group, casing_obj)

                pipe_drop_index += 1
                pipe_casing_index += 1

            for idx in range(wire_count):
                along = base_offset + spread[idx]
                along = self._shift_offset_away_from_openings(
                    ctx=ctx,
                    side=side,
                    offset=along,
                    clearance=0.45,
                )
                wire_length = max(0.2, tray_outward)
                wire = SceneObject(
                    id=f"exterior_service_wire_{wire_drop_index}",
                    type="exterior_service_wire",
                    transform=Transform(
                        position=self._outside_side_position(
                            ctx,
                            side=side,
                            side_offset=along,
                            outward=wire_length / 2.0,
                            z=ctx.base_z + cable_height,
                        )
                    ),
                    mesh=self._outward_segment_mesh(
                        side=side,
                        length=wire_length,
                        thickness=cable_thickness,
                    ),
                )
                generated.append(wire)
                self._upsert_child(group, wire)
                wire_drop_index += 1

        return generated

    def _is_side_offset_blocked_by_opening(self, side: str, offset: float, clearance: float) -> bool:
        for opening in self._openings_by_side.get(side, []):
            if abs(offset - opening.offset) <= (opening.width / 2.0 + clearance):
                return True
        return False

    def _shift_offset_away_from_openings(
        self,
        *,
        ctx: _ExteriorContext,
        side: str,
        offset: float,
        clearance: float,
    ) -> float:
        half_span = ctx.exterior_width / 2.0 if side in {"south", "north"} else ctx.exterior_depth / 2.0
        min_offset = -half_span + 0.9
        max_offset = half_span - 0.9
        resolved = max(min_offset, min(max_offset, offset))
        openings = list(self._openings_by_side.get(side, []))
        if not openings:
            return resolved
        for _ in range(6):
            blockers = [
                opening
                for opening in openings
                if abs(resolved - opening.offset) <= (opening.width / 2.0 + clearance)
            ]
            if not blockers:
                break
            closest = min(blockers, key=lambda opening: abs(resolved - opening.offset))
            sign = -1.0 if resolved <= closest.offset else 1.0
            resolved += sign * (closest.width / 2.0 + clearance + 0.08)
            resolved = max(min_offset, min(max_offset, resolved))
        return resolved

    def _room_exterior_anchor(self, ctx: _ExteriorContext, room: _RoomVolume) -> tuple[str, float]:
        half_w = ctx.interior_width / 2.0
        half_d = ctx.interior_depth / 2.0
        south_bound = ctx.center_y - half_d
        north_bound = ctx.center_y + half_d
        west_bound = ctx.center_x - half_w
        east_bound = ctx.center_x + half_w

        distances = {
            "south": abs(room.min_y - south_bound),
            "north": abs(north_bound - room.max_y),
            "west": abs(room.min_x - west_bound),
            "east": abs(east_bound - room.max_x),
        }
        side = min(distances, key=distances.get)
        if side in {"south", "north"}:
            side_offset = ((room.min_x + room.max_x) / 2.0) - ctx.center_x
            half_span = ctx.exterior_width / 2.0
        else:
            side_offset = ((room.min_y + room.max_y) / 2.0) - ctx.center_y
            half_span = ctx.exterior_depth / 2.0
        margin = 0.9
        side_offset = max(-half_span + margin, min(half_span - margin, side_offset))
        return side, side_offset

    def _outside_side_position(
        self,
        ctx: _ExteriorContext,
        *,
        side: str,
        side_offset: float,
        outward: float,
        z: float,
    ) -> tuple[float, float, float]:
        half_w = ctx.exterior_width / 2.0
        half_d = ctx.exterior_depth / 2.0
        x = ctx.center_x
        y = ctx.center_y
        if side == "south":
            x = ctx.center_x + max(-half_w + 0.4, min(half_w - 0.4, side_offset))
            y = ctx.center_y - half_d - outward
        elif side == "north":
            x = ctx.center_x + max(-half_w + 0.4, min(half_w - 0.4, side_offset))
            y = ctx.center_y + half_d + outward
        elif side == "west":
            x = ctx.center_x - half_w - outward
            y = ctx.center_y + max(-half_d + 0.4, min(half_d - 0.4, side_offset))
        else:
            x = ctx.center_x + half_w + outward
            y = ctx.center_y + max(-half_d + 0.4, min(half_d - 0.4, side_offset))
        return (x, y, z)

    def _side_outer_face(self, ctx: _ExteriorContext, side: str) -> float:
        if side == "south":
            return ctx.center_y - ctx.exterior_depth / 2.0
        if side == "north":
            return ctx.center_y + ctx.exterior_depth / 2.0
        if side == "west":
            return ctx.center_x - ctx.exterior_width / 2.0
        return ctx.center_x + ctx.exterior_width / 2.0

    def _side_normal_coordinate(self, side: str, x: float, y: float) -> float:
        return y if side in {"south", "north"} else x

    def _outward_segment_mesh(self, *, side: str, length: float, thickness: float) -> Mesh:
        if side in {"south", "north"}:
            return create_box(width=thickness, height=thickness, depth=length)
        return create_box(width=length, height=thickness, depth=thickness)

    def generate_terrain(self, scene: Scene) -> list[SceneObject]:
        group = self._ensure_group(scene)
        ctx = self._ensure_context(scene)
        apron_width = _to_non_negative_float(
            self.settings.get("apron_width"),
            2.2,
            "exterior.apron_width",
        )
        apron_thickness = _to_positive_float(
            self.settings.get("apron_thickness"),
            0.08,
            "exterior.apron_thickness",
        )
        if apron_width <= 0.0:
            return []

        apron = SceneObject(
            id="exterior_apron",
            type="exterior_apron",
            transform=Transform(position=(ctx.center_x, ctx.center_y, ctx.base_z + apron_thickness / 2.0)),
            mesh=create_box(
                width=ctx.exterior_width + apron_width * 2.0,
                height=apron_thickness,
                depth=ctx.exterior_depth + apron_width * 2.0,
            ),
        )
        self._upsert_child(group, apron)
        return [apron]

    def _equipment_density(self) -> float:
        value = _resolve_ranged_float(
            self.settings.get("equipment_density"),
            default=1.0,
            label="exterior.equipment_density",
            seed=self.seed,
            salt="equipment_density",
        )
        return max(0.1, value)

    def _window_density(self) -> float:
        facade_cfg = _to_mapping(self.settings.get("facade"), "exterior.facade")
        value = _resolve_ranged_float(
            facade_cfg.get("window_density"),
            default=0.5,
            label="exterior.facade.window_density",
            seed=self.seed,
            salt="window_density",
        )
        return max(0.0, min(1.0, value))

    def _build_facade_sections(
        self,
        *,
        ctx: _ExteriorContext,
        openings_by_side: Mapping[str, list[_FacadeOpening]],
    ) -> dict[str, list[_FacadeSection]]:
        facade_cfg = _to_mapping(self.settings.get("facade"), "exterior.facade")
        section_width = _to_positive_float(
            facade_cfg.get("section_width", facade_cfg.get("panel_width", 4.2)),
            4.2,
            "exterior.facade.section_width",
        )
        touching_by_side = self._touching_rooms_by_side(ctx)

        sections_by_side: dict[str, list[_FacadeSection]] = {
            "south": [],
            "north": [],
            "west": [],
            "east": [],
        }

        for side in ("south", "north", "west", "east"):
            span = ctx.exterior_width if side in {"south", "north"} else ctx.exterior_depth
            count = max(1, int(ceil(span / max(0.25, section_width))))
            actual = span / count
            start = -span / 2.0

            side_touching = touching_by_side.get(side, [])
            side_openings = openings_by_side.get(side, [])

            for index in range(count):
                sec_start = start + index * actual
                sec_end = sec_start + actual
                sec_center = (sec_start + sec_end) / 2.0

                dominant_biome = "generic"
                dominant_overlap = 0.0
                for room, room_start, room_end in side_touching:
                    overlap = max(0.0, min(sec_end, room_end) - max(sec_start, room_start))
                    if overlap > dominant_overlap:
                        dominant_overlap = overlap
                        dominant_biome = room.biome

                kind_from_openings = self._section_opening_kind(sec_start, sec_end, side_openings)
                if kind_from_openings is not None:
                    section_kind = kind_from_openings
                else:
                    section_kind = self._section_kind_for_biome(
                        biome=dominant_biome,
                        section_index=index,
                        section_center=sec_center,
                        side_span=span,
                    )

                sections_by_side[side].append(
                    _FacadeSection(
                        side=side,
                        start=sec_start,
                        end=sec_end,
                        biome=dominant_biome,
                        kind=section_kind,
                    )
                )

        return sections_by_side

    def _section_opening_kind(
        self,
        section_start: float,
        section_end: float,
        openings: list[_FacadeOpening],
    ) -> str | None:
        kind_priority = {"gate": 3, "door": 2, "window": 1}
        selected_kind: str | None = None
        selected_priority = -1
        for opening in openings:
            open_start = opening.offset - opening.width / 2.0
            open_end = opening.offset + opening.width / 2.0
            overlap = min(section_end, open_end) - max(section_start, open_start)
            if overlap <= 0.06:
                continue
            priority = kind_priority.get(opening.kind, 0)
            if priority > selected_priority:
                selected_priority = priority
                selected_kind = opening.kind
        return selected_kind

    def _section_kind_for_biome(
        self,
        *,
        biome: str,
        section_index: int,
        section_center: float,
        side_span: float,
    ) -> str:
        normalized = biome.strip().lower()
        if normalized in {"control", "office"}:
            return "window"
        if normalized in {"boiler"}:
            return "window" if section_index % 5 == 2 else "blind"
        if normalized in {"storage", "warehouse", "maintenance"}:
            if abs(section_center) <= side_span * 0.28 and section_index % 2 == 0:
                return "gate"
            return "blind"
        if normalized in {"laboratory"}:
            return "window" if section_index % 2 == 0 else "blind"
        if normalized in {"electrical"}:
            return "window" if section_index % 4 == 1 else "blind"
        return "window" if section_index % 3 == 1 else "blind"

    def _touching_rooms_by_side(
        self,
        ctx: _ExteriorContext,
    ) -> dict[str, list[tuple[_RoomVolume, float, float]]]:
        touching: dict[str, list[tuple[_RoomVolume, float, float]]] = {
            "south": [],
            "north": [],
            "west": [],
            "east": [],
        }
        if not ctx.room_volumes:
            return touching

        occupied_min_x = min(room.min_x for room in ctx.room_volumes)
        occupied_max_x = max(room.max_x for room in ctx.room_volumes)
        occupied_min_y = min(room.min_y for room in ctx.room_volumes)
        occupied_max_y = max(room.max_y for room in ctx.room_volumes)

        south_boundary = occupied_min_y
        north_boundary = occupied_max_y
        west_boundary = occupied_min_x
        east_boundary = occupied_max_x
        tolerance = max(1.2, ctx.wall_offset + ctx.wall_thickness + 0.8)

        for room in ctx.room_volumes:
            if abs(room.min_y - south_boundary) <= tolerance:
                touching["south"].append((room, room.min_x - ctx.center_x, room.max_x - ctx.center_x))
            if abs(room.max_y - north_boundary) <= tolerance:
                touching["north"].append((room, room.min_x - ctx.center_x, room.max_x - ctx.center_x))
            if abs(room.min_x - west_boundary) <= tolerance:
                touching["west"].append((room, room.min_y - ctx.center_y, room.max_y - ctx.center_y))
            if abs(room.max_x - east_boundary) <= tolerance:
                touching["east"].append((room, room.min_y - ctx.center_y, room.max_y - ctx.center_y))

        for side in ("south", "north", "west", "east"):
            span = ctx.exterior_width if side in {"south", "north"} else ctx.exterior_depth
            half = span / 2.0
            clamped: list[tuple[_RoomVolume, float, float]] = []
            for room, start, end in touching[side]:
                a = max(-half, start)
                b = min(half, end)
                if b - a <= 0.15:
                    continue
                clamped.append((room, a, b))
            touching[side] = clamped
        return touching

    def _section_window_offsets(
        self,
        section_start: float,
        section_end: float,
        biome: str,
    ) -> list[float]:
        facade_cfg = _to_mapping(self.settings.get("facade"), "exterior.facade")
        windows_cfg = _to_mapping(facade_cfg.get("windows"), "exterior.facade.windows")
        default_width = _to_positive_float(
            windows_cfg.get("width"),
            1.55,
            "exterior.facade.windows.width",
        )
        spacing = _to_positive_float(
            windows_cfg.get("spacing"),
            1.8,
            "exterior.facade.windows.spacing",
        )
        segment_width = max(0.2, section_end - section_start)
        biome_scale = 1.0
        biome_name = biome.strip().lower()
        if biome_name == "control":
            biome_scale = 1.12
        elif biome_name == "boiler":
            biome_scale = 0.65
        elif biome_name in {"storage", "warehouse", "maintenance"}:
            biome_scale = 0.85

        width = max(0.5, default_width * biome_scale)
        base_count = max(1, int((segment_width + spacing) // max(0.4, width + spacing)))
        density = self._window_density()
        density_scale = density / 0.5 if density <= 0.5 else density
        count = int(round(base_count * max(0.0, density_scale)))
        if count <= 0:
            return []
        max_offset = max(0.0, segment_width / 2.0 - width / 2.0 - 0.08)
        local_positions = _symmetric_positions(count, max_offset)
        center = (section_start + section_end) / 2.0
        return [center + pos for pos in local_positions]

    def _build_window_object(
        self,
        *,
        ctx: _ExteriorContext,
        side: str,
        offset: float,
        index: int,
        section_index: int,
        wall_x: float,
        wall_y: float,
    ) -> SceneObject | None:
        facade_cfg = _to_mapping(self.settings.get("facade"), "exterior.facade")
        windows_cfg = _to_mapping(facade_cfg.get("windows"), "exterior.facade.windows")
        width = _to_positive_float(windows_cfg.get("width"), 1.55, "exterior.facade.windows.width")
        height = _to_positive_float(windows_cfg.get("height"), 1.35, "exterior.facade.windows.height")
        thickness = _to_positive_float(
            windows_cfg.get("thickness"),
            0.06,
            "exterior.facade.windows.thickness",
        )
        sill_height = _to_non_negative_float(
            windows_cfg.get("sill_height"),
            1.3,
            "exterior.facade.windows.sill_height",
        )
        z = ctx.base_z + sill_height + height / 2.0

        if side in {"south", "north"}:
            sign = -1.0 if side == "south" else 1.0
            x = ctx.center_x + offset
            y = ctx.center_y + sign * (wall_y + ctx.wall_thickness / 2.0 + thickness / 2.0 + 0.03)
            mesh = create_box(width=width, height=height, depth=thickness)
        else:
            sign = -1.0 if side == "west" else 1.0
            x = ctx.center_x + sign * (wall_x + ctx.wall_thickness / 2.0 + thickness / 2.0 + 0.03)
            y = ctx.center_y + offset
            mesh = create_box(width=thickness, height=height, depth=width)

        return SceneObject(
            id=f"exterior_window_{side}_{section_index}_{index}",
            type="exterior_window",
            transform=Transform(position=(x, y, z)),
            mesh=mesh,
        )

    def _build_door_or_gate_leaf(
        self,
        *,
        ctx: _ExteriorContext,
        opening: _FacadeOpening,
        wall_x: float,
        wall_y: float,
        depth: float,
        kind: str,
        index: int,
    ) -> SceneObject | None:
        if kind not in {"door", "gate"}:
            return None
        width = max(0.5, opening.width * (0.95 if kind == "door" else 0.98))
        height = max(1.8, opening.height * (0.98 if kind == "door" else 0.995))
        z = ctx.base_z + height / 2.0
        if opening.side in {"south", "north"}:
            sign = -1.0 if opening.side == "south" else 1.0
            x = ctx.center_x + opening.offset
            y = ctx.center_y + sign * (wall_y + ctx.wall_thickness / 2.0 + depth / 2.0)
            mesh = create_box(width=width, height=height, depth=depth)
        else:
            sign = -1.0 if opening.side == "west" else 1.0
            x = ctx.center_x + sign * (wall_x + ctx.wall_thickness / 2.0 + depth / 2.0)
            y = ctx.center_y + opening.offset
            mesh = create_box(width=depth, height=height, depth=width)

        return SceneObject(
            id=f"exterior_{kind}_{opening.side}_{index}",
            type=f"exterior_{kind}",
            transform=Transform(position=(x, y, z)),
            mesh=mesh,
        )

    def _roof_type_config(self) -> tuple[str, dict[str, object]]:
        roof_cfg = _to_mapping(self.settings.get("roof"), "exterior.roof")
        roof_type_raw = roof_cfg.get("type", self.settings.get("roof_type", "flat"))
        if isinstance(roof_type_raw, (list, tuple)):
            choices = [str(item).strip().lower() for item in roof_type_raw if str(item).strip()]
            if choices:
                index = min(len(choices) - 1, int(_hash_unit(self.seed, "roof.type") * len(choices)))
                roof_type = choices[index]
            else:
                roof_type = "flat"
        else:
            roof_type = str(roof_type_raw).strip().lower() or "flat"
        if roof_type not in {"flat", "sawtooth", "gabled"}:
            roof_type = "flat"
        return roof_type, roof_cfg

    def _roof_peak_height(
        self,
        *,
        ctx: _ExteriorContext,
        roof_type: str,
        roof_cfg: Mapping[str, object],
        thickness: float,
        lift: float,
    ) -> float:
        base_z = ctx.base_z + ctx.building_height + lift
        if roof_type == "flat":
            return base_z + thickness
        if roof_type == "gabled":
            rise = _to_positive_float(
                roof_cfg.get("gabled_rise", roof_cfg.get("rise")),
                max(1.2, min(ctx.exterior_width, ctx.exterior_depth) * 0.12),
                "exterior.roof.gabled_rise",
            )
            return base_z + rise

        rise = _to_positive_float(
            roof_cfg.get("sawtooth_rise", roof_cfg.get("rise")),
            max(1.0, min(ctx.exterior_width, ctx.exterior_depth) * 0.1),
            "exterior.roof.sawtooth_rise",
        )
        return base_z + rise

    def _create_gabled_roof_mesh(
        self,
        *,
        width: float,
        depth: float,
        thickness: float,
        rise: float,
        ridge_axis: str,
    ) -> Mesh:
        rise = max(rise, thickness + 0.05)
        if ridge_axis == "y":
            polygon_xz = [
                (-width / 2.0, 0.0),
                (0.0, rise),
                (width / 2.0, 0.0),
                (width / 2.0, -thickness),
                (0.0, rise - thickness),
                (-width / 2.0, -thickness),
            ]
            return self._extrude_polygon_along_y(depth, polygon_xz)

        polygon_yz = [
            (-depth / 2.0, 0.0),
            (0.0, rise),
            (depth / 2.0, 0.0),
            (depth / 2.0, -thickness),
            (0.0, rise - thickness),
            (-depth / 2.0, -thickness),
        ]
        return self._extrude_polygon_along_x(width, polygon_yz)

    def _create_sawtooth_roof_mesh(
        self,
        *,
        width: float,
        depth: float,
        thickness: float,
        rise: float,
        tooth_count: int,
        axis: str,
        clerestory_ratio: float,
    ) -> Mesh:
        rise = max(rise, thickness + 0.05)
        clerestory_ratio = max(0.08, min(0.85, clerestory_ratio))
        tooth_count = max(2, tooth_count)

        if axis == "y":
            span = depth
            top_polyline = self._build_sawtooth_polyline(
                span=span,
                rise=rise,
                tooth_count=tooth_count,
                clerestory_ratio=clerestory_ratio,
            )
            polygon_yz = [*top_polyline, *[(y, z - thickness) for y, z in reversed(top_polyline)]]
            return self._extrude_polygon_along_x(width, polygon_yz)

        span = width
        top_polyline = self._build_sawtooth_polyline(
            span=span,
            rise=rise,
            tooth_count=tooth_count,
            clerestory_ratio=clerestory_ratio,
        )
        polygon_xz = [*top_polyline, *[(x, z - thickness) for x, z in reversed(top_polyline)]]
        return self._extrude_polygon_along_y(depth, polygon_xz)

    def _build_sawtooth_polyline(
        self,
        *,
        span: float,
        rise: float,
        tooth_count: int,
        clerestory_ratio: float,
    ) -> list[tuple[float, float]]:
        left = -span / 2.0
        step = span / tooth_count
        rise_run = step * clerestory_ratio

        points: list[tuple[float, float]] = [(left, 0.0)]
        x = left
        for _ in range(tooth_count):
            peak_x = min(span / 2.0, x + rise_run)
            end_x = min(span / 2.0, x + step)
            if peak_x > points[-1][0] + 1e-6:
                points.append((peak_x, rise))
            if end_x > points[-1][0] + 1e-6:
                points.append((end_x, 0.0))
            x = end_x

        if points[-1][0] < span / 2.0 - 1e-6:
            points.append((span / 2.0, 0.0))
        return points

    def _extrude_polygon_along_x(self, width: float, polygon_yz: list[tuple[float, float]]) -> Mesh:
        if len(polygon_yz) < 3:
            raise ValueError("polygon_yz must contain at least three points.")

        half_w = width / 2.0
        vertices: list[tuple[float, float, float]] = []
        for y, z in polygon_yz:
            vertices.append((-half_w, y, z))
            vertices.append((half_w, y, z))

        faces: list[tuple[int, int, int]] = []
        n = len(polygon_yz)
        for i in range(n):
            j = (i + 1) % n
            li = i * 2
            ri = li + 1
            lj = j * 2
            rj = lj + 1
            faces.append((li, lj, rj))
            faces.append((li, rj, ri))

        for i in range(1, n - 1):
            faces.append((0, 2 * (i + 1), 2 * i))
            faces.append((1, 2 * i + 1, 2 * (i + 1) + 1))

        return Mesh(vertices=vertices, faces=faces)

    def _extrude_polygon_along_y(self, depth: float, polygon_xz: list[tuple[float, float]]) -> Mesh:
        if len(polygon_xz) < 3:
            raise ValueError("polygon_xz must contain at least three points.")

        half_d = depth / 2.0
        vertices: list[tuple[float, float, float]] = []
        for x, z in polygon_xz:
            vertices.append((x, -half_d, z))
            vertices.append((x, half_d, z))

        faces: list[tuple[int, int, int]] = []
        n = len(polygon_xz)
        for i in range(n):
            j = (i + 1) % n
            fi = i * 2
            bi = fi + 1
            fj = j * 2
            bj = fj + 1
            faces.append((fi, fj, bj))
            faces.append((fi, bj, bi))

        for i in range(1, n - 1):
            faces.append((0, 2 * i, 2 * (i + 1)))
            faces.append((1, 2 * (i + 1) + 1, 2 * i + 1))

        return Mesh(vertices=vertices, faces=faces)

    def _collect_shell_openings(
        self,
        scene: Scene,
        ctx: _ExteriorContext,
        *,
        opening_clearance: float,
    ) -> dict[str, list[_FacadeOpening]]:
        side_openings: dict[str, list[_FacadeOpening]] = defaultdict(list)

        for planned in self._planned_gate_openings(ctx):
            side_openings[planned.side].append(
                _FacadeOpening(
                    side=planned.side,
                    offset=planned.offset,
                    width=planned.width + opening_clearance * 2.0,
                    height=min(ctx.building_height, planned.height + opening_clearance),
                    kind=planned.kind,
                )
            )

        for opening in self._functional_room_openings(ctx):
            side_openings[opening.side].append(
                _FacadeOpening(
                    side=opening.side,
                    offset=opening.offset,
                    width=opening.width + opening_clearance * 2.0,
                    height=min(ctx.building_height, opening.height + opening_clearance),
                    kind=opening.kind,
                )
            )

        default_door_width = _to_positive_float(
            self.settings.get("door_width"),
            1.25,
            "exterior.door_width",
        )
        snap_distance = max(0.5, ctx.wall_offset + ctx.wall_thickness + opening_clearance + 0.25)
        for obj, matrix in self._iter_objects_with_world(scene):
            if obj.type != "door_opening":
                continue
            wx, wy, wz = _transform_point(matrix, (0.0, 0.0, 0.0))
            side, distance = self._nearest_envelope_side(ctx, wx, wy)
            if distance > snap_distance:
                continue

            opening_height = max(1.8, min(ctx.building_height * 0.9, wz * 2.0))
            if side in {"south", "north"}:
                opening_offset = wx - ctx.center_x
            else:
                opening_offset = wy - ctx.center_y
            side_openings[side].append(
                _FacadeOpening(
                    side=side,
                    offset=opening_offset,
                    width=default_door_width + opening_clearance * 2.0,
                    height=min(ctx.building_height, opening_height + opening_clearance),
                    kind="door",
                )
            )

        normalized: dict[str, list[_FacadeOpening]] = {
            "south": [],
            "north": [],
            "west": [],
            "east": [],
        }
        for side in ("south", "north", "west", "east"):
            span = ctx.exterior_width if side in {"south", "north"} else ctx.exterior_depth
            normalized[side] = self._normalize_side_openings(span, side_openings.get(side, []))
        return normalized

    def _planned_gate_openings(self, ctx: _ExteriorContext) -> list[_FacadeOpening]:
        gates_cfg = _to_mapping(self.settings.get("gates"), "exterior.gates")
        if not _to_bool(gates_cfg.get("enabled", True), default=True):
            return []

        side = str(gates_cfg.get("side", "south")).strip().lower() or "south"
        if side not in {"south", "north", "west", "east"}:
            side = "south"

        gate_width = _to_positive_float(gates_cfg.get("width"), 3.8, "exterior.gates.width")
        gate_height = _to_positive_float(gates_cfg.get("height"), 4.2, "exterior.gates.height")
        edge_margin = _to_non_negative_float(
            gates_cfg.get("edge_margin"),
            2.2,
            "exterior.gates.edge_margin",
        )
        gap = _to_non_negative_float(gates_cfg.get("gap"), 1.6, "exterior.gates.gap")

        span = (ctx.exterior_width if side in {"south", "north"} else ctx.exterior_depth) - edge_margin * 2.0
        max_count = max(1, int((span + gap) // max(gate_width + gap, 0.1)))

        count_raw = gates_cfg.get("count", "auto")
        if isinstance(count_raw, str) and count_raw.strip().lower() == "auto":
            estimate = max(1, int(round(ctx.interior_width / 28.0)))
            gate_count = min(max_count, estimate)
        else:
            gate_count = min(max_count, max(1, _to_non_negative_int(count_raw, 1, "exterior.gates.count")))

        max_offset = max(0.0, span / 2.0 - gate_width / 2.0)
        offsets = _symmetric_positions(gate_count, max_offset)
        return [
            _FacadeOpening(
                side=side,
                offset=offset,
                width=gate_width,
                height=min(ctx.building_height, gate_height),
                kind="gate",
            )
            for offset in offsets
        ]

    def _functional_room_openings(self, ctx: _ExteriorContext) -> list[_FacadeOpening]:
        touching_by_side = self._touching_rooms_by_side(ctx)
        gates_cfg = _to_mapping(self.settings.get("gates"), "exterior.gates")
        gate_width = _to_positive_float(gates_cfg.get("width"), 3.8, "exterior.gates.width")
        gate_height = _to_positive_float(gates_cfg.get("height"), 4.2, "exterior.gates.height")
        door_width = _to_positive_float(self.settings.get("door_width"), 1.25, "exterior.door_width")
        door_height = _to_positive_float(
            self.settings.get("door_height", 2.3),
            2.3,
            "exterior.door_height",
        )
        facade_cfg = _to_mapping(self.settings.get("facade"), "exterior.facade")
        functional_cfg = _to_mapping(
            facade_cfg.get("functional"),
            "exterior.facade.functional",
        )
        if not _to_bool(functional_cfg.get("enabled", True), default=True):
            return []

        openings: list[_FacadeOpening] = []
        gate_biomes = {"storage", "warehouse", "maintenance"}
        min_gate_segment = _to_positive_float(
            functional_cfg.get("min_gate_segment"),
            gate_width * 1.15,
            "exterior.facade.functional.min_gate_segment",
        )

        for side in ("south", "north", "west", "east"):
            for room, seg_start, seg_end in touching_by_side.get(side, []):
                segment_len = seg_end - seg_start
                if segment_len <= 0.25:
                    continue

                biome = room.biome.strip().lower()
                if biome in gate_biomes and segment_len >= min_gate_segment:
                    count = 1 if segment_len < gate_width * 2.6 else 2
                    width = min(gate_width * 1.15, segment_len * (0.72 if count == 1 else 0.42))
                    max_offset = max(0.0, segment_len / 2.0 - width / 2.0 - 0.1)
                    segment_center = (seg_start + seg_end) / 2.0
                    for delta in _symmetric_positions(count, max_offset):
                        openings.append(
                            _FacadeOpening(
                                side=side,
                                offset=segment_center + delta,
                                width=width,
                                height=max(3.4, gate_height),
                                kind="gate",
                            )
                        )
                    continue

                # Every functional facade zone gets at least one technical door.
                if segment_len >= door_width + 0.25:
                    openings.append(
                        _FacadeOpening(
                            side=side,
                            offset=(seg_start + seg_end) / 2.0,
                            width=min(door_width, segment_len * 0.7),
                            height=door_height,
                            kind="door",
                        )
                    )
        return openings

    def _nearest_envelope_side(self, ctx: _ExteriorContext, x: float, y: float) -> tuple[str, float]:
        half_w = ctx.exterior_width / 2.0
        half_d = ctx.exterior_depth / 2.0
        distances = {
            "south": abs(y - (ctx.center_y - half_d)),
            "north": abs(y - (ctx.center_y + half_d)),
            "west": abs(x - (ctx.center_x - half_w)),
            "east": abs(x - (ctx.center_x + half_w)),
        }
        side = min(distances, key=distances.get)
        return side, distances[side]

    def _normalize_side_openings(
        self,
        span: float,
        openings: list[_FacadeOpening],
    ) -> list[_FacadeOpening]:
        if not openings:
            return []

        half_span = span / 2.0
        min_edge_margin = 0.24
        prepared: list[tuple[float, float, float, str]] = []
        for opening in openings:
            width = max(0.3, opening.width)
            height = max(1.8, opening.height)
            start = opening.offset - width / 2.0
            end = opening.offset + width / 2.0
            start = max(-half_span + min_edge_margin, start)
            end = min(half_span - min_edge_margin, end)
            if end - start < 0.25:
                continue
            prepared.append((start, end, height, opening.kind))

        if not prepared:
            return []

        prepared.sort(key=lambda item: item[0])
        merged: list[tuple[float, float, float, str]] = [prepared[0]]
        for start, end, height, kind in prepared[1:]:
            prev_start, prev_end, prev_height, prev_kind = merged[-1]
            if start <= prev_end + 0.08:
                merged[-1] = (
                    prev_start,
                    max(prev_end, end),
                    max(prev_height, height),
                    "gate" if "gate" in {prev_kind, kind} else prev_kind,
                )
                continue
            merged.append((start, end, height, kind))

        normalized: list[_FacadeOpening] = []
        side = openings[0].side
        for start, end, height, kind in merged:
            normalized.append(
                _FacadeOpening(
                    side=side,
                    offset=(start + end) / 2.0,
                    width=end - start,
                    height=height,
                    kind=kind,
                )
            )
        return normalized

    def _build_side_wall_mesh(
        self,
        *,
        side: str,
        ctx: _ExteriorContext,
        openings: list[_FacadeOpening],
    ) -> Mesh:
        length = ctx.exterior_width if side in {"south", "north"} else ctx.exterior_depth
        return self._build_wall_mesh_with_openings(
            length=length,
            height=ctx.building_height,
            thickness=ctx.wall_thickness,
            openings=openings,
        )

    def _build_wall_mesh_with_openings(
        self,
        *,
        length: float,
        height: float,
        thickness: float,
        openings: list[_FacadeOpening],
    ) -> Mesh:
        if not openings:
            return create_wall(length=length, height=height, thickness=thickness)

        intervals = sorted(
            (
                (
                    opening.offset - opening.width / 2.0,
                    opening.offset + opening.width / 2.0,
                    max(1.8, min(height, opening.height)),
                )
                for opening in openings
            ),
            key=lambda item: item[0],
        )

        half = length / 2.0
        merged: list[tuple[float, float, float]] = []
        for start, end, opening_height in intervals:
            start = max(-half, start)
            end = min(half, end)
            if end - start <= 0.2:
                continue
            if not merged:
                merged.append((start, end, opening_height))
                continue
            prev_start, prev_end, prev_height = merged[-1]
            if start <= prev_end + 0.06:
                merged[-1] = (prev_start, max(prev_end, end), max(prev_height, opening_height))
            else:
                merged.append((start, end, opening_height))

        if not merged:
            return create_wall(length=length, height=height, thickness=thickness)

        parts: list[Mesh] = []
        cursor = -half
        min_width = 0.06
        for start, end, opening_height in merged:
            if start - cursor > min_width:
                parts.append(
                    self._wall_slice_mesh(
                        width=start - cursor,
                        total_height=height,
                        thickness=thickness,
                        center_x=(cursor + start) / 2.0,
                        bottom=0.0,
                        top=height,
                    )
                )

            if height - opening_height > 0.06:
                parts.append(
                    self._wall_slice_mesh(
                        width=end - start,
                        total_height=height,
                        thickness=thickness,
                        center_x=(start + end) / 2.0,
                        bottom=opening_height,
                        top=height,
                    )
                )
            cursor = end

        if half - cursor > min_width:
            parts.append(
                self._wall_slice_mesh(
                    width=half - cursor,
                    total_height=height,
                    thickness=thickness,
                    center_x=(cursor + half) / 2.0,
                    bottom=0.0,
                    top=height,
                )
            )

        if not parts:
            return create_wall(length=length, height=height, thickness=thickness)

        merged_mesh = Mesh()
        for part in parts:
            merged_mesh.merge(part)
        return merged_mesh

    def _wall_slice_mesh(
        self,
        *,
        width: float,
        total_height: float,
        thickness: float,
        center_x: float,
        bottom: float,
        top: float,
    ) -> Mesh:
        segment_height = max(0.05, top - bottom)
        mesh = create_box(width=width, height=segment_height, depth=thickness)
        local_z = ((bottom + top) / 2.0) - total_height / 2.0
        mesh.vertices = [
            (x + center_x, y, z + local_z)
            for x, y, z in mesh.vertices
        ]
        return mesh

    def _solid_intervals(
        self,
        span: float,
        openings: list[_FacadeOpening],
    ) -> list[tuple[float, float]]:
        half = span / 2.0
        if not openings:
            return [(-half, half)]

        intervals = sorted(
            (
                (
                    opening.offset - opening.width / 2.0,
                    opening.offset + opening.width / 2.0,
                )
                for opening in openings
            ),
            key=lambda item: item[0],
        )
        solids: list[tuple[float, float]] = []
        cursor = -half
        for start, end in intervals:
            start = max(-half, start)
            end = min(half, end)
            if start > cursor + 0.05:
                solids.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < half - 0.05:
            solids.append((cursor, half))
        return solids

    def _segment_facade_height(
        self,
        ctx: _ExteriorContext,
        *,
        side: str,
        segment_offset: float,
    ) -> float:
        if not ctx.room_volumes:
            return ctx.building_height

        target_heights: list[float] = []
        if side in {"south", "north"}:
            x = ctx.center_x + segment_offset
            side_y = ctx.center_y - ctx.interior_depth / 2.0 if side == "south" else ctx.center_y + ctx.interior_depth / 2.0
            for room in ctx.room_volumes:
                if x < room.min_x - 0.3 or x > room.max_x + 0.3:
                    continue
                edge_distance = abs(room.min_y - side_y) if side == "south" else abs(room.max_y - side_y)
                if edge_distance > max(2.0, ctx.wall_offset + 1.2):
                    continue
                target_heights.append(room.max_z - ctx.base_z + 0.15)
        else:
            y = ctx.center_y + segment_offset
            side_x = ctx.center_x - ctx.interior_width / 2.0 if side == "west" else ctx.center_x + ctx.interior_width / 2.0
            for room in ctx.room_volumes:
                if y < room.min_y - 0.3 or y > room.max_y + 0.3:
                    continue
                edge_distance = abs(room.min_x - side_x) if side == "west" else abs(room.max_x - side_x)
                if edge_distance > max(2.0, ctx.wall_offset + 1.2):
                    continue
                target_heights.append(room.max_z - ctx.base_z + 0.15)

        if not target_heights:
            return ctx.building_height
        return max(2.5, min(ctx.building_height, max(target_heights)))

    def _build_opening_frame(
        self,
        *,
        ctx: _ExteriorContext,
        opening: _FacadeOpening,
        wall_x: float,
        wall_y: float,
        frame_thickness: float,
        frame_depth: float,
        frame_gap: float,
        frame_index: int,
    ) -> list[SceneObject]:
        objects: list[SceneObject] = []
        if opening.side in {"south", "north"}:
            sign = -1.0 if opening.side == "south" else 1.0
            y = ctx.center_y + sign * (
                wall_y + ctx.wall_thickness / 2.0 + frame_gap + frame_depth / 2.0
            )
            left_x = ctx.center_x + opening.offset - opening.width / 2.0 + frame_thickness / 2.0
            right_x = ctx.center_x + opening.offset + opening.width / 2.0 - frame_thickness / 2.0
            jamb_z = ctx.base_z + opening.height / 2.0
            lintel_z = ctx.base_z + opening.height + frame_thickness / 2.0

            objects.append(
                SceneObject(
                    id=f"exterior_gate_frame_{opening.side}_{frame_index}_left",
                    type="exterior_gate_frame",
                    transform=Transform(position=(left_x, y, jamb_z)),
                    mesh=create_box(width=frame_thickness, height=opening.height, depth=frame_depth),
                )
            )
            objects.append(
                SceneObject(
                    id=f"exterior_gate_frame_{opening.side}_{frame_index}_right",
                    type="exterior_gate_frame",
                    transform=Transform(position=(right_x, y, jamb_z)),
                    mesh=create_box(width=frame_thickness, height=opening.height, depth=frame_depth),
                )
            )
            objects.append(
                SceneObject(
                    id=f"exterior_gate_frame_{opening.side}_{frame_index}_top",
                    type="exterior_gate_frame",
                    transform=Transform(position=(ctx.center_x + opening.offset, y, lintel_z)),
                    mesh=create_box(width=opening.width, height=frame_thickness, depth=frame_depth),
                )
            )
            return objects

        sign = -1.0 if opening.side == "west" else 1.0
        x = ctx.center_x + sign * (
            wall_x + ctx.wall_thickness / 2.0 + frame_gap + frame_depth / 2.0
        )
        low_y = ctx.center_y + opening.offset - opening.width / 2.0 + frame_thickness / 2.0
        high_y = ctx.center_y + opening.offset + opening.width / 2.0 - frame_thickness / 2.0
        jamb_z = ctx.base_z + opening.height / 2.0
        lintel_z = ctx.base_z + opening.height + frame_thickness / 2.0

        objects.append(
            SceneObject(
                id=f"exterior_gate_frame_{opening.side}_{frame_index}_low",
                type="exterior_gate_frame",
                transform=Transform(position=(x, low_y, jamb_z)),
                mesh=create_box(width=frame_depth, height=opening.height, depth=frame_thickness),
            )
        )
        objects.append(
            SceneObject(
                id=f"exterior_gate_frame_{opening.side}_{frame_index}_high",
                type="exterior_gate_frame",
                transform=Transform(position=(x, high_y, jamb_z)),
                mesh=create_box(width=frame_depth, height=opening.height, depth=frame_thickness),
            )
        )
        objects.append(
            SceneObject(
                id=f"exterior_gate_frame_{opening.side}_{frame_index}_top",
                type="exterior_gate_frame",
                transform=Transform(position=(x, ctx.center_y + opening.offset, lintel_z)),
                mesh=create_box(width=frame_depth, height=frame_thickness, depth=opening.width),
            )
        )
        return objects

    def _build_gate_canopy(
        self,
        *,
        ctx: _ExteriorContext,
        opening: _FacadeOpening,
        wall_x: float,
        wall_y: float,
        canopy_depth: float,
        canopy_thickness: float,
        canopy_width_factor: float,
        z_offset: float,
        canopy_index: int,
    ) -> SceneObject | None:
        if opening.kind != "gate":
            return None

        canopy_z = ctx.base_z + opening.height + z_offset + canopy_thickness / 2.0
        if opening.side in {"south", "north"}:
            sign = -1.0 if opening.side == "south" else 1.0
            x = ctx.center_x + opening.offset
            y = ctx.center_y + sign * (wall_y + ctx.wall_thickness / 2.0 + canopy_depth / 2.0)
            mesh = create_box(
                width=max(0.3, opening.width * canopy_width_factor),
                height=canopy_thickness,
                depth=canopy_depth,
            )
        else:
            sign = -1.0 if opening.side == "west" else 1.0
            x = ctx.center_x + sign * (wall_x + ctx.wall_thickness / 2.0 + canopy_depth / 2.0)
            y = ctx.center_y + opening.offset
            mesh = create_box(
                width=canopy_depth,
                height=canopy_thickness,
                depth=max(0.3, opening.width * canopy_width_factor),
            )
        return SceneObject(
            id=f"exterior_canopy_{canopy_index}",
            type="exterior_canopy",
            transform=Transform(position=(x, y, canopy_z)),
            mesh=mesh,
        )

    def _apply_exterior_rules(self, scene: Scene, *, allow_rebuild: bool) -> bool:
        ctx = self._ensure_context(scene)
        rules_cfg = _to_mapping(self.settings.get("rules"), "exterior.rules")
        if not _to_bool(rules_cfg.get("enabled", True), default=True):
            return False

        auto_fix = _to_bool(rules_cfg.get("auto_fix", True), default=True)
        shell_rule = ShellContainmentRule(
            margin=_to_non_negative_float(
                rules_cfg.get("shell_margin"),
                0.05,
                "exterior.rules.shell_margin",
            )
        )
        collision_rule = NoCollisionRule(
            padding=_to_non_negative_float(
                rules_cfg.get("collision_padding"),
                0.06,
                "exterior.rules.collision_padding",
            ),
            correction_step=_to_positive_float(
                rules_cfg.get("collision_step"),
                0.45,
                "exterior.rules.collision_step",
            ),
        )
        access_rule = AccessRule(
            min_entrances=max(
                1,
                _to_non_negative_int(
                    rules_cfg.get("min_entrances"),
                    1,
                    "exterior.rules.min_entrances",
                ),
            ),
            clearance=_to_non_negative_float(
                rules_cfg.get("access_clearance"),
                1.6,
                "exterior.rules.access_clearance",
            ),
        )
        pipe_rule = PipeContinuityRule(
            tolerance=_to_non_negative_float(
                rules_cfg.get("pipe_tolerance"),
                0.35,
                "exterior.rules.pipe_tolerance",
            ),
            link_thickness=_to_positive_float(
                rules_cfg.get("pipe_link_thickness"),
                0.18,
                "exterior.rules.pipe_link_thickness",
            ),
        )
        ground_types_raw = rules_cfg.get("ground_contact_types")
        if isinstance(ground_types_raw, (list, tuple)):
            ground_types = frozenset(str(item).strip() for item in ground_types_raw if str(item).strip())
        else:
            ground_types = frozenset(
                {
                    "exterior_transformer",
                    "exterior_repair_pad",
                    "exterior_door",
                    "exterior_gate",
                }
            )
        ground_rule = GroundContactRule(
            tolerance=_to_non_negative_float(
                rules_cfg.get("ground_tolerance"),
                0.03,
                "exterior.rules.ground_tolerance",
            ),
            supported_types=ground_types,
        )

        if self._violates_shell_containment(ctx, shell_rule):
            if auto_fix and allow_rebuild:
                expanded_ctx = self._expand_context_for_shell_containment(ctx, shell_rule)
                if expanded_ctx is not None:
                    self._context = expanded_ctx
                    return True

        group = self._ensure_group(scene)
        if auto_fix:
            self._auto_fix_access(scene, group, ctx, access_rule)
            self._auto_fix_pipe_continuity(group, ctx, pipe_rule)
            self._auto_fix_ground_contact(group, ctx, ground_rule)
            self._auto_fix_no_collisions(group, ctx, collision_rule)
            self._auto_fix_access(scene, group, ctx, access_rule)
        return False

    def _violates_shell_containment(self, ctx: _ExteriorContext, rule: ShellContainmentRule) -> bool:
        if not ctx.room_volumes:
            return False
        half_w = ctx.exterior_width / 2.0 - rule.margin
        half_d = ctx.exterior_depth / 2.0 - rule.margin
        max_height = ctx.base_z + ctx.building_height - rule.margin
        min_x = ctx.center_x - half_w
        max_x = ctx.center_x + half_w
        min_y = ctx.center_y - half_d
        max_y = ctx.center_y + half_d
        for room in ctx.room_volumes:
            if room.min_x < min_x or room.max_x > max_x:
                return True
            if room.min_y < min_y or room.max_y > max_y:
                return True
            if room.max_z > max_height:
                return True
        return False

    def _expand_context_for_shell_containment(
        self,
        ctx: _ExteriorContext,
        rule: ShellContainmentRule,
    ) -> _ExteriorContext | None:
        if not ctx.room_volumes:
            return None
        req_half_w = max(
            max(abs(room.min_x - ctx.center_x), abs(room.max_x - ctx.center_x))
            for room in ctx.room_volumes
        ) + rule.margin + ctx.wall_thickness / 2.0
        req_half_d = max(
            max(abs(room.min_y - ctx.center_y), abs(room.max_y - ctx.center_y))
            for room in ctx.room_volumes
        ) + rule.margin + ctx.wall_thickness / 2.0
        req_height = max(room.max_z for room in ctx.room_volumes) - ctx.base_z + rule.margin

        new_exterior_w = max(ctx.exterior_width, req_half_w * 2.0)
        new_exterior_d = max(ctx.exterior_depth, req_half_d * 2.0)
        new_building_h = max(ctx.building_height, req_height)
        if (
            abs(new_exterior_w - ctx.exterior_width) <= 1e-9
            and abs(new_exterior_d - ctx.exterior_depth) <= 1e-9
            and abs(new_building_h - ctx.building_height) <= 1e-9
        ):
            return None
        return replace(
            ctx,
            exterior_width=new_exterior_w,
            exterior_depth=new_exterior_d,
            building_height=new_building_h,
        )

    def _auto_fix_access(
        self,
        scene: Scene,
        group: SceneObject,
        ctx: _ExteriorContext,
        rule: AccessRule,
    ) -> None:
        entrances = [child for child in group.children if child.type in {"exterior_door", "exterior_gate"}]
        if len(entrances) < rule.min_entrances:
            opening = _FacadeOpening(
                side="south",
                offset=0.0,
                width=_to_positive_float(self.settings.get("door_width"), 1.25, "exterior.door_width"),
                height=_to_positive_float(self.settings.get("door_height", 2.3), 2.3, "exterior.door_height"),
                kind="door",
            )
            openings = list(self._openings_by_side.get("south", []))
            openings.append(opening)
            self._openings_by_side["south"] = self._normalize_side_openings(ctx.exterior_width, openings)
            south_wall = scene.get_object("exterior_wall_south")
            if south_wall is not None:
                south_wall.mesh = self._build_side_wall_mesh(
                    side="south",
                    ctx=ctx,
                    openings=self._openings_by_side["south"],
                )
            half_wall_x = ctx.exterior_width / 2.0 - ctx.wall_thickness / 2.0
            half_wall_y = ctx.exterior_depth / 2.0 - ctx.wall_thickness / 2.0
            door_index = len([child for child in group.children if child.type == "exterior_door"]) + 1
            door_obj = self._build_door_or_gate_leaf(
                ctx=ctx,
                opening=opening,
                wall_x=half_wall_x,
                wall_y=half_wall_y,
                depth=0.08,
                kind="door",
                index=door_index,
            )
            if door_obj is not None:
                self._upsert_child(group, door_obj)
                entrances.append(door_obj)
            marker = SceneObject(
                id=f"exterior_opening_door_south_auto_{door_index}",
                type="exterior_opening",
                transform=Transform(
                    position=(
                        ctx.center_x + opening.offset,
                        ctx.center_y - half_wall_y,
                        ctx.base_z + opening.height / 2.0,
                    )
                ),
                mesh=None,
            )
            self._upsert_child(group, marker)

        blocker_types = {
            "exterior_transformer",
            "exterior_repair_pad",
            "exterior_chimney",
        }
        for entrance in [child for child in group.children if child.type in {"exterior_door", "exterior_gate"}]:
            side = self._infer_opening_side(entrance.id)
            if side is None:
                continue
            e_bounds = self._child_world_bounds(group, entrance)
            if e_bounds is None:
                continue
            e_center_x = (e_bounds.min_x + e_bounds.max_x) / 2.0
            e_center_y = (e_bounds.min_y + e_bounds.max_y) / 2.0
            e_width = max(0.2, e_bounds.max_x - e_bounds.min_x)
            e_depth = max(0.2, e_bounds.max_y - e_bounds.min_y)
            for obj in group.children:
                if obj.type not in blocker_types or obj.mesh is None:
                    continue
                o_bounds = self._child_world_bounds(group, obj)
                if o_bounds is None:
                    continue
                o_center_x = (o_bounds.min_x + o_bounds.max_x) / 2.0
                o_center_y = (o_bounds.min_y + o_bounds.max_y) / 2.0
                o_width = max(0.2, o_bounds.max_x - o_bounds.min_x)
                o_depth = max(0.2, o_bounds.max_y - o_bounds.min_y)
                if side in {"south", "north"}:
                    if abs(o_center_x - e_center_x) > (e_width + o_width) / 2.0 + 0.15:
                        continue
                    sign = -1.0 if side == "south" else 1.0
                    forward = (o_center_y - e_center_y) * sign
                    min_forward = (e_depth + o_depth) / 2.0 + rule.clearance
                    if forward < min_forward:
                        delta = min_forward - forward + 0.05
                        ox, oy, oz = obj.transform.position
                        obj.transform.position = (ox, oy + sign * delta, oz)
                else:
                    if abs(o_center_y - e_center_y) > (e_depth + o_depth) / 2.0 + 0.15:
                        continue
                    sign = -1.0 if side == "west" else 1.0
                    forward = (o_center_x - e_center_x) * sign
                    min_forward = (e_width + o_width) / 2.0 + rule.clearance
                    if forward < min_forward:
                        delta = min_forward - forward + 0.05
                        ox, oy, oz = obj.transform.position
                        obj.transform.position = (ox + sign * delta, oy, oz)

    def _auto_fix_pipe_continuity(
        self,
        group: SceneObject,
        ctx: _ExteriorContext,
        rule: PipeContinuityRule,
    ) -> None:
        sources = [obj for obj in group.children if obj.type == "exterior_boiler_pipe" and obj.mesh is not None]
        targets = [
            obj
            for obj in group.children
            if obj.type in {"exterior_roof_pipe", "exterior_chimney", "exterior_pipe_link"} and obj.mesh is not None
        ]
        link_index = len([obj for obj in group.children if obj.type == "exterior_pipe_link"]) + 1
        for source in sources:
            s_bounds = self._child_world_bounds(group, source)
            if s_bounds is None:
                continue
            connected = False
            for target in targets:
                t_bounds = self._child_world_bounds(group, target)
                if t_bounds is None:
                    continue
                if self._bounds_overlap(s_bounds, t_bounds, rule.tolerance):
                    connected = True
                    break
            if connected:
                continue

            source_pos = source.transform.position
            nearest = self._nearest_pipe_target(group, source_pos)
            if nearest is None:
                continue
            target_pos = nearest.transform.position
            segments = self._build_pipe_connection_segments(
                source_pos=source_pos,
                target_pos=target_pos,
                thickness=rule.link_thickness,
                start_index=link_index,
            )
            for segment in segments:
                self._upsert_child(group, segment)
                targets.append(segment)
                link_index += 1

    def _auto_fix_ground_contact(
        self,
        group: SceneObject,
        ctx: _ExteriorContext,
        rule: GroundContactRule,
    ) -> None:
        for obj in group.children:
            if obj.type not in rule.supported_types or obj.mesh is None:
                continue
            bounds = self._child_world_bounds(group, obj)
            if bounds is None:
                continue
            delta = ctx.base_z - bounds.min_z
            if abs(delta) <= rule.tolerance:
                continue
            px, py, pz = obj.transform.position
            obj.transform.position = (px, py, pz + delta)

    def _auto_fix_no_collisions(
        self,
        group: SceneObject,
        ctx: _ExteriorContext,
        rule: NoCollisionRule,
    ) -> None:
        collision_types = {
            "exterior_transformer",
            "exterior_repair_pad",
            "exterior_chimney",
            "exterior_boiler_pipe",
            "exterior_roof_pipe",
            "exterior_technical_block",
            "exterior_vent_shaft",
            "exterior_lab_vent",
            "exterior_lab_technical_block",
            "exterior_pipe_link",
            "exterior_power_cable",
        }
        candidates = [obj for obj in group.children if obj.type in collision_types and obj.mesh is not None]
        for left_idx in range(len(candidates)):
            for right_idx in range(left_idx + 1, len(candidates)):
                right = candidates[right_idx]
                attempts = 0
                while attempts < 10:
                    left_bounds = self._child_world_bounds(group, candidates[left_idx])
                    right_bounds = self._child_world_bounds(group, right)
                    if left_bounds is None or right_bounds is None:
                        break
                    if not self._bounds_overlap(left_bounds, right_bounds, rule.padding):
                        break
                    self._move_outward_from_building(ctx, right, rule.correction_step)
                    attempts += 1

    def _build_pipe_connection_segments(
        self,
        *,
        source_pos: tuple[float, float, float],
        target_pos: tuple[float, float, float],
        thickness: float,
        start_index: int,
    ) -> list[SceneObject]:
        sx, sy, sz = source_pos
        tx, ty, tz = target_pos
        objects: list[SceneObject] = []
        if abs(tx - sx) > 0.08:
            objects.append(
                SceneObject(
                    id=f"exterior_pipe_link_{start_index + len(objects)}",
                    type="exterior_pipe_link",
                    transform=Transform(position=((sx + tx) / 2.0, sy, sz)),
                    mesh=create_box(width=abs(tx - sx), height=thickness, depth=thickness),
                )
            )
        if abs(ty - sy) > 0.08:
            objects.append(
                SceneObject(
                    id=f"exterior_pipe_link_{start_index + len(objects)}",
                    type="exterior_pipe_link",
                    transform=Transform(position=(tx, (sy + ty) / 2.0, sz)),
                    mesh=create_box(width=thickness, height=thickness, depth=abs(ty - sy)),
                )
            )
        if abs(tz - sz) > 0.08:
            objects.append(
                SceneObject(
                    id=f"exterior_pipe_link_{start_index + len(objects)}",
                    type="exterior_pipe_link",
                    transform=Transform(position=(tx, ty, (sz + tz) / 2.0)),
                    mesh=create_box(width=thickness, height=abs(tz - sz), depth=thickness),
                )
            )
        return objects

    def _nearest_pipe_target(
        self,
        group: SceneObject,
        source_pos: tuple[float, float, float],
    ) -> SceneObject | None:
        sx, sy, sz = source_pos
        nearest_obj: SceneObject | None = None
        nearest_distance = float("inf")
        for candidate in group.children:
            if candidate.type not in {"exterior_roof_pipe", "exterior_chimney"}:
                continue
            cx, cy, cz = candidate.transform.position
            distance = (cx - sx) ** 2 + (cy - sy) ** 2 + (cz - sz) ** 2
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_obj = candidate
        return nearest_obj

    def _infer_opening_side(self, object_id: str) -> str | None:
        lowered = object_id.lower()
        for side in ("south", "north", "west", "east"):
            if f"_{side}_" in lowered:
                return side
        return None

    def _bounds_overlap(self, left: _Bounds, right: _Bounds, padding: float) -> bool:
        return (
            left.max_x + padding > right.min_x
            and right.max_x + padding > left.min_x
            and left.max_y + padding > right.min_y
            and right.max_y + padding > left.min_y
            and left.max_z + padding > right.min_z
            and right.max_z + padding > left.min_z
        )

    def _move_outward_from_building(
        self,
        ctx: _ExteriorContext,
        obj: SceneObject,
        delta: float,
    ) -> None:
        px, py, pz = obj.transform.position
        side, _ = self._nearest_envelope_side(ctx, px, py)
        if side == "south":
            obj.transform.position = (px, py - delta, pz)
        elif side == "north":
            obj.transform.position = (px, py + delta, pz)
        elif side == "west":
            obj.transform.position = (px - delta, py, pz)
        else:
            obj.transform.position = (px + delta, py, pz)

    def _child_world_bounds(self, group: SceneObject, child: SceneObject) -> _Bounds | None:
        group_matrix = _local_matrix(group.transform)
        child_world = _mat_mul(group_matrix, _local_matrix(child.transform))
        minima_x: list[float] = []
        maxima_x: list[float] = []
        minima_y: list[float] = []
        maxima_y: list[float] = []
        minima_z: list[float] = []
        maxima_z: list[float] = []

        stack: list[tuple[SceneObject, Matrix4]] = [(child, child_world)]
        while stack:
            node, world = stack.pop()
            if node.mesh is not None:
                for vx, vy, vz in node.mesh.vertices:
                    wx, wy, wz = _transform_point(world, (vx, vy, vz))
                    minima_x.append(wx)
                    maxima_x.append(wx)
                    minima_y.append(wy)
                    maxima_y.append(wy)
                    minima_z.append(wz)
                    maxima_z.append(wz)
            for nested in reversed(node.children):
                stack.append((nested, _mat_mul(world, _local_matrix(nested.transform))))

        if not minima_x:
            return None
        return _Bounds(
            min_x=min(minima_x),
            max_x=max(maxima_x),
            min_y=min(minima_y),
            max_y=max(maxima_y),
            min_z=min(minima_z),
            max_z=max(maxima_z),
        )

    def _ensure_group(self, scene: Scene) -> SceneObject:
        group = scene.get_object(self.group_id)
        if group is None:
            group = SceneObject(id=self.group_id, type="exterior_group")
            scene.add_object(group)
        return group

    def _ensure_context(self, scene: Scene) -> _ExteriorContext:
        if self._context is None:
            self._context = self._resolve_context(
                scene,
                interior_width=None,
                interior_depth=None,
                interior_max_height=None,
            )
        return self._context

    def _resolve_context(
        self,
        scene: Scene,
        *,
        interior_width: float | None,
        interior_depth: float | None,
        interior_max_height: float | None,
    ) -> _ExteriorContext:
        room_volumes = tuple(self._measure_room_volumes(scene))
        measured = self._bounds_from_room_volumes(room_volumes)
        if measured is None:
            measured = self._measure_interior_bounds(scene)

        if measured is not None:
            center_x = measured.center_x
            center_y = measured.center_y
            base_z = min(0.0, measured.min_z)
            measured_width = measured.width
            measured_depth = measured.depth
            measured_height = max(0.01, measured.max_z - base_z)
        else:
            center_x = 0.0
            center_y = 0.0
            base_z = 0.0
            measured_width = 20.0
            measured_depth = 20.0
            measured_height = 6.0

        width_hint = float(interior_width) if interior_width is not None else None
        depth_hint = float(interior_depth) if interior_depth is not None else None
        height_hint = float(interior_max_height) if interior_max_height is not None else None

        interior_w = max(measured_width, width_hint if width_hint is not None else measured_width)
        interior_d = max(measured_depth, depth_hint if depth_hint is not None else measured_depth)
        interior_h = max(measured_height, height_hint if height_hint is not None else measured_height)
        building_cfg = _to_mapping(self.settings.get("building"), "exterior.building")

        wall_thickness = _to_positive_float(
            self.settings.get("wall_thickness"),
            0.34,
            "exterior.wall_thickness",
        )
        wall_offset_raw = building_cfg.get("margin", self.settings.get("wall_offset"))
        wall_offset = _resolve_ranged_float(
            wall_offset_raw,
            default=0.12,
            label="exterior.building.margin",
            seed=self.seed,
            salt="building.margin",
        )
        if wall_offset < 0.0:
            raise ValueError("exterior.building.margin must be >= 0.")
        wall_headroom = _to_non_negative_float(
            self.settings.get("wall_headroom"),
            0.25,
            "exterior.wall_headroom",
        )
        height_variation = _resolve_ranged_float(
            building_cfg.get("height_variation"),
            default=0.0,
            label="exterior.building.height_variation",
            seed=self.seed,
            salt="building.height_variation",
        )
        if height_variation < 0.0:
            raise ValueError("exterior.building.height_variation must be >= 0.")

        exterior_w = interior_w + 2.0 * (wall_offset + wall_thickness)
        exterior_d = interior_d + 2.0 * (wall_offset + wall_thickness)
        building_h = interior_h + wall_headroom + height_variation

        return _ExteriorContext(
            center_x=center_x,
            center_y=center_y,
            base_z=base_z,
            interior_width=interior_w,
            interior_depth=interior_d,
            interior_height=interior_h,
            exterior_width=exterior_w,
            exterior_depth=exterior_d,
            building_height=building_h,
            wall_thickness=wall_thickness,
            wall_offset=wall_offset,
            room_volumes=room_volumes,
        )

    def _measure_room_volumes(self, scene: Scene) -> list[_RoomVolume]:
        volumes: list[_RoomVolume] = []
        for root in scene.objects:
            if not root.type.strip().lower().startswith("room_"):
                continue
            bounds = self._measure_subtree_bounds(root)
            if bounds is None:
                continue
            biome = root.type.strip().lower()[len("room_"):] or "generic"
            volumes.append(
                _RoomVolume(
                    room_id=root.id,
                    biome=biome,
                    min_x=bounds.min_x,
                    max_x=bounds.max_x,
                    min_y=bounds.min_y,
                    max_y=bounds.max_y,
                    min_z=bounds.min_z,
                    max_z=bounds.max_z,
                )
            )
        return volumes

    def _bounds_from_room_volumes(self, volumes: tuple[_RoomVolume, ...]) -> _Bounds | None:
        if not volumes:
            return None
        return _Bounds(
            min_x=min(room.min_x for room in volumes),
            max_x=max(room.max_x for room in volumes),
            min_y=min(room.min_y for room in volumes),
            max_y=max(room.max_y for room in volumes),
            min_z=min(room.min_z for room in volumes),
            max_z=max(room.max_z for room in volumes),
        )

    def _measure_subtree_bounds(self, root: SceneObject) -> _Bounds | None:
        minima_x: list[float] = []
        maxima_x: list[float] = []
        minima_y: list[float] = []
        maxima_y: list[float] = []
        minima_z: list[float] = []
        maxima_z: list[float] = []

        stack: list[tuple[SceneObject, Matrix4]] = [(root, _identity_matrix())]
        while stack:
            obj, parent_matrix = stack.pop()
            local = _local_matrix(obj.transform)
            world = _mat_mul(parent_matrix, local)

            if obj.mesh is not None:
                for vx, vy, vz in obj.mesh.vertices:
                    wx, wy, wz = _transform_point(world, (vx, vy, vz))
                    minima_x.append(wx)
                    maxima_x.append(wx)
                    minima_y.append(wy)
                    maxima_y.append(wy)
                    minima_z.append(wz)
                    maxima_z.append(wz)

            for child in reversed(obj.children):
                stack.append((child, world))

        if not minima_x:
            return None
        return _Bounds(
            min_x=min(minima_x),
            max_x=max(maxima_x),
            min_y=min(minima_y),
            max_y=max(maxima_y),
            min_z=min(minima_z),
            max_z=max(maxima_z),
        )

    def _measure_interior_bounds(self, scene: Scene) -> _Bounds | None:
        minima: list[float] = []
        maxima: list[float] = []
        minima_y: list[float] = []
        maxima_y: list[float] = []
        minima_z: list[float] = []
        maxima_z: list[float] = []

        for obj, matrix in self._iter_objects_with_world(scene):
            if obj.mesh is None:
                continue
            if not self._is_structural(obj):
                continue
            for vx, vy, vz in obj.mesh.vertices:
                wx, wy, wz = _transform_point(matrix, (vx, vy, vz))
                minima.append(wx)
                maxima.append(wx)
                minima_y.append(wy)
                maxima_y.append(wy)
                minima_z.append(wz)
                maxima_z.append(wz)

        if not minima:
            return None

        return _Bounds(
            min_x=min(minima),
            max_x=max(maxima),
            min_y=min(minima_y),
            max_y=max(maxima_y),
            min_z=min(minima_z),
            max_z=max(maxima_z),
        )

    def _is_structural(self, obj: SceneObject) -> bool:
        object_type = obj.type.strip().lower()
        if object_type.startswith("exterior_") or object_type.startswith("site_"):
            return False
        if object_type in _STRUCTURAL_TYPES:
            return True
        return object_type.startswith("corridor_") or object_type.startswith("shell_")

    def _iter_objects_with_world(self, scene: Scene) -> Iterable[tuple[SceneObject, Matrix4]]:
        identity = _identity_matrix()
        stack: list[tuple[SceneObject, Matrix4]] = [(root, identity) for root in reversed(scene.objects)]
        while stack:
            obj, parent_matrix = stack.pop()
            local = _local_matrix(obj.transform)
            world = _mat_mul(parent_matrix, local)
            yield obj, world
            for child in reversed(obj.children):
                stack.append((child, world))

    def _upsert_child(self, group: SceneObject, child: SceneObject) -> None:
        group.remove_child(child.id)
        group.add_child(child)
