from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..parametric.infrastructure_primitives import create_junction_node, create_vertical_drop
from ..parametric.primitives import create_beam, create_box, create_column
from ..scene.scene_graph import Scene, SceneObject, Transform


def _to_mapping(value: object, label: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    return {str(k): v for k, v in value.items()}


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


def _to_positive_int(value: object, default: int, label: str) -> int:
    if value is None:
        return default
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label} must be > 0.")
    return number


def _to_float_range(
    value: object,
    default: tuple[float, float],
    label: str,
    *,
    allow_zero: bool = False,
) -> tuple[float, float]:
    if value is None:
        minimum, maximum = default
    elif isinstance(value, Mapping):
        if "min" not in value or "max" not in value:
            raise ValueError(f"{label} mapping must contain 'min' and 'max'.")
        minimum = float(value["min"])
        maximum = float(value["max"])
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        minimum = float(value[0])
        maximum = float(value[1])
    else:
        scalar = float(value)
        minimum = scalar
        maximum = scalar

    if minimum > maximum:
        minimum, maximum = maximum, minimum
    if allow_zero:
        if minimum < 0.0:
            raise ValueError(f"{label} minimum must be >= 0.")
    else:
        if minimum <= 0.0:
            raise ValueError(f"{label} minimum must be > 0.")
    return (minimum, maximum)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _symmetric_offsets(count: int, max_offset: float) -> list[float]:
    if count <= 1:
        return [0.0]
    if max_offset <= 1e-9:
        return [0.0 for _ in range(count)]
    step = (2.0 * max_offset) / (count - 1)
    return [(-max_offset + step * index) for index in range(count)]


@dataclass(frozen=True)
class _RoomGeometry:
    center_x: float
    center_y: float
    width: float
    depth: float
    height: float


@dataclass(frozen=True)
class _TraySegment:
    start_x: float
    start_y: float
    end_x: float
    end_y: float


@dataclass(frozen=True)
class _TrayObjectSegment:
    obj: SceneObject
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    along_x: bool
    length: float
    width: float
    thickness: float


@dataclass(frozen=True)
class _ObstacleVolume:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    top_z: float


@dataclass(frozen=True)
class ClearanceRule:
    min_clearance: float

    def required_tray_z(self, obstacle_top_z: float, tray_half_thickness: float) -> float:
        return obstacle_top_z + self.min_clearance + tray_half_thickness


@dataclass(frozen=True)
class HeightRule:
    min_height: float
    top_margin: float

    def clamp_center_z(self, z: float, room_height: float) -> float:
        max_center = max(self.min_height, room_height - self.top_margin)
        return _clamp(z, self.min_height, max_center)

    def max_center(self, room_height: float) -> float:
        return max(self.min_height, room_height - self.top_margin)


@dataclass(frozen=True)
class ConnectivityRule:
    require_vertical_drop: bool = True

    def is_connected(
        self,
        room_bounds: tuple[float, float, float, float],
        drop_points: list[tuple[float, float]],
        link_points: list[tuple[float, float]],
    ) -> bool:
        min_x, max_x, min_y, max_y = room_bounds
        for x, y in drop_points:
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return True
        if not self.require_vertical_drop:
            return bool(link_points)
        for x, y in link_points:
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return True
        return False


@dataclass(frozen=True)
class SupportRule:
    spacing: float
    long_segment_length: float
    tolerance: float = 0.22

    def needs_supports(self, length: float) -> bool:
        return length >= self.long_segment_length


@dataclass(frozen=True)
class CollisionAvoidanceRule:
    min_spacing: float
    xy_padding: float = 0.03

    def requires_separation(self, z_a: float, z_b: float) -> bool:
        return abs(z_a - z_b) + 1e-9 < self.min_spacing


_DEFAULT_KEY_OBJECT_TYPES = {
    "electrical_cabinet",
    "boiler_unit",
    "machine",
    "machine_part",
    "transformer_unit",
    "workbench",
    "lab_bench",
    "control_panel",
    "console",
    "rack",
    "conveyor",
}

_STRUCTURAL_OBJECT_TYPES = {
    "floor",
    "ceiling",
    "wall",
    "door_opening",
    "window_opening",
}

_OBJECT_CONNECTION_EXACT_TYPES = {
    "electrical_cabinet",
    "desk",
    "boiler_unit",
}

_OBJECT_CONNECTION_KEYWORDS = {
    "cabinet",
    "desk",
    "table",
    "boiler",
    "шкаф",
    "стол",
    "котел",
    "котёл",
}

_OBJECT_PIPE_CONNECTION_KEYWORDS = {
    "boiler",
    "котел",
    "котёл",
}

_OBJECT_PIPE_CONNECTION_EXACT_TYPES = {
    "boiler_unit",
    "tank",
    "heat_exchanger",
    "chimney",
    "pump",
}

_BIOME_CONNECTION_TARGET_TYPES: dict[str, set[str]] = {
    "boiler": {
        "boiler_unit",
        "tank",
        "pump",
        "heat_exchanger",
        "valve",
        "pipe",
        "pipe_segment",
    },
    "electrical": {
        "electrical_cabinet",
        "switch_panel",
        "junction_box",
        "transformer_unit",
        "small_control_panel",
        "backup_battery",
    },
    "control": {
        "control_panel",
        "desk",
        "monitor",
        "rack",
        "console",
    },
    "laboratory": {
        "lab_bench",
        "equipment_unit",
        "fume_hood",
        "sink",
        "cabinet",
        "shelf",
    },
    "maintenance": {
        "workbench",
        "machine_part",
        "tool_rack",
        "pallet",
        "toolbox",
    },
}

_DEFAULT_BIOME_INFRA_PROFILE: dict[str, object] = {
    "tray_width_scale": 1.0,
    "room_link_width_scale": 1.0,
    "pipe_radius_scale": 1.0,
    "support_spacing_scale": 1.0,
    "branch_count_scale": 1.0,
    "branch_ratio_scale": 1.0,
    "drop_clearance_scale": 1.0,
    "min_drop_scale": 1.0,
    "include_pipes": None,
    "strict_linearity": False,
    "extra_pipe_lines": 0,
    "hidden_layout": False,
    "multi_target_connections": False,
    "max_targets": 1,
    "hanging_cables": 0,
    "hanging_ratio": 0.35,
}

_DEFAULT_BIOME_INFRA_PROFILES: dict[str, dict[str, object]] = {
    "boiler": {
        "pipe_radius_scale": 1.75,
        "branch_count_scale": 1.8,
        "branch_ratio_scale": 1.2,
        "include_pipes": True,
        "extra_pipe_lines": 2,
        "drop_clearance_scale": 1.1,
        "max_targets": 3,
    },
    "electrical": {
        "tray_width_scale": 1.2,
        "room_link_width_scale": 1.25,
        "support_spacing_scale": 0.8,
        "branch_count_scale": 2.0,
        "strict_linearity": True,
        "include_pipes": True,
    },
    "control": {
        "tray_width_scale": 0.72,
        "room_link_width_scale": 0.68,
        "support_spacing_scale": 1.7,
        "branch_count_scale": 0.55,
        "branch_ratio_scale": 0.62,
        "include_pipes": True,
        "hidden_layout": True,
    },
    "laboratory": {
        "tray_width_scale": 0.82,
        "room_link_width_scale": 0.85,
        "pipe_radius_scale": 0.65,
        "branch_count_scale": 1.15,
        "include_pipes": True,
        "multi_target_connections": True,
        "max_targets": 6,
    },
    "maintenance": {
        "tray_width_scale": 0.95,
        "pipe_radius_scale": 0.9,
        "support_spacing_scale": 1.25,
        "branch_count_scale": 0.85,
        "branch_ratio_scale": 0.95,
        "include_pipes": True,
        "extra_pipe_lines": 1,
        "hanging_cables": 3,
        "hanging_ratio": 0.46,
    },
}


class InfrastructureGenerator:
    """
    Global infrastructure pass for the whole scene.

    Creates a shared backbone and connects all rooms with trays/pipes/supports,
    independent from biome-specific generators.
    """

    def __init__(self, settings: Mapping[str, object] | None = None) -> None:
        self.settings = _to_mapping(settings, "infrastructure settings")
        self._seed = int(self.settings.get("seed", 0) or 0)
        self._id_counter = 0
        self._active_network: dict[str, object] | None = None
        self._connected_object_ids: set[str] = set()

    def _seeded_ratio(self, salt: str) -> float:
        state = (self._seed & 0xFFFFFFFF) ^ 0x9E3779B9
        for char in salt:
            state = (state * 1664525 + ord(char) + 1013904223) & 0xFFFFFFFF
        return state / 0xFFFFFFFF

    def _resolve_range_setting(
        self,
        value: object,
        default: tuple[float, float],
        label: str,
        *,
        salt: str,
        allow_zero: bool = False,
    ) -> float:
        minimum, maximum = _to_float_range(
            value,
            default,
            label,
            allow_zero=allow_zero,
        )
        if abs(maximum - minimum) <= 1e-9:
            return minimum
        t = self._seeded_ratio(salt)
        return minimum + (maximum - minimum) * t

    def _room_biome(self, room: SceneObject) -> str:
        room_type = room.type.strip().lower()
        if room_type.startswith("room_") and len(room_type) > len("room_"):
            return room_type[len("room_") :]
        return "generic"

    def _effective_biome_name(self, room: SceneObject | None) -> str:
        raw = str(self.settings.get("biome_type", "auto")).strip().lower()
        if raw and raw not in {"auto", "mixed", "scene"}:
            return raw
        if room is None:
            return "generic"
        return self._room_biome(room)

    def _biome_profile(self, room: SceneObject | None) -> dict[str, object]:
        biome = self._effective_biome_name(room)
        profile = dict(_DEFAULT_BIOME_INFRA_PROFILE)
        profile.update(_DEFAULT_BIOME_INFRA_PROFILES.get(biome, {}))

        raw_profiles = self.settings.get("biome_profiles")
        if isinstance(raw_profiles, Mapping):
            raw_profile = raw_profiles.get(biome)
            if raw_profile is None:
                raw_profile = raw_profiles.get("default")
            if isinstance(raw_profile, Mapping):
                profile.update({str(k): v for k, v in raw_profile.items()})
        profile["biome"] = biome
        return profile

    def generate(self, scene: Scene) -> Scene:
        network = self.generate_backbone(scene)
        group = network.get("group")
        if not isinstance(group, SceneObject):
            return scene

        room_roots = self._iter_room_roots(scene)
        for room in room_roots:
            self.connect_room(room, network)
            self.add_secondary_branches(room)
            self.connect_objects(room, network)
        self._apply_infrastructure_rules(scene, network, room_roots)
        return scene

    def generate_backbone(self, scene: Scene) -> dict[str, object]:
        self._connected_object_ids.clear()
        group_id = str(self.settings.get("group_id", "global_infrastructure")).strip() or "global_infrastructure"
        scene.remove_object(group_id)
        room_roots = self._iter_room_roots(scene)
        if not room_roots:
            self._active_network = {"group": None, "rooms": {}}
            return self._active_network

        room_geometries = {room.id: self._room_geometry(room) for room in room_roots}
        min_x = min(geo.center_x - geo.width / 2.0 for geo in room_geometries.values())
        max_x = max(geo.center_x + geo.width / 2.0 for geo in room_geometries.values())
        min_y = min(geo.center_y - geo.depth / 2.0 for geo in room_geometries.values())
        max_y = max(geo.center_y + geo.depth / 2.0 for geo in room_geometries.values())

        margin = _to_positive_float(
            self.settings.get("backbone_margin"),
            1.0,
            "infrastructure.backbone_margin",
        )
        tray_width = self._resolve_range_setting(
            self.settings.get("tray_width"),
            (0.35, 0.35),
            "infrastructure.tray_width",
            salt="tray_width",
        )
        tray_thickness = self._resolve_range_setting(
            self.settings.get("tray_thickness"),
            (0.12, 0.12),
            "infrastructure.tray_thickness",
            salt="tray_thickness",
        )
        min_segment_length = _to_positive_float(
            self.settings.get("min_segment_length"),
            0.25,
            "infrastructure.min_segment_length",
        )
        supports_settings = _to_mapping(
            self.settings.get("supports"),
            "infrastructure.supports",
        )
        support_spacing = self._resolve_range_setting(
            supports_settings.get("spacing", self.settings.get("support_spacing")),
            (4.0, 4.0),
            "infrastructure.supports.spacing",
            salt="support_spacing",
        )
        support_radius = _to_positive_float(
            self.settings.get("support_radius"),
            0.06,
            "infrastructure.support_radius",
        )
        wall_clearance = _to_positive_float(
            self.settings.get("wall_clearance"),
            max(tray_width * 0.65, 0.2),
            "infrastructure.wall_clearance",
        )
        parallel_channels = _to_positive_int(
            self.settings.get("parallel_channels"),
            1,
            "infrastructure.parallel_channels",
        )
        parallel_channels = max(1, min(parallel_channels, 8))
        channel_spacing = _to_positive_float(
            self.settings.get("channel_spacing"),
            max(tray_width * 2.2, 1.5),
            "infrastructure.channel_spacing",
        )
        cross_spacing = _to_positive_float(
            self.settings.get("cross_spacing"),
            max(support_spacing * 1.5, 5.0),
            "infrastructure.cross_spacing",
        )
        junction_size = _to_positive_float(
            self.settings.get("junction_size"),
            max(tray_width * 0.8, 0.2),
            "infrastructure.junction_size",
        )
        add_cross_connections = _to_bool(
            self.settings.get("cross_connections", True),
            default=True,
        )
        density = self._resolve_range_setting(
            self.settings.get("density"),
            (1.0, 1.0),
            "infrastructure.density",
            salt="density",
        )
        branch_frequency = self._resolve_range_setting(
            self.settings.get("branch_frequency"),
            (1.0, 1.0),
            "infrastructure.branch_frequency",
            salt="branch_frequency",
        )
        vertical_settings = _to_mapping(
            self.settings.get("vertical"),
            "infrastructure.vertical",
        )
        drop_frequency = self._resolve_range_setting(
            vertical_settings.get("drop_frequency", self.settings.get("drop_frequency")),
            (1.0, 1.0),
            "infrastructure.vertical.drop_frequency",
            salt="drop_frequency",
        )

        min_room_height = min(geo.height for geo in room_geometries.values())
        routing_level = str(self.settings.get("routing_level", "ceiling")).strip().lower()
        ceiling_offset = _to_positive_float(
            self.settings.get("ceiling_offset"),
            0.25,
            "infrastructure.ceiling_offset",
        )
        top_clearance = _to_positive_float(
            self.settings.get("top_clearance"),
            0.35,
            "infrastructure.top_clearance",
        )
        level_offset = top_clearance
        if routing_level in {"sub_ceiling", "sub-ceiling", "under_ceiling", "under-ceiling", "offset"}:
            level_offset += ceiling_offset
        ceiling_based_tray_z = min_room_height - level_offset - tray_thickness / 2.0
        tray_height_setting = self.settings.get("tray_height")
        if tray_height_setting is None:
            desired_tray_height = ceiling_based_tray_z
        else:
            desired_tray_height = self._resolve_range_setting(
                tray_height_setting,
                (ceiling_based_tray_z, ceiling_based_tray_z),
                "infrastructure.tray_height",
                salt="tray_height",
            )
        max_tray_height = max(1.2 + tray_thickness / 2.0, min_room_height - tray_thickness / 2.0 - 0.05)
        tray_z = _clamp(desired_tray_height, 1.8, max_tray_height)

        include_pipes = _to_bool(self.settings.get("include_pipes", True), default=True)
        pipe_radius = _to_positive_float(
            self.settings.get("pipe_radius"),
            0.09,
            "infrastructure.pipe_radius",
        )
        pipe_drop = _to_positive_float(
            self.settings.get("pipe_drop"),
            tray_thickness * 0.6 + pipe_radius * 1.5,
            "infrastructure.pipe_drop",
        )
        pipe_z = max(pipe_radius + 0.05, tray_z - pipe_drop)

        scene_profile = self._biome_profile(None)
        scene_biome = str(scene_profile.get("biome", "generic")).strip().lower()
        if scene_biome != "generic":
            tray_width *= max(0.3, float(scene_profile.get("tray_width_scale", 1.0)))
            support_spacing *= max(0.3, float(scene_profile.get("support_spacing_scale", 1.0)))
            pipe_radius *= max(0.3, float(scene_profile.get("pipe_radius_scale", 1.0)))
            pipe_z = max(pipe_radius + 0.05, tray_z - pipe_drop)
            density_scale = max(0.3, float(scene_profile.get("branch_count_scale", 1.0)))
            parallel_channels = max(1, min(8, int(round(parallel_channels * density_scale))))
            channel_spacing = max(
                tray_width * 1.6,
                channel_spacing / max(0.6, density_scale),
            )
            include_override = scene_profile.get("include_pipes")
            if include_override is not None:
                include_pipes = _to_bool(include_override, default=include_pipes)
            if _to_bool(scene_profile.get("strict_linearity"), default=False):
                add_cross_connections = False

        # Global density controls how many backbone channels are created.
        parallel_channels = max(1, min(16, int(round(parallel_channels * density))))

        inner_min_x = min_x + wall_clearance
        inner_max_x = max_x - wall_clearance
        inner_min_y = min_y + wall_clearance
        inner_max_y = max_y - wall_clearance
        if inner_max_x - inner_min_x < min_segment_length:
            inner_min_x = min_x
            inner_max_x = max_x
        if inner_max_y - inner_min_y < min_segment_length:
            inner_min_y = min_y
            inner_max_y = max_y

        run_min_x = inner_min_x - margin
        run_max_x = inner_max_x + margin
        run_min_y = inner_min_y - margin
        run_max_y = inner_max_y + margin

        span_x = max(run_max_x - run_min_x, min_segment_length)
        span_y = max(run_max_y - run_min_y, min_segment_length)
        long_axis_x = span_x >= span_y

        long_start = run_min_x if long_axis_x else run_min_y
        long_end = run_max_x if long_axis_x else run_max_y
        short_min = run_min_y if long_axis_x else run_min_x
        short_max = run_max_y if long_axis_x else run_max_x
        short_center = (short_min + short_max) / 2.0
        available_half_span = max((short_max - short_min) / 2.0 - tray_width / 2.0, 0.0)

        if parallel_channels <= 1:
            channel_offsets = [0.0]
        else:
            requested_half = channel_spacing * (parallel_channels - 1) / 2.0
            channel_offsets = _symmetric_offsets(parallel_channels, requested_half)
            max_requested = max(abs(offset) for offset in channel_offsets)
            if max_requested > 1e-9 and max_requested > available_half_span:
                scale = available_half_span / max_requested
                channel_offsets = [offset * scale for offset in channel_offsets]

        channel_positions: list[float] = []
        for offset in channel_offsets:
            position = _clamp(
                short_center + offset,
                short_min + tray_width / 2.0,
                short_max - tray_width / 2.0,
            )
            if all(abs(position - existing) > 1e-6 for existing in channel_positions):
                channel_positions.append(position)
        if not channel_positions:
            channel_positions = [short_center]

        backbone_x = (run_min_x + run_max_x) / 2.0
        backbone_y = (run_min_y + run_max_y) / 2.0
        span_long = max(long_end - long_start, min_segment_length)

        group = SceneObject(id=group_id, type="infrastructure_group")
        scene.add_object(group)

        support_top = max(tray_z - tray_thickness / 2.0, 0.2)
        for channel in channel_positions:
            if long_axis_x:
                center_x = (long_start + long_end) / 2.0
                center_y = channel
            else:
                center_x = channel
                center_y = (long_start + long_end) / 2.0

            self._add_tray_segment(
                group=group,
                object_type="infra_cable_tray_backbone",
                center_x=center_x,
                center_y=center_y,
                z=tray_z,
                length=span_long,
                along_x=long_axis_x,
                tray_width=tray_width,
                tray_thickness=tray_thickness,
            )
            self._add_linear_supports(
                group=group,
                along_x=long_axis_x,
                fixed_coord=channel,
                start=long_start,
                end=long_end,
                support_top=support_top,
                spacing=support_spacing,
                radius=support_radius,
            )
            if include_pipes:
                self._add_pipe_segment(
                    group=group,
                    object_type="infra_pipe_backbone",
                    center_x=center_x,
                    center_y=center_y,
                    z=pipe_z,
                    length=span_long,
                    along_x=long_axis_x,
                    radius=pipe_radius,
                )

        cross_positions = self._equal_interval_positions(long_start, long_end, cross_spacing)
        if add_cross_connections and len(channel_positions) > 1 and cross_positions:
            channel_min = min(channel_positions)
            channel_max = max(channel_positions)
            cross_length = max(channel_max - channel_min, min_segment_length)
            cross_center = (channel_min + channel_max) / 2.0
            for long_position in cross_positions:
                if long_axis_x:
                    center_x = long_position
                    center_y = cross_center
                else:
                    center_x = cross_center
                    center_y = long_position

                self._add_tray_segment(
                    group=group,
                    object_type="infra_cable_tray_cross",
                    center_x=center_x,
                    center_y=center_y,
                    z=tray_z,
                    length=cross_length,
                    along_x=not long_axis_x,
                    tray_width=tray_width,
                    tray_thickness=tray_thickness,
                )
                self._add_linear_supports(
                    group=group,
                    along_x=not long_axis_x,
                    fixed_coord=long_position,
                    start=channel_min,
                    end=channel_max,
                    support_top=support_top,
                    spacing=support_spacing,
                    radius=support_radius,
                )
                if include_pipes:
                    self._add_pipe_segment(
                        group=group,
                        object_type="infra_pipe_cross",
                        center_x=center_x,
                        center_y=center_y,
                        z=pipe_z,
                        length=cross_length,
                        along_x=not long_axis_x,
                        radius=pipe_radius,
                    )
                for channel in channel_positions:
                    junction_x = long_position if long_axis_x else channel
                    junction_y = channel if long_axis_x else long_position
                    self._add_junction_node(
                        group=group,
                        x=junction_x,
                        y=junction_y,
                        z=tray_z,
                        size=junction_size,
                    )

        network: dict[str, object] = {
            "group": group,
            "group_id": group_id,
            "rooms": room_geometries,
            "axis_long_x": long_axis_x,
            "long_start": long_start,
            "long_end": long_end,
            "short_min": short_min,
            "short_max": short_max,
            "channel_positions": channel_positions,
            "backbone_x": backbone_x,
            "backbone_y": backbone_y,
            "tray_z": tray_z,
            "tray_width": tray_width,
            "tray_thickness": tray_thickness,
            "junction_size": junction_size,
            "room_link_width": _to_positive_float(
                self.settings.get("room_link_width"),
                max(tray_width * 0.85, 0.18),
                "infrastructure.room_link_width",
            ),
            "include_pipes": include_pipes,
            "pipe_z": pipe_z,
            "pipe_radius": pipe_radius,
            "support_radius": support_radius,
            "support_spacing": support_spacing,
            "min_segment_length": min_segment_length,
            "density": density,
            "branch_frequency": branch_frequency,
            "drop_frequency": drop_frequency,
            "secondary_branches": _to_bool(
                self.settings.get("secondary_branches", True),
                default=True,
            ),
            "secondary_branch_count": _to_positive_int(
                self.settings.get("secondary_branch_count"),
                2,
                "infrastructure.secondary_branch_count",
            ),
            "secondary_branch_ratio": _to_positive_float(
                self.settings.get("secondary_branch_ratio"),
                0.32,
                "infrastructure.secondary_branch_ratio",
            ),
        }
        self._active_network = network
        return network

    def _apply_infrastructure_rules(
        self,
        scene: Scene,
        network: Mapping[str, object],
        room_roots: list[SceneObject],
    ) -> None:
        group = network.get("group")
        if not isinstance(group, SceneObject):
            return

        rules_cfg = _to_mapping(self.settings.get("rules"), "infrastructure.rules")
        auto_fix = _to_bool(rules_cfg.get("auto_fix", True), default=True)

        room_map_raw = network.get("rooms")
        room_map: dict[str, _RoomGeometry] = {}
        if isinstance(room_map_raw, Mapping):
            for room_id, geometry in room_map_raw.items():
                if isinstance(geometry, _RoomGeometry):
                    room_map[str(room_id)] = geometry
        for room in room_roots:
            if room.id not in room_map:
                room_map[room.id] = self._room_geometry(room)

        room_bounds = {room_id: self._room_bounds(geometry) for room_id, geometry in room_map.items()}
        room_heights = {room_id: geometry.height for room_id, geometry in room_map.items()}
        room_by_id = {room.id: room for room in room_roots}

        clearance_rule = ClearanceRule(
            min_clearance=_to_positive_float(
                rules_cfg.get("clearance", self.settings.get("drop_clearance", 0.25)),
                0.25,
                "infrastructure.rules.clearance",
            )
        )
        height_rule = HeightRule(
            min_height=_to_positive_float(
                rules_cfg.get("min_height", 0.35),
                0.35,
                "infrastructure.rules.min_height",
            ),
            top_margin=_to_positive_float(
                rules_cfg.get("top_margin", 0.2),
                0.2,
                "infrastructure.rules.top_margin",
            ),
        )
        connectivity_rule = ConnectivityRule(
            require_vertical_drop=_to_bool(
                rules_cfg.get("require_vertical_drop", True),
                default=True,
            )
        )
        support_spacing = _to_positive_float(
            rules_cfg.get("support_spacing", network.get("support_spacing", 2.0)),
            float(network.get("support_spacing", 2.0)),
            "infrastructure.rules.support_spacing",
        )
        support_rule = SupportRule(
            spacing=support_spacing,
            long_segment_length=_to_positive_float(
                rules_cfg.get("long_segment_length", support_spacing * 1.2),
                max(support_spacing * 1.2, 0.5),
                "infrastructure.rules.long_segment_length",
            ),
            tolerance=_to_positive_float(
                rules_cfg.get("support_tolerance", 0.22),
                0.22,
                "infrastructure.rules.support_tolerance",
            ),
        )
        collision_rule = CollisionAvoidanceRule(
            min_spacing=_to_positive_float(
                rules_cfg.get("collision_spacing", max(float(network.get("tray_thickness", 0.12)) * 1.3, 0.08)),
                max(float(network.get("tray_thickness", 0.12)) * 1.3, 0.08),
                "infrastructure.rules.collision_spacing",
            ),
            xy_padding=_to_positive_float(
                rules_cfg.get("collision_padding", 0.03),
                0.03,
                "infrastructure.rules.collision_padding",
            ),
        )

        if auto_fix:
            self._auto_fix_connectivity(
                group=group,
                network=network,
                room_roots=room_roots,
                room_bounds=room_bounds,
                connectivity_rule=connectivity_rule,
            )
            self._auto_fix_collision_avoidance(
                group=group,
                room_bounds=room_bounds,
                room_heights=room_heights,
                collision_rule=collision_rule,
                height_rule=height_rule,
            )
            self._auto_fix_clearance(
                group=group,
                room_by_id=room_by_id,
                room_bounds=room_bounds,
                room_heights=room_heights,
                clearance_rule=clearance_rule,
                height_rule=height_rule,
            )
            self._auto_fix_height(
                group=group,
                room_bounds=room_bounds,
                room_heights=room_heights,
                height_rule=height_rule,
            )
            self._auto_fix_supports(
                group=group,
                room_bounds=room_bounds,
                support_rule=support_rule,
                support_radius=float(network.get("support_radius", 0.06)),
            )

    def _auto_fix_connectivity(
        self,
        group: SceneObject,
        network: Mapping[str, object],
        room_roots: list[SceneObject],
        room_bounds: Mapping[str, tuple[float, float, float, float]],
        connectivity_rule: ConnectivityRule,
    ) -> None:
        drop_points, link_points = self._collect_connection_points(group)
        secondary_enabled = _to_bool(network.get("secondary_branches"), default=True)
        for room in room_roots:
            bounds = room_bounds.get(room.id)
            if bounds is None:
                continue
            if connectivity_rule.is_connected(bounds, drop_points, link_points):
                continue
            self.connect_room(room, network)
            if secondary_enabled:
                self.add_secondary_branches(room)
            self.connect_objects(room, network)
            drop_points, link_points = self._collect_connection_points(group)

    def _auto_fix_collision_avoidance(
        self,
        group: SceneObject,
        room_bounds: Mapping[str, tuple[float, float, float, float]],
        room_heights: Mapping[str, float],
        collision_rule: CollisionAvoidanceRule,
        height_rule: HeightRule,
    ) -> None:
        segments = self._tray_object_segments(group)
        for index in range(len(segments)):
            current = segments[index]
            for other_index in range(index + 1, len(segments)):
                other = segments[other_index]
                if not self._segments_overlap_xy(current, other, collision_rule.xy_padding):
                    continue
                z_current = current.obj.transform.position[2]
                z_other = other.obj.transform.position[2]
                if not collision_rule.requires_separation(z_current, z_other):
                    continue
                center_x = (other.start_x + other.end_x) / 2.0
                center_y = (other.start_y + other.end_y) / 2.0
                room_id = self._room_id_for_point(center_x, center_y, room_bounds)
                room_height = room_heights.get(room_id, max(room_heights.values(), default=4.0))
                desired = max(z_other, z_current + collision_rule.min_spacing)
                clamped = height_rule.clamp_center_z(desired, room_height)
                if clamped <= z_other + 1e-9:
                    continue
                self._set_object_center_z(other.obj, clamped)

    def _auto_fix_clearance(
        self,
        group: SceneObject,
        room_by_id: Mapping[str, SceneObject],
        room_bounds: Mapping[str, tuple[float, float, float, float]],
        room_heights: Mapping[str, float],
        clearance_rule: ClearanceRule,
        height_rule: HeightRule,
    ) -> None:
        obstacle_map: dict[str, list[_ObstacleVolume]] = {}
        for room_id, room in room_by_id.items():
            obstacle_map[room_id] = self._room_obstacle_volumes(room)

        segments = self._tray_object_segments(group)
        for segment in segments:
            center_x = (segment.start_x + segment.end_x) / 2.0
            center_y = (segment.start_y + segment.end_y) / 2.0
            room_id = self._room_id_for_point(center_x, center_y, room_bounds)
            if room_id is None:
                continue
            obstacles = obstacle_map.get(room_id, [])
            required_z = segment.obj.transform.position[2]
            tray_half = segment.thickness / 2.0
            for obstacle in obstacles:
                if not self._segment_hits_box(
                    (segment.start_x, segment.start_y),
                    (segment.end_x, segment.end_y),
                    (obstacle.min_x, obstacle.max_x, obstacle.min_y, obstacle.max_y),
                    clearance_rule.min_clearance,
                ):
                    continue
                min_z = clearance_rule.required_tray_z(obstacle.top_z, tray_half)
                if min_z > required_z:
                    required_z = min_z
            if required_z <= segment.obj.transform.position[2] + 1e-9:
                continue
            room_height = room_heights.get(room_id, max(room_heights.values(), default=4.0))
            fixed_z = height_rule.clamp_center_z(required_z, room_height)
            if fixed_z > segment.obj.transform.position[2] + 1e-9:
                self._set_object_center_z(segment.obj, fixed_z)

    def _auto_fix_height(
        self,
        group: SceneObject,
        room_bounds: Mapping[str, tuple[float, float, float, float]],
        room_heights: Mapping[str, float],
        height_rule: HeightRule,
    ) -> None:
        for obj in group.children:
            if not obj.type.startswith("infra_"):
                continue
            x, y, z = obj.transform.position
            room_id = self._room_id_for_point(x, y, room_bounds)
            room_height = room_heights.get(room_id, max(room_heights.values(), default=4.0))
            if obj.type.startswith("infra_vertical_drop") or obj.type.startswith("infra_hanging_cable"):
                if obj.mesh is None or not obj.mesh.vertices:
                    continue
                zs = [vertex[2] * obj.transform.scale[2] for vertex in obj.mesh.vertices]
                length = max(0.1, max(zs) - min(zs))
                allowed_top = height_rule.max_center(room_height)
                bottom = max(0.0, min(z, allowed_top - 0.1))
                length = max(0.1, min(length, allowed_top - bottom))
                obj.transform = Transform(
                    position=(x, y, bottom),
                    rotation=obj.transform.rotation,
                    scale=obj.transform.scale,
                )
                obj.mesh = create_vertical_drop(length=length)
                continue
            if obj.type.startswith("infra_cable_tray") or obj.type.startswith("infra_pipe"):
                fixed_z = height_rule.clamp_center_z(z, room_height)
                if abs(fixed_z - z) > 1e-9:
                    self._set_object_center_z(obj, fixed_z)

    def _auto_fix_supports(
        self,
        group: SceneObject,
        room_bounds: Mapping[str, tuple[float, float, float, float]],
        support_rule: SupportRule,
        support_radius: float,
    ) -> None:
        support_objects = [obj for obj in group.children if obj.type == "infra_support"]
        segments = self._tray_object_segments(group)
        for segment in segments:
            if not support_rule.needs_supports(segment.length):
                continue

            positions = self._equal_interval_positions(
                segment.start_x if segment.along_x else segment.start_y,
                segment.end_x if segment.along_x else segment.end_y,
                support_rule.spacing,
            )
            fixed_coord = segment.start_y if segment.along_x else segment.start_x
            support_top = max(segment.obj.transform.position[2] - segment.thickness / 2.0, 0.2)
            for value in positions:
                sx = value if segment.along_x else fixed_coord
                sy = fixed_coord if segment.along_x else value
                if self._has_support_near(support_objects, sx, sy, support_rule.tolerance):
                    continue
                room_id = self._room_id_for_point(sx, sy, room_bounds)
                if room_id is None:
                    continue
                self._add_support(group=group, x=sx, y=sy, top=support_top, radius=support_radius)
                support_objects.append(group.children[-1])

    def connect_room(self, room: SceneObject, network: Mapping[str, object]) -> None:
        group = network.get("group")
        if not isinstance(group, SceneObject):
            return

        room_map_raw = network.get("rooms")
        room_map = room_map_raw if isinstance(room_map_raw, Mapping) else {}
        room_geometry = room_map.get(room.id)
        if not isinstance(room_geometry, _RoomGeometry):
            room_geometry = self._room_geometry(room)

        biome_profile = self._biome_profile(room)
        biome_name = str(biome_profile.get("biome", "generic")).strip().lower() or "generic"

        tray_z = float(network.get("tray_z", 2.5))
        tray_width = float(network.get("tray_width", 0.35))
        tray_thickness = float(network.get("tray_thickness", 0.12))
        link_width = float(network.get("room_link_width", tray_width))
        min_segment_length = float(network.get("min_segment_length", 0.25))
        include_pipes = _to_bool(network.get("include_pipes"), default=True)
        pipe_z = float(network.get("pipe_z", tray_z - tray_thickness))
        pipe_radius = float(network.get("pipe_radius", 0.08))
        support_radius = float(network.get("support_radius", 0.06))
        support_spacing = float(network.get("support_spacing", 4.0))
        junction_size = float(network.get("junction_size", max(tray_width * 0.8, 0.2)))

        tray_width *= max(0.3, float(biome_profile.get("tray_width_scale", 1.0)))
        link_width *= max(0.3, float(biome_profile.get("room_link_width_scale", 1.0)))
        pipe_radius *= max(0.3, float(biome_profile.get("pipe_radius_scale", 1.0)))
        support_spacing *= max(0.3, float(biome_profile.get("support_spacing_scale", 1.0)))

        include_override = biome_profile.get("include_pipes")
        if include_override is not None:
            include_pipes = _to_bool(include_override, default=include_pipes)

        strict_linearity = _to_bool(
            biome_profile.get("strict_linearity"),
            default=False,
        )
        hidden_layout = _to_bool(
            biome_profile.get("hidden_layout"),
            default=False,
        )
        extra_pipe_lines = max(0, int(float(biome_profile.get("extra_pipe_lines", 0))))
        multi_target_connections = _to_bool(
            biome_profile.get("multi_target_connections"),
            default=False,
        )
        max_targets = max(1, int(float(biome_profile.get("max_targets", 1))))
        drop_frequency = max(0.1, float(network.get("drop_frequency", 1.0)))
        if hidden_layout:
            junction_size = max(junction_size * 0.85, 0.12)
            support_spacing *= 1.3

        min_vertical_drop = _to_positive_float(
            self.settings.get("min_vertical_drop_length"),
            max(tray_thickness * 2.0, 0.35),
            "infrastructure.min_vertical_drop_length",
        )
        min_vertical_drop *= max(0.35, float(biome_profile.get("min_drop_scale", 1.0)))
        drop_clearance = _to_positive_float(
            self.settings.get("drop_clearance"),
            max(link_width * 0.9, 0.2),
            "infrastructure.drop_clearance",
        )
        drop_clearance *= max(0.35, float(biome_profile.get("drop_clearance_scale", 1.0)))

        room_bounds = self._room_bounds(room_geometry)
        target_types = _BIOME_CONNECTION_TARGET_TYPES.get(biome_name, _DEFAULT_KEY_OBJECT_TYPES)
        effective_multi_target = multi_target_connections or drop_frequency > 1.0
        requested_targets = 1 if not effective_multi_target else max_targets
        requested_targets = max(1, int(round(requested_targets * drop_frequency)))
        requested_targets = max(1, min(requested_targets, 12))
        target_candidates = self._room_connection_targets(
            room=room,
            room_geometry=room_geometry,
            key_types=target_types,
            max_targets=requested_targets,
        )
        if not target_candidates:
            target_candidates = [(room_geometry.center_x, room_geometry.center_y, 0.0, None)]
        target_x, target_y, target_z, target_object_id = target_candidates[0]

        nearest = self._nearest_tray_point(group, target_x=target_x, target_y=target_y)
        if nearest is None:
            source_x = room_geometry.center_x
            source_y = room_geometry.center_y
        else:
            source_x, source_y = nearest

        entry_x, entry_y = self._entry_point_on_room_bounds(source_x, source_y, room_bounds)
        obstacle_boxes = self._room_obstacle_boxes(
            room,
            excluded_ids={target_object_id} if target_object_id else set(),
        )
        drop_x, drop_y, inside_path = self._select_drop_point(
            target_x=target_x,
            target_y=target_y,
            entry_x=entry_x,
            entry_y=entry_y,
            room_bounds=room_bounds,
            obstacle_boxes=obstacle_boxes,
            clearance=drop_clearance,
        )

        prefer_x_first = _to_bool(network.get("axis_long_x"), default=True)
        if strict_linearity:
            outside_path = self._axis_path(
                start=(source_x, source_y),
                end=(entry_x, entry_y),
                prefer_x_first=prefer_x_first,
            )
            inside_path = self._axis_path(
                start=(entry_x, entry_y),
                end=(drop_x, drop_y),
                prefer_x_first=prefer_x_first,
            )
        else:
            outside_path = self._best_axis_path(
                start=(source_x, source_y),
                end=(entry_x, entry_y),
                obstacle_boxes=obstacle_boxes if self._point_in_bounds(source_x, source_y, room_bounds) else (),
                clearance=drop_clearance,
            )
        route = self._clean_polyline([*outside_path, *inside_path[1:]])

        support_top = max(tray_z - tray_thickness / 2.0, 0.2)
        created_any_segment = False
        for index in range(len(route) - 1):
            start = route[index]
            end = route[index + 1]
            supports_enabled = not (biome_name == "maintenance" and index % 2 == 1)
            created = self._add_horizontal_link_segment(
                group=group,
                start=start,
                end=end,
                z=tray_z,
                tray_width=link_width,
                tray_thickness=tray_thickness,
                support_top=support_top,
                support_spacing=support_spacing,
                support_radius=support_radius,
                supports_enabled=supports_enabled,
                include_pipes=include_pipes,
                pipe_z=pipe_z,
                pipe_radius=pipe_radius,
                extra_pipe_lines=extra_pipe_lines,
            )
            created_any_segment = created_any_segment or created

        if not created_any_segment:
            self._add_tray_segment(
                group=group,
                object_type="infra_cable_tray_room_link",
                center_x=entry_x,
                center_y=entry_y,
                z=tray_z,
                length=max(link_width, min_segment_length),
                along_x=True,
                tray_width=link_width,
                tray_thickness=tray_thickness,
            )

        self._add_junction_node(
            group=group,
            x=entry_x,
            y=entry_y,
            z=tray_z,
            size=junction_size,
        )

        self._add_vertical_drop(
            group=group,
            x=drop_x,
            y=drop_y,
            tray_z=tray_z,
            target_z=target_z,
            min_vertical_drop=min_vertical_drop,
            object_type="infra_vertical_drop",
        )

        if effective_multi_target and len(target_candidates) > 1:
            for extra_target_x, extra_target_y, extra_target_z, extra_target_id in target_candidates[1:]:
                branch_obstacles = self._room_obstacle_boxes(
                    room,
                    excluded_ids={extra_target_id} if extra_target_id else set(),
                )
                branch_drop_x, branch_drop_y, branch_inside_path = self._select_drop_point(
                    target_x=extra_target_x,
                    target_y=extra_target_y,
                    entry_x=entry_x,
                    entry_y=entry_y,
                    room_bounds=room_bounds,
                    obstacle_boxes=branch_obstacles,
                    clearance=drop_clearance,
                )
                if strict_linearity:
                    branch_inside_path = self._axis_path(
                        start=(entry_x, entry_y),
                        end=(branch_drop_x, branch_drop_y),
                        prefer_x_first=prefer_x_first,
                    )
                branch_route = self._clean_polyline(
                    [(entry_x, entry_y), *branch_inside_path[1:]]
                )
                for index in range(len(branch_route) - 1):
                    self._add_horizontal_link_segment(
                        group=group,
                        start=branch_route[index],
                        end=branch_route[index + 1],
                        z=tray_z,
                        tray_width=link_width,
                        tray_thickness=tray_thickness,
                        support_top=support_top,
                        support_spacing=support_spacing,
                        support_radius=support_radius,
                        supports_enabled=not hidden_layout,
                        include_pipes=include_pipes,
                        pipe_z=pipe_z,
                        pipe_radius=pipe_radius,
                        extra_pipe_lines=extra_pipe_lines,
                    )
                self._add_vertical_drop(
                    group=group,
                    x=branch_drop_x,
                    y=branch_drop_y,
                    tray_z=tray_z,
                    target_z=extra_target_z,
                    min_vertical_drop=min_vertical_drop,
                    object_type="infra_vertical_drop",
                )

        hanging_cables = max(0, int(float(biome_profile.get("hanging_cables", 0))))
        if hanging_cables > 0:
            hanging_ratio = _clamp(
                float(biome_profile.get("hanging_ratio", 0.35)),
                0.1,
                0.95,
            )
            self._add_hanging_cables(
                group=group,
                room_geometry=room_geometry,
                tray_z=tray_z,
                count=hanging_cables,
                ratio=hanging_ratio,
                min_vertical_drop=min_vertical_drop,
            )

    def connect_objects(self, room: SceneObject, network: Mapping[str, object]) -> None:
        group = network.get("group")
        if not isinstance(group, SceneObject):
            return

        object_cfg = _to_mapping(
            self.settings.get("object_connections"),
            "infrastructure.object_connections",
        )
        if not _to_bool(object_cfg.get("enabled", True), default=True):
            return

        room_map_raw = network.get("rooms")
        room_map = room_map_raw if isinstance(room_map_raw, Mapping) else {}
        room_geometry = room_map.get(room.id)
        if not isinstance(room_geometry, _RoomGeometry):
            room_geometry = self._room_geometry(room)
        room_bounds = self._room_bounds(room_geometry)

        tray_z = float(network.get("tray_z", 2.5))
        tray_thickness = float(network.get("tray_thickness", 0.12))
        pipe_radius = float(network.get("pipe_radius", 0.08))
        include_pipes = _to_bool(network.get("include_pipes"), default=True)

        cable_radius = _to_positive_float(
            object_cfg.get("cable_radius"),
            max(pipe_radius * 0.35, 0.02),
            "infrastructure.object_connections.cable_radius",
        )
        pipe_connection_radius = _to_positive_float(
            object_cfg.get("pipe_connection_radius"),
            max(pipe_radius, cable_radius * 1.4),
            "infrastructure.object_connections.pipe_connection_radius",
        )
        route_clearance = _to_positive_float(
            object_cfg.get("clearance"),
            max(float(self.settings.get("drop_clearance", 0.25)) * 0.6, 0.08),
            "infrastructure.object_connections.clearance",
        )
        min_vertical_drop = _to_positive_float(
            self.settings.get("min_vertical_drop_length"),
            max(tray_thickness * 2.0, 0.35),
            "infrastructure.min_vertical_drop_length",
        )
        max_targets_per_room = _to_positive_int(
            object_cfg.get("max_targets_per_room"),
            256,
            "infrastructure.object_connections.max_targets_per_room",
        )
        connection_z = max(cable_radius * 2.0, tray_z - tray_thickness * 0.65)

        connected_in_room = 0
        for obj, world_x, world_y, world_z in self._iter_room_world_objects(room):
            if connected_in_room >= max_targets_per_room:
                break
            if obj.id in self._connected_object_ids:
                continue
            if not self._object_requires_infrastructure_connection(obj.type):
                continue

            anchor_local = self._select_connection_anchor(obj)
            target_x, target_y, target_z = self._anchor_world_position(
                obj=obj,
                world_x=world_x,
                world_y=world_y,
                world_z=world_z,
                anchor_local=anchor_local,
            )
            target_x, target_y = self._clamp_point_to_bounds(target_x, target_y, room_bounds)

            nearest = self._nearest_tray_point(
                group,
                target_x=target_x,
                target_y=target_y,
                include_room_links=True,
            )
            if nearest is None:
                continue

            prefer_pipe = self._object_prefers_pipe_connection(obj.type)
            obstacle_boxes = self._room_obstacle_boxes(room, excluded_ids={obj.id})
            route = self._clean_polyline(
                self._best_axis_path(
                    start=nearest,
                    end=(target_x, target_y),
                    obstacle_boxes=obstacle_boxes,
                    clearance=route_clearance,
                )
            )
            for index in range(len(route) - 1):
                self._add_object_connection_segment(
                    group=group,
                    start=route[index],
                    end=route[index + 1],
                    z=connection_z,
                    prefer_pipe=prefer_pipe,
                    include_pipes=include_pipes,
                    cable_radius=cable_radius,
                    pipe_radius=pipe_connection_radius,
                )

            self._add_vertical_drop(
                group=group,
                x=target_x,
                y=target_y,
                tray_z=tray_z,
                target_z=max(0.0, target_z),
                min_vertical_drop=min_vertical_drop,
                object_type="infra_vertical_drop_object_link",
            )
            self._connected_object_ids.add(obj.id)
            connected_in_room += 1

    def _add_object_connection_segment(
        self,
        group: SceneObject,
        start: tuple[float, float],
        end: tuple[float, float],
        z: float,
        prefer_pipe: bool,
        include_pipes: bool,
        cable_radius: float,
        pipe_radius: float,
    ) -> None:
        start_x, start_y = start
        end_x, end_y = end
        dx = end_x - start_x
        dy = end_y - start_y
        if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
            return

        along_x = abs(dx) >= abs(dy)
        length = abs(dx) if along_x else abs(dy)
        center_x = (start_x + end_x) / 2.0
        center_y = (start_y + end_y) / 2.0

        if prefer_pipe and include_pipes:
            self._add_pipe_segment(
                group=group,
                object_type="infra_pipe_object_link",
                center_x=center_x,
                center_y=center_y,
                z=z,
                length=length,
                along_x=along_x,
                radius=pipe_radius,
            )
            return

        self._add_cable_segment(
            group=group,
            object_type="infra_cable_object_link",
            center_x=center_x,
            center_y=center_y,
            z=z,
            length=length,
            along_x=along_x,
            radius=cable_radius,
        )

    def _add_cable_segment(
        self,
        group: SceneObject,
        object_type: str,
        center_x: float,
        center_y: float,
        z: float,
        length: float,
        along_x: bool,
        radius: float,
    ) -> None:
        length = max(length, 0.0)
        if length <= 1e-9:
            return

        mesh = create_beam(
            length=length,
            profile_type={"type": "circular", "radius": radius, "segments": 12},
        )
        rotation = (0.0, 0.0, 0.0) if along_x else (0.0, 0.0, 90.0)
        group.add_child(
            SceneObject(
                id=self._next_id(object_type),
                type=object_type,
                transform=Transform(position=(center_x, center_y, z), rotation=rotation),
                mesh=mesh,
            )
        )

    def _object_requires_infrastructure_connection(self, object_type: str) -> bool:
        normalized = object_type.strip().lower()
        if not normalized:
            return False
        if normalized in _STRUCTURAL_OBJECT_TYPES:
            return False
        if normalized.startswith("infra_") or normalized.startswith("aux_"):
            return False
        if normalized in _OBJECT_CONNECTION_EXACT_TYPES:
            return True
        return any(keyword in normalized for keyword in _OBJECT_CONNECTION_KEYWORDS)

    def _object_prefers_pipe_connection(self, object_type: str) -> bool:
        normalized = object_type.strip().lower()
        if normalized in _OBJECT_PIPE_CONNECTION_EXACT_TYPES:
            return True
        return any(keyword in normalized for keyword in _OBJECT_PIPE_CONNECTION_KEYWORDS)

    def _select_connection_anchor(self, obj: SceneObject) -> tuple[float, float, float] | None:
        obj.ensure_anchor_points()
        if not obj.anchor_points:
            return None

        preferred_order: tuple[str, ...]
        if self._object_prefers_pipe_connection(obj.type):
            preferred_order = ("top", "side", "bottom")
        else:
            preferred_order = ("top", "side", "bottom")

        for anchor_type in preferred_order:
            matches = [anchor for anchor in obj.anchor_points if anchor.type == anchor_type]
            if matches:
                return matches[0].position
        return obj.anchor_points[0].position

    def _anchor_world_position(
        self,
        obj: SceneObject,
        world_x: float,
        world_y: float,
        world_z: float,
        anchor_local: tuple[float, float, float] | None,
    ) -> tuple[float, float, float]:
        if anchor_local is None:
            return (world_x, world_y, max(0.0, world_z))

        sx, sy, sz = obj.transform.scale
        ax, ay, az = anchor_local
        return (
            world_x + ax * sx,
            world_y + ay * sy,
            max(0.0, world_z + az * sz),
        )

    def _clamp_point_to_bounds(
        self,
        x: float,
        y: float,
        bounds: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        min_x, max_x, min_y, max_y = bounds
        if min_x > max_x:
            min_x, max_x = max_x, min_x
        if min_y > max_y:
            min_y, max_y = max_y, min_y
        return (_clamp(x, min_x, max_x), _clamp(y, min_y, max_y))

    def _add_horizontal_link_segment(
        self,
        group: SceneObject,
        start: tuple[float, float],
        end: tuple[float, float],
        z: float,
        tray_width: float,
        tray_thickness: float,
        support_top: float,
        support_spacing: float,
        support_radius: float,
        supports_enabled: bool,
        include_pipes: bool,
        pipe_z: float,
        pipe_radius: float,
        extra_pipe_lines: int,
    ) -> bool:
        start_x, start_y = start
        end_x, end_y = end
        dx = end_x - start_x
        dy = end_y - start_y
        if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
            return False

        along_x = abs(dx) >= abs(dy)
        length = abs(dx) if along_x else abs(dy)
        center_x = (start_x + end_x) / 2.0
        center_y = (start_y + end_y) / 2.0

        self._add_tray_segment(
            group=group,
            object_type="infra_cable_tray_room_link",
            center_x=center_x,
            center_y=center_y,
            z=z,
            length=length,
            along_x=along_x,
            tray_width=tray_width,
            tray_thickness=tray_thickness,
        )
        if supports_enabled:
            if along_x:
                self._add_linear_supports(
                    group=group,
                    along_x=True,
                    fixed_coord=center_y,
                    start=min(start_x, end_x),
                    end=max(start_x, end_x),
                    support_top=support_top,
                    spacing=support_spacing,
                    radius=support_radius,
                )
            else:
                self._add_linear_supports(
                    group=group,
                    along_x=False,
                    fixed_coord=center_x,
                    start=min(start_y, end_y),
                    end=max(start_y, end_y),
                    support_top=support_top,
                    spacing=support_spacing,
                    radius=support_radius,
                )
        if include_pipes:
            self._add_pipe_segment(
                group=group,
                object_type="infra_pipe_room_link",
                center_x=center_x,
                center_y=center_y,
                z=pipe_z,
                length=length,
                along_x=along_x,
                radius=pipe_radius,
            )
            if extra_pipe_lines > 0:
                line_count = min(extra_pipe_lines, 3)
                spacing = max(pipe_radius * 3.0, 0.09)
                offsets = _symmetric_offsets(line_count + 1, spacing * line_count * 0.5)
                for offset in offsets:
                    if abs(offset) <= 1e-9:
                        continue
                    aux_x = center_x
                    aux_y = center_y
                    if along_x:
                        aux_y += offset
                    else:
                        aux_x += offset
                    self._add_pipe_segment(
                        group=group,
                        object_type="infra_pipe_room_link_aux",
                        center_x=aux_x,
                        center_y=aux_y,
                        z=pipe_z,
                        length=length,
                        along_x=along_x,
                        radius=pipe_radius * 0.9,
                    )
        return length > 1e-9

    def _room_bounds(self, room_geometry: _RoomGeometry) -> tuple[float, float, float, float]:
        half_w = room_geometry.width / 2.0
        half_d = room_geometry.depth / 2.0
        return (
            room_geometry.center_x - half_w,
            room_geometry.center_x + half_w,
            room_geometry.center_y - half_d,
            room_geometry.center_y + half_d,
        )

    def _entry_point_on_room_bounds(
        self,
        source_x: float,
        source_y: float,
        room_bounds: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        min_x, max_x, min_y, max_y = room_bounds
        if source_x <= min_x:
            return (min_x, _clamp(source_y, min_y, max_y))
        if source_x >= max_x:
            return (max_x, _clamp(source_y, min_y, max_y))
        if source_y <= min_y:
            return (_clamp(source_x, min_x, max_x), min_y)
        if source_y >= max_y:
            return (_clamp(source_x, min_x, max_x), max_y)

        distances = (
            (abs(source_x - min_x), (min_x, source_y)),
            (abs(max_x - source_x), (max_x, source_y)),
            (abs(source_y - min_y), (source_x, min_y)),
            (abs(max_y - source_y), (source_x, max_y)),
        )
        _, point = min(distances, key=lambda item: item[0])
        return (
            _clamp(point[0], min_x, max_x),
            _clamp(point[1], min_y, max_y),
        )

    def _nearest_tray_point(
        self,
        group: SceneObject,
        target_x: float,
        target_y: float,
        include_room_links: bool = False,
    ) -> tuple[float, float] | None:
        tray_segments = self._tray_segments(group, include_room_links=include_room_links)
        if not tray_segments:
            return None

        best_point: tuple[float, float] | None = None
        best_distance = float("inf")
        for segment in tray_segments:
            point = self._nearest_point_on_axis_segment(
                start_x=segment.start_x,
                start_y=segment.start_y,
                end_x=segment.end_x,
                end_y=segment.end_y,
                target_x=target_x,
                target_y=target_y,
            )
            dx = point[0] - target_x
            dy = point[1] - target_y
            distance_sq = dx * dx + dy * dy
            if distance_sq < best_distance:
                best_distance = distance_sq
                best_point = point
        return best_point

    def _collect_connection_points(
        self,
        group: SceneObject,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        drop_points: list[tuple[float, float]] = []
        link_points: list[tuple[float, float]] = []
        for child in group.children:
            if child.type.startswith("infra_vertical_drop") or child.type.startswith("infra_hanging_cable"):
                drop_points.append((child.transform.position[0], child.transform.position[1]))
            elif child.type.startswith("infra_cable_tray_room_link") or child.type == "infra_cable_tray_entry":
                link_points.append((child.transform.position[0], child.transform.position[1]))
        return (drop_points, link_points)

    def _tray_object_segments(self, group: SceneObject) -> list[_TrayObjectSegment]:
        segments: list[_TrayObjectSegment] = []
        for child in group.children:
            if not child.type.startswith("infra_cable_tray"):
                continue
            if child.mesh is None or not child.mesh.vertices:
                continue

            scaled_x = [vertex[0] * child.transform.scale[0] for vertex in child.mesh.vertices]
            scaled_y = [vertex[1] * child.transform.scale[1] for vertex in child.mesh.vertices]
            scaled_z = [vertex[2] * child.transform.scale[2] for vertex in child.mesh.vertices]
            span_x = max(scaled_x) - min(scaled_x)
            span_y = max(scaled_y) - min(scaled_y)
            span_z = max(scaled_z) - min(scaled_z)
            if span_x <= 1e-9 and span_y <= 1e-9:
                continue

            along_x = span_x >= span_y
            length = max(span_x, span_y)
            width = min(span_x, span_y)
            half = length / 2.0
            center_x = child.transform.position[0]
            center_y = child.transform.position[1]
            if along_x:
                start_x = center_x - half
                end_x = center_x + half
                start_y = center_y
                end_y = center_y
            else:
                start_x = center_x
                end_x = center_x
                start_y = center_y - half
                end_y = center_y + half
            segments.append(
                _TrayObjectSegment(
                    obj=child,
                    start_x=start_x,
                    start_y=start_y,
                    end_x=end_x,
                    end_y=end_y,
                    along_x=along_x,
                    length=length,
                    width=max(width, 0.01),
                    thickness=max(span_z, 0.01),
                )
            )
        return segments

    def _segments_overlap_xy(
        self,
        first: _TrayObjectSegment,
        second: _TrayObjectSegment,
        padding: float,
    ) -> bool:
        first_box = self._segment_xy_bounds(first, padding)
        second_box = self._segment_xy_bounds(second, padding)
        return not (
            first_box[1] < second_box[0]
            or second_box[1] < first_box[0]
            or first_box[3] < second_box[2]
            or second_box[3] < first_box[2]
        )

    def _segment_xy_bounds(
        self,
        segment: _TrayObjectSegment,
        padding: float,
    ) -> tuple[float, float, float, float]:
        half_width = segment.width / 2.0 + padding
        if segment.along_x:
            min_x = min(segment.start_x, segment.end_x) - padding
            max_x = max(segment.start_x, segment.end_x) + padding
            min_y = segment.start_y - half_width
            max_y = segment.start_y + half_width
        else:
            min_x = segment.start_x - half_width
            max_x = segment.start_x + half_width
            min_y = min(segment.start_y, segment.end_y) - padding
            max_y = max(segment.start_y, segment.end_y) + padding
        return (min_x, max_x, min_y, max_y)

    def _room_id_for_point(
        self,
        x: float,
        y: float,
        room_bounds: Mapping[str, tuple[float, float, float, float]],
    ) -> str | None:
        for room_id, bounds in room_bounds.items():
            if self._point_in_bounds(x, y, bounds):
                return room_id
        return None

    def _set_object_center_z(self, obj: SceneObject, z: float) -> None:
        x, y, _ = obj.transform.position
        obj.transform = Transform(
            position=(x, y, z),
            rotation=obj.transform.rotation,
            scale=obj.transform.scale,
        )

    def _room_obstacle_volumes(
        self,
        room: SceneObject,
        excluded_ids: set[str] | None = None,
    ) -> list[_ObstacleVolume]:
        ignored_types = {
            "floor",
            "ceiling",
            "wall",
            "door_opening",
            "window_opening",
        }
        excluded = excluded_ids or set()
        volumes: list[_ObstacleVolume] = []
        for obj, world_x, world_y, world_z in self._iter_room_world_objects(room):
            if obj.id in excluded:
                continue
            if obj.type in ignored_types or obj.type.startswith("infra_"):
                continue
            if obj.mesh is None or not obj.mesh.vertices:
                continue
            xs = [world_x + vertex[0] * obj.transform.scale[0] for vertex in obj.mesh.vertices]
            ys = [world_y + vertex[1] * obj.transform.scale[1] for vertex in obj.mesh.vertices]
            zs = [world_z + vertex[2] * obj.transform.scale[2] for vertex in obj.mesh.vertices]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            if max_x - min_x <= 1e-6 or max_y - min_y <= 1e-6:
                continue
            volumes.append(
                _ObstacleVolume(
                    min_x=min_x,
                    max_x=max_x,
                    min_y=min_y,
                    max_y=max_y,
                    top_z=max(zs),
                )
            )
        return volumes

    def _has_support_near(
        self,
        support_objects: list[SceneObject],
        x: float,
        y: float,
        tolerance: float,
    ) -> bool:
        tolerance_sq = tolerance * tolerance
        for support in support_objects:
            dx = support.transform.position[0] - x
            dy = support.transform.position[1] - y
            if dx * dx + dy * dy <= tolerance_sq:
                return True
        return False

    def _tray_segments(
        self,
        group: SceneObject,
        include_room_links: bool = False,
    ) -> list[_TraySegment]:
        segments: list[_TraySegment] = []
        ignored_types: set[str] = set()
        if not include_room_links:
            ignored_types.update(
                {
                    "infra_cable_tray_room_link",
                    "infra_cable_tray_entry",
                }
            )
        for child in group.children:
            if not child.type.startswith("infra_cable_tray"):
                continue
            if child.type in ignored_types:
                continue
            if child.mesh is None or not child.mesh.vertices:
                continue

            scaled_x = [vertex[0] * child.transform.scale[0] for vertex in child.mesh.vertices]
            scaled_y = [vertex[1] * child.transform.scale[1] for vertex in child.mesh.vertices]
            span_x = max(scaled_x) - min(scaled_x)
            span_y = max(scaled_y) - min(scaled_y)
            if span_x <= 1e-9 and span_y <= 1e-9:
                continue

            center_x = child.transform.position[0]
            center_y = child.transform.position[1]
            along_x = span_x >= span_y
            length = max(span_x, span_y)
            half = length / 2.0
            if along_x:
                segments.append(
                    _TraySegment(
                        start_x=center_x - half,
                        start_y=center_y,
                        end_x=center_x + half,
                        end_y=center_y,
                    )
                )
            else:
                segments.append(
                    _TraySegment(
                        start_x=center_x,
                        start_y=center_y - half,
                        end_x=center_x,
                        end_y=center_y + half,
                    )
                )
        return segments

    def _nearest_point_on_axis_segment(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        target_x: float,
        target_y: float,
    ) -> tuple[float, float]:
        if abs(start_x - end_x) >= abs(start_y - end_y):
            return (_clamp(target_x, min(start_x, end_x), max(start_x, end_x)), start_y)
        return (start_x, _clamp(target_y, min(start_y, end_y), max(start_y, end_y)))

    def _room_connection_targets(
        self,
        room: SceneObject,
        room_geometry: _RoomGeometry,
        key_types: set[str] | None = None,
        max_targets: int = 1,
    ) -> list[tuple[float, float, float, str | None]]:
        center_x = room_geometry.center_x
        center_y = room_geometry.center_y
        valid_types = key_types if key_types is not None else _DEFAULT_KEY_OBJECT_TYPES
        max_count = max(1, max_targets)

        ranked: list[tuple[float, float, float, str | None, float]] = []
        for obj, world_x, world_y, world_z in self._iter_room_world_objects(room):
            object_type = obj.type.strip().lower()
            if object_type not in valid_types:
                continue
            target_z = max(0.0, world_z)
            if obj.mesh is not None and obj.mesh.vertices:
                scaled_z = [vertex[2] * obj.transform.scale[2] for vertex in obj.mesh.vertices]
                local_min_z = min(scaled_z)
                local_max_z = max(scaled_z)
                target_z = max(0.0, world_z + (local_min_z + local_max_z) / 2.0)

            dx = world_x - center_x
            dy = world_y - center_y
            distance_sq = dx * dx + dy * dy
            ranked.append((world_x, world_y, target_z, obj.id, distance_sq))

        ranked.sort(key=lambda item: item[4])
        if not ranked:
            return [(center_x, center_y, 0.0, None)]

        targets: list[tuple[float, float, float, str | None]] = []
        for world_x, world_y, target_z, object_id, _ in ranked[:max_count]:
            targets.append((world_x, world_y, target_z, object_id))
        return targets

    def _room_connection_target(
        self,
        room: SceneObject,
        room_geometry: _RoomGeometry,
    ) -> tuple[float, float, float, str | None]:
        targets = self._room_connection_targets(
            room=room,
            room_geometry=room_geometry,
            key_types=None,
            max_targets=1,
        )
        return targets[0]

    def _iter_room_world_objects(
        self,
        room: SceneObject,
    ) -> list[tuple[SceneObject, float, float, float]]:
        output: list[tuple[SceneObject, float, float, float]] = []

        def walk(node: SceneObject, base_x: float, base_y: float, base_z: float) -> None:
            world_x = base_x + node.transform.position[0]
            world_y = base_y + node.transform.position[1]
            world_z = base_z + node.transform.position[2]
            output.append((node, world_x, world_y, world_z))
            for child in node.children:
                walk(child, world_x, world_y, world_z)

        root_x, root_y, root_z = room.transform.position
        for child in room.children:
            walk(child, root_x, root_y, root_z)
        return output

    def _room_obstacle_boxes(
        self,
        room: SceneObject,
        excluded_ids: set[str],
    ) -> list[tuple[float, float, float, float]]:
        ignored_types = {
            "floor",
            "ceiling",
            "wall",
            "door_opening",
            "window_opening",
        }
        boxes: list[tuple[float, float, float, float]] = []
        for obj, world_x, world_y, _ in self._iter_room_world_objects(room):
            if obj.id in excluded_ids:
                continue
            if obj.type in ignored_types or obj.type.startswith("infra_"):
                continue
            if obj.mesh is None or not obj.mesh.vertices:
                continue

            transformed_x = [world_x + vertex[0] * obj.transform.scale[0] for vertex in obj.mesh.vertices]
            transformed_y = [world_y + vertex[1] * obj.transform.scale[1] for vertex in obj.mesh.vertices]
            min_x = min(transformed_x)
            max_x = max(transformed_x)
            min_y = min(transformed_y)
            max_y = max(transformed_y)
            if max_x - min_x <= 1e-6 or max_y - min_y <= 1e-6:
                continue
            boxes.append((min_x, max_x, min_y, max_y))
        return boxes

    def _select_drop_point(
        self,
        target_x: float,
        target_y: float,
        entry_x: float,
        entry_y: float,
        room_bounds: tuple[float, float, float, float],
        obstacle_boxes: list[tuple[float, float, float, float]],
        clearance: float,
    ) -> tuple[float, float, list[tuple[float, float]]]:
        min_x, max_x, min_y, max_y = room_bounds
        inset = min(clearance * 0.5, min((max_x - min_x) * 0.2, (max_y - min_y) * 0.2))
        if inset < 0.05:
            inset = 0.05
        low_x = min_x + inset
        high_x = max_x - inset
        low_y = min_y + inset
        high_y = max_y - inset
        if low_x > high_x:
            low_x = min_x
            high_x = max_x
        if low_y > high_y:
            low_y = min_y
            high_y = max_y

        offset_values = [0.0, clearance, -clearance, 2.0 * clearance, -2.0 * clearance]
        best_choice: tuple[float, float, list[tuple[float, float]], tuple[int, float, float]] | None = None
        for offset_x in offset_values:
            for offset_y in offset_values:
                candidate_x = _clamp(target_x + offset_x, low_x, high_x)
                candidate_y = _clamp(target_y + offset_y, low_y, high_y)
                if self._point_hits_obstacles(candidate_x, candidate_y, obstacle_boxes, clearance * 0.35):
                    continue
                path = self._best_axis_path(
                    start=(entry_x, entry_y),
                    end=(candidate_x, candidate_y),
                    obstacle_boxes=obstacle_boxes,
                    clearance=clearance,
                )
                intersection_count = self._count_path_intersections(path, obstacle_boxes, clearance * 0.35)
                path_length = self._polyline_length(path)
                drift = abs(candidate_x - target_x) + abs(candidate_y - target_y)
                score = (intersection_count, path_length, drift)
                if best_choice is None or score < best_choice[3]:
                    best_choice = (candidate_x, candidate_y, path, score)

        if best_choice is not None:
            return (best_choice[0], best_choice[1], best_choice[2])

        fallback_x = _clamp(target_x, low_x, high_x)
        fallback_y = _clamp(target_y, low_y, high_y)
        fallback_path = self._best_axis_path(
            start=(entry_x, entry_y),
            end=(fallback_x, fallback_y),
            obstacle_boxes=obstacle_boxes,
            clearance=clearance,
        )
        return (fallback_x, fallback_y, fallback_path)

    def _best_axis_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        obstacle_boxes: list[tuple[float, float, float, float]]
        | tuple[tuple[float, float, float, float], ...],
        clearance: float,
    ) -> list[tuple[float, float]]:
        first = self._axis_path(start, end, prefer_x_first=True)
        second = self._axis_path(start, end, prefer_x_first=False)
        first_score = (
            self._count_path_intersections(first, obstacle_boxes, clearance),
            self._polyline_length(first),
        )
        second_score = (
            self._count_path_intersections(second, obstacle_boxes, clearance),
            self._polyline_length(second),
        )
        return first if first_score <= second_score else second

    def _axis_path(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        prefer_x_first: bool,
    ) -> list[tuple[float, float]]:
        start_x, start_y = start
        end_x, end_y = end
        if abs(start_x - end_x) <= 1e-9 and abs(start_y - end_y) <= 1e-9:
            return [start]
        if abs(start_x - end_x) <= 1e-9 or abs(start_y - end_y) <= 1e-9:
            return [start, end]
        if prefer_x_first:
            return [start, (end_x, start_y), end]
        return [start, (start_x, end_y), end]

    def _point_hits_obstacles(
        self,
        x: float,
        y: float,
        obstacle_boxes: list[tuple[float, float, float, float]],
        clearance: float,
    ) -> bool:
        for min_x, max_x, min_y, max_y in obstacle_boxes:
            if min_x - clearance <= x <= max_x + clearance and min_y - clearance <= y <= max_y + clearance:
                return True
        return False

    def _count_path_intersections(
        self,
        path: list[tuple[float, float]],
        obstacle_boxes: list[tuple[float, float, float, float]]
        | tuple[tuple[float, float, float, float], ...],
        clearance: float,
    ) -> int:
        if len(path) < 2:
            return 0
        collisions = 0
        for index in range(len(path) - 1):
            start = path[index]
            end = path[index + 1]
            for box in obstacle_boxes:
                if self._segment_hits_box(start, end, box, clearance):
                    collisions += 1
        return collisions

    def _segment_hits_box(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        box: tuple[float, float, float, float],
        clearance: float,
    ) -> bool:
        min_x, max_x, min_y, max_y = box
        min_x -= clearance
        max_x += clearance
        min_y -= clearance
        max_y += clearance
        start_x, start_y = start
        end_x, end_y = end

        if abs(start_x - end_x) <= 1e-9:
            x = start_x
            if x < min_x or x > max_x:
                return False
            low_y = min(start_y, end_y)
            high_y = max(start_y, end_y)
            return max(low_y, min_y) <= min(high_y, max_y)

        if abs(start_y - end_y) <= 1e-9:
            y = start_y
            if y < min_y or y > max_y:
                return False
            low_x = min(start_x, end_x)
            high_x = max(start_x, end_x)
            return max(low_x, min_x) <= min(high_x, max_x)

        return False

    def _polyline_length(self, path: list[tuple[float, float]]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for index in range(len(path) - 1):
            dx = path[index + 1][0] - path[index][0]
            dy = path[index + 1][1] - path[index][1]
            total += abs(dx) + abs(dy)
        return total

    def _clean_polyline(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        filtered: list[tuple[float, float]] = []
        for point in points:
            if filtered and abs(filtered[-1][0] - point[0]) <= 1e-9 and abs(filtered[-1][1] - point[1]) <= 1e-9:
                continue
            filtered.append(point)
        if len(filtered) <= 2:
            return filtered

        simplified: list[tuple[float, float]] = [filtered[0]]
        for index in range(1, len(filtered) - 1):
            prev_x, prev_y = simplified[-1]
            cur_x, cur_y = filtered[index]
            next_x, next_y = filtered[index + 1]
            collinear_x = abs(prev_x - cur_x) <= 1e-9 and abs(cur_x - next_x) <= 1e-9
            collinear_y = abs(prev_y - cur_y) <= 1e-9 and abs(cur_y - next_y) <= 1e-9
            if collinear_x or collinear_y:
                continue
            simplified.append(filtered[index])
        simplified.append(filtered[-1])
        return simplified

    def _point_in_bounds(
        self,
        x: float,
        y: float,
        bounds: tuple[float, float, float, float],
    ) -> bool:
        min_x, max_x, min_y, max_y = bounds
        return min_x <= x <= max_x and min_y <= y <= max_y

    def _add_vertical_drop(
        self,
        group: SceneObject,
        x: float,
        y: float,
        tray_z: float,
        target_z: float,
        min_vertical_drop: float,
        object_type: str = "infra_vertical_drop",
    ) -> None:
        drop_target_z = min(target_z, tray_z - 0.05)
        drop_bottom_z = min(drop_target_z, tray_z - min_vertical_drop)
        drop_bottom_z = max(0.0, drop_bottom_z)
        drop_length = max(tray_z - drop_bottom_z, min_vertical_drop)
        if drop_length <= 1e-9:
            return
        group.add_child(
            SceneObject(
                id=self._next_id(object_type),
                type=object_type,
                transform=Transform(position=(x, y, drop_bottom_z)),
                mesh=create_vertical_drop(length=drop_length),
            )
        )

    def _add_hanging_cables(
        self,
        group: SceneObject,
        room_geometry: _RoomGeometry,
        tray_z: float,
        count: int,
        ratio: float,
        min_vertical_drop: float,
    ) -> None:
        count = max(1, count)
        spread_x = room_geometry.width * 0.32
        spread_y = room_geometry.depth * 0.32
        for index in range(count):
            t = 0.5 if count == 1 else index / (count - 1)
            x = room_geometry.center_x - spread_x + 2.0 * spread_x * t
            # Zig-zag along Y to create a partially chaotic hanging pattern.
            y_shift = spread_y * (0.6 if index % 2 == 0 else -0.6)
            y = room_geometry.center_y + y_shift
            length = max(min_vertical_drop * 0.75, room_geometry.height * ratio * (0.8 + 0.1 * (index % 3)))
            bottom_z = max(0.0, tray_z - length)
            self._add_vertical_drop(
                group=group,
                x=x,
                y=y,
                tray_z=tray_z,
                target_z=bottom_z,
                min_vertical_drop=max(min_vertical_drop * 0.6, 0.2),
                object_type="infra_hanging_cable",
            )

    def add_secondary_branches(self, room: SceneObject) -> None:
        network = self._active_network
        if not isinstance(network, Mapping):
            return
        group = network.get("group")
        if not isinstance(group, SceneObject):
            return
        if not _to_bool(network.get("secondary_branches"), default=True):
            return

        room_map_raw = network.get("rooms")
        room_map = room_map_raw if isinstance(room_map_raw, Mapping) else {}
        room_geometry = room_map.get(room.id)
        if not isinstance(room_geometry, _RoomGeometry):
            room_geometry = self._room_geometry(room)
        biome_profile = self._biome_profile(room)
        biome_name = str(biome_profile.get("biome", "generic")).strip().lower() or "generic"

        tray_z = float(network.get("tray_z", 2.5))
        tray_width = float(network.get("tray_width", 0.35))
        tray_thickness = float(network.get("tray_thickness", 0.12))
        min_segment_length = float(network.get("min_segment_length", 0.25))
        support_radius = float(network.get("support_radius", 0.06))
        support_spacing = float(network.get("support_spacing", 4.0))
        include_pipes = _to_bool(network.get("include_pipes"), default=True)
        pipe_z = float(network.get("pipe_z", tray_z - tray_thickness))
        pipe_radius = float(network.get("pipe_radius", 0.08))
        tray_width *= max(0.3, float(biome_profile.get("tray_width_scale", 1.0)))
        pipe_radius *= max(0.3, float(biome_profile.get("pipe_radius_scale", 1.0)))
        support_spacing *= max(0.3, float(biome_profile.get("support_spacing_scale", 1.0)))
        include_override = biome_profile.get("include_pipes")
        if include_override is not None:
            include_pipes = _to_bool(include_override, default=include_pipes)
        strict_linearity = _to_bool(biome_profile.get("strict_linearity"), default=False)
        hidden_layout = _to_bool(biome_profile.get("hidden_layout"), default=False)
        extra_pipe_lines = max(0, int(float(biome_profile.get("extra_pipe_lines", 0))))
        if hidden_layout:
            support_spacing *= 1.25

        branch_count = int(network.get("secondary_branch_count", 2))
        branch_count = max(1, min(branch_count, 6))
        branch_ratio = _clamp(float(network.get("secondary_branch_ratio", 0.32)), 0.1, 0.49)
        branch_frequency = max(0.1, float(network.get("branch_frequency", 1.0)))
        branch_count = max(1, min(24, int(round(branch_count * branch_frequency))))
        branch_count = max(
            1,
            min(24, int(round(branch_count * max(0.2, float(biome_profile.get("branch_count_scale", 1.0)))))),
        )
        branch_ratio = _clamp(
            branch_ratio * max(0.25, float(biome_profile.get("branch_ratio_scale", 1.0))),
            0.08,
            0.72,
        )

        along_x = room_geometry.width >= room_geometry.depth
        if strict_linearity:
            along_x = _to_bool(network.get("axis_long_x"), default=True)
        primary_size = room_geometry.width if along_x else room_geometry.depth
        secondary_size = room_geometry.depth if along_x else room_geometry.width
        branch_length = max(primary_size * branch_ratio, min_segment_length)
        branch_length = min(branch_length, primary_size * 0.5)
        branch_length = max(branch_length, min_segment_length)
        secondary_spread = max(secondary_size * 0.28, 0.0)
        offsets = _symmetric_offsets(branch_count, secondary_spread)

        for offset in offsets:
            if along_x:
                center_x = room_geometry.center_x
                center_y = room_geometry.center_y + offset
            else:
                center_x = room_geometry.center_x + offset
                center_y = room_geometry.center_y

            self._add_tray_segment(
                group=group,
                object_type="infra_cable_tray_branch",
                center_x=center_x,
                center_y=center_y,
                z=tray_z,
                length=branch_length,
                along_x=along_x,
                tray_width=tray_width,
                tray_thickness=tray_thickness,
            )

            half_len = branch_length / 2.0
            if along_x:
                self._add_linear_supports(
                    group=group,
                    along_x=True,
                    fixed_coord=center_y,
                    start=center_x - half_len,
                    end=center_x + half_len,
                    support_top=max(tray_z - tray_thickness / 2.0, 0.2),
                    spacing=support_spacing,
                    radius=support_radius,
                )
            else:
                self._add_linear_supports(
                    group=group,
                    along_x=False,
                    fixed_coord=center_x,
                    start=center_y - half_len,
                    end=center_y + half_len,
                    support_top=max(tray_z - tray_thickness / 2.0, 0.2),
                    spacing=support_spacing,
                    radius=support_radius,
                )

            if include_pipes:
                self._add_pipe_segment(
                    group=group,
                    object_type="infra_pipe_branch",
                    center_x=center_x,
                    center_y=center_y,
                    z=pipe_z,
                    length=branch_length,
                    along_x=along_x,
                    radius=pipe_radius,
                )
                if extra_pipe_lines > 0 and biome_name in {"boiler", "maintenance"}:
                    line_count = min(extra_pipe_lines, 3)
                    spacing = max(pipe_radius * 2.8, 0.08)
                    offsets = _symmetric_offsets(line_count + 1, spacing * line_count * 0.5)
                    for pipe_offset in offsets:
                        if abs(pipe_offset) <= 1e-9:
                            continue
                        aux_x = center_x
                        aux_y = center_y
                        if along_x:
                            aux_y += pipe_offset
                        else:
                            aux_x += pipe_offset
                        self._add_pipe_segment(
                            group=group,
                            object_type="infra_pipe_branch_aux",
                            center_x=aux_x,
                            center_y=aux_y,
                            z=pipe_z,
                            length=branch_length,
                            along_x=along_x,
                            radius=pipe_radius * 0.88,
                        )

    def _iter_room_roots(self, scene: Scene) -> list[SceneObject]:
        return [obj for obj in scene.objects if obj.type.startswith("room_")]

    def _room_geometry(self, room: SceneObject) -> _RoomGeometry:
        center_x, center_y, _ = room.transform.position

        width = 0.0
        depth = 0.0
        for child in room.children:
            if child.type != "floor" or child.mesh is None or not child.mesh.vertices:
                continue
            xs = [vertex[0] for vertex in child.mesh.vertices]
            ys = [vertex[1] for vertex in child.mesh.vertices]
            width = (max(xs) - min(xs)) * abs(child.transform.scale[0])
            depth = (max(ys) - min(ys)) * abs(child.transform.scale[1])
            break

        if width <= 1e-6 or depth <= 1e-6:
            width = _to_positive_float(self.settings.get("fallback_room_width"), 8.0, "infrastructure.fallback_room_width")
            depth = _to_positive_float(self.settings.get("fallback_room_depth"), 8.0, "infrastructure.fallback_room_depth")

        height = 0.0
        for child in room.children:
            if child.type == "ceiling":
                height = max(height, child.transform.position[2])
            if child.mesh is None or not child.mesh.vertices:
                continue
            local_top = max(vertex[2] for vertex in child.mesh.vertices) * abs(child.transform.scale[2])
            top = child.transform.position[2] + local_top
            height = max(height, top)
        if height <= 1e-6:
            height = _to_positive_float(self.settings.get("fallback_room_height"), 4.0, "infrastructure.fallback_room_height")

        return _RoomGeometry(
            center_x=center_x,
            center_y=center_y,
            width=max(width, 1.0),
            depth=max(depth, 1.0),
            height=max(height, 2.0),
        )

    def _add_tray_segment(
        self,
        group: SceneObject,
        object_type: str,
        center_x: float,
        center_y: float,
        z: float,
        length: float,
        along_x: bool,
        tray_width: float,
        tray_thickness: float,
    ) -> None:
        length = max(length, 0.0)
        if length <= 1e-9:
            return

        if along_x:
            mesh = create_box(width=length, height=tray_thickness, depth=tray_width)
        else:
            mesh = create_box(width=tray_width, height=tray_thickness, depth=length)

        group.add_child(
            SceneObject(
                id=self._next_id(object_type),
                type=object_type,
                transform=Transform(position=(center_x, center_y, z)),
                mesh=mesh,
            )
        )

        if _to_bool(self.settings.get("tray_cable_bundle_enabled", True), default=True):
            bundle_count = max(
                1,
                _to_positive_int(
                    self.settings.get("tray_cable_bundle_count"),
                    2,
                    "infrastructure.tray_cable_bundle_count",
                ),
            )
            bundle_radius = _to_positive_float(
                self.settings.get("tray_cable_bundle_radius"),
                max(0.014, min(tray_width, tray_thickness) * 0.16),
                "infrastructure.tray_cable_bundle_radius",
            )
            max_offset = _to_positive_float(
                self.settings.get("tray_cable_bundle_max_offset"),
                max(tray_width * 0.22, bundle_radius * 1.25),
                "infrastructure.tray_cable_bundle_max_offset",
            )
            offsets = _symmetric_offsets(bundle_count, max_offset)
            bundle_z = z + tray_thickness / 2.0 + bundle_radius * 1.2
            for offset in offsets:
                bundle_x = center_x + offset if not along_x else center_x
                bundle_y = center_y + offset if along_x else center_y
                self._add_cable_bundle_segment(
                    group=group,
                    center_x=bundle_x,
                    center_y=bundle_y,
                    z=bundle_z,
                    length=length,
                    along_x=along_x,
                    radius=bundle_radius,
                )

    def _add_pipe_segment(
        self,
        group: SceneObject,
        object_type: str,
        center_x: float,
        center_y: float,
        z: float,
        length: float,
        along_x: bool,
        radius: float,
    ) -> None:
        length = max(length, 0.0)
        if length <= 1e-9:
            return

        mesh = create_beam(
            length=length,
            profile_type={"type": "circular", "radius": radius, "segments": 18},
        )
        rotation = (0.0, 0.0, 0.0) if along_x else (0.0, 0.0, 90.0)
        group.add_child(
            SceneObject(
                id=self._next_id(object_type),
                type=object_type,
                transform=Transform(position=(center_x, center_y, z), rotation=rotation),
                mesh=mesh,
            )
        )

        if _to_bool(self.settings.get("pipe_casing_enabled", True), default=True):
            casing_thickness = _to_positive_float(
                self.settings.get("pipe_casing_thickness"),
                max(radius * 0.28, 0.015),
                "infrastructure.pipe_casing_thickness",
            )
            casing_scale = _to_positive_float(
                self.settings.get("pipe_casing_scale"),
                1.22,
                "infrastructure.pipe_casing_scale",
            )
            casing_radius = max(radius + casing_thickness, radius * casing_scale)
            casing_mesh = create_beam(
                length=length,
                profile_type={"type": "circular", "radius": casing_radius, "segments": 18},
            )
            group.add_child(
                SceneObject(
                    id=self._next_id("infra_pipe_casing"),
                    type="infra_pipe_casing",
                    transform=Transform(position=(center_x, center_y, z), rotation=rotation),
                    mesh=casing_mesh,
                )
            )

    def _add_cable_bundle_segment(
        self,
        group: SceneObject,
        center_x: float,
        center_y: float,
        z: float,
        length: float,
        along_x: bool,
        radius: float,
    ) -> None:
        segment_length = max(length, 0.0)
        if segment_length <= 1e-9:
            return

        bundle_mesh = create_beam(
            length=segment_length,
            profile_type={"type": "circular", "radius": radius, "segments": 14},
        )
        rotation = (0.0, 0.0, 0.0) if along_x else (0.0, 0.0, 90.0)
        group.add_child(
            SceneObject(
                id=self._next_id("infra_cable_bundle"),
                type="infra_cable_bundle",
                transform=Transform(position=(center_x, center_y, z), rotation=rotation),
                mesh=bundle_mesh,
            )
        )

    def _add_linear_supports(
        self,
        group: SceneObject,
        along_x: bool,
        fixed_coord: float,
        start: float,
        end: float,
        support_top: float,
        spacing: float,
        radius: float,
    ) -> None:
        start_value = min(start, end)
        end_value = max(start, end)
        for value in self._equal_interval_positions(start_value, end_value, spacing):
            if along_x:
                x = value
                y = fixed_coord
            else:
                x = fixed_coord
                y = value
            self._add_support(group=group, x=x, y=y, top=support_top, radius=radius)

    def _equal_interval_positions(self, start: float, end: float, spacing: float) -> list[float]:
        start_value = min(start, end)
        end_value = max(start, end)
        span = end_value - start_value
        if span <= 1e-9:
            return [start_value]

        interval = max(spacing, 0.1)
        segment_count = max(1, int(round(span / interval)))
        return [
            start_value + span * index / segment_count
            for index in range(segment_count + 1)
        ]

    def _add_support(
        self,
        group: SceneObject,
        x: float,
        y: float,
        top: float,
        radius: float,
    ) -> None:
        support_height = max(top, radius * 4.0)
        group.add_child(
            SceneObject(
                id=self._next_id("infra_support"),
                type="infra_support",
                transform=Transform(position=(x, y, support_height / 2.0)),
                mesh=create_column(radius=radius, height=support_height, segments=12),
            )
        )

    def _add_junction_node(
        self,
        group: SceneObject,
        x: float,
        y: float,
        z: float,
        size: float,
    ) -> None:
        group.add_child(
            SceneObject(
                id=self._next_id("infra_junction_node"),
                type="infra_junction_node",
                transform=Transform(position=(x, y, z)),
                mesh=create_junction_node(size=size),
            )
        )

    def _next_id(self, prefix: str) -> str:
        self._id_counter += 1
        return f"{prefix}_{self._id_counter}"
