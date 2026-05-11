from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from ..parametric.primitives import create_box, create_floor, create_wall
from ..scene.scene_graph import SceneObject, Transform

_VALID_DOOR_SIDES = {"north", "south", "east", "west"}


@dataclass(frozen=True)
class RoomParams:
    width: float
    depth: float
    height: float
    wall_thickness: float
    door_count: int
    window_count: int
    door_sides: tuple[str, ...] = ()
    door_height: float | None = None


class RoomGenerator:
    """Generate room building blocks as scene objects with meshes."""

    def __init__(self) -> None:
        self._room_index = 0

    def generate_room(self, params: RoomParams | Mapping[str, object]) -> list[SceneObject]:
        cfg = self._coerce_params(params)
        room_id = self._next_room_id()

        objects: list[SceneObject] = []
        objects.extend(self._generate_floor_and_ceiling(room_id, cfg))
        objects.extend(self._generate_walls(room_id, cfg))
        objects.extend(self._generate_openings(room_id, cfg))
        return objects

    def _next_room_id(self) -> str:
        self._room_index += 1
        return f"room_{self._room_index}"

    def _coerce_params(self, params: RoomParams | Mapping[str, object]) -> RoomParams:
        if isinstance(params, RoomParams):
            cfg = params
        elif isinstance(params, Mapping):
            cfg = RoomParams(
                width=float(params["width"]),
                depth=float(params["depth"]),
                height=float(params["height"]),
                wall_thickness=float(params["wall_thickness"]),
                door_count=int(params["door_count"]),
                window_count=int(params["window_count"]),
                door_sides=self._coerce_door_sides(params.get("door_sides", ())),
                door_height=(
                    float(params["door_height"])
                    if "door_height" in params and params["door_height"] is not None
                    else None
                ),
            )
        else:
            raise TypeError("params must be RoomParams or mapping.")

        self._validate(cfg)
        return cfg

    def _coerce_door_sides(self, raw: object) -> tuple[str, ...]:
        if raw is None:
            return ()
        if not isinstance(raw, (list, tuple)):
            raise TypeError("door_sides must be a list/tuple of side names.")
        return tuple(str(item).strip().lower() for item in raw if str(item).strip())

    def _validate(self, params: RoomParams) -> None:
        if params.width <= 0.0:
            raise ValueError("width must be > 0.")
        if params.depth <= 0.0:
            raise ValueError("depth must be > 0.")
        if params.height <= 0.0:
            raise ValueError("height must be > 0.")
        if params.wall_thickness <= 0.0:
            raise ValueError("wall_thickness must be > 0.")
        if params.door_count < 0:
            raise ValueError("door_count must be >= 0.")
        if params.window_count < 0:
            raise ValueError("window_count must be >= 0.")
        for side in params.door_sides:
            if side not in _VALID_DOOR_SIDES:
                raise ValueError("door_sides values must be one of: north, south, east, west.")
        if params.door_height is not None:
            if params.door_height <= 0.0:
                raise ValueError("door_height must be > 0 when provided.")
            if params.door_height >= params.height:
                raise ValueError("door_height must be < room height.")

        min_span = min(params.width, params.depth)
        if params.wall_thickness >= min_span / 2.0:
            raise ValueError("wall_thickness must be < min(width, depth) / 2.")

    def _generate_floor_and_ceiling(self, room_id: str, params: RoomParams) -> list[SceneObject]:
        floor = SceneObject(
            id=f"{room_id}_floor",
            type="floor",
            transform=Transform(position=(0.0, 0.0, 0.0)),
            mesh=create_floor(width=params.width, depth=params.depth),
        )
        ceiling = SceneObject(
            id=f"{room_id}_ceiling",
            type="ceiling",
            transform=Transform(position=(0.0, 0.0, params.height)),
            mesh=create_floor(width=params.width, depth=params.depth),
        )
        return [floor, ceiling]

    def _generate_walls(self, room_id: str, params: RoomParams) -> list[SceneObject]:
        wall_z = params.height / 2.0
        y_offset = params.depth / 2.0 - params.wall_thickness / 2.0
        x_offset = params.width / 2.0 - params.wall_thickness / 2.0
        door_sides = self._resolve_door_sides(params)
        side_counts = Counter(door_sides)
        door_height = self._resolve_door_height(params)

        walls: list[SceneObject] = []
        for side in ("south", "north", "west", "east"):
            openings_count = side_counts.get(side, 0)
            if openings_count <= 0:
                walls.append(self._build_full_wall(room_id, side, params, wall_z, x_offset, y_offset))
                continue
            walls.extend(
                self._build_wall_segments_with_openings(
                    room_id=room_id,
                    side=side,
                    openings_count=openings_count,
                    params=params,
                    wall_z=wall_z,
                    door_height=door_height,
                    x_offset=x_offset,
                    y_offset=y_offset,
                )
            )
        return walls

    def _build_full_wall(
        self,
        room_id: str,
        side: str,
        params: RoomParams,
        wall_z: float,
        x_offset: float,
        y_offset: float,
    ) -> SceneObject:
        if side in {"south", "north"}:
            y = -y_offset if side == "south" else y_offset
            return SceneObject(
                id=f"{room_id}_wall_{side}",
                type="wall",
                transform=Transform(position=(0.0, y, wall_z)),
                mesh=create_wall(
                    length=params.width,
                    height=params.height,
                    thickness=params.wall_thickness,
                ),
            )

        x = -x_offset if side == "west" else x_offset
        return SceneObject(
            id=f"{room_id}_wall_{side}",
            type="wall",
            transform=Transform(
                position=(x, 0.0, wall_z),
                rotation=(0.0, 0.0, 90.0),
            ),
            mesh=create_wall(
                length=params.depth,
                height=params.height,
                thickness=params.wall_thickness,
            ),
        )

    def _build_wall_segments_with_openings(
        self,
        room_id: str,
        side: str,
        openings_count: int,
        params: RoomParams,
        wall_z: float,
        door_height: float,
        x_offset: float,
        y_offset: float,
    ) -> list[SceneObject]:
        span = (
            params.width - 2.0 * params.wall_thickness
            if side in {"south", "north"}
            else params.depth - 2.0 * params.wall_thickness
        )
        if span <= 0.0:
            raise ValueError("Not enough room span to place wall openings.")

        section = span / (2 * openings_count + 1)
        if section <= 0.0:
            raise ValueError("Computed wall segment length must be > 0.")

        segments: list[SceneObject] = []
        left = -span / 2.0
        total_sections = 2 * openings_count + 1
        lintel_height = max(0.0, params.height - door_height)
        lintel_z = door_height + lintel_height / 2.0
        for idx in range(total_sections):
            center = left + section * (idx + 0.5)
            is_opening = idx % 2 == 1

            if side in {"south", "north"}:
                y = -y_offset if side == "south" else y_offset
                position_full = (center, y, wall_z)
                position_lintel = (center, y, lintel_z)
                rotation = (0.0, 0.0, 0.0)
            else:
                x = -x_offset if side == "west" else x_offset
                position_full = (x, center, wall_z)
                position_lintel = (x, center, lintel_z)
                rotation = (0.0, 0.0, 90.0)

            if not is_opening:
                segments.append(
                    SceneObject(
                        id=f"{room_id}_wall_{side}_segment_{idx + 1}",
                        type="wall",
                        transform=Transform(position=position_full, rotation=rotation),
                        mesh=create_wall(
                            length=section,
                            height=params.height,
                            thickness=params.wall_thickness,
                        ),
                    )
                )
                continue

            if lintel_height > 0.02:
                segments.append(
                    SceneObject(
                        id=f"{room_id}_wall_{side}_lintel_{idx + 1}",
                        type="wall",
                        transform=Transform(position=position_lintel, rotation=rotation),
                        mesh=create_wall(
                            length=section,
                            height=lintel_height,
                            thickness=params.wall_thickness,
                        ),
                    )
                )

        return segments

    def _generate_openings(self, room_id: str, params: RoomParams) -> list[SceneObject]:
        openings: list[SceneObject] = []

        if params.door_count > 0:
            openings.extend(self._generate_door_openings(room_id, params))

        if params.window_count > 0:
            openings.extend(self._generate_window_openings(room_id, params))

        return openings

    def _generate_door_openings(self, room_id: str, params: RoomParams) -> list[SceneObject]:
        door_sides = self._resolve_door_sides(params)
        door_height = self._resolve_door_height(params)
        door_z = door_height / 2.0

        openings: list[SceneObject] = []
        side_counts = Counter(door_sides)
        for side in ("south", "north", "west", "east"):
            count = side_counts.get(side, 0)
            if count <= 0:
                continue
            openings.extend(
                self._generate_side_door_openings(
                    room_id=room_id,
                    params=params,
                    side=side,
                    count=count,
                    door_height=door_height,
                    door_z=door_z,
                )
            )

        return openings

    def _resolve_door_height(self, params: RoomParams) -> float:
        if params.door_height is not None:
            base = params.door_height
        else:
            base = min(2.3, params.height * 0.72)
        return max(1.8, min(base, params.height * 0.82))

    def _resolve_door_sides(self, params: RoomParams) -> tuple[str, ...]:
        if params.door_count <= 0:
            return ()

        base = params.door_sides if params.door_sides else ("south",)
        sides = list(base)
        while len(sides) < params.door_count:
            sides.append(base[(len(sides) - len(base)) % len(base)])
        return tuple(sides)

    def _generate_side_door_openings(
        self,
        room_id: str,
        params: RoomParams,
        side: str,
        count: int,
        door_height: float,
        door_z: float,
    ) -> list[SceneObject]:
        span = (
            params.width - 2.0 * params.wall_thickness
            if side in {"south", "north"}
            else params.depth - 2.0 * params.wall_thickness
        )
        if span <= 0.0:
            raise ValueError("Not enough room span to place door openings.")

        section = span / (2 * count + 1)
        half_w = params.width / 2.0 - params.wall_thickness / 2.0
        half_d = params.depth / 2.0 - params.wall_thickness / 2.0

        openings: list[SceneObject] = []
        left = -span / 2.0
        for index in range(count):
            center = left + section * (2 * index + 1.5)
            if side == "south":
                position = (center, -half_d, door_z)
                rotation = (0.0, 0.0, 0.0)
            elif side == "north":
                position = (center, half_d, door_z)
                rotation = (0.0, 0.0, 0.0)
            elif side == "west":
                position = (-half_w, center, door_z)
                rotation = (0.0, 0.0, 90.0)
            else:  # side == "east"
                position = (half_w, center, door_z)
                rotation = (0.0, 0.0, 90.0)

            openings.append(
                SceneObject(
                    id=f"{room_id}_door_opening_{side}_{index + 1}",
                    type="door_opening",
                    transform=Transform(position=position, rotation=rotation),
                    mesh=None,
                )
            )
        return openings

    def _generate_window_openings(self, room_id: str, params: RoomParams) -> list[SceneObject]:
        span = params.width - 2.0 * params.wall_thickness
        if span <= 0.0:
            raise ValueError("Not enough room width to place window openings.")

        section = span / (2 * params.window_count + 1)
        window_width = section
        window_height = params.height * (params.window_count + 1) / (
            params.door_count + params.window_count + 2
        )
        window_y = params.depth / 2.0 - params.wall_thickness / 2.0
        window_z = params.height / 2.0

        openings: list[SceneObject] = []
        left = -span / 2.0
        for index in range(params.window_count):
            center_x = left + section * (2 * index + 1.5)
            openings.append(
                SceneObject(
                    id=f"{room_id}_window_opening_{index + 1}",
                    type="window_opening",
                    transform=Transform(position=(center_x, window_y, window_z)),
                    mesh=create_box(
                        width=window_width,
                        height=window_height,
                        depth=params.wall_thickness,
                    ),
                )
            )

        return openings
