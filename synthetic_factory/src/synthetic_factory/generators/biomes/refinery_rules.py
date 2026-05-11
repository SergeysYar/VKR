from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Callable, Mapping, Sequence

from ...parametric.refinery_primitives import create_refinery_pipe
from .shared import BiomeRoom
from .refinery_utils import (
    Vector3,
    iter_horizontal_segments,
    lerp,
    nearest_component_pair,
    path_intersects_paths,
)

EmitFn = Callable[[str, object, Vector3, Vector3], None]


@dataclass(frozen=True)
class MinClearanceRule:
    min_distance: float

    def enforce_axis_positions(
        self,
        axis_positions: list[float],
        axis_limit: float,
        object_radius: float,
    ) -> tuple[list[float], bool]:
        if len(axis_positions) <= 1:
            return (list(axis_positions), False)

        required = max(0.0, 2.0 * object_radius + self.min_distance)
        if required <= 1e-9:
            return (sorted(axis_positions), False)

        values = sorted(float(value) for value in axis_positions)
        changed = False
        for idx in range(1, len(values)):
            minimum = values[idx - 1] + required
            if values[idx] < minimum:
                values[idx] = minimum
                changed = True

        overflow = values[-1] - axis_limit
        if overflow > 1e-9:
            values = [value - overflow for value in values]
            changed = True

        underflow = -axis_limit - values[0]
        if underflow > 1e-9:
            values = [value + underflow for value in values]
            changed = True

        clamped = [max(-axis_limit, min(axis_limit, value)) for value in values]
        if any(abs(left - right) > 1e-9 for left, right in zip(values, clamped)):
            changed = True
        return (clamped, changed)


@dataclass(frozen=True)
class AccessRule:
    walkway_width: float
    margin: float = 0.05

    def blocked_lane(self, object_depth: float) -> float:
        return self.walkway_width / 2.0 + object_depth / 2.0 + self.margin

    def place_outside_lane(self, lateral: float, object_depth: float) -> float:
        blocked = self.blocked_lane(object_depth)
        if abs(lateral) >= blocked:
            return lateral
        return blocked if lateral >= 0.0 else -blocked

    def ensure_lateral_access(self, lateral: float, object_depth: float, cross_limit: float) -> float:
        corrected = self.place_outside_lane(lateral, object_depth)
        if cross_limit <= 0.0:
            return corrected
        return max(-cross_limit, min(cross_limit, corrected))


@dataclass(frozen=True)
class HeightRule:
    room_height: float
    margin: float = 0.04

    def clamp_size(self, value: float, minimum: float) -> float:
        upper = max(minimum, self.room_height - 2.0 * self.margin)
        return max(minimum, min(float(value), upper))

    def clamp_center(self, center_z: float, object_height: float) -> float:
        half = max(object_height / 2.0, 0.0)
        low = self.margin + half
        high = self.room_height - self.margin - half
        if low > high:
            return self.room_height / 2.0
        return max(low, min(float(center_z), high))

    def clamp_path(self, path: Sequence[Vector3]) -> list[Vector3]:
        return [
            (
                point[0],
                point[1],
                max(self.margin, min(self.room_height - self.margin, point[2])),
            )
            for point in path
        ]


HeightConstraint = HeightRule


@dataclass(frozen=True)
class ConnectivityRule:
    def ensure_backbone(self, pipe_count: int, emit: EmitFn, room: BiomeRoom) -> int:
        if pipe_count > 0:
            return 0
        mesh = create_refinery_pipe(radius=0.18, length=max(room.width, room.depth) * 0.6, bend_angle=0.0)
        emit("refinery_pipe_backbone", mesh, (0.0, 0.0, room.height * 0.6), (0.0, 0.0, 0.0))
        return 1

    def ensure_all_nodes_connected(
        self,
        node_points: Mapping[str, Vector3],
        edges: Sequence[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        node_ids = list(node_points.keys())
        if len(node_ids) <= 1:
            return []

        adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
        for source_id, target_id in edges:
            if source_id in adjacency and target_id in adjacency:
                adjacency[source_id].add(target_id)
                adjacency[target_id].add(source_id)

        components: list[set[str]] = []
        visited: set[str] = set()
        for node_id in node_ids:
            if node_id in visited:
                continue
            stack = [node_id]
            component: set[str] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(adjacency[current] - visited)
            components.append(component)

        if len(components) <= 1:
            return []

        new_edges: list[tuple[str, str]] = []
        anchor_component = components[0]
        for component in components[1:]:
            source_id, target_id = nearest_component_pair(anchor_component, component, node_points)
            new_edges.append((source_id, target_id))
            anchor_component = anchor_component.union(component)
        return new_edges


@dataclass(frozen=True)
class NoPipeCollisionRule:
    margin: float = 0.12
    vertical_step: float = 0.28

    def resolve(
        self,
        path: Sequence[Vector3],
        existing_paths: Sequence[Sequence[Vector3]],
        height_rule: HeightRule,
    ) -> list[Vector3]:
        corrected = list(path)
        attempt = 0
        while attempt < 18 and path_intersects_paths(corrected, existing_paths, self.margin):
            corrected = [(point[0], point[1], point[2] + self.vertical_step) for point in corrected]
            corrected = height_rule.clamp_path(corrected)
            attempt += 1
        return corrected


@dataclass(frozen=True)
class SupportRule:
    spacing: float
    min_support_height: float = 0.42

    def support_points(self, path: Sequence[Vector3]) -> list[tuple[Vector3, float]]:
        points: list[tuple[Vector3, float]] = []
        spacing = max(self.spacing, 0.5)
        for start, end, length in iter_horizontal_segments(path):
            if length <= spacing * 0.9:
                continue
            count = max(1, int(length // spacing))
            yaw = degrees(atan2(end[1] - start[1], end[0] - start[0]))
            for index in range(1, count + 1):
                t = index / (count + 1)
                px, py, pz = lerp(start, end, t)
                points.append(((px, py, max(pz, self.min_support_height)), yaw))
        return points
