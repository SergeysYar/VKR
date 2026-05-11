from __future__ import annotations

from math import atan2, degrees, sqrt
from typing import Mapping, Sequence

from ...parametric.primitives import create_box, create_column
from ...parametric.refinery_primitives import create_valve

Vector3 = tuple[float, float, float]


def translate_mesh(mesh: object, dx: float, dy: float, dz: float) -> None:
    mesh.transform(
        (
            (1.0, 0.0, 0.0, dx),
            (0.0, 1.0, 0.0, dy),
            (0.0, 0.0, 1.0, dz),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def create_valve_cluster_mesh(size: float) -> object:
    body = create_box(
        width=max(size * 0.54, 0.16),
        height=max(size * 0.34, 0.12),
        depth=max(size * 0.54, 0.16),
    )
    stem = create_column(radius=max(size * 0.1, 0.03), height=max(size * 0.62, 0.16), segments=12)
    translate_mesh(stem, 0.0, 0.0, max(size * 0.36, 0.08))
    body.merge(stem)

    arm_size = max(size * 0.42, 0.14)
    offsets = (
        (size * 0.34, 0.0, 0.0),
        (-size * 0.34, 0.0, 0.0),
        (0.0, size * 0.34, 0.0),
    )
    for dx, dy, dz in offsets:
        arm = create_valve(size=arm_size)
        translate_mesh(arm, dx, dy, dz)
        body.merge(arm)
    return body


def create_small_control_box_mesh(size: float) -> object:
    width = max(size, 0.22)
    depth = max(size * 0.58, 0.16)
    height = max(size * 0.62, 0.18)
    box = create_box(width=width, height=height, depth=depth)
    panel = create_box(width=width * 0.9, height=height * 0.62, depth=max(depth * 0.1, 0.02))
    translate_mesh(panel, 0.0, depth / 2.0 - max(depth * 0.1, 0.02) / 2.0, height * 0.06)
    box.merge(panel)

    knob = create_column(radius=max(size * 0.08, 0.02), height=max(size * 0.08, 0.02), segments=10)
    translate_mesh(knob, width * 0.22, depth / 2.0, 0.0)
    box.merge(knob)
    return box


def create_sensor_unit_mesh(size: float) -> object:
    mast = create_column(radius=max(size * 0.18, 0.012), height=max(size * 1.2, 0.08), segments=10)
    head = create_box(
        width=max(size * 0.55, 0.04),
        height=max(size * 0.32, 0.03),
        depth=max(size * 0.34, 0.03),
    )
    translate_mesh(head, 0.0, 0.0, max(size * 0.75, 0.05))
    mast.merge(head)
    return mast


def route_corners(route: Sequence[Vector3]) -> list[tuple[Vector3, float]]:
    corners: list[tuple[Vector3, float]] = []
    if len(route) < 3:
        return corners
    for idx in range(1, len(route) - 1):
        prev_point = route[idx - 1]
        current = route[idx]
        next_point = route[idx + 1]
        prev_len = distance(prev_point, current)
        next_len = distance(current, next_point)
        if prev_len <= 1e-9 or next_len <= 1e-9:
            continue

        prev_dir = (
            (current[0] - prev_point[0]) / prev_len,
            (current[1] - prev_point[1]) / prev_len,
            (current[2] - prev_point[2]) / prev_len,
        )
        next_dir = (
            (next_point[0] - current[0]) / next_len,
            (next_point[1] - current[1]) / next_len,
            (next_point[2] - current[2]) / next_len,
        )
        alignment = prev_dir[0] * next_dir[0] + prev_dir[1] * next_dir[1] + prev_dir[2] * next_dir[2]
        if abs(alignment) >= 0.97:
            continue

        direction = (next_point[0] - prev_point[0], next_point[1] - prev_point[1])
        yaw = degrees(atan2(direction[1], direction[0]))
        corners.append((current, yaw))
    return corners


def nearest_component_pair(
    left_component: set[str],
    right_component: set[str],
    node_points: Mapping[str, Vector3],
) -> tuple[str, str]:
    best_pair = (next(iter(left_component)), next(iter(right_component)))
    best_distance = float("inf")
    for left_id in left_component:
        left_point = node_points.get(left_id)
        if left_point is None:
            continue
        for right_id in right_component:
            right_point = node_points.get(right_id)
            if right_point is None:
                continue
            dist = distance(left_point, right_point)
            if dist < best_distance:
                best_distance = dist
                best_pair = (left_id, right_id)
    return best_pair


def iter_horizontal_segments(path: Sequence[Vector3]) -> list[tuple[Vector3, Vector3, float]]:
    segments: list[tuple[Vector3, Vector3, float]] = []
    if len(path) < 2:
        return segments
    for idx in range(len(path) - 1):
        start = path[idx]
        end = path[idx + 1]
        if abs(start[2] - end[2]) > 1e-9:
            continue
        length = distance(start, end)
        if length <= 1e-9:
            continue
        segments.append((start, end, length))
    return segments


def path_intersects_paths(
    candidate: Sequence[Vector3],
    existing_paths: Sequence[Sequence[Vector3]],
    margin: float,
) -> bool:
    if len(candidate) < 2 or not existing_paths:
        return False
    segments_a = iter_horizontal_segments(candidate)
    if not segments_a:
        return False

    margin = max(float(margin), 1e-6)
    for other_path in existing_paths:
        segments_b = iter_horizontal_segments(other_path)
        for seg_a_start, seg_a_end, _ in segments_a:
            min_ax = min(seg_a_start[0], seg_a_end[0]) - margin
            max_ax = max(seg_a_start[0], seg_a_end[0]) + margin
            min_ay = min(seg_a_start[1], seg_a_end[1]) - margin
            max_ay = max(seg_a_start[1], seg_a_end[1]) + margin
            z_a = (seg_a_start[2] + seg_a_end[2]) / 2.0

            for seg_b_start, seg_b_end, _ in segments_b:
                z_b = (seg_b_start[2] + seg_b_end[2]) / 2.0
                if abs(z_a - z_b) > margin * 0.75:
                    continue
                min_bx = min(seg_b_start[0], seg_b_end[0]) - margin
                max_bx = max(seg_b_start[0], seg_b_end[0]) + margin
                min_by = min(seg_b_start[1], seg_b_end[1]) - margin
                max_by = max(seg_b_start[1], seg_b_end[1]) + margin

                overlap_x = min(max_ax, max_bx) - max(min_ax, min_bx)
                overlap_y = min(max_ay, max_by) - max(min_ay, min_by)
                if overlap_x >= 0.0 and overlap_y >= 0.0:
                    return True
    return False


def distance(left: Vector3, right: Vector3) -> float:
    return sqrt((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2 + (right[2] - left[2]) ** 2)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def compose_edge_route(
    *,
    start: Vector3,
    end: Vector3,
    axis_along_x: bool,
    lane_offset: float,
    lane_z: float,
    axis_limit: float,
    cross_limit: float,
) -> list[Vector3]:
    lane_z = max(lane_z, start[2] + 0.22, end[2] + 0.22)
    lane_cross = clamp(lane_offset, -cross_limit, cross_limit)
    axis_mid = clamp((start[0] + end[0]) / 2.0, -axis_limit, axis_limit) if axis_along_x else clamp((start[1] + end[1]) / 2.0, -axis_limit, axis_limit)

    if axis_along_x:
        return dedupe_path(
            [
                start,
                (start[0], start[1], lane_z),
                (start[0], lane_cross, lane_z),
                (axis_mid, lane_cross, lane_z),
                (end[0], lane_cross, lane_z),
                (end[0], end[1], lane_z),
                (end[0], end[1], end[2]),
            ]
        )

    return dedupe_path(
        [
            start,
            (start[0], start[1], lane_z),
            (lane_cross, start[1], lane_z),
            (lane_cross, axis_mid, lane_z),
            (lane_cross, end[1], lane_z),
            (end[0], end[1], lane_z),
            (end[0], end[1], end[2]),
        ]
    )


def dedupe_path(points: Sequence[Vector3]) -> list[Vector3]:
    if not points:
        return []
    result: list[Vector3] = [points[0]]
    for point in points[1:]:
        if distance(result[-1], point) <= 1e-9:
            continue
        result.append(point)
    return result


def smooth_path(points: Sequence[Vector3], chamfer: float) -> list[Vector3]:
    cleaned = dedupe_path(points)
    if len(cleaned) <= 2:
        return cleaned
    chamfer = max(0.01, float(chamfer))

    result: list[Vector3] = [cleaned[0]]
    for idx in range(1, len(cleaned) - 1):
        prev_point = cleaned[idx - 1]
        current = cleaned[idx]
        next_point = cleaned[idx + 1]
        prev_len = distance(prev_point, current)
        next_len = distance(current, next_point)
        if prev_len <= 1e-9 or next_len <= 1e-9:
            continue

        prev_dir = (
            (current[0] - prev_point[0]) / prev_len,
            (current[1] - prev_point[1]) / prev_len,
            (current[2] - prev_point[2]) / prev_len,
        )
        next_dir = (
            (next_point[0] - current[0]) / next_len,
            (next_point[1] - current[1]) / next_len,
            (next_point[2] - current[2]) / next_len,
        )
        alignment = prev_dir[0] * next_dir[0] + prev_dir[1] * next_dir[1] + prev_dir[2] * next_dir[2]
        if abs(alignment) >= 0.995:
            result.append(current)
            continue

        back = min(chamfer, prev_len * 0.45)
        forward = min(chamfer, next_len * 0.45)
        pre_point = (
            current[0] - prev_dir[0] * back,
            current[1] - prev_dir[1] * back,
            current[2] - prev_dir[2] * back,
        )
        post_point = (
            current[0] + next_dir[0] * forward,
            current[1] + next_dir[1] * forward,
            current[2] + next_dir[2] * forward,
        )
        result.append(pre_point)
        result.append(current)
        result.append(post_point)

    result.append(cleaned[-1])
    return dedupe_path(result)


def polyline_length(path: Sequence[Vector3]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(distance(path[idx], path[idx + 1]) for idx in range(len(path) - 1))


def largest_horizontal_segment(path: Sequence[Vector3]) -> tuple[Vector3, Vector3, float] | None:
    best: tuple[Vector3, Vector3, float] | None = None
    best_length = 0.0
    for idx in range(len(path) - 1):
        start = path[idx]
        end = path[idx + 1]
        if abs(start[2] - end[2]) > 1e-9:
            continue
        length = distance(start, end)
        if length <= best_length:
            continue
        best = (start, end, length)
        best_length = length
    return best


def lerp(start: Vector3, end: Vector3, t: float) -> Vector3:
    ratio = clamp(float(t), 0.0, 1.0)
    return (
        start[0] + (end[0] - start[0]) * ratio,
        start[1] + (end[1] - start[1]) * ratio,
        start[2] + (end[2] - start[2]) * ratio,
    )
