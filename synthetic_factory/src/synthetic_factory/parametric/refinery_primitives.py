from __future__ import annotations

from math import cos, pi, radians, sin, sqrt
from typing import Iterable, Sequence

from ..geometry.mesh import Mesh
from .primitives import create_beam, create_box, create_column

MatrixLike = Iterable[Iterable[float]]
Vector3 = tuple[float, float, float]


def _require_positive(value: float, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return number


def _require_non_negative(value: float, name: str) -> float:
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return number


def _apply_transform(mesh: Mesh, transform: MatrixLike | None) -> Mesh:
    if transform is not None:
        mesh.transform(transform)
    return mesh


def _translate(mesh: Mesh, dx: float, dy: float, dz: float) -> None:
    mesh.vertices = [(x + dx, y + dy, z + dz) for x, y, z in mesh.vertices]


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(v: Vector3) -> Vector3:
    length = sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _tube_between(
    start: Vector3,
    end: Vector3,
    radius: float,
    segment_count: int = 20,
) -> Mesh:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    length = sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1e-9:
        raise ValueError("Pipe segment length must be > 0.")

    mesh = create_beam(
        length=length,
        profile_type={
            "type": "circular",
            "radius": radius,
            "segments": max(8, int(segment_count)),
        },
    )
    x_axis = (dx / length, dy / length, dz / length)
    up = (0.0, 0.0, 1.0)
    if abs(x_axis[2]) > 0.98:
        up = (0.0, 1.0, 0.0)

    y_axis = _normalize(_cross(up, x_axis))
    z_axis = _normalize(_cross(x_axis, y_axis))
    center = (
        (start[0] + end[0]) / 2.0,
        (start[1] + end[1]) / 2.0,
        (start[2] + end[2]) / 2.0,
    )

    mesh.transform(
        (
            (x_axis[0], y_axis[0], z_axis[0], center[0]),
            (x_axis[1], y_axis[1], z_axis[1], center[1]),
            (x_axis[2], y_axis[2], z_axis[2], center[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    return mesh


def _polyline_tube(
    points: Sequence[Vector3],
    radius: float,
    segments: int,
) -> Mesh:
    mesh = Mesh()
    for idx in range(len(points) - 1):
        start = points[idx]
        end = points[idx + 1]
        if sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2 + (end[2] - start[2]) ** 2) <= 1e-9:
            continue
        mesh.merge(_tube_between(start, end, radius=radius, segment_count=segments))
    return mesh


def create_distillation_column(
    *,
    height: float,
    radius: float,
    stage_count: int = 6,
    tray_count: int | None = None,
    segment_count: int = 32,
    transform: MatrixLike | None = None,
) -> Mesh:
    """
    Distillation column with staged rings and multiple side nozzles.

    Primary request params:
    - height
    - radius
    - stage_count
    """
    radius = _require_positive(radius, "radius")
    height = _require_positive(height, "height")
    if tray_count is not None:
        stage_count = int(tray_count)
    stage_count = max(2, int(stage_count))
    segment_count = max(8, int(segment_count))

    body = create_column(radius=radius, height=height, segments=segment_count)

    cap_height = max(radius * 0.34, height * 0.02)
    top_cap = create_column(radius=radius * 1.04, height=cap_height, segments=segment_count)
    bottom_cap = create_column(radius=radius * 1.04, height=cap_height, segments=segment_count)
    _translate(top_cap, 0.0, 0.0, height / 2.0 + cap_height / 2.0)
    _translate(bottom_cap, 0.0, 0.0, -height / 2.0 - cap_height / 2.0)
    body.merge(top_cap)
    body.merge(bottom_cap)

    ring_h = max(cap_height * 0.52, 0.04)
    ring_r = radius * 1.085
    for idx in range(stage_count):
        t = idx / (stage_count - 1) if stage_count > 1 else 0.5
        z = -height / 2.0 + t * height
        ring = create_column(radius=ring_r, height=ring_h, segments=max(10, segment_count - 8))
        _translate(ring, 0.0, 0.0, z)
        body.merge(ring)

    nozzle_radius = max(radius * 0.12, 0.05)
    nozzle_len = max(radius * 1.6, 0.35)
    nozzle_levels = (
        -height * 0.28,
        -height * 0.05,
        height * 0.22,
    )
    for z in nozzle_levels:
        for sign in (-1.0, 1.0):
            nozzle = _tube_between(
                (sign * radius * 0.95, 0.0, z),
                (sign * (radius * 0.95 + nozzle_len), 0.0, z),
                radius=nozzle_radius,
                segment_count=16,
            )
            body.merge(nozzle)

    return _apply_transform(body, transform)


def create_reactor(
    *,
    radius: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """
    Vertical reactor vessel.

    Request params:
    - radius
    - height
    """
    radius = _require_positive(radius, "radius")
    height = _require_positive(height, "height")

    vessel = create_column(radius=radius, height=height, segments=28)
    dome_h = max(radius * 0.5, height * 0.06)
    top = create_column(radius=radius * 0.95, height=dome_h, segments=24)
    _translate(top, 0.0, 0.0, height / 2.0 + dome_h / 2.0)
    vessel.merge(top)

    skirt_h = max(height * 0.12, radius * 0.45)
    skirt = create_column(radius=radius * 1.05, height=skirt_h, segments=24)
    _translate(skirt, 0.0, 0.0, -height / 2.0 - skirt_h / 2.0)
    vessel.merge(skirt)

    nozzle_r = max(radius * 0.1, 0.05)
    nozzle_len = max(radius * 1.4, 0.28)
    inlet = _tube_between(
        (radius * 0.95, 0.0, height * 0.18),
        (radius * 0.95 + nozzle_len, 0.0, height * 0.18),
        radius=nozzle_r,
        segment_count=14,
    )
    outlet = _tube_between(
        (-radius * 0.95, 0.0, -height * 0.18),
        (-(radius * 0.95 + nozzle_len), 0.0, -height * 0.18),
        radius=nozzle_r,
        segment_count=14,
    )
    vessel.merge(inlet)
    vessel.merge(outlet)

    return _apply_transform(vessel, transform)


def create_heat_exchanger(
    *,
    length: float,
    radius: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """
    Horizontal shell-and-tube style exchanger.

    Request params:
    - length
    - radius
    """
    length = _require_positive(length, "length")
    radius = _require_positive(radius, "radius")

    body = create_beam(
        length=length,
        profile_type={"type": "circular", "radius": radius, "segments": 24},
    )
    cap_len = max(length * 0.045, radius * 0.42)
    left = create_beam(
        length=cap_len,
        profile_type={"type": "circular", "radius": radius * 1.06, "segments": 18},
    )
    right = create_beam(
        length=cap_len,
        profile_type={"type": "circular", "radius": radius * 1.06, "segments": 18},
    )
    _translate(left, -length / 2.0 - cap_len / 2.0, 0.0, 0.0)
    _translate(right, length / 2.0 + cap_len / 2.0, 0.0, 0.0)
    body.merge(left)
    body.merge(right)

    fin_count = max(3, int(length / max(radius * 0.65, 0.1)))
    fin_len = max(radius * 0.14, length / (fin_count * 9.0))
    for idx in range(fin_count):
        t = idx / (fin_count - 1) if fin_count > 1 else 0.5
        x = -length / 2.0 + t * length
        fin = create_beam(
            length=fin_len,
            profile_type={"type": "circular", "radius": radius * 1.22, "segments": 16},
        )
        _translate(fin, x, 0.0, 0.0)
        body.merge(fin)

    leg_w = max(radius * 0.36, 0.05)
    leg_h = max(radius * 0.9, 0.12)
    for x in (-length * 0.34, length * 0.34):
        leg = create_box(width=leg_w, height=leg_h, depth=leg_w)
        _translate(leg, x, 0.0, -radius - leg_h / 2.0)
        body.merge(leg)

    return _apply_transform(body, transform)


def create_storage_tank(
    *,
    radius: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """
    Vertical storage tank with base ring and top hatch.

    Request params:
    - radius
    - height
    """
    radius = _require_positive(radius, "radius")
    height = _require_positive(height, "height")

    tank = create_column(radius=radius, height=height, segments=30)
    base_h = max(height * 0.05, radius * 0.26)
    base = create_column(radius=radius * 1.1, height=base_h, segments=24)
    _translate(base, 0.0, 0.0, -height / 2.0 - base_h / 2.0)
    tank.merge(base)

    hatch = create_column(radius=radius * 0.24, height=max(radius * 0.18, 0.04), segments=14)
    _translate(hatch, 0.0, 0.0, height / 2.0 + max(radius * 0.18, 0.04) / 2.0)
    tank.merge(hatch)

    nozzle = _tube_between(
        (radius * 0.9, 0.0, -height * 0.1),
        (radius * 1.8, 0.0, -height * 0.1),
        radius=max(radius * 0.11, 0.045),
        segment_count=14,
    )
    tank.merge(nozzle)
    return _apply_transform(tank, transform)


def create_pipe(
    *,
    radius: float,
    length: float,
    bend_radius: float = 0.0,
    segments: int = 16,
    transform: MatrixLike | None = None,
) -> Mesh:
    """
    Extended refinery pipe with configurable bend radius.

    Request params:
    - radius
    - length
    - bend_radius
    - segments
    """
    radius = _require_positive(radius, "radius")
    length = _require_positive(length, "length")
    bend_radius = _require_non_negative(bend_radius, "bend_radius")
    segments = max(8, int(segments))

    if bend_radius <= 1e-9:
        straight = create_beam(
            length=length,
            profile_type={"type": "circular", "radius": radius, "segments": segments},
        )
        return _apply_transform(straight, transform)

    arc_angle = pi / 2.0
    arc_len = bend_radius * arc_angle
    straight_len = max((length - arc_len) / 2.0, radius * 1.8)
    total_len = 2.0 * straight_len + arc_len
    if total_len > length + 1e-9:
        scale = length / total_len
        straight_len *= scale
        bend_radius *= scale
        arc_len = bend_radius * arc_angle

    points: list[Vector3] = [(-straight_len - bend_radius, 0.0, 0.0), (-bend_radius, 0.0, 0.0)]
    arc_steps = max(6, segments // 2)
    for idx in range(1, arc_steps + 1):
        t = idx / arc_steps
        theta = t * arc_angle
        x = -bend_radius * cos(theta)
        y = bend_radius * sin(theta)
        points.append((x, y, 0.0))
    end = points[-1]
    points.append((end[0], end[1] + straight_len, 0.0))

    mesh = _polyline_tube(points, radius=radius, segments=segments)
    return _apply_transform(mesh, transform)


def create_pipe_from_path(
    *,
    points: Sequence[Vector3],
    radius: float,
    segments: int = 16,
    transform: MatrixLike | None = None,
) -> Mesh:
    """
    Build a pipe along an arbitrary polyline path.

    This utility enables complex multi-turn connections between refinery units.
    """
    radius = _require_positive(radius, "radius")
    if len(points) < 2:
        raise ValueError("points must contain at least 2 path vertices.")
    mesh = _polyline_tube(points, radius=radius, segments=max(8, int(segments)))
    return _apply_transform(mesh, transform)


def create_valve(
    *,
    size: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Inline refinery valve."""
    size = _require_positive(size, "size")
    body = create_beam(
        length=size * 0.95,
        profile_type={"type": "circular", "radius": size * 0.2, "segments": 20},
    )
    stem = create_column(radius=max(size * 0.06, 0.02), height=size * 0.65, segments=14)
    _translate(stem, 0.0, 0.0, size * 0.33)
    body.merge(stem)
    handle = create_box(width=size * 0.62, height=max(size * 0.08, 0.02), depth=size * 0.18)
    _translate(handle, 0.0, 0.0, size * 0.62)
    body.merge(handle)
    return _apply_transform(body, transform)


def create_pump(
    *,
    size: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Refinery pump with motor and nozzles."""
    size = _require_positive(size, "size")
    body = create_box(width=size * 0.9, height=size * 0.56, depth=size * 0.62)
    base = create_box(width=size * 1.22, height=size * 0.12, depth=size * 0.82)
    _translate(base, 0.0, 0.0, -size * 0.34)
    body.merge(base)

    motor = create_beam(
        length=size * 0.62,
        profile_type={"type": "circular", "radius": size * 0.19, "segments": 18},
    )
    _translate(motor, size * 0.6, 0.0, 0.0)
    body.merge(motor)

    nozzle_r = max(size * 0.09, 0.03)
    nozzle_l = max(size * 0.36, 0.14)
    inlet = _tube_between(
        (-size * 0.45, 0.0, 0.0),
        (-size * 0.45 - nozzle_l, 0.0, 0.0),
        radius=nozzle_r,
        segment_count=14,
    )
    outlet = _tube_between(
        (size * 0.45, 0.0, 0.0),
        (size * 0.45 + nozzle_l, 0.0, 0.0),
        radius=nozzle_r,
        segment_count=14,
    )
    body.merge(inlet)
    body.merge(outlet)
    return _apply_transform(body, transform)


def create_platform(
    *,
    width: float,
    depth: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """
    Service platform primitive.

    Request params:
    - width
    - depth
    - height
    """
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")

    deck_thickness = max(min(height * 0.18, 0.22), 0.06)
    support_height = max(height - deck_thickness, 0.1)
    mesh = create_box(width=width, height=deck_thickness, depth=depth)
    _translate(mesh, 0.0, 0.0, support_height / 2.0)

    post_w = max(min(width, depth) * 0.06, 0.04)
    post_z = -deck_thickness / 2.0
    offsets = (
        (width / 2.0 - post_w / 2.0, depth / 2.0 - post_w / 2.0),
        (width / 2.0 - post_w / 2.0, -depth / 2.0 + post_w / 2.0),
        (-width / 2.0 + post_w / 2.0, depth / 2.0 - post_w / 2.0),
        (-width / 2.0 + post_w / 2.0, -depth / 2.0 + post_w / 2.0),
    )
    for ox, oy in offsets:
        post = create_box(width=post_w, height=support_height, depth=post_w)
        _translate(post, ox, oy, post_z)
        mesh.merge(post)

    return _apply_transform(mesh, transform)


def create_ladder(
    *,
    height: float,
    step_count: int = 10,
    transform: MatrixLike | None = None,
) -> Mesh:
    height = _require_positive(height, "height")
    step_count = max(1, int(step_count))

    rung_width = max(0.52, height * 0.12)
    rung_depth = max(0.08, height * 0.02)
    rung_thickness = max(0.035, height * 0.012)
    rail_w = max(0.045, height * 0.012)

    mesh = Mesh()
    left_rail = create_box(width=rail_w, height=height, depth=rung_depth)
    right_rail = create_box(width=rail_w, height=height, depth=rung_depth)
    _translate(left_rail, -rung_width / 2.0, 0.0, 0.0)
    _translate(right_rail, rung_width / 2.0, 0.0, 0.0)
    mesh.merge(left_rail)
    mesh.merge(right_rail)

    for idx in range(step_count):
        z = -height / 2.0 + (height * idx / (step_count - 1)) if step_count > 1 else 0.0
        rung = create_box(width=rung_width, height=rung_thickness, depth=rung_depth)
        _translate(rung, 0.0, 0.0, z)
        mesh.merge(rung)

    return _apply_transform(mesh, transform)


def create_stair(
    *,
    height: float,
    step_count: int = 8,
    transform: MatrixLike | None = None,
) -> Mesh:
    height = _require_positive(height, "height")
    step_count = max(1, int(step_count))

    stair_length = max(height * 1.15, 1.25)
    stair_width = max(height * 0.34, 0.85)
    rise = height / step_count
    run = stair_length / step_count

    mesh = Mesh()
    for idx in range(step_count):
        tread_h = max(rise * 0.2, 0.03)
        tread = create_box(width=run, height=tread_h, depth=stair_width)
        x = -stair_length / 2.0 + run * idx + run / 2.0
        z = -height / 2.0 + rise * idx + tread_h / 2.0
        _translate(tread, x, 0.0, z)
        mesh.merge(tread)
    return _apply_transform(mesh, transform)


def create_refinery_support(
    *,
    height: float,
    spacing: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Compatibility helper used by refinery biome for pipe supports."""
    height = _require_positive(height, "height")
    spacing = _require_positive(spacing, "spacing")

    post_w = max(min(height, spacing) * 0.09, 0.05)
    beam_h = max(post_w * 0.9, 0.03)
    mesh = Mesh()

    left = create_box(width=post_w, height=height, depth=post_w)
    right = create_box(width=post_w, height=height, depth=post_w)
    _translate(left, 0.0, -spacing / 2.0, 0.0)
    _translate(right, 0.0, spacing / 2.0, 0.0)
    mesh.merge(left)
    mesh.merge(right)

    beam = create_box(width=post_w, height=beam_h, depth=spacing + post_w)
    _translate(beam, 0.0, 0.0, height / 2.0 - beam_h / 2.0)
    mesh.merge(beam)
    return _apply_transform(mesh, transform)


def create_junction_node(
    *,
    size: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    size = _require_positive(size, "size")
    box = create_box(width=size, height=size, depth=size)
    ring = create_column(radius=size * 0.32, height=size * 0.26, segments=14)
    _translate(ring, 0.0, 0.0, size / 2.0 + size * 0.13)
    box.merge(ring)
    return _apply_transform(box, transform)


# Backward-compatible wrappers currently used by refinery biome and tests.
def create_process_vessel(
    *,
    radius: float,
    length: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    return create_heat_exchanger(length=length, radius=radius, transform=transform)


def create_refinery_pipe(
    *,
    radius: float,
    length: float,
    bend_angle: float = 0.0,
    segment_count: int = 16,
    transform: MatrixLike | None = None,
) -> Mesh:
    bend_angle = float(bend_angle)
    if abs(bend_angle) <= 1e-6:
        return create_pipe(
            radius=radius,
            length=length,
            bend_radius=0.0,
            segments=segment_count,
            transform=transform,
        )

    angle = abs(radians(bend_angle))
    inferred_radius = max(length / max(angle, 1e-6), radius * 2.1)

    samples = max(6, int(abs(bend_angle) / 8.0) + 2)
    points: list[Vector3] = []
    signed = radians(bend_angle)
    for idx in range(samples):
        t = idx / (samples - 1)
        theta = -signed / 2.0 + signed * t
        x = inferred_radius * sin(theta)
        y = inferred_radius * (1.0 - cos(theta))
        points.append((x, y, 0.0))

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    centered = [(x - cx, y - cy, z) for x, y, z in points]
    mesh = _polyline_tube(centered, radius=radius, segments=segment_count)
    return _apply_transform(mesh, transform)


def create_refinery_platform(
    *,
    width: float,
    depth: float,
    thickness: float = 0.14,
    transform: MatrixLike | None = None,
) -> Mesh:
    return create_platform(width=width, depth=depth, height=thickness, transform=transform)
