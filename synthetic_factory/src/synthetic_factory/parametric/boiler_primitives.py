from __future__ import annotations

from math import cos, radians, sin, sqrt
from typing import Iterable, Mapping

from ..geometry.mesh import Mesh
from .primitives import create_beam, create_box, create_column

MatrixLike = Iterable[Iterable[float]]
_AXIS_ORIENTATIONS = {"x", "-x", "y", "-y", "z", "-z"}


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


def _require_steps(value: int, name: str = "step_count") -> int:
    count = int(value)
    if count <= 0:
        raise ValueError(f"{name} must be > 0.")
    return count


def _apply_transform(mesh: Mesh, transform: MatrixLike | None) -> Mesh:
    if transform is not None:
        mesh.transform(transform)
    return mesh


def _translate(mesh: Mesh, dx: float, dy: float, dz: float) -> None:
    mesh.vertices = [(x + dx, y + dy, z + dz) for x, y, z in mesh.vertices]


def _rotate_x(mesh: Mesh, angle_deg: float) -> None:
    angle = radians(angle_deg)
    c = cos(angle)
    s = sin(angle)
    mesh.vertices = [
        (
            x,
            y * c - z * s,
            y * s + z * c,
        )
        for x, y, z in mesh.vertices
    ]


def _rotate_y(mesh: Mesh, angle_deg: float) -> None:
    angle = radians(angle_deg)
    c = cos(angle)
    s = sin(angle)
    mesh.vertices = [
        (
            x * c + z * s,
            y,
            -x * s + z * c,
        )
        for x, y, z in mesh.vertices
    ]


def _rotate_z(mesh: Mesh, angle_deg: float) -> None:
    angle = radians(angle_deg)
    c = cos(angle)
    s = sin(angle)
    mesh.vertices = [
        (
            x * c - y * s,
            x * s + y * c,
            z,
        )
        for x, y, z in mesh.vertices
    ]


def _apply_axis_orientation(mesh: Mesh, orientation: str) -> None:
    axis = str(orientation).strip().lower()
    if axis not in _AXIS_ORIENTATIONS:
        raise ValueError(f"orientation must be one of: {sorted(_AXIS_ORIENTATIONS)}")

    if axis == "x":
        return
    if axis == "-x":
        _rotate_z(mesh, 180.0)
        return
    if axis == "y":
        _rotate_z(mesh, 90.0)
        return
    if axis == "-y":
        _rotate_z(mesh, -90.0)
        return
    if axis == "z":
        _rotate_y(mesh, -90.0)
        return
    _rotate_y(mesh, 90.0)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _tube_between(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    segment_count: int,
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
            "segments": segment_count,
        },
    )

    x_axis = (dx / length, dy / length, dz / length)
    up = (0.0, 0.0, 1.0)
    if abs(x_axis[2]) > 0.99:
        up = (0.0, 1.0, 0.0)

    y_axis = _normalize(_cross(up, x_axis))
    z_axis = _normalize(_cross(x_axis, y_axis))
    center = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, (start[2] + end[2]) / 2.0)

    mesh.transform(
        (
            (x_axis[0], y_axis[0], z_axis[0], center[0]),
            (x_axis[1], y_axis[1], z_axis[1], center[1]),
            (x_axis[2], y_axis[2], z_axis[2], center[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    return mesh


def create_boiler_unit(
    radius: float,
    height: float,
    segment_count: int = 24,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Vertical segmented cylinder (boiler unit)."""
    radius = _require_positive(radius, "radius")
    height = _require_positive(height, "height")
    segment_count = max(3, int(segment_count))
    mesh = create_column(radius=radius, height=height, segments=segment_count)
    return _apply_transform(mesh, transform)


def create_pipe(
    radius: float,
    length: float,
    bend_angle: float = 0.0,
    segment_count: int = 24,
    transform: MatrixLike | None = None,
) -> Mesh:
    """
    Pipe primitive with optional bend.

    `bend_angle` is in degrees. `0` creates a straight segment.
    """
    radius = _require_positive(radius, "radius")
    length = _require_positive(length, "length")
    _require_non_negative(abs(float(bend_angle)), "bend_angle")
    segment_count = max(6, int(segment_count))

    if abs(bend_angle) < 1e-6:
        mesh = create_beam(
            length=length,
            profile_type={"type": "circular", "radius": radius, "segments": segment_count},
        )
        return _apply_transform(mesh, transform)

    signed_angle = radians(float(bend_angle))
    abs_angle = abs(signed_angle)
    arc_radius = max(length / abs_angle, radius * 2.2)
    samples = max(8, int(abs(float(bend_angle)) / 8.0) + 2)

    points: list[tuple[float, float, float]] = []
    for idx in range(samples):
        t = idx / (samples - 1)
        theta = -signed_angle / 2.0 + signed_angle * t
        x = arc_radius * sin(theta)
        y = arc_radius * (1.0 - cos(theta))
        points.append((x, y, 0.0))

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    points = [(x - center_x, y - center_y, z) for x, y, z in points]

    mesh = Mesh()
    for idx in range(len(points) - 1):
        segment = _tube_between(points[idx], points[idx + 1], radius, segment_count)
        mesh.merge(segment)

    return _apply_transform(mesh, transform)


def create_valve(
    radius: float,
    handle_size: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Valve node with cylindrical body and handle."""
    radius = _require_positive(radius, "radius")
    handle_size = _require_positive(handle_size, "handle_size")

    body_length = radius * 1.5
    body = create_beam(
        length=body_length,
        profile_type={"type": "circular", "radius": radius, "segments": 24},
    )

    flange_radius = radius * 1.2
    flange_length = radius * 0.28
    left_flange = create_beam(
        length=flange_length,
        profile_type={"type": "circular", "radius": flange_radius, "segments": 20},
    )
    right_flange = create_beam(
        length=flange_length,
        profile_type={"type": "circular", "radius": flange_radius, "segments": 20},
    )
    _translate(left_flange, -(body_length / 2.0 + flange_length / 2.0), 0.0, 0.0)
    _translate(right_flange, body_length / 2.0 + flange_length / 2.0, 0.0, 0.0)
    body.merge(left_flange)
    body.merge(right_flange)

    stem = create_column(radius=max(radius * 0.16, 0.02), height=handle_size, segments=16)
    _translate(stem, 0.0, 0.0, radius + handle_size / 2.0)
    body.merge(stem)

    handle = create_box(
        width=handle_size,
        height=max(radius * 0.12, 0.02),
        depth=max(radius * 0.35, 0.03),
    )
    _translate(handle, 0.0, 0.0, radius + handle_size)
    body.merge(handle)

    return _apply_transform(body, transform)


def create_platform(
    width: float,
    depth: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Service platform (rectangular deck)."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")
    mesh = create_box(width=width, height=height, depth=depth)
    return _apply_transform(mesh, transform)


def create_ladder(
    height: float,
    step_count: int,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Simple ladder with two rails and evenly spaced rungs."""
    height = _require_positive(height, "height")
    step_count = _require_steps(step_count)

    rung_width = max(0.5, height * 0.12)
    rung_depth = max(0.08, height * 0.02)
    rung_thickness = max(0.04, height * 0.012)
    rail_width = max(0.05, height * 0.012)

    mesh = Mesh()
    left_rail = create_box(width=rail_width, height=height, depth=rung_depth)
    right_rail = create_box(width=rail_width, height=height, depth=rung_depth)
    _translate(left_rail, -rung_width / 2.0, 0.0, 0.0)
    _translate(right_rail, rung_width / 2.0, 0.0, 0.0)
    mesh.merge(left_rail)
    mesh.merge(right_rail)

    for idx in range(step_count):
        if step_count == 1:
            z = 0.0
        else:
            z = -height / 2.0 + (height * idx / (step_count - 1))
        rung = create_box(width=rung_width, height=rung_thickness, depth=rung_depth)
        _translate(rung, 0.0, 0.0, z)
        mesh.merge(rung)

    return _apply_transform(mesh, transform)


def create_stair(
    height: float,
    step_count: int,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Simple industrial stair as a stack of box steps."""
    height = _require_positive(height, "height")
    step_count = _require_steps(step_count)

    stair_length = max(height * 1.1, 1.2)
    stair_width = max(height * 0.35, 0.8)
    rise = height / step_count
    run = stair_length / step_count

    mesh = Mesh()
    for idx in range(step_count):
        step_height = max(rise * 0.18, 0.03)
        step = create_box(
            width=run,
            height=step_height,
            depth=stair_width,
        )
        x = -stair_length / 2.0 + run * idx + run / 2.0
        z = -height / 2.0 + rise * idx + step_height / 2.0
        _translate(step, x, 0.0, z)
        mesh.merge(step)

    return _apply_transform(mesh, transform)


def create_support_beam(
    length: float,
    profile_type: str | Mapping[str, object],
    transform: MatrixLike | None = None,
) -> Mesh:
    """Support beam wrapper around generic beam primitive."""
    length = _require_positive(length, "length")
    mesh = create_beam(length=length, profile_type=profile_type)
    return _apply_transform(mesh, transform)


def create_tank(
    radius: float,
    height: float,
    segment_count: int = 28,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Vertical tank primitive."""
    radius = _require_positive(radius, "radius")
    height = _require_positive(height, "height")
    segment_count = max(6, int(segment_count))
    body = create_column(radius=radius, height=height, segments=segment_count)

    top_cap = create_column(radius=radius * 0.92, height=max(height * 0.08, 0.05), segments=segment_count)
    bottom_cap = create_column(radius=radius * 0.92, height=max(height * 0.08, 0.05), segments=segment_count)
    _translate(top_cap, 0.0, 0.0, height / 2.0 + max(height * 0.08, 0.05) / 2.0)
    _translate(bottom_cap, 0.0, 0.0, -height / 2.0 - max(height * 0.08, 0.05) / 2.0)
    body.merge(top_cap)
    body.merge(bottom_cap)

    return _apply_transform(body, transform)


def create_bunker(
    width: float,
    height: float,
    outlet_radius: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Storage bunker with funnel and pipe-compatible outlet neck."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    outlet_radius = _require_positive(outlet_radius, "outlet_radius")
    outlet_diameter = outlet_radius * 2.0
    if outlet_diameter >= width:
        raise ValueError("outlet_radius must be < width / 2.")

    depth = width * 0.82
    top_h = height * 0.52
    funnel_h = height * 0.35
    neck_h = max(height - top_h - funnel_h, height * 0.13)
    funnel_levels = 3

    mesh = Mesh()

    top_block = create_box(width=width, height=top_h, depth=depth)
    _translate(top_block, 0.0, 0.0, height / 2.0 - top_h / 2.0)
    mesh.merge(top_block)

    top_plane = height / 2.0 - top_h
    target_w = min(width * 0.92, max(outlet_diameter * 1.25, outlet_diameter + width * 0.08))
    target_d = min(depth * 0.92, max(outlet_diameter * 1.12, outlet_diameter + depth * 0.08))
    for idx in range(funnel_levels):
        t = (idx + 0.5) / funnel_levels
        level_w = width - (width - target_w) * t
        level_d = depth - (depth - target_d) * t
        level_h = funnel_h / funnel_levels
        level = create_box(width=level_w, height=level_h, depth=level_d)
        _translate(level, 0.0, 0.0, top_plane - level_h * (idx + 0.5))
        mesh.merge(level)

    outlet = create_column(radius=outlet_radius, height=neck_h, segments=22)
    _translate(outlet, 0.0, 0.0, -height / 2.0 + neck_h / 2.0)
    mesh.merge(outlet)

    collar_h = max(min(outlet_radius * 0.7, neck_h * 0.35), neck_h * 0.15)
    collar = create_column(radius=outlet_radius * 1.28, height=collar_h, segments=20)
    _translate(collar, 0.0, 0.0, -height / 2.0 + neck_h + collar_h / 2.0)
    mesh.merge(collar)

    return _apply_transform(mesh, transform)


def create_pump(
    size: float,
    orientation: str,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Pump body with aligned intake/output nozzles for pipe network integration."""
    size = _require_positive(size, "size")

    casing_len = size * 0.9
    motor_len = size * 0.62
    nozzle_len = size * 0.42
    nozzle_radius = size * 0.12

    mesh = Mesh()

    base_h = size * 0.16
    base = create_box(width=size * 1.24, height=base_h, depth=size * 0.88)
    _translate(base, 0.0, 0.0, -size * 0.34)
    mesh.merge(base)

    casing = create_box(width=casing_len, height=size * 0.58, depth=size * 0.72)
    mesh.merge(casing)

    motor = create_beam(
        length=motor_len,
        profile_type={"type": "circular", "radius": size * 0.24, "segments": 20},
    )
    _translate(motor, casing_len / 2.0 + motor_len / 2.0 - size * 0.08, 0.0, 0.0)
    mesh.merge(motor)

    inlet = create_beam(
        length=nozzle_len,
        profile_type={"type": "circular", "radius": nozzle_radius, "segments": 16},
    )
    outlet = create_beam(
        length=nozzle_len,
        profile_type={"type": "circular", "radius": nozzle_radius, "segments": 16},
    )
    _translate(inlet, -casing_len / 2.0 - nozzle_len / 2.0, 0.0, 0.0)
    _translate(outlet, casing_len / 2.0 + nozzle_len / 2.0, 0.0, 0.0)
    mesh.merge(inlet)
    mesh.merge(outlet)

    top_port = _tube_between(
        (0.0, 0.0, size * 0.29),
        (0.0, 0.0, size * 0.58),
        radius=nozzle_radius * 0.72,
        segment_count=16,
    )
    mesh.merge(top_port)

    _apply_axis_orientation(mesh, orientation)
    return _apply_transform(mesh, transform)


def create_heat_exchanger(
    length: float,
    radius: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Horizontal heat exchanger shell with fins and service nozzles."""
    length = _require_positive(length, "length")
    radius = _require_positive(radius, "radius")

    mesh = create_beam(
        length=length,
        profile_type={"type": "circular", "radius": radius, "segments": 26},
    )

    cap_len = max(length * 0.04, radius * 0.42)
    left_cap = create_beam(
        length=cap_len,
        profile_type={"type": "circular", "radius": radius * 1.08, "segments": 20},
    )
    right_cap = create_beam(
        length=cap_len,
        profile_type={"type": "circular", "radius": radius * 1.08, "segments": 20},
    )
    _translate(left_cap, -length / 2.0 - cap_len / 2.0, 0.0, 0.0)
    _translate(right_cap, length / 2.0 + cap_len / 2.0, 0.0, 0.0)
    mesh.merge(left_cap)
    mesh.merge(right_cap)

    fin_count = max(3, int(length / max(radius * 0.6, 0.1)))
    fin_len = max(radius * 0.12, length / (fin_count * 9.0))
    for idx in range(fin_count):
        ratio = idx / (fin_count - 1) if fin_count > 1 else 0.5
        x_pos = -length / 2.0 + ratio * length
        fin = create_beam(
            length=fin_len,
            profile_type={"type": "circular", "radius": radius * 1.22, "segments": 18},
        )
        _translate(fin, x_pos, 0.0, 0.0)
        mesh.merge(fin)

    nozzle_radius = radius * 0.32
    nozzle_height = radius * 1.8
    for x_pos in (-length * 0.22, length * 0.22):
        nozzle = _tube_between(
            (x_pos, 0.0, radius),
            (x_pos, 0.0, radius + nozzle_height),
            radius=nozzle_radius,
            segment_count=16,
        )
        mesh.merge(nozzle)

    leg_w = max(radius * 0.36, 0.04)
    leg_h = radius * 0.95
    for x_pos in (-length * 0.35, length * 0.35):
        leg = create_box(width=leg_w, height=leg_h, depth=leg_w)
        _translate(leg, x_pos, 0.0, -radius - leg_h / 2.0)
        mesh.merge(leg)

    return _apply_transform(mesh, transform)


def create_valve_cluster(
    count: int,
    spacing: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Valve manifold with repeated valve nodes and branch connectors."""
    count = _require_steps(count, "count")
    spacing = _require_positive(spacing, "spacing")

    valve_radius = max(spacing * 0.14, 0.05)
    handle_size = max(valve_radius * 2.4, spacing * 0.25)
    span = spacing * (count - 1)
    manifold_length = span + spacing * 1.2
    branch_len = spacing * 0.55

    mesh = create_pipe(
        radius=valve_radius * 0.72,
        length=manifold_length,
        bend_angle=0.0,
        segment_count=18,
    )
    start_x = -span / 2.0
    for idx in range(count):
        x_pos = start_x + idx * spacing if count > 1 else 0.0
        valve = create_valve(radius=valve_radius, handle_size=handle_size)
        _translate(valve, x_pos, 0.0, 0.0)
        mesh.merge(valve)

        branch = _tube_between(
            (x_pos, 0.0, -branch_len / 2.0),
            (x_pos, 0.0, branch_len / 2.0),
            radius=valve_radius * 0.52,
            segment_count=16,
        )
        mesh.merge(branch)

    return _apply_transform(mesh, transform)


def create_pipe_support(
    height: float,
    spacing: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Portal-style support frame for overhead pipe runs."""
    height = _require_positive(height, "height")
    spacing = _require_positive(spacing, "spacing")

    post_w = max(min(height, spacing) * 0.08, 0.04)
    beam_h = max(post_w * 0.85, 0.03)
    base_h = max(beam_h * 0.5, 0.02)

    mesh = Mesh()

    left_post = create_box(width=post_w, height=height, depth=post_w)
    right_post = create_box(width=post_w, height=height, depth=post_w)
    _translate(left_post, 0.0, -spacing / 2.0, 0.0)
    _translate(right_post, 0.0, spacing / 2.0, 0.0)
    mesh.merge(left_post)
    mesh.merge(right_post)

    crossbar = create_box(width=post_w, height=beam_h, depth=spacing + post_w * 1.2)
    _translate(crossbar, 0.0, 0.0, height / 2.0 - beam_h / 2.0)
    mesh.merge(crossbar)

    left_base = create_box(width=post_w * 1.8, height=base_h, depth=post_w * 1.8)
    right_base = create_box(width=post_w * 1.8, height=base_h, depth=post_w * 1.8)
    _translate(left_base, 0.0, -spacing / 2.0, -height / 2.0 + base_h / 2.0)
    _translate(right_base, 0.0, spacing / 2.0, -height / 2.0 + base_h / 2.0)
    mesh.merge(left_base)
    mesh.merge(right_base)

    saddle_radius = max(spacing * 0.09, post_w * 0.38)
    saddle = create_beam(
        length=spacing * 0.68,
        profile_type={"type": "circular", "radius": saddle_radius, "segments": 16},
    )
    _translate(saddle, 0.0, 0.0, height / 2.0 + saddle_radius * 0.35)
    mesh.merge(saddle)

    return _apply_transform(mesh, transform)


def create_control_box(
    size: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Boiler-side control box with conduit ports for cable/pipe tie-ins."""
    size = _require_positive(size, "size")

    body_w = size
    body_h = size * 0.72
    body_d = size * 0.46
    mesh = create_box(width=body_w, height=body_h, depth=body_d)

    panel_t = max(size * 0.05, 0.01)
    front_panel = create_box(width=body_w * 0.94, height=body_h * 0.9, depth=panel_t)
    _translate(front_panel, 0.0, body_d / 2.0 + panel_t / 2.0, 0.0)
    mesh.merge(front_panel)

    port_radius = size * 0.09
    port_len = size * 0.3
    left_port = create_beam(
        length=port_len,
        profile_type={"type": "circular", "radius": port_radius, "segments": 14},
    )
    right_port = create_beam(
        length=port_len,
        profile_type={"type": "circular", "radius": port_radius, "segments": 14},
    )
    _translate(left_port, -body_w / 2.0 - port_len / 2.0, 0.0, -body_h * 0.1)
    _translate(right_port, body_w / 2.0 + port_len / 2.0, 0.0, -body_h * 0.1)
    mesh.merge(left_port)
    mesh.merge(right_port)

    top_port = _tube_between(
        (0.0, 0.0, body_h / 2.0),
        (0.0, 0.0, body_h / 2.0 + port_len),
        radius=port_radius * 0.86,
        segment_count=14,
    )
    mesh.merge(top_port)

    return _apply_transform(mesh, transform)


def create_chimney(
    height: float,
    radius: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Tall exhaust stack with side intake nozzle for pipe system connectivity."""
    height = _require_positive(height, "height")
    radius = _require_positive(radius, "radius")

    mesh = create_column(radius=radius, height=height, segments=30)

    base_h = max(radius * 0.35, height * 0.03)
    base = create_column(radius=radius * 1.22, height=base_h, segments=24)
    _translate(base, 0.0, 0.0, -height / 2.0 - base_h / 2.0)
    mesh.merge(base)

    lip_h = max(radius * 0.22, height * 0.02)
    lip = create_column(radius=radius * 1.1, height=lip_h, segments=24)
    _translate(lip, 0.0, 0.0, height / 2.0 + lip_h / 2.0)
    mesh.merge(lip)

    inlet_len = max(radius * 1.8, height * 0.12)
    inlet = _tube_between(
        (radius * 0.95, 0.0, -height * 0.2),
        (radius * 0.95 + inlet_len, 0.0, -height * 0.2),
        radius=radius * 0.35,
        segment_count=18,
    )
    mesh.merge(inlet)

    return _apply_transform(mesh, transform)
