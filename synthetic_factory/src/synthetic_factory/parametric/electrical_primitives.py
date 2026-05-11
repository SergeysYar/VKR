from __future__ import annotations

from math import cos, radians, sin, sqrt
from typing import Iterable

from ..geometry.mesh import Mesh
from .primitives import create_beam, create_box, create_column

MatrixLike = Iterable[Iterable[float]]

_DOOR_TYPES = {"single", "double", "solid"}
_AIRFLOW_DIRECTIONS = {
    "front_to_back",
    "back_to_front",
    "left_to_right",
    "right_to_left",
    "bottom_to_top",
    "top_to_bottom",
}


def _require_positive(value: float, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return number


def _apply_transform(mesh: Mesh, transform: MatrixLike | None) -> Mesh:
    if transform is not None:
        mesh.transform(transform)
    return mesh


def _translate(mesh: Mesh, dx: float, dy: float, dz: float) -> None:
    mesh.vertices = [(x + dx, y + dy, z + dz) for x, y, z in mesh.vertices]


def _set_instance_metadata(mesh: Mesh, primitive: str, **params: object) -> None:
    key_parts = [primitive]
    for name in sorted(params):
        key_parts.append(f"{name}={params[name]}")
    setattr(
        mesh,
        "metadata",
        {
            "primitive": primitive,
            "instance_key": "|".join(key_parts),
            "params": dict(params),
        },
    )


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
    segment_count: int = 16,
) -> Mesh:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    length = sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1e-9:
        raise ValueError("segment length must be > 0.")

    mesh = create_beam(
        length=length,
        profile_type={"type": "circular", "radius": radius, "segments": max(8, segment_count)},
    )

    x_axis = (dx / length, dy / length, dz / length)
    up = (0.0, 0.0, 1.0)
    if abs(x_axis[2]) > 0.98:
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


def create_electrical_cabinet(
    width: float,
    depth: float,
    height: float,
    door_type: str = "single",
    transform: MatrixLike | None = None,
) -> Mesh:
    """
    Minimal electrical cabinet.

    Axes:
    - width: X
    - depth: Y
    - height: Z
    """
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")
    door = str(door_type).strip().lower()
    if door not in _DOOR_TYPES:
        raise ValueError(f"door_type must be one of: {sorted(_DOOR_TYPES)}")

    plinth_h = max(height * 0.04, 0.03)
    body_h = max(height - plinth_h, height * 0.75)
    shell = create_box(width=width, height=body_h, depth=depth)
    _translate(shell, 0.0, 0.0, plinth_h + body_h / 2.0)

    plinth = create_box(width=width * 0.98, height=plinth_h, depth=depth * 0.95)
    _translate(plinth, 0.0, 0.0, plinth_h / 2.0)
    shell.merge(plinth)

    door_thickness = max(depth * 0.04, 0.012)
    door_plate = create_box(width=width * 0.96, height=body_h * 0.9, depth=door_thickness)
    _translate(
        door_plate,
        0.0,
        depth / 2.0 - door_thickness / 2.0,
        plinth_h + body_h * 0.5,
    )
    shell.merge(door_plate)

    handle_w = max(width * 0.03, 0.01)
    handle_h = max(body_h * 0.14, 0.08)
    handle_d = max(door_thickness * 1.3, 0.014)
    if door == "single":
        handle = create_box(width=handle_w, height=handle_h, depth=handle_d)
        _translate(
            handle,
            width * 0.34,
            depth / 2.0 + handle_d / 2.0,
            plinth_h + body_h * 0.5,
        )
        shell.merge(handle)
    elif door == "double":
        seam = create_box(width=max(width * 0.012, 0.006), height=body_h * 0.9, depth=door_thickness * 1.05)
        _translate(seam, 0.0, depth / 2.0 - door_thickness / 2.0, plinth_h + body_h * 0.5)
        shell.merge(seam)
        for side in (-1.0, 1.0):
            handle = create_box(width=handle_w, height=handle_h, depth=handle_d)
            _translate(
                handle,
                side * width * 0.2,
                depth / 2.0 + handle_d / 2.0,
                plinth_h + body_h * 0.5,
            )
            shell.merge(handle)

    _set_instance_metadata(
        shell,
        "electrical_cabinet",
        width=round(width, 6),
        depth=round(depth, 6),
        height=round(height, 6),
        door_type=door,
    )
    return _apply_transform(shell, transform)


def create_electrical_cable_tray(
    width: float,
    height: float,
    length: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """U-shaped tray suitable for overhead or underfloor cable routes."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    length = _require_positive(length, "length")

    wall_t = max(min(width, height) * 0.16, 0.01)
    bottom = create_box(width=length, height=wall_t, depth=width)
    _translate(bottom, 0.0, 0.0, wall_t / 2.0)

    side_depth = max(wall_t, 0.01)
    side_offset = width / 2.0 - side_depth / 2.0
    for side in (-1.0, 1.0):
        wall = create_box(width=length, height=height, depth=side_depth)
        _translate(wall, 0.0, side * side_offset, height / 2.0)
        bottom.merge(wall)

    _set_instance_metadata(
        bottom,
        "electrical_cable_tray",
        width=round(width, 6),
        height=round(height, 6),
        length=round(length, 6),
    )
    return _apply_transform(bottom, transform)


def create_cable_bundle(
    radius: float,
    length: float,
    curvature: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Cable bundle as straight or bent round profile."""
    radius = _require_positive(radius, "radius")
    length = _require_positive(length, "length")
    curvature_deg = float(curvature)

    if abs(curvature_deg) < 1e-6:
        mesh = create_beam(
            length=length,
            profile_type={"type": "circular", "radius": radius, "segments": 14},
        )
        _set_instance_metadata(
            mesh,
            "cable_bundle",
            radius=round(radius, 6),
            length=round(length, 6),
            curvature=0.0,
        )
        return _apply_transform(mesh, transform)

    angle = radians(curvature_deg)
    abs_angle = max(abs(angle), 1e-6)
    arc_radius = max(length / abs_angle, radius * 2.0)
    samples = max(4, int(abs(curvature_deg) / 10.0) + 2)

    points: list[tuple[float, float, float]] = []
    for idx in range(samples):
        t = idx / (samples - 1)
        theta = -angle / 2.0 + angle * t
        x = arc_radius * sin(theta)
        y = arc_radius * (1.0 - cos(theta))
        points.append((x, y, 0.0))

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    points = [(x - cx, y - cy, z) for x, y, z in points]

    mesh = Mesh()
    for idx in range(len(points) - 1):
        mesh.merge(_tube_between(points[idx], points[idx + 1], radius=radius, segment_count=12))

    _set_instance_metadata(
        mesh,
        "cable_bundle",
        radius=round(radius, 6),
        length=round(length, 6),
        curvature=round(curvature_deg, 6),
    )
    return _apply_transform(mesh, transform)


def create_switch_panel(
    width: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Wall-mounted switch panel with repeated switch blocks."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")

    panel_t = max(min(width, height) * 0.03, 0.01)
    panel = create_box(width=width, height=height, depth=panel_t)
    _translate(panel, 0.0, 0.0, height / 2.0)

    cols = max(2, int(width / 0.24))
    rows = max(2, int(height / 0.28))
    switch_w = width * 0.08
    switch_h = height * 0.08
    switch_d = panel_t * 0.7
    x_span = max(width * 0.36, switch_w / 2.0)
    z_span = max(height * 0.32, switch_h / 2.0)
    xs = [(-x_span + (2.0 * x_span * idx / (cols - 1))) for idx in range(cols)] if cols > 1 else [0.0]
    zs = [(height / 2.0 - z_span + (2.0 * z_span * idx / (rows - 1))) for idx in range(rows)] if rows > 1 else [height / 2.0]
    front_y = panel_t / 2.0 + switch_d / 2.0
    for z in zs:
        for x in xs:
            switch = create_box(width=switch_w, height=switch_h, depth=switch_d)
            _translate(switch, x, front_y, z)
            panel.merge(switch)

    _set_instance_metadata(
        panel,
        "switch_panel",
        width=round(width, 6),
        height=round(height, 6),
    )
    return _apply_transform(panel, transform)


def create_junction_box(
    size: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Compact junction box with front lid."""
    size = _require_positive(size, "size")
    body = create_box(width=size, height=size, depth=size)
    _translate(body, 0.0, 0.0, size / 2.0)

    lid_t = max(size * 0.08, 0.01)
    lid = create_box(width=size * 0.94, height=size * 0.94, depth=lid_t)
    _translate(lid, 0.0, size / 2.0 + lid_t / 2.0, size / 2.0)
    body.merge(lid)

    _set_instance_metadata(body, "junction_box", size=round(size, 6))
    return _apply_transform(body, transform)


def create_floor_tile(
    size: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """False-floor tile module."""
    size = _require_positive(size, "size")
    thickness = max(size * 0.05, 0.02)
    tile = create_box(width=size, height=thickness, depth=size)
    _translate(tile, 0.0, 0.0, thickness / 2.0)

    seam_t = max(size * 0.01, 0.003)
    seam_h = max(thickness * 0.25, 0.002)
    seam_x = create_box(width=size * 0.9, height=seam_h, depth=seam_t)
    seam_y = create_box(width=seam_t, height=seam_h, depth=size * 0.9)
    _translate(seam_x, 0.0, 0.0, thickness + seam_h / 2.0)
    _translate(seam_y, 0.0, 0.0, thickness + seam_h / 2.0)
    tile.merge(seam_x)
    tile.merge(seam_y)

    _set_instance_metadata(tile, "floor_tile", size=round(size, 6))
    return _apply_transform(tile, transform)


def create_ceiling_tray_support(
    height: float,
    spacing: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Ceiling support frame for cable trays."""
    height = _require_positive(height, "height")
    spacing = _require_positive(spacing, "spacing")

    post_w = max(min(height, spacing) * 0.04, 0.015)
    bridge_h = max(post_w * 0.8, 0.01)
    support = Mesh()

    left_post = create_box(width=post_w, height=height, depth=post_w)
    right_post = create_box(width=post_w, height=height, depth=post_w)
    _translate(left_post, -spacing / 2.0, 0.0, height / 2.0)
    _translate(right_post, spacing / 2.0, 0.0, height / 2.0)
    support.merge(left_post)
    support.merge(right_post)

    bridge = create_box(width=spacing + post_w, height=bridge_h, depth=post_w)
    _translate(bridge, 0.0, 0.0, height - bridge_h / 2.0)
    support.merge(bridge)

    _set_instance_metadata(
        support,
        "ceiling_tray_support",
        height=round(height, 6),
        spacing=round(spacing, 6),
    )
    return _apply_transform(support, transform)


def create_cooling_unit(
    width: float,
    height: float,
    airflow_direction: str = "front_to_back",
    transform: MatrixLike | None = None,
) -> Mesh:
    """Optional cooling unit with simple directional vents."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    direction = str(airflow_direction).strip().lower()
    if direction not in _AIRFLOW_DIRECTIONS:
        raise ValueError(f"airflow_direction must be one of: {sorted(_AIRFLOW_DIRECTIONS)}")

    depth = max(width * 0.55, 0.35)
    body = create_box(width=width, height=height, depth=depth)
    _translate(body, 0.0, 0.0, height / 2.0)

    vent_t = max(min(width, height, depth) * 0.04, 0.01)
    vent_w = width * 0.7
    vent_h = height * 0.42

    inlet = create_box(width=vent_w, height=vent_h, depth=vent_t)
    outlet = create_box(width=vent_w, height=vent_h, depth=vent_t)

    if direction == "front_to_back":
        _translate(inlet, 0.0, depth / 2.0 + vent_t / 2.0, height * 0.58)
        _translate(outlet, 0.0, -depth / 2.0 - vent_t / 2.0, height * 0.58)
    elif direction == "back_to_front":
        _translate(inlet, 0.0, -depth / 2.0 - vent_t / 2.0, height * 0.58)
        _translate(outlet, 0.0, depth / 2.0 + vent_t / 2.0, height * 0.58)
    elif direction == "left_to_right":
        inlet = create_box(width=vent_t, height=vent_h, depth=depth * 0.7)
        outlet = create_box(width=vent_t, height=vent_h, depth=depth * 0.7)
        _translate(inlet, -width / 2.0 - vent_t / 2.0, 0.0, height * 0.58)
        _translate(outlet, width / 2.0 + vent_t / 2.0, 0.0, height * 0.58)
    elif direction == "right_to_left":
        inlet = create_box(width=vent_t, height=vent_h, depth=depth * 0.7)
        outlet = create_box(width=vent_t, height=vent_h, depth=depth * 0.7)
        _translate(inlet, width / 2.0 + vent_t / 2.0, 0.0, height * 0.58)
        _translate(outlet, -width / 2.0 - vent_t / 2.0, 0.0, height * 0.58)
    elif direction == "bottom_to_top":
        inlet = create_box(width=width * 0.65, height=vent_t, depth=depth * 0.65)
        outlet = create_box(width=width * 0.65, height=vent_t, depth=depth * 0.65)
        _translate(inlet, 0.0, 0.0, vent_t / 2.0)
        _translate(outlet, 0.0, 0.0, height + vent_t / 2.0)
    else:  # top_to_bottom
        inlet = create_box(width=width * 0.65, height=vent_t, depth=depth * 0.65)
        outlet = create_box(width=width * 0.65, height=vent_t, depth=depth * 0.65)
        _translate(inlet, 0.0, 0.0, height + vent_t / 2.0)
        _translate(outlet, 0.0, 0.0, vent_t / 2.0)

    body.merge(inlet)
    body.merge(outlet)

    _set_instance_metadata(
        body,
        "cooling_unit",
        width=round(width, 6),
        height=round(height, 6),
        airflow_direction=direction,
    )
    return _apply_transform(body, transform)


def create_floor_cable_entry(
    radius: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Cable penetration node for raised floor / slab entry points."""
    radius = _require_positive(radius, "radius")
    height = _require_positive(height, "height")

    flange_h = max(height * 0.2, 0.01)
    flange = create_box(width=radius * 3.0, height=flange_h, depth=radius * 3.0)
    _translate(flange, 0.0, 0.0, flange_h / 2.0)

    neck = create_column(radius=radius, height=height, segments=16)
    _translate(neck, 0.0, 0.0, height / 2.0)
    flange.merge(neck)

    _set_instance_metadata(
        flange,
        "floor_cable_entry",
        radius=round(radius, 6),
        height=round(height, 6),
    )
    return _apply_transform(flange, transform)


def create_transformer_unit(
    width: float,
    depth: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Compact transformer block with top cooling fins and terminal bushings."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")

    base_h = max(height * 0.08, 0.03)
    body_h = max(height - base_h, height * 0.72)
    body = create_box(width=width, height=body_h, depth=depth)
    _translate(body, 0.0, 0.0, base_h + body_h / 2.0)

    base = create_box(width=width * 1.04, height=base_h, depth=depth * 1.02)
    _translate(base, 0.0, 0.0, base_h / 2.0)
    body.merge(base)

    fin_count = max(3, int(width / 0.15))
    fin_w = max(width / (fin_count * 2.8), 0.02)
    fin_h = max(height * 0.12, 0.05)
    fin_d = max(depth * 0.65, 0.08)
    fin_span = max(width * 0.34, fin_w)
    fin_xs = [(-fin_span + (2.0 * fin_span * idx / (fin_count - 1))) for idx in range(fin_count)] if fin_count > 1 else [0.0]
    for fin_x in fin_xs:
        fin = create_box(width=fin_w, height=fin_h, depth=fin_d)
        _translate(fin, fin_x, 0.0, height + fin_h / 2.0)
        body.merge(fin)

    bushing_h = max(height * 0.16, 0.08)
    bushing_r = max(min(width, depth) * 0.06, 0.018)
    bushing_offset = depth * 0.22
    left_bushing = create_column(radius=bushing_r, height=bushing_h, segments=12)
    right_bushing = create_column(radius=bushing_r, height=bushing_h, segments=12)
    _translate(left_bushing, -width * 0.18, bushing_offset, height + bushing_h / 2.0)
    _translate(right_bushing, width * 0.18, bushing_offset, height + bushing_h / 2.0)
    body.merge(left_bushing)
    body.merge(right_bushing)

    _set_instance_metadata(
        body,
        "transformer_unit",
        width=round(width, 6),
        depth=round(depth, 6),
        height=round(height, 6),
    )
    return _apply_transform(body, transform)


def create_backup_battery(
    width: float,
    depth: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Backup battery cabinet with stacked internal module seams."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")

    body = create_box(width=width, height=height, depth=depth)
    _translate(body, 0.0, 0.0, height / 2.0)

    seam_h = max(height * 0.025, 0.008)
    seam_d = max(depth * 0.86, 0.03)
    module_count = max(2, int(height / 0.35))
    z_span = max(height * 0.34, seam_h)
    zs = [height / 2.0 - z_span + (2.0 * z_span * idx / (module_count - 1)) for idx in range(module_count)] if module_count > 1 else [height / 2.0]
    for seam_z in zs:
        seam = create_box(width=width * 0.9, height=seam_h, depth=seam_d)
        _translate(seam, 0.0, depth * 0.48, seam_z)
        body.merge(seam)

    terminal_h = max(height * 0.08, 0.02)
    terminal_w = max(width * 0.12, 0.02)
    terminal_d = max(depth * 0.12, 0.02)
    for side in (-1.0, 1.0):
        terminal = create_box(width=terminal_w, height=terminal_h, depth=terminal_d)
        _translate(terminal, side * width * 0.18, -depth * 0.25, height + terminal_h / 2.0)
        body.merge(terminal)

    _set_instance_metadata(
        body,
        "backup_battery",
        width=round(width, 6),
        depth=round(depth, 6),
        height=round(height, 6),
    )
    return _apply_transform(body, transform)


def create_small_control_panel(
    width: float,
    height: float,
    depth: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Small local control panel for nearby electrical modules."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    depth = _require_positive(depth, "depth")

    panel = create_box(width=width, height=height, depth=depth)
    _translate(panel, 0.0, 0.0, height / 2.0)

    bezel_t = max(depth * 0.3, 0.01)
    bezel = create_box(width=width * 0.88, height=height * 0.82, depth=bezel_t)
    _translate(bezel, 0.0, depth / 2.0 + bezel_t / 2.0, height * 0.55)
    panel.merge(bezel)

    knob_r = max(min(width, height) * 0.05, 0.01)
    knob_h = max(depth * 0.45, 0.01)
    for idx in range(3):
        kx = -width * 0.22 + idx * width * 0.22
        knob = create_column(radius=knob_r, height=knob_h, segments=10)
        _translate(knob, kx, depth / 2.0 + knob_h / 2.0, height * 0.28)
        panel.merge(knob)

    _set_instance_metadata(
        panel,
        "small_control_panel",
        width=round(width, 6),
        height=round(height, 6),
        depth=round(depth, 6),
    )
    return _apply_transform(panel, transform)
