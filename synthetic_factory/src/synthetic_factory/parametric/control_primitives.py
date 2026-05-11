from __future__ import annotations

from math import cos, radians, sin
from typing import Iterable

from ..geometry.mesh import Mesh
from .primitives import create_beam, create_box

MatrixLike = Iterable[Iterable[float]]


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


def create_desk(
    width: float,
    depth: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Minimal operator desk: tabletop + four legs."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")

    tabletop_thickness = min(max(height * 0.08, 0.03), height * 0.35)
    leg_width = max(min(width, depth) * 0.08, 0.03)
    leg_height = max(height - tabletop_thickness, 0.03)

    mesh = Mesh()
    tabletop = create_box(width=width, height=tabletop_thickness, depth=depth)
    _translate(tabletop, 0.0, 0.0, leg_height + tabletop_thickness / 2.0)
    mesh.merge(tabletop)

    offset_x = max(width / 2.0 - leg_width / 2.0, 0.0)
    offset_y = max(depth / 2.0 - leg_width / 2.0, 0.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            leg = create_box(width=leg_width, height=leg_height, depth=leg_width)
            _translate(leg, sx * offset_x, sy * offset_y, leg_height / 2.0)
            mesh.merge(leg)

    return _apply_transform(mesh, transform)


def create_chair(
    seat_height: float,
    back_height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Minimal chair: seat + backrest + four legs."""
    seat_height = _require_positive(seat_height, "seat_height")
    back_height = _require_positive(back_height, "back_height")

    seat_width = max(seat_height * 1.2, 0.42)
    seat_depth = max(seat_height * 1.1, 0.42)
    seat_thickness = min(max(seat_height * 0.2, 0.03), seat_height * 0.4)
    leg_width = max(min(seat_width, seat_depth) * 0.08, 0.025)

    mesh = Mesh()
    seat = create_box(width=seat_width, height=seat_thickness, depth=seat_depth)
    _translate(seat, 0.0, 0.0, seat_height - seat_thickness / 2.0)
    mesh.merge(seat)

    back_thickness = max(seat_depth * 0.12, 0.03)
    back = create_box(width=seat_width, height=back_height, depth=back_thickness)
    _translate(
        back,
        0.0,
        -seat_depth / 2.0 + back_thickness / 2.0,
        seat_height + back_height / 2.0,
    )
    mesh.merge(back)

    leg_height = max(seat_height - seat_thickness, 0.02)
    leg_x = max(seat_width / 2.0 - leg_width / 2.0, 0.0)
    leg_y = max(seat_depth / 2.0 - leg_width / 2.0, 0.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            leg = create_box(width=leg_width, height=leg_height, depth=leg_width)
            _translate(leg, sx * leg_x, sy * leg_y, leg_height / 2.0)
            mesh.merge(leg)

    return _apply_transform(mesh, transform)


def create_monitor(
    width: float,
    height: float,
    thickness: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Minimal monitor: screen block + stand + base."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    thickness = _require_positive(thickness, "thickness")

    mesh = Mesh()
    screen = create_box(width=width, height=height, depth=thickness)
    _translate(screen, 0.0, 0.0, height * 0.65)
    mesh.merge(screen)

    stem_height = max(height * 0.35, 0.1)
    stem_width = max(width * 0.08, 0.03)
    stem_depth = max(thickness * 0.9, 0.03)
    stem = create_box(width=stem_width, height=stem_height, depth=stem_depth)
    _translate(stem, 0.0, 0.0, stem_height / 2.0)
    mesh.merge(stem)

    base = create_box(
        width=max(width * 0.35, 0.12),
        height=max(stem_width * 0.6, 0.02),
        depth=max(thickness * 2.5, 0.08),
    )
    _translate(base, 0.0, 0.0, max(stem_width * 0.6, 0.02) / 2.0)
    mesh.merge(base)

    return _apply_transform(mesh, transform)


def create_control_panel(
    width: float,
    height: float,
    tilt_angle: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Minimal tilted control panel with pedestal base."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    _require_non_negative(abs(float(tilt_angle)), "tilt_angle")
    if abs(float(tilt_angle)) >= 89.0:
        raise ValueError("tilt_angle must be between -89 and 89 degrees.")

    panel_thickness = max(width * 0.04, 0.03)
    face = create_box(width=width, height=height, depth=panel_thickness)
    _rotate_x(face, float(tilt_angle))
    min_z = min(z for _, _, z in face.vertices)
    _translate(face, 0.0, 0.0, -min_z)

    base_height = max(height * 0.18, 0.05)
    pedestal = create_box(
        width=max(width * 0.24, 0.14),
        height=base_height,
        depth=max(panel_thickness * 2.0, 0.06),
    )
    _translate(pedestal, 0.0, 0.0, base_height / 2.0)

    mesh = Mesh()
    mesh.merge(face)
    mesh.merge(pedestal)
    return _apply_transform(mesh, transform)


def create_rack(
    height: float,
    width: float,
    depth: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Minimal server rack: body + two front rails."""
    height = _require_positive(height, "height")
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")

    mesh = Mesh()
    body = create_box(width=width, height=height, depth=depth)
    _translate(body, 0.0, 0.0, height / 2.0)
    mesh.merge(body)

    rail_width = max(width * 0.06, 0.015)
    rail_depth = max(depth * 0.08, 0.015)
    rail_x = max(width / 2.0 - rail_width / 2.0, 0.0)
    rail_y = depth / 2.0 - rail_depth / 2.0
    for sx in (-1.0, 1.0):
        rail = create_box(width=rail_width, height=height, depth=rail_depth)
        _translate(rail, sx * rail_x, rail_y, height / 2.0)
        mesh.merge(rail)

    return _apply_transform(mesh, transform)


def create_cable_tray(
    length: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Minimal U-shaped cable tray."""
    length = _require_positive(length, "length")
    height = _require_positive(height, "height")

    tray_width = max(height * 1.6, 0.08)
    wall_thickness = max(height * 0.15, 0.015)

    mesh = Mesh()
    bottom = create_box(width=length, height=wall_thickness, depth=tray_width)
    _translate(bottom, 0.0, 0.0, wall_thickness / 2.0)
    mesh.merge(bottom)

    side_depth = max(wall_thickness, 0.01)
    side_offset_y = tray_width / 2.0 - side_depth / 2.0
    for side in (-1.0, 1.0):
        wall = create_box(width=length, height=height, depth=side_depth)
        _translate(wall, 0.0, side * side_offset_y, height / 2.0)
        mesh.merge(wall)

    return _apply_transform(mesh, transform)


def create_wall_panel(
    width: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Minimal flat wall panel."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    thickness = max(min(width, height) * 0.04, 0.02)

    mesh = create_box(width=width, height=height, depth=thickness)
    return _apply_transform(mesh, transform)


def create_light_panel(
    size: float,
    intensity: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Minimal square light panel with intensity metadata."""
    size = _require_positive(size, "size")
    intensity = _require_non_negative(intensity, "intensity")
    thickness = max(size * 0.08, 0.015)

    mesh = create_box(width=size, height=thickness, depth=size)
    setattr(mesh, "metadata", {"intensity": float(intensity)})
    return _apply_transform(mesh, transform)


def create_keyboard(
    width: float,
    depth: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Minimal keyboard slab."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")
    return _apply_transform(create_box(width=width, height=height, depth=depth), transform)


def create_console_block(
    width: float,
    depth: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Compact console module for desk-mounted controls."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")

    body = create_box(width=width, height=height, depth=depth)
    top = create_box(
        width=max(width * 0.84, 0.05),
        height=max(height * 0.18, 0.01),
        depth=max(depth * 0.86, 0.05),
    )
    _translate(top, 0.0, 0.0, height / 2.0 + max(height * 0.18, 0.01) / 2.0)
    body.merge(top)
    return _apply_transform(body, transform)


def create_small_screen(
    width: float,
    height: float,
    thickness: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Small utility screen."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    thickness = _require_positive(thickness, "thickness")

    mesh = Mesh()
    screen = create_box(width=width, height=height, depth=thickness)
    _translate(screen, 0.0, 0.0, height * 0.62)
    mesh.merge(screen)

    stand = create_box(
        width=max(width * 0.1, 0.02),
        height=max(height * 0.22, 0.05),
        depth=max(thickness * 1.2, 0.02),
    )
    _translate(stand, 0.0, 0.0, max(height * 0.22, 0.05) / 2.0)
    mesh.merge(stand)
    return _apply_transform(mesh, transform)


def create_wall_display(
    width: float,
    height: float,
    thickness: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Flat wall-mounted display panel."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    thickness = _require_positive(thickness, "thickness")
    return _apply_transform(create_box(width=width, height=height, depth=thickness), transform)


def create_ups_unit(
    width: float,
    depth: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Small UPS tower next to workstation."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")

    body = create_box(width=width, height=height, depth=depth)
    vent_count = max(3, int(height / 0.15))
    vent_h = max(height * 0.02, 0.006)
    vent_d = max(depth * 0.05, 0.006)
    vent_w = max(width * 0.7, 0.03)
    span = max(height * 0.28, vent_h)
    for idx in range(vent_count):
        t = idx / (vent_count - 1) if vent_count > 1 else 0.5
        z = -span + 2.0 * span * t
        vent = create_box(width=vent_w, height=vent_h, depth=vent_d)
        _translate(vent, 0.0, depth / 2.0 + vent_d / 2.0, z)
        body.merge(vent)
    return _apply_transform(body, transform)


def create_console_cable_bundle(
    length: float,
    radius: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Round cable bundle segment for under-desk wiring."""
    length = _require_positive(length, "length")
    radius = _require_positive(radius, "radius")
    mesh = create_beam(
        length=length,
        profile_type={"type": "circular", "radius": radius, "segments": 12},
    )
    return _apply_transform(mesh, transform)
