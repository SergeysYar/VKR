from __future__ import annotations

from typing import Iterable

from ..geometry.mesh import Mesh
from .primitives import create_beam, create_box, create_column

MatrixLike = Iterable[Iterable[float]]


def _require_positive(value: float, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return number


def _require_levels(value: int, name: str = "levels") -> int:
    count = int(value)
    if count <= 0:
        raise ValueError(f"{name} must be > 0.")
    return count


def _require_count(value: int, name: str = "count") -> int:
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


def create_lab_bench(
    width: float,
    depth: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """
    Detailed laboratory bench:
    tabletop + splash guard + frame + lower shelf + 4 legs.
    """
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")

    top_t = max(min(height * 0.08, height * 0.22), 0.012)
    leg_size = max(min(width, depth) * 0.06, 0.01)
    shelf_t = max(top_t * 0.75, 0.008)
    guard_t = max(top_t * 0.6, 0.006)

    leg_height = max(height - top_t, 0.02)
    shelf_z = max(leg_height * 0.34, shelf_t / 2.0)

    mesh = Mesh()

    tabletop = create_box(width=width, height=top_t, depth=depth)
    _translate(tabletop, 0.0, 0.0, leg_height + top_t / 2.0)
    mesh.merge(tabletop)

    guard_h = max(top_t * 1.4, min(height * 0.18, 0.12))
    splash_guard = create_box(width=width * 0.96, height=guard_h, depth=guard_t)
    _translate(
        splash_guard,
        0.0,
        -depth / 2.0 + guard_t / 2.0,
        leg_height + top_t + guard_h / 2.0,
    )
    mesh.merge(splash_guard)

    shelf = create_box(width=width * 0.86, height=shelf_t, depth=depth * 0.72)
    _translate(shelf, 0.0, 0.0, shelf_z)
    mesh.merge(shelf)

    leg_x = max(width / 2.0 - leg_size / 2.0, 0.0)
    leg_y = max(depth / 2.0 - leg_size / 2.0, 0.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            leg = create_box(width=leg_size, height=leg_height, depth=leg_size)
            _translate(leg, sx * leg_x, sy * leg_y, leg_height / 2.0)
            mesh.merge(leg)

    stretcher_h = max(leg_size * 0.6, 0.006)
    stretcher_z = max(shelf_z - shelf_t, stretcher_h / 2.0)
    stretcher_x = create_box(width=max(width * 0.8, leg_size), height=stretcher_h, depth=leg_size * 0.8)
    _translate(stretcher_x, 0.0, 0.0, stretcher_z)
    mesh.merge(stretcher_x)
    stretcher_y = create_box(width=leg_size * 0.8, height=stretcher_h, depth=max(depth * 0.65, leg_size))
    _translate(stretcher_y, 0.0, 0.0, stretcher_z)
    mesh.merge(stretcher_y)

    _set_instance_metadata(
        mesh,
        "lab_bench",
        width=round(width, 6),
        depth=round(depth, 6),
        height=round(height, 6),
    )
    return _apply_transform(mesh, transform)


def create_equipment_unit(
    width: float,
    height: float,
    type: str = "generic",
    transform: MatrixLike | None = None,
) -> Mesh:
    """
    Generic lab equipment unit:
    housing + front panel + display + knobs.
    """
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    unit_type = str(type).strip().lower()
    if not unit_type:
        raise ValueError("type must be a non-empty string.")

    depth = max(width * 0.62, 0.05)
    body = create_box(width=width, height=height, depth=depth)
    _translate(body, 0.0, 0.0, height / 2.0)

    panel_t = max(depth * 0.06, 0.004)
    panel = create_box(width=width * 0.94, height=height * 0.88, depth=panel_t)
    _translate(panel, 0.0, depth / 2.0 + panel_t / 2.0, height * 0.52)
    body.merge(panel)

    display_w = max(width * 0.42, 0.018)
    display_h = max(height * 0.14, 0.01)
    display = create_box(width=display_w, height=display_h, depth=max(panel_t * 0.6, 0.002))
    _translate(
        display,
        0.0,
        depth / 2.0 + panel_t * 0.8,
        height * 0.68,
    )
    body.merge(display)

    knob_radius = max(min(width, height) * 0.035, 0.0025)
    knob_length = max(panel_t * 1.3, 0.003)
    knob_y = depth / 2.0 + panel_t * 0.7
    knob_z = max(height * 0.38, knob_radius * 2.0)
    for sx in (-1.0, 0.0, 1.0):
        knob = create_beam(
            length=knob_length,
            profile_type={"type": "circular", "radius": knob_radius, "segments": 12},
        )
        _translate(knob, sx * width * 0.18, knob_y, knob_z)
        body.merge(knob)

    vent_h = max(height * 0.04, 0.004)
    vent = create_box(width=width * 0.7, height=vent_h, depth=max(panel_t * 0.4, 0.002))
    _translate(vent, 0.0, depth / 2.0 + panel_t * 0.8, height * 0.2)
    body.merge(vent)

    _set_instance_metadata(
        body,
        "equipment_unit",
        width=round(width, 6),
        height=round(height, 6),
        type=unit_type,
    )
    return _apply_transform(body, transform)


def create_fume_hood(
    width: float,
    height: float,
    depth: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Detailed fume hood with base cabinet, sash opening and top duct."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    depth = _require_positive(depth, "depth")

    mesh = Mesh()

    base_h = max(height * 0.38, 0.08)
    chamber_h = max(height - base_h, 0.08)
    frame_t = max(min(width, depth) * 0.05, 0.008)

    base = create_box(width=width, height=base_h, depth=depth)
    _translate(base, 0.0, 0.0, base_h / 2.0)
    mesh.merge(base)

    left_col = create_box(width=frame_t, height=chamber_h, depth=depth)
    right_col = create_box(width=frame_t, height=chamber_h, depth=depth)
    x_offset = max(width / 2.0 - frame_t / 2.0, 0.0)
    _translate(left_col, -x_offset, 0.0, base_h + chamber_h / 2.0)
    _translate(right_col, x_offset, 0.0, base_h + chamber_h / 2.0)
    mesh.merge(left_col)
    mesh.merge(right_col)

    top_canopy = create_box(width=width, height=frame_t * 1.25, depth=depth)
    _translate(top_canopy, 0.0, 0.0, height - frame_t * 0.625)
    mesh.merge(top_canopy)

    back_panel = create_box(width=width - frame_t * 2.0, height=chamber_h, depth=frame_t)
    _translate(
        back_panel,
        0.0,
        -depth / 2.0 + frame_t / 2.0,
        base_h + chamber_h / 2.0,
    )
    mesh.merge(back_panel)

    sash = create_box(
        width=max(width - frame_t * 2.6, frame_t),
        height=max(chamber_h * 0.44, frame_t),
        depth=max(frame_t * 0.5, 0.004),
    )
    _translate(
        sash,
        0.0,
        depth / 2.0 - frame_t / 2.0,
        base_h + chamber_h * 0.62,
    )
    mesh.merge(sash)

    duct_r = max(min(width, depth) * 0.12, 0.02)
    duct_h = max(height * 0.16, 0.06)
    duct = create_column(radius=duct_r, height=duct_h, segments=16)
    _translate(duct, 0.0, -depth * 0.2, height + duct_h / 2.0)
    mesh.merge(duct)

    _set_instance_metadata(
        mesh,
        "fume_hood",
        width=round(width, 6),
        height=round(height, 6),
        depth=round(depth, 6),
    )
    return _apply_transform(mesh, transform)


def create_shelf(
    width: float,
    height: float,
    levels: int,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Multi-level shelf with posts and deck plates."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    levels = _require_levels(levels, "levels")

    depth = max(width * 0.36, 0.08)
    post_w = max(min(width, depth) * 0.08, 0.006)
    deck_t = max(height * 0.02, 0.005)

    mesh = Mesh()

    post_x = max(width / 2.0 - post_w / 2.0, 0.0)
    post_y = max(depth / 2.0 - post_w / 2.0, 0.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            post = create_box(width=post_w, height=height, depth=post_w)
            _translate(post, sx * post_x, sy * post_y, height / 2.0)
            mesh.merge(post)

    for idx in range(levels):
        if levels == 1:
            z = height * 0.55
        else:
            z = deck_t / 2.0 + (height - deck_t) * idx / (levels - 1)
        deck = create_box(width=width, height=deck_t, depth=depth)
        _translate(deck, 0.0, 0.0, z)
        mesh.merge(deck)

    back_brace_t = max(post_w * 0.6, 0.004)
    back_brace = create_box(width=width * 0.9, height=back_brace_t, depth=back_brace_t)
    _translate(back_brace, 0.0, -depth / 2.0 + back_brace_t / 2.0, height * 0.7)
    mesh.merge(back_brace)

    _set_instance_metadata(
        mesh,
        "shelf",
        width=round(width, 6),
        height=round(height, 6),
        levels=levels,
    )
    return _apply_transform(mesh, transform)


def create_cabinet(
    width: float,
    height: float,
    depth: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Storage cabinet with plinth, doors and handles."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    depth = _require_positive(depth, "depth")

    mesh = Mesh()

    plinth_h = max(height * 0.06, 0.008)
    body_h = max(height - plinth_h, 0.03)
    body = create_box(width=width, height=body_h, depth=depth)
    _translate(body, 0.0, 0.0, plinth_h + body_h / 2.0)
    mesh.merge(body)

    plinth = create_box(width=width * 0.96, height=plinth_h, depth=depth * 0.94)
    _translate(plinth, 0.0, 0.0, plinth_h / 2.0)
    mesh.merge(plinth)

    door_t = max(depth * 0.04, 0.004)
    door_h = body_h * 0.9
    door_z = plinth_h + body_h * 0.5
    door_w = max(width * 0.47, 0.01)
    offset_x = max(width * 0.24, door_w / 2.0)
    for sx in (-1.0, 1.0):
        door = create_box(width=door_w, height=door_h, depth=door_t)
        _translate(door, sx * offset_x, depth / 2.0 - door_t / 2.0, door_z)
        mesh.merge(door)

        handle = create_box(
            width=max(door_w * 0.06, 0.003),
            height=max(door_h * 0.22, 0.01),
            depth=max(door_t * 1.25, 0.003),
        )
        _translate(
            handle,
            sx * max(door_w * 0.2, 0.005),
            depth / 2.0 + max(door_t * 0.6, 0.002),
            door_z,
        )
        mesh.merge(handle)

    seam = create_box(width=max(width * 0.01, 0.002), height=door_h, depth=door_t * 1.05)
    _translate(seam, 0.0, depth / 2.0 - door_t / 2.0, door_z)
    mesh.merge(seam)

    top_cap = create_box(width=width, height=max(plinth_h * 0.8, 0.004), depth=depth)
    _translate(top_cap, 0.0, 0.0, height - max(plinth_h * 0.8, 0.004) / 2.0)
    mesh.merge(top_cap)

    _set_instance_metadata(
        mesh,
        "cabinet",
        width=round(width, 6),
        height=round(height, 6),
        depth=round(depth, 6),
    )
    return _apply_transform(mesh, transform)


def create_sink(
    width: float,
    depth: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Small sink module with countertop, basin walls and faucet."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")

    counter_t = max(min(width, depth) * 0.08, 0.006)
    basin_d = max(counter_t * 0.9, 0.005)
    wall_t = max(min(width, depth) * 0.05, 0.004)
    mesh = Mesh()

    counter = create_box(width=width, height=counter_t, depth=depth)
    _translate(counter, 0.0, 0.0, counter_t / 2.0)
    mesh.merge(counter)

    inner_w = max(width - wall_t * 2.0, wall_t)
    inner_d = max(depth - wall_t * 2.0, wall_t)
    inner_bottom = create_box(width=inner_w, height=wall_t, depth=inner_d)
    _translate(inner_bottom, 0.0, 0.0, -basin_d + wall_t / 2.0)
    mesh.merge(inner_bottom)

    side_h = basin_d + wall_t / 2.0
    for sx in (-1.0, 1.0):
        side = create_box(width=wall_t, height=side_h, depth=inner_d)
        _translate(side, sx * (inner_w / 2.0 - wall_t / 2.0), 0.0, -basin_d / 2.0)
        mesh.merge(side)
    for sy in (-1.0, 1.0):
        side = create_box(width=inner_w, height=side_h, depth=wall_t)
        _translate(side, 0.0, sy * (inner_d / 2.0 - wall_t / 2.0), -basin_d / 2.0)
        mesh.merge(side)

    faucet_r = max(min(width, depth) * 0.03, 0.0025)
    faucet_h = max(counter_t * 3.2, 0.03)
    stem = create_column(radius=faucet_r, height=faucet_h, segments=14)
    _translate(stem, width * 0.18, -depth * 0.25, counter_t + faucet_h / 2.0)
    mesh.merge(stem)
    spout = create_beam(
        length=max(width * 0.28, 0.015),
        profile_type={"type": "circular", "radius": max(faucet_r * 0.8, 0.002), "segments": 12},
    )
    _translate(
        spout,
        width * 0.04,
        -depth * 0.25,
        counter_t + faucet_h * 0.9,
    )
    mesh.merge(spout)

    _set_instance_metadata(
        mesh,
        "sink",
        width=round(width, 6),
        depth=round(depth, 6),
    )
    return _apply_transform(mesh, transform)


def create_pipe(
    radius: float,
    length: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Thin engineering pipe with couplings."""
    radius = _require_positive(radius, "radius")
    length = _require_positive(length, "length")

    body = create_beam(
        length=length,
        profile_type={"type": "circular", "radius": radius, "segments": 16},
    )

    coupling_len = max(length * 0.08, radius * 1.4)
    coupling_radius = radius * 1.18
    left = create_beam(
        length=coupling_len,
        profile_type={"type": "circular", "radius": coupling_radius, "segments": 14},
    )
    right = create_beam(
        length=coupling_len,
        profile_type={"type": "circular", "radius": coupling_radius, "segments": 14},
    )
    _translate(left, -(length / 2.0 - coupling_len / 2.0), 0.0, 0.0)
    _translate(right, length / 2.0 - coupling_len / 2.0, 0.0, 0.0)
    body.merge(left)
    body.merge(right)

    _set_instance_metadata(
        body,
        "lab_pipe",
        radius=round(radius, 6),
        length=round(length, 6),
    )
    return _apply_transform(body, transform)


def create_cable(
    radius: float,
    length: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Thin cable with small terminal sleeves."""
    radius = _require_positive(radius, "radius")
    length = _require_positive(length, "length")

    mesh = create_beam(
        length=length,
        profile_type={"type": "circular", "radius": radius, "segments": 12},
    )

    sleeve_len = max(length * 0.06, radius * 1.8)
    sleeve_r = radius * 1.12
    for sx in (-1.0, 1.0):
        sleeve = create_beam(
            length=sleeve_len,
            profile_type={"type": "circular", "radius": sleeve_r, "segments": 12},
        )
        _translate(sleeve, sx * (length / 2.0 - sleeve_len / 2.0), 0.0, 0.0)
        mesh.merge(sleeve)

    _set_instance_metadata(
        mesh,
        "lab_cable",
        radius=round(radius, 6),
        length=round(length, 6),
    )
    return _apply_transform(mesh, transform)


def create_light_fixture(
    size: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Ceiling light fixture: frame + diffuser + two mounts."""
    size = _require_positive(size, "size")

    width = size
    depth = max(size * 0.5, 0.02)
    frame_t = max(size * 0.06, 0.004)
    frame_h = max(size * 0.12, 0.006)

    mesh = Mesh()

    frame = create_box(width=width, height=frame_h, depth=depth)
    _translate(frame, 0.0, 0.0, frame_h / 2.0)
    mesh.merge(frame)

    diffuser = create_box(
        width=max(width - frame_t * 1.4, frame_t),
        height=max(frame_h * 0.45, 0.003),
        depth=max(depth - frame_t * 1.4, frame_t),
    )
    _translate(diffuser, 0.0, 0.0, frame_h * 0.28)
    mesh.merge(diffuser)

    mount_r = max(frame_t * 0.35, 0.002)
    mount_h = max(size * 0.18, 0.01)
    mount_x = max(width * 0.26, mount_r)
    for sx in (-1.0, 1.0):
        mount = create_column(radius=mount_r, height=mount_h, segments=10)
        _translate(mount, sx * mount_x, 0.0, frame_h + mount_h / 2.0)
        mesh.merge(mount)

    _set_instance_metadata(mesh, "light_fixture", size=round(size, 6))
    return _apply_transform(mesh, transform)


def create_bottle_cluster(
    bottle_radius: float,
    bottle_height: float,
    count: int,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Bottle set for benches and shelves."""
    bottle_radius = _require_positive(bottle_radius, "bottle_radius")
    bottle_height = _require_positive(bottle_height, "bottle_height")
    count = _require_count(count, "count")

    cols = max(1, int(round(count ** 0.5)))
    rows = (count + cols - 1) // cols
    step_x = max(bottle_radius * 2.8, 0.018)
    step_y = max(bottle_radius * 2.6, 0.018)
    start_x = -step_x * (cols - 1) / 2.0
    start_y = -step_y * (rows - 1) / 2.0

    body_h = bottle_height * 0.8
    neck_h = bottle_height * 0.16
    cap_h = max(bottle_height - body_h - neck_h, bottle_height * 0.04)
    neck_r = max(bottle_radius * 0.44, 0.002)
    cap_w = max(neck_r * 1.2, 0.002)

    mesh = Mesh()
    for idx in range(count):
        cx = start_x + (idx % cols) * step_x
        cy = start_y + (idx // cols) * step_y

        body = create_column(radius=bottle_radius, height=body_h, segments=12)
        _translate(body, cx, cy, body_h / 2.0)
        mesh.merge(body)

        neck = create_column(radius=neck_r, height=neck_h, segments=10)
        _translate(neck, cx, cy, body_h + neck_h / 2.0)
        mesh.merge(neck)

        cap = create_box(width=cap_w, height=cap_h, depth=cap_w)
        _translate(cap, cx, cy, body_h + neck_h + cap_h / 2.0)
        mesh.merge(cap)

    _set_instance_metadata(
        mesh,
        "bottle_cluster",
        bottle_radius=round(bottle_radius, 6),
        bottle_height=round(bottle_height, 6),
        count=count,
    )
    return _apply_transform(mesh, transform)


def create_small_instrument(
    width: float,
    depth: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Compact tabletop device with display and controls."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")

    body = create_box(width=width, height=height, depth=depth)
    _translate(body, 0.0, 0.0, height / 2.0)

    display = create_box(
        width=max(width * 0.42, 0.008),
        height=max(height * 0.18, 0.005),
        depth=max(depth * 0.08, 0.003),
    )
    _translate(display, 0.0, depth / 2.0 + max(depth * 0.04, 0.002), height * 0.64)
    body.merge(display)

    top_h = max(height * 0.18, 0.004)
    top = create_box(
        width=max(width * 0.72, 0.012),
        height=top_h,
        depth=max(depth * 0.62, 0.012),
    )
    _translate(top, 0.0, 0.0, height + top_h / 2.0)
    body.merge(top)

    knob_r = max(min(width, depth) * 0.06, 0.0018)
    knob_len = max(depth * 0.08, 0.003)
    for sx in (-1.0, 1.0):
        knob = create_beam(
            length=knob_len,
            profile_type={"type": "circular", "radius": knob_r, "segments": 10},
        )
        _translate(knob, sx * width * 0.18, depth / 2.0 + knob_len / 2.0, height * 0.36)
        body.merge(knob)

    _set_instance_metadata(
        body,
        "small_instrument",
        width=round(width, 6),
        depth=round(depth, 6),
        height=round(height, 6),
    )
    return _apply_transform(body, transform)


def create_tube_connection(
    radius: float,
    length: float,
    curvature: float = 0.0,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Flexible thin tube segment with optional smooth bend."""
    radius = _require_positive(radius, "radius")
    length = _require_positive(length, "length")
    bend = max(-1.0, min(1.0, float(curvature)))

    mesh = create_beam(
        length=length,
        profile_type={"type": "circular", "radius": radius, "segments": 14},
    )

    sleeve_len = max(length * 0.09, radius * 2.0)
    sleeve_r = radius * 1.12
    for sx in (-1.0, 1.0):
        sleeve = create_beam(
            length=sleeve_len,
            profile_type={"type": "circular", "radius": sleeve_r, "segments": 12},
        )
        _translate(sleeve, sx * (length / 2.0 - sleeve_len / 2.0), 0.0, 0.0)
        mesh.merge(sleeve)

    if abs(bend) > 1e-9:
        half = max(length / 2.0, 1e-9)
        amplitude = bend * max(length * 0.12, radius * 1.8)
        mesh.vertices = [
            (
                x,
                y,
                z + amplitude * max(0.0, 1.0 - (x / half) ** 2),
            )
            for x, y, z in mesh.vertices
        ]

    _set_instance_metadata(
        mesh,
        "tube_connection",
        radius=round(radius, 6),
        length=round(length, 6),
        curvature=round(bend, 6),
    )
    return _apply_transform(mesh, transform)


def create_waste_container(
    width: float,
    depth: float,
    height: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Open waste container with rim and side walls."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")

    wall_t = max(min(width, depth) * 0.08, 0.004)
    base_t = max(wall_t * 0.95, 0.003)
    wall_h = max(height - base_t, base_t)

    mesh = Mesh()
    base = create_box(width=width, height=base_t, depth=depth)
    _translate(base, 0.0, 0.0, base_t / 2.0)
    mesh.merge(base)

    side_x = max(width / 2.0 - wall_t / 2.0, 0.0)
    side_y = max(depth / 2.0 - wall_t / 2.0, 0.0)
    side_z = base_t + wall_h / 2.0
    left = create_box(width=wall_t, height=wall_h, depth=depth)
    right = create_box(width=wall_t, height=wall_h, depth=depth)
    _translate(left, -side_x, 0.0, side_z)
    _translate(right, side_x, 0.0, side_z)
    mesh.merge(left)
    mesh.merge(right)

    front = create_box(width=max(width - wall_t * 2.0, wall_t), height=wall_h, depth=wall_t)
    back = create_box(width=max(width - wall_t * 2.0, wall_t), height=wall_h, depth=wall_t)
    _translate(front, 0.0, side_y, side_z)
    _translate(back, 0.0, -side_y, side_z)
    mesh.merge(front)
    mesh.merge(back)

    rim_h = max(base_t * 0.55, 0.002)
    rim = create_box(width=width, height=rim_h, depth=depth)
    _translate(rim, 0.0, 0.0, height - rim_h / 2.0)
    mesh.merge(rim)

    _set_instance_metadata(
        mesh,
        "waste_container",
        width=round(width, 6),
        depth=round(depth, 6),
        height=round(height, 6),
    )
    return _apply_transform(mesh, transform)


def create_wall_mounted_unit(
    width: float,
    height: float,
    depth: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Wall-mounted technical unit."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    depth = _require_positive(depth, "depth")

    mesh = Mesh()
    back_depth = max(depth * 0.18, 0.006)
    back = create_box(width=width * 1.08, height=height * 1.04, depth=back_depth)
    _translate(back, 0.0, -depth / 2.0 + back_depth / 2.0, height / 2.0)
    mesh.merge(back)

    body = create_box(width=width, height=height, depth=depth)
    _translate(body, 0.0, 0.0, height / 2.0)
    mesh.merge(body)

    panel = create_box(
        width=max(width * 0.74, 0.01),
        height=max(height * 0.46, 0.01),
        depth=max(depth * 0.08, 0.003),
    )
    _translate(panel, 0.0, depth / 2.0 + max(depth * 0.04, 0.0015), height * 0.58)
    mesh.merge(panel)

    port = create_column(radius=max(min(width, height) * 0.04, 0.002), height=max(depth * 0.12, 0.003), segments=10)
    _translate(port, width * 0.22, depth / 2.0 + max(depth * 0.05, 0.002), height * 0.3)
    mesh.merge(port)

    _set_instance_metadata(
        mesh,
        "wall_mounted_unit",
        width=round(width, 6),
        height=round(height, 6),
        depth=round(depth, 6),
    )
    return _apply_transform(mesh, transform)
