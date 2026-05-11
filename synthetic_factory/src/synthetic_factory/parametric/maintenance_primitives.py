from __future__ import annotations

from math import cos, pi, sin, sqrt
from random import Random
from typing import Iterable

from ..geometry.mesh import Mesh
from .primitives import create_beam, create_box, create_column

MatrixLike = Iterable[Iterable[float]]

_SPARE_PART_TYPES = {
    "generic",
    "gear",
    "shaft",
    "plate",
    "valve",
    "bearing",
    "housing",
}


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


def _require_levels(value: int, name: str = "levels") -> int:
    levels = int(value)
    if levels <= 0:
        raise ValueError(f"{name} must be > 0.")
    return levels


def _require_complexity(value: int, name: str = "complexity") -> int:
    complexity = int(value)
    if complexity <= 0:
        raise ValueError(f"{name} must be > 0.")
    return complexity


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
    segment_count: int = 14,
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


def _sample_jitter(rng: Random, amplitude: float) -> float:
    if amplitude <= 1e-12:
        return 0.0
    return rng.uniform(-amplitude, amplitude)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def create_workbench(
    width: float,
    depth: float,
    height: float,
    transform: MatrixLike | None = None,
    variation: float = 0.25,
    scatter: float = 0.0,
    seed: int = 0,
) -> Mesh:
    """Maintenance workbench with configurable detail variability."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    height = _require_positive(height, "height")
    variation = _require_non_negative(variation, "variation")
    scatter = _require_non_negative(scatter, "scatter")
    rng = Random(int(seed))

    var = _clamp(variation, 0.0, 1.0)
    scatter_amp = min(scatter, max(width, depth) * 0.4)

    top_t = max(height * (0.07 + var * 0.03), 0.03)
    leg_w = max(min(width, depth) * (0.07 + var * 0.02), 0.03)
    leg_h = max(height - top_t, 0.06)
    mesh = Mesh()

    tabletop = create_box(width=width, height=top_t, depth=depth)
    _translate(tabletop, 0.0, 0.0, leg_h + top_t / 2.0)
    mesh.merge(tabletop)

    leg_x = max(width / 2.0 - leg_w / 2.0, 0.0)
    leg_y = max(depth / 2.0 - leg_w / 2.0, 0.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            leg = create_box(width=leg_w, height=leg_h, depth=leg_w)
            _translate(
                leg,
                sx * leg_x + _sample_jitter(rng, scatter_amp * 0.06),
                sy * leg_y + _sample_jitter(rng, scatter_amp * 0.06),
                leg_h / 2.0,
            )
            mesh.merge(leg)

    drawer_count = 1 + int(round(var * 2.0))
    drawer_w = max(width * 0.24, 0.12)
    drawer_h = max(height * 0.16, 0.08)
    drawer_d = max(depth * 0.28, 0.1)
    drawer_offsets = [(-0.5 + (idx + 0.5) / drawer_count) * width * 0.7 for idx in range(drawer_count)]
    for offset in drawer_offsets:
        drawer = create_box(width=drawer_w, height=drawer_h, depth=drawer_d)
        _translate(
            drawer,
            offset + _sample_jitter(rng, scatter_amp * 0.1),
            -depth / 2.0 + drawer_d / 2.0 + _sample_jitter(rng, scatter_amp * 0.05),
            max(drawer_h / 2.0 + 0.03, leg_h * 0.45),
        )
        mesh.merge(drawer)

    if var > 0.35:
        vise_w = max(width * 0.14, 0.12)
        vise_h = max(height * 0.12, 0.06)
        vise_d = max(depth * 0.18, 0.08)
        vise = create_box(width=vise_w, height=vise_h, depth=vise_d)
        _translate(
            vise,
            width * 0.32 + _sample_jitter(rng, scatter_amp * 0.15),
            depth * 0.34 + _sample_jitter(rng, scatter_amp * 0.15),
            leg_h + top_t + vise_h / 2.0,
        )
        mesh.merge(vise)

    _set_instance_metadata(
        mesh,
        "maintenance_workbench",
        width=round(width, 6),
        depth=round(depth, 6),
        height=round(height, 6),
        variation=round(var, 6),
        scatter=round(scatter, 6),
        seed=int(seed),
    )
    return _apply_transform(mesh, transform)


def create_tool_rack(
    width: float,
    height: float,
    transform: MatrixLike | None = None,
    variation: float = 0.25,
    scatter: float = 0.0,
    seed: int = 0,
) -> Mesh:
    """Tool rack with variable number of shelves and bins."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    variation = _require_non_negative(variation, "variation")
    scatter = _require_non_negative(scatter, "scatter")
    rng = Random(int(seed))

    var = _clamp(variation, 0.0, 1.0)
    depth = max(width * (0.38 + var * 0.08), 0.24)
    post_w = max(min(width, depth) * 0.08, 0.02)
    shelf_t = max(height * 0.03, 0.02)
    shelf_levels = max(2, 3 + int(round(var * 3.0)))
    scatter_amp = min(scatter, width * 0.35)

    mesh = Mesh()
    post_x = max(width / 2.0 - post_w / 2.0, 0.0)
    post_y = max(depth / 2.0 - post_w / 2.0, 0.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            post = create_box(width=post_w, height=height, depth=post_w)
            _translate(post, sx * post_x, sy * post_y, height / 2.0)
            mesh.merge(post)

    for level in range(shelf_levels):
        z = shelf_t / 2.0 + (height - shelf_t) * (level / max(shelf_levels - 1, 1))
        shelf = create_box(width=width, height=shelf_t, depth=depth)
        _translate(shelf, 0.0, 0.0, z)
        mesh.merge(shelf)

    bin_count = max(2, int(2 + var * 5.0))
    for idx in range(bin_count):
        bin_w = max(width * 0.16, 0.08)
        bin_d = max(depth * 0.36, 0.08)
        bin_h = max(height * 0.1, 0.06)
        bx = (-0.5 + (idx + 0.5) / bin_count) * (width * 0.85) + _sample_jitter(rng, scatter_amp * 0.15)
        by = depth * 0.3 + _sample_jitter(rng, scatter_amp * 0.12)
        bz = height * (0.18 + 0.62 * ((idx % max(shelf_levels - 1, 1)) / max(shelf_levels - 1, 1)))
        bin_mesh = create_box(width=bin_w, height=bin_h, depth=bin_d)
        _translate(bin_mesh, bx, by, bz)
        mesh.merge(bin_mesh)

    _set_instance_metadata(
        mesh,
        "maintenance_tool_rack",
        width=round(width, 6),
        height=round(height, 6),
        variation=round(var, 6),
        scatter=round(scatter, 6),
        seed=int(seed),
    )
    return _apply_transform(mesh, transform)


def create_toolbox(
    size: float,
    transform: MatrixLike | None = None,
    variation: float = 0.25,
    scatter: float = 0.0,
    seed: int = 0,
) -> Mesh:
    """Toolbox with optional compartment scatter details."""
    size = _require_positive(size, "size")
    variation = _require_non_negative(variation, "variation")
    scatter = _require_non_negative(scatter, "scatter")
    rng = Random(int(seed))

    var = _clamp(variation, 0.0, 1.0)
    width = size
    depth = max(size * 0.55, 0.1)
    height = max(size * 0.42, 0.08)
    shell_t = max(size * 0.06, 0.01)
    scatter_amp = min(scatter, size * 0.6)

    mesh = Mesh()
    body = create_box(width=width, height=height, depth=depth)
    _translate(body, 0.0, 0.0, height / 2.0)
    mesh.merge(body)

    lid_h = max(height * 0.24, 0.03)
    lid = create_box(width=width * 0.98, height=lid_h, depth=depth * 0.94)
    _translate(lid, 0.0, 0.0, height + lid_h / 2.0)
    mesh.merge(lid)

    handle = create_box(
        width=max(width * 0.28, 0.05),
        height=max(shell_t * 0.9, 0.01),
        depth=max(depth * 0.18, 0.02),
    )
    _translate(handle, 0.0, depth * 0.35, height + lid_h + shell_t)
    mesh.merge(handle)

    latch_count = 1 if var < 0.45 else 2
    for idx in range(latch_count):
        lx = 0.0 if latch_count == 1 else (-1.0 if idx == 0 else 1.0) * width * 0.22
        latch = create_box(width=max(width * 0.08, 0.02), height=max(shell_t * 0.8, 0.01), depth=max(depth * 0.08, 0.01))
        _translate(latch, lx, depth / 2.0 + shell_t * 0.45, height * 0.78)
        mesh.merge(latch)

    loose_count = int(round(var * 3.0 + min(scatter_amp / max(size, 1e-9), 1.0) * 3.0))
    loose_count = min(max(loose_count, 0), 5)
    for idx in range(loose_count):
        part_size = max(size * 0.08, 0.015)
        part = create_box(width=part_size, height=part_size * 0.65, depth=part_size * 0.9)
        _translate(
            part,
            _sample_jitter(rng, scatter_amp * 0.4),
            _sample_jitter(rng, scatter_amp * 0.3),
            height + lid_h + part_size * 0.4 + idx * 0.001,
        )
        mesh.merge(part)

    _set_instance_metadata(
        mesh,
        "maintenance_toolbox",
        size=round(size, 6),
        variation=round(var, 6),
        scatter=round(scatter, 6),
        seed=int(seed),
    )
    return _apply_transform(mesh, transform)


def _build_spare_part_mesh(part_type: str, size: float, variation: float, rng: Random) -> Mesh:
    if part_type == "gear":
        core_r = max(size * 0.24, 0.02)
        core_h = max(size * 0.2, 0.02)
        gear = create_column(radius=core_r, height=core_h, segments=18)
        tooth_count = max(8, int(8 + variation * 10.0))
        tooth_w = max(size * 0.07, 0.01)
        tooth_h = max(core_h * 0.7, 0.01)
        tooth_d = max(size * 0.08, 0.01)
        radius = core_r + tooth_d * 0.55
        for idx in range(tooth_count):
            angle = 2.0 * pi * idx / tooth_count
            tooth = create_box(width=tooth_w, height=tooth_h, depth=tooth_d)
            _translate(tooth, radius * cos(angle), radius * sin(angle), 0.0)
            gear.merge(tooth)
        return gear

    if part_type == "shaft":
        length = max(size * 0.95, 0.08)
        radius = max(size * 0.12, 0.01)
        shaft = create_beam(
            length=length,
            profile_type={"type": "circular", "radius": radius, "segments": 16},
        )
        flange_r = radius * (1.4 + variation * 0.6)
        flange_t = max(radius * 0.6, 0.01)
        for side in (-1.0, 1.0):
            flange = create_beam(
                length=flange_t,
                profile_type={"type": "circular", "radius": flange_r, "segments": 16},
            )
            _translate(flange, side * (length / 2.0 - flange_t / 2.0), 0.0, 0.0)
            shaft.merge(flange)
        return shaft

    if part_type == "plate":
        width = max(size * 0.82, 0.08)
        depth = max(size * 0.62, 0.08)
        thickness = max(size * 0.12, 0.01)
        plate = create_box(width=width, height=thickness, depth=depth)
        rib_count = 1 + int(round(variation * 2.0))
        for idx in range(rib_count):
            rib = create_box(width=width * 0.16, height=thickness * 1.8, depth=depth * 0.9)
            x = (-0.5 + (idx + 0.5) / rib_count) * width * 0.7
            _translate(rib, x, 0.0, thickness * 0.9)
            plate.merge(rib)
        return plate

    if part_type == "valve":
        ring_r = max(size * 0.22, 0.02)
        ring_t = max(size * 0.08, 0.01)
        valve = create_column(radius=ring_r, height=ring_t, segments=18)
        hub = create_column(radius=ring_r * 0.35, height=ring_t * 1.6, segments=12)
        valve.merge(hub)
        spoke_count = 4 + int(round(variation * 2.0))
        for idx in range(spoke_count):
            angle = 2.0 * pi * idx / spoke_count
            spoke = create_box(width=ring_r * 1.4, height=ring_t * 0.65, depth=ring_t * 0.65)
            _translate(spoke, ring_r * 0.3 * cos(angle), ring_r * 0.3 * sin(angle), 0.0)
            valve.merge(spoke)
        return valve

    if part_type == "bearing":
        outer_r = max(size * 0.24, 0.02)
        inner_r = max(size * 0.12, 0.01)
        ring_h = max(size * 0.14, 0.01)
        bearing = create_column(radius=outer_r, height=ring_h, segments=20)
        inner = create_column(radius=inner_r, height=ring_h * 1.1, segments=14)
        bearing.merge(inner)
        ball_count = max(6, int(6 + variation * 6.0))
        ball_r = max(size * 0.03, 0.004)
        orbit = (outer_r + inner_r) * 0.5
        for idx in range(ball_count):
            angle = 2.0 * pi * idx / ball_count
            ball = create_column(radius=ball_r, height=ball_r * 1.4, segments=10)
            _translate(ball, orbit * cos(angle), orbit * sin(angle), 0.0)
            bearing.merge(ball)
        return bearing

    # housing / generic fallback
    body = create_box(
        width=max(size * 0.74, 0.08),
        height=max(size * 0.48, 0.05),
        depth=max(size * 0.58, 0.06),
    )
    port_count = 1 + int(round(variation * 3.0))
    for idx in range(port_count):
        radius = max(size * 0.04, 0.004)
        port = create_column(radius=radius, height=max(size * 0.2, 0.02), segments=12)
        px = (-0.5 + (idx + 0.5) / port_count) * size * 0.5
        py = size * 0.3
        _translate(port, px, py, size * 0.12 + _sample_jitter(rng, size * 0.04))
        body.merge(port)
    return body


def create_spare_part(
    type: str,
    size: float,
    transform: MatrixLike | None = None,
    variation: float = 0.25,
    scatter: float = 0.0,
    seed: int = 0,
) -> Mesh:
    """Spare part with type-driven shape families and optional local scatter."""
    size = _require_positive(size, "size")
    variation = _require_non_negative(variation, "variation")
    scatter = _require_non_negative(scatter, "scatter")
    part_type = str(type).strip().lower()
    if not part_type:
        raise ValueError("type must be a non-empty string.")
    if part_type not in _SPARE_PART_TYPES:
        raise ValueError(f"type must be one of: {sorted(_SPARE_PART_TYPES)}")

    rng = Random(int(seed))
    var = _clamp(variation, 0.0, 1.0)
    if part_type == "generic":
        choices = ["gear", "shaft", "plate", "valve", "bearing", "housing"]
        part_type = choices[int(rng.random() * len(choices)) % len(choices)]
    if part_type == "housing":
        part_type = "generic"

    mesh = _build_spare_part_mesh(part_type, size, var, rng)
    scatter_amp = min(scatter, size * 0.9)
    fragment_count = int(round(min(scatter_amp / max(size, 1e-9), 1.0) * 4.0))
    fragment_count = min(max(fragment_count, 0), 4)
    for idx in range(fragment_count):
        frag = create_box(
            width=max(size * 0.08, 0.01),
            height=max(size * 0.05, 0.01),
            depth=max(size * 0.07, 0.01),
        )
        _translate(
            frag,
            _sample_jitter(rng, scatter_amp * 0.5),
            _sample_jitter(rng, scatter_amp * 0.5),
            max(size * 0.02, 0.005) + idx * 0.001,
        )
        mesh.merge(frag)

    _set_instance_metadata(
        mesh,
        "maintenance_spare_part",
        type=part_type,
        size=round(size, 6),
        variation=round(var, 6),
        scatter=round(scatter, 6),
        seed=int(seed),
    )
    return _apply_transform(mesh, transform)


def create_machine_part(
    size: float,
    complexity: int,
    transform: MatrixLike | None = None,
    variation: float = 0.25,
    scatter: float = 0.0,
    seed: int = 0,
) -> Mesh:
    """Machine sub-assembly with complexity-driven topology."""
    size = _require_positive(size, "size")
    complexity = _require_complexity(complexity, "complexity")
    variation = _require_non_negative(variation, "variation")
    scatter = _require_non_negative(scatter, "scatter")
    rng = Random(int(seed))

    var = _clamp(variation, 0.0, 1.0)
    complexity = min(complexity, 12)
    scatter_amp = min(scatter, size * 0.8)

    body_w = max(size * 0.72, 0.08)
    body_h = max(size * 0.46, 0.06)
    body_d = max(size * 0.58, 0.08)
    mesh = create_box(width=body_w, height=body_h, depth=body_d)

    module_count = max(1, complexity)
    for idx in range(module_count):
        mode = idx % 3
        shift = (-0.5 + (idx + 0.5) / module_count)
        x = shift * body_w * 0.92 + _sample_jitter(rng, scatter_amp * 0.2)
        y = _sample_jitter(rng, scatter_amp * 0.18)
        z = body_h * (0.5 + 0.15 * (idx % 2))
        if mode == 0:
            module = create_box(
                width=max(size * 0.13, 0.02),
                height=max(size * (0.11 + var * 0.06), 0.02),
                depth=max(size * 0.1, 0.02),
            )
        elif mode == 1:
            module = create_column(
                radius=max(size * 0.05, 0.01),
                height=max(size * (0.16 + var * 0.08), 0.02),
                segments=12,
            )
        else:
            module = create_beam(
                length=max(size * 0.2, 0.04),
                profile_type={"type": "circular", "radius": max(size * 0.03, 0.008), "segments": 12},
            )
        _translate(module, x, y, z)
        mesh.merge(module)

    bolt_count = max(4, complexity * 2)
    bolt_r = max(size * 0.015, 0.003)
    for idx in range(bolt_count):
        bx = _sample_jitter(rng, body_w * 0.45)
        by = _sample_jitter(rng, body_d * 0.42)
        bolt = create_column(radius=bolt_r, height=max(size * 0.05, 0.01), segments=10)
        _translate(bolt, bx, by, body_h * 0.55)
        mesh.merge(bolt)

    fragment_count = int(round(min(scatter_amp / max(size, 1e-9), 1.0) * complexity * 0.4))
    fragment_count = min(max(fragment_count, 0), 8)
    for idx in range(fragment_count):
        frag = create_box(
            width=max(size * 0.06, 0.01),
            height=max(size * 0.04, 0.01),
            depth=max(size * 0.05, 0.01),
        )
        _translate(
            frag,
            _sample_jitter(rng, scatter_amp * 0.6),
            _sample_jitter(rng, scatter_amp * 0.6),
            _sample_jitter(rng, scatter_amp * 0.2),
        )
        mesh.merge(frag)

    _set_instance_metadata(
        mesh,
        "maintenance_machine_part",
        size=round(size, 6),
        complexity=complexity,
        variation=round(var, 6),
        scatter=round(scatter, 6),
        seed=int(seed),
    )
    return _apply_transform(mesh, transform)


def create_crane_hook(
    height: float,
    transform: MatrixLike | None = None,
    variation: float = 0.2,
    scatter: float = 0.0,
    seed: int = 0,
) -> Mesh:
    """Suspended crane hook: shank + curved hook."""
    height = _require_positive(height, "height")
    variation = _require_non_negative(variation, "variation")
    scatter = _require_non_negative(scatter, "scatter")
    rng = Random(int(seed))
    var = _clamp(variation, 0.0, 1.0)
    scatter_amp = min(scatter, height * 0.3)

    shank_r = max(height * (0.024 + var * 0.008), 0.01)
    shank_h = max(height * 0.68, 0.06)
    head_h = max(height * 0.12, 0.03)
    ring_r = shank_r * (1.4 + var * 0.4)

    mesh = Mesh()
    shank = create_column(radius=shank_r, height=shank_h, segments=16)
    _translate(shank, 0.0, 0.0, shank_h / 2.0)
    mesh.merge(shank)

    head = create_box(width=ring_r * 2.2, height=head_h, depth=ring_r * 1.6)
    _translate(head, 0.0, 0.0, shank_h + head_h / 2.0)
    mesh.merge(head)

    arc_radius = max(height * (0.18 + var * 0.06), shank_r * 4.0)
    hook_r = max(shank_r * 0.8, 0.008)
    samples = max(8, int(10 + var * 8.0))
    points: list[tuple[float, float, float]] = []
    for idx in range(samples):
        t = idx / (samples - 1)
        angle = (0.2 + t * 1.3) * pi
        x = arc_radius * cos(angle)
        y = 0.0
        z = arc_radius * sin(angle)
        points.append((x, y, z))
    base_z = shank_h - arc_radius * 0.32
    for idx in range(len(points) - 1):
        p0 = points[idx]
        p1 = points[idx + 1]
        seg = _tube_between(
            (p0[0], p0[1], p0[2] + base_z),
            (p1[0], p1[1], p1[2] + base_z),
            radius=hook_r,
            segment_count=14,
        )
        mesh.merge(seg)

    if scatter_amp > 1e-9:
        links = max(1, int(round(var * 2.0)))
        for idx in range(links):
            link = create_box(
                width=hook_r * 1.9,
                height=hook_r * 1.9,
                depth=hook_r * 1.4,
            )
            _translate(
                link,
                _sample_jitter(rng, scatter_amp * 0.3),
                _sample_jitter(rng, scatter_amp * 0.2),
                shank_h + head_h + idx * hook_r * 1.5,
            )
            mesh.merge(link)

    _set_instance_metadata(
        mesh,
        "maintenance_crane_hook",
        height=round(height, 6),
        variation=round(var, 6),
        scatter=round(scatter, 6),
        seed=int(seed),
    )
    return _apply_transform(mesh, transform)


def create_pallet(
    width: float,
    depth: float,
    transform: MatrixLike | None = None,
    variation: float = 0.2,
    scatter: float = 0.0,
    seed: int = 0,
) -> Mesh:
    """Industrial pallet with slats and optional loose load blocks."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    variation = _require_non_negative(variation, "variation")
    scatter = _require_non_negative(scatter, "scatter")
    rng = Random(int(seed))

    var = _clamp(variation, 0.0, 1.0)
    scatter_amp = min(scatter, max(width, depth) * 0.5)
    slat_h = max(min(width, depth) * 0.07, 0.015)
    slat_count = max(3, int(4 + var * 3.0))
    stringer_h = max(slat_h * 1.15, 0.015)
    mesh = Mesh()

    for idx in range(slat_count):
        x = (-0.5 + (idx + 0.5) / slat_count) * width * 0.94 + _sample_jitter(rng, scatter_amp * 0.06)
        slat = create_box(width=max(width / slat_count * 0.86, 0.05), height=slat_h, depth=depth)
        _translate(slat, x, 0.0, slat_h / 2.0)
        mesh.merge(slat)

    stringer_count = 3 if var < 0.5 else 4
    for idx in range(stringer_count):
        y = (-0.5 + (idx + 0.5) / stringer_count) * depth * 0.8
        stringer = create_box(width=width * 0.94, height=stringer_h, depth=max(depth * 0.14, 0.05))
        _translate(stringer, 0.0, y, slat_h + stringer_h / 2.0)
        mesh.merge(stringer)

    if scatter_amp > 1e-9:
        block_count = min(4, int(round(scatter_amp / max(max(width, depth), 1e-9) * 5.0 + var * 2.0)))
        for idx in range(block_count):
            block = create_box(
                width=max(width * 0.1, 0.04),
                height=max(slat_h * 1.3, 0.02),
                depth=max(depth * 0.1, 0.04),
            )
            _translate(
                block,
                _sample_jitter(rng, scatter_amp * 0.6),
                _sample_jitter(rng, scatter_amp * 0.6),
                slat_h + stringer_h + block.vertices[0][2] * -1.0 + 0.02,
            )
            mesh.merge(block)

    _set_instance_metadata(
        mesh,
        "maintenance_pallet",
        width=round(width, 6),
        depth=round(depth, 6),
        variation=round(var, 6),
        scatter=round(scatter, 6),
        seed=int(seed),
    )
    return _apply_transform(mesh, transform)


def create_storage_shelf(
    width: float,
    height: float,
    levels: int,
    transform: MatrixLike | None = None,
    variation: float = 0.25,
    scatter: float = 0.0,
    seed: int = 0,
) -> Mesh:
    """Storage shelf with level control and scattered small bins."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    levels = _require_levels(levels, "levels")
    variation = _require_non_negative(variation, "variation")
    scatter = _require_non_negative(scatter, "scatter")
    rng = Random(int(seed))

    var = _clamp(variation, 0.0, 1.0)
    scatter_amp = min(scatter, width * 0.5)
    depth = max(width * (0.34 + var * 0.12), 0.2)
    post_w = max(min(width, depth) * 0.07, 0.015)
    shelf_t = max(height * 0.022, 0.012)
    mesh = Mesh()

    px = max(width / 2.0 - post_w / 2.0, 0.0)
    py = max(depth / 2.0 - post_w / 2.0, 0.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            post = create_box(width=post_w, height=height, depth=post_w)
            _translate(post, sx * px, sy * py, height / 2.0)
            mesh.merge(post)

    for level in range(levels):
        z = shelf_t / 2.0 + (height - shelf_t) * level / max(levels - 1, 1)
        deck = create_box(width=width, height=shelf_t, depth=depth)
        _translate(deck, 0.0, 0.0, z)
        mesh.merge(deck)

    bin_rows = 1 + int(round(var * 2.0))
    bins_per_row = max(2, int(2 + var * 4.0))
    for row in range(bin_rows):
        z = height * (0.2 + 0.6 * row / max(bin_rows - 1, 1))
        for idx in range(bins_per_row):
            bw = max(width * 0.14, 0.05)
            bd = max(depth * 0.32, 0.05)
            bh = max(height * 0.08, 0.04)
            bx = (-0.5 + (idx + 0.5) / bins_per_row) * width * 0.85 + _sample_jitter(rng, scatter_amp * 0.14)
            by = depth * 0.28 + _sample_jitter(rng, scatter_amp * 0.1)
            bin_mesh = create_box(width=bw, height=bh, depth=bd)
            _translate(bin_mesh, bx, by, z)
            mesh.merge(bin_mesh)

    brace = create_box(width=width * 0.9, height=max(post_w * 0.6, 0.01), depth=max(post_w * 0.6, 0.01))
    _translate(brace, 0.0, -depth / 2.0 + post_w * 0.3, height * 0.7)
    mesh.merge(brace)

    _set_instance_metadata(
        mesh,
        "maintenance_storage_shelf",
        width=round(width, 6),
        height=round(height, 6),
        levels=levels,
        variation=round(var, 6),
        scatter=round(scatter, 6),
        seed=int(seed),
    )
    return _apply_transform(mesh, transform)


def _build_loose_curve_mesh(
    primitive_name: str,
    length: float,
    curvature: float,
    radius: float,
    variation: float,
    scatter: float,
    seed: int,
    with_couplings: bool,
) -> Mesh:
    rng = Random(int(seed))
    var = _clamp(variation, 0.0, 1.0)
    scatter_amp = min(scatter, length * 0.5)
    curvature = float(curvature)

    samples = max(6, int(length / 0.25) + 5)
    half = length / 2.0
    phase = rng.uniform(0.0, 2.0 * pi)
    points: list[tuple[float, float, float]] = []
    for idx in range(samples):
        t = idx / (samples - 1)
        x = -half + length * t
        arch = (1.0 - (2.0 * t - 1.0) ** 2)
        y = curvature * length * 0.18 * arch
        y += sin(2.0 * pi * t + phase) * length * (0.03 + var * 0.05)
        y += _sample_jitter(rng, scatter_amp * 0.2)
        z = -abs(curvature) * length * 0.06 * arch
        z += cos(3.0 * pi * t + phase) * length * (0.01 + var * 0.02)
        z += _sample_jitter(rng, scatter_amp * 0.08)
        points.append((x, y, z))

    mesh = Mesh()
    for idx in range(len(points) - 1):
        mesh.merge(_tube_between(points[idx], points[idx + 1], radius=radius, segment_count=14))

    end_sleeve_len = max(length * 0.06, radius * 2.2)
    for side in (-1.0, 1.0):
        end = create_beam(
            length=end_sleeve_len,
            profile_type={
                "type": "circular",
                "radius": radius * (1.16 if with_couplings else 1.08),
                "segments": 12,
            },
        )
        _translate(end, side * (half - end_sleeve_len / 2.0), 0.0, 0.0)
        mesh.merge(end)

    _set_instance_metadata(
        mesh,
        primitive_name,
        length=round(length, 6),
        curvature=round(curvature, 6),
        radius=round(radius, 6),
        variation=round(var, 6),
        scatter=round(scatter, 6),
        seed=int(seed),
    )
    return mesh


def create_cable_loose(
    length: float,
    curvature: float,
    transform: MatrixLike | None = None,
    variation: float = 0.25,
    scatter: float = 0.0,
    seed: int = 0,
) -> Mesh:
    """Loose cable segment with smooth bends and jitter."""
    length = _require_positive(length, "length")
    variation = _require_non_negative(variation, "variation")
    scatter = _require_non_negative(scatter, "scatter")
    radius = max(length * 0.012, 0.004)
    mesh = _build_loose_curve_mesh(
        "maintenance_cable_loose",
        length=length,
        curvature=curvature,
        radius=radius,
        variation=variation,
        scatter=scatter,
        seed=seed,
        with_couplings=False,
    )
    return _apply_transform(mesh, transform)


def create_pipe_loose(
    length: float,
    curvature: float,
    transform: MatrixLike | None = None,
    variation: float = 0.2,
    scatter: float = 0.0,
    seed: int = 0,
) -> Mesh:
    """Loose pipe segment with stronger couplings and curvature."""
    length = _require_positive(length, "length")
    variation = _require_non_negative(variation, "variation")
    scatter = _require_non_negative(scatter, "scatter")
    radius = max(length * 0.02, 0.007)
    mesh = _build_loose_curve_mesh(
        "maintenance_pipe_loose",
        length=length,
        curvature=curvature,
        radius=radius,
        variation=variation,
        scatter=scatter,
        seed=seed,
        with_couplings=True,
    )
    return _apply_transform(mesh, transform)
