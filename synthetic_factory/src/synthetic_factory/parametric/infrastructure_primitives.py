from __future__ import annotations

from typing import Iterable

from ..geometry.mesh import Mesh
from .primitives import create_beam, create_box, create_column

MatrixLike = Iterable[Iterable[float]]
Vector3 = tuple[float, float, float]


def _require_positive(value: float, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return number


def _transform_point(matrix: MatrixLike, point: Vector3) -> Vector3:
    rows = [tuple(float(v) for v in row) for row in matrix]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("transform must be a 4x4 matrix.")

    x, y, z = point
    tx = rows[0][0] * x + rows[0][1] * y + rows[0][2] * z + rows[0][3]
    ty = rows[1][0] * x + rows[1][1] * y + rows[1][2] * z + rows[1][3]
    tz = rows[2][0] * x + rows[2][1] * y + rows[2][2] * z + rows[2][3]
    tw = rows[3][0] * x + rows[3][1] * y + rows[3][2] * z + rows[3][3]
    if abs(tw) < 1e-12:
        return (tx, ty, tz)
    return (tx / tw, ty / tw, tz / tw)


def _translate(mesh: Mesh, dx: float, dy: float, dz: float) -> None:
    mesh.vertices = [(x + dx, y + dy, z + dz) for x, y, z in mesh.vertices]


def _set_metadata(mesh: Mesh, primitive: str, **params: object) -> None:
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


def _set_connection_points(mesh: Mesh, points: dict[str, Vector3]) -> None:
    metadata = getattr(mesh, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["connectable"] = True
    metadata["connection_points"] = dict(points)
    setattr(mesh, "metadata", metadata)


def _apply_transform(mesh: Mesh, transform: MatrixLike | None) -> Mesh:
    if transform is None:
        return mesh
    mesh.transform(transform)
    metadata = getattr(mesh, "metadata", {})
    if isinstance(metadata, dict):
        points = metadata.get("connection_points")
        if isinstance(points, dict):
            metadata["connection_points"] = {
                str(name): _transform_point(transform, point)  # type: ignore[arg-type]
                for name, point in points.items()
            }
            setattr(mesh, "metadata", metadata)
    return mesh


def create_cable_tray(
    width: float,
    height: float,
    length: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Universal cable tray segment with start/end connectors."""
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    length = _require_positive(length, "length")

    wall_t = max(min(width, height) * 0.16, 0.01)
    mesh = create_box(width=length, height=wall_t, depth=width)
    _translate(mesh, 0.0, 0.0, wall_t / 2.0)

    side_depth = max(wall_t, 0.01)
    side_offset = width / 2.0 - side_depth / 2.0
    for side in (-1.0, 1.0):
        wall = create_box(width=length, height=height, depth=side_depth)
        _translate(wall, 0.0, side * side_offset, height / 2.0)
        mesh.merge(wall)

    _set_metadata(
        mesh,
        "infrastructure_cable_tray",
        width=round(width, 6),
        height=round(height, 6),
        length=round(length, 6),
    )
    _set_connection_points(
        mesh,
        {
            "start": (-length / 2.0, 0.0, height / 2.0),
            "end": (length / 2.0, 0.0, height / 2.0),
        },
    )
    return _apply_transform(mesh, transform)


def create_tray_support(
    height: float,
    spacing: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Portal support for trays; spacing controls support span."""
    height = _require_positive(height, "height")
    spacing = _require_positive(spacing, "spacing")

    post_radius = max(min(spacing, height) * 0.06, 0.015)
    top_beam_h = max(post_radius * 1.7, 0.02)
    span = max(spacing, post_radius * 2.5)

    mesh = Mesh()
    y_offset = span / 2.0
    for side in (-1.0, 1.0):
        post = create_column(radius=post_radius, height=height, segments=12)
        _translate(post, 0.0, side * y_offset, height / 2.0)
        mesh.merge(post)

    top_beam = create_box(width=post_radius * 2.2, height=top_beam_h, depth=span + post_radius * 1.8)
    _translate(top_beam, 0.0, 0.0, height - top_beam_h / 2.0)
    mesh.merge(top_beam)

    _set_metadata(
        mesh,
        "infrastructure_tray_support",
        height=round(height, 6),
        spacing=round(spacing, 6),
    )
    _set_connection_points(
        mesh,
        {
            "top_left": (0.0, -y_offset, height),
            "top_right": (0.0, y_offset, height),
        },
    )
    return _apply_transform(mesh, transform)


def create_vertical_drop(
    length: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Vertical cable drop segment with top/bottom connectors."""
    length = _require_positive(length, "length")

    radius = max(length * 0.015, 0.006)
    mesh = create_column(radius=radius, height=length, segments=12)
    _translate(mesh, 0.0, 0.0, length / 2.0)

    cap_h = max(radius * 1.4, 0.006)
    top_cap = create_box(width=radius * 2.2, height=cap_h, depth=radius * 2.2)
    bottom_cap = create_box(width=radius * 2.2, height=cap_h, depth=radius * 2.2)
    _translate(top_cap, 0.0, 0.0, length - cap_h / 2.0)
    _translate(bottom_cap, 0.0, 0.0, cap_h / 2.0)
    mesh.merge(top_cap)
    mesh.merge(bottom_cap)

    _set_metadata(
        mesh,
        "infrastructure_vertical_drop",
        length=round(length, 6),
    )
    _set_connection_points(
        mesh,
        {
            "top": (0.0, 0.0, length),
            "bottom": (0.0, 0.0, 0.0),
        },
    )
    return _apply_transform(mesh, transform)


def create_junction_node(
    size: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Junction node with six-way connectivity."""
    size = _require_positive(size, "size")

    mesh = create_box(width=size, height=size, depth=size)
    stem_radius = max(size * 0.08, 0.006)
    stem_length = max(size * 0.32, stem_radius * 2.0)
    axis_half = size / 2.0 + stem_length / 2.0

    for axis in ("x", "-x", "y", "-y", "z", "-z"):
        stem = create_beam(
            length=stem_length,
            profile_type={"type": "circular", "radius": stem_radius, "segments": 10},
        )
        if axis == "x":
            _translate(stem, axis_half, 0.0, 0.0)
        elif axis == "-x":
            _translate(stem, -axis_half, 0.0, 0.0)
        elif axis == "y":
            stem.transform(
                (
                    (0.0, -1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, axis_half),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )
        elif axis == "-y":
            stem.transform(
                (
                    (0.0, 1.0, 0.0, 0.0),
                    (-1.0, 0.0, 0.0, -axis_half),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )
        elif axis == "z":
            stem.transform(
                (
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (-1.0, 0.0, 0.0, axis_half),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )
        else:
            stem.transform(
                (
                    (0.0, 0.0, -1.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, -axis_half),
                    (0.0, 0.0, 0.0, 1.0),
                )
            )
        mesh.merge(stem)

    _set_metadata(
        mesh,
        "infrastructure_junction_node",
        size=round(size, 6),
    )
    _set_connection_points(
        mesh,
        {
            "x_pos": (size / 2.0 + stem_length, 0.0, 0.0),
            "x_neg": (-size / 2.0 - stem_length, 0.0, 0.0),
            "y_pos": (0.0, size / 2.0 + stem_length, 0.0),
            "y_neg": (0.0, -size / 2.0 - stem_length, 0.0),
            "z_pos": (0.0, 0.0, size / 2.0 + stem_length),
            "z_neg": (0.0, 0.0, -size / 2.0 - stem_length),
        },
    )
    return _apply_transform(mesh, transform)


def create_pipe_main(
    radius: float,
    length: float,
    transform: MatrixLike | None = None,
) -> Mesh:
    """Main pipe segment with start/end connectors and collars."""
    radius = _require_positive(radius, "radius")
    length = _require_positive(length, "length")

    mesh = create_beam(
        length=length,
        profile_type={"type": "circular", "radius": radius, "segments": 20},
    )

    collar_length = max(radius * 1.25, length * 0.035)
    collar_radius = radius * 1.18
    for side in (-1.0, 1.0):
        collar = create_beam(
            length=collar_length,
            profile_type={"type": "circular", "radius": collar_radius, "segments": 16},
        )
        _translate(collar, side * (length / 2.0 - collar_length / 2.0), 0.0, 0.0)
        mesh.merge(collar)

    _set_metadata(
        mesh,
        "infrastructure_pipe_main",
        radius=round(radius, 6),
        length=round(length, 6),
    )
    _set_connection_points(
        mesh,
        {
            "start": (-length / 2.0, 0.0, 0.0),
            "end": (length / 2.0, 0.0, 0.0),
        },
    )
    return _apply_transform(mesh, transform)
