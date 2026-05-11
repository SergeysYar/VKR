from __future__ import annotations

from math import cos, sin, tau
from typing import Mapping

from ..geometry.mesh import Mesh

ProfileType = str | Mapping[str, object]


def _require_positive(value: float, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return number


def _require_segments(value: int, name: str = "segments") -> int:
    segments = int(value)
    if segments < 3:
        raise ValueError(f"{name} must be >= 3.")
    return segments


def _create_box_centered(width: float, height: float, depth: float) -> Mesh:
    hx = width / 2.0
    hy = depth / 2.0
    hz = height / 2.0

    vertices = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]

    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 6, 2),
        (3, 7, 6),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    ]

    return Mesh(vertices=vertices, faces=faces)


def _translate(mesh: Mesh, dx: float, dy: float, dz: float) -> None:
    mesh.vertices = [(x + dx, y + dy, z + dz) for x, y, z in mesh.vertices]


def _coerce_profile(profile_type: ProfileType) -> dict[str, object]:
    if isinstance(profile_type, Mapping):
        spec = {str(k).strip().lower(): v for k, v in profile_type.items()}
    elif isinstance(profile_type, str):
        spec = _parse_profile_string(profile_type)
    else:
        raise TypeError("profile_type must be either string or mapping.")

    profile = spec.get("type")
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("profile_type must define a profile type.")
    spec["type"] = profile.strip().lower()
    return spec


def _parse_profile_string(raw_profile: str) -> dict[str, object]:
    raw = raw_profile.strip()
    if not raw:
        raise ValueError("profile_type string cannot be empty.")

    profile_name, delimiter, params_blob = raw.partition(":")
    profile_name = profile_name.strip().lower()
    if not profile_name:
        raise ValueError("profile_type string must start with profile name.")

    parsed: dict[str, object] = {"type": profile_name}
    if not delimiter:
        return parsed

    for chunk in params_blob.split(","):
        pair = chunk.strip()
        if not pair:
            continue
        key, separator, raw_value = pair.partition("=")
        if not separator:
            raise ValueError(
                "profile_type params must be key=value, separated by commas."
            )
        param_name = key.strip().lower()
        if not param_name:
            raise ValueError("profile_type parameter name cannot be empty.")
        parsed[param_name] = _parse_scalar(raw_value.strip())

    return parsed


def _parse_scalar(raw: str) -> object:
    if not raw:
        raise ValueError("profile_type parameter value cannot be empty.")
    try:
        number = float(raw)
    except ValueError:
        return raw
    if number.is_integer():
        return int(number)
    return number


def _get_positive(spec: Mapping[str, object], key: str) -> float:
    if key not in spec:
        raise ValueError(f"profile_type must provide '{key}'.")
    return _require_positive(float(spec[key]), key)


def _get_segments(
    spec: Mapping[str, object],
    key: str = "segments",
    default: int | None = None,
) -> int:
    if key not in spec:
        if default is None:
            raise ValueError(f"profile_type must provide '{key}'.")
        return _require_segments(default, key)
    return _require_segments(int(spec[key]), key)


def _create_cylinder_z(radius: float, height: float, segments: int) -> Mesh:
    hz = height / 2.0
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    for i in range(segments):
        angle = tau * i / segments
        x = radius * cos(angle)
        y = radius * sin(angle)
        vertices.append((x, y, -hz))
        vertices.append((x, y, hz))

    bottom_center_idx = len(vertices)
    vertices.append((0.0, 0.0, -hz))
    top_center_idx = len(vertices)
    vertices.append((0.0, 0.0, hz))

    for i in range(segments):
        next_i = (i + 1) % segments
        b = i * 2
        t = b + 1
        b_next = next_i * 2
        t_next = b_next + 1

        faces.append((b, b_next, t_next))
        faces.append((b, t_next, t))

        faces.append((top_center_idx, t, t_next))
        faces.append((bottom_center_idx, b_next, b))

    return Mesh(vertices=vertices, faces=faces)


def _create_cylinder_x(length: float, radius: float, segments: int) -> Mesh:
    hx = length / 2.0
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    for i in range(segments):
        angle = tau * i / segments
        y = radius * cos(angle)
        z = radius * sin(angle)
        vertices.append((-hx, y, z))
        vertices.append((hx, y, z))

    left_center_idx = len(vertices)
    vertices.append((-hx, 0.0, 0.0))
    right_center_idx = len(vertices)
    vertices.append((hx, 0.0, 0.0))

    for i in range(segments):
        next_i = (i + 1) % segments
        left = i * 2
        right = left + 1
        left_next = next_i * 2
        right_next = left_next + 1

        faces.append((left, left_next, right_next))
        faces.append((left, right_next, right))

        faces.append((right_center_idx, right, right_next))
        faces.append((left_center_idx, left_next, left))

    return Mesh(vertices=vertices, faces=faces)


def create_box(width: float, height: float, depth: float) -> Mesh:
    """
    Create a box centered at origin.

    Axes:
    - width: X
    - depth: Y
    - height: Z
    """
    width = _require_positive(width, "width")
    height = _require_positive(height, "height")
    depth = _require_positive(depth, "depth")
    return _create_box_centered(width=width, height=height, depth=depth)


def create_wall(length: float, height: float, thickness: float) -> Mesh:
    """Create wall as a box primitive."""
    length = _require_positive(length, "length")
    height = _require_positive(height, "height")
    thickness = _require_positive(thickness, "thickness")
    return create_box(width=length, height=height, depth=thickness)


def create_column(radius: float, height: float, segments: int = 24) -> Mesh:
    """
    Create vertical cylindrical column centered at origin.

    `segments` controls roundness.
    """
    radius = _require_positive(radius, "radius")
    height = _require_positive(height, "height")
    segments = _require_segments(segments)
    return _create_cylinder_z(radius=radius, height=height, segments=segments)


def create_floor(width: float, depth: float) -> Mesh:
    """Create rectangular floor plane in XY at Z=0."""
    width = _require_positive(width, "width")
    depth = _require_positive(depth, "depth")
    hx = width / 2.0
    hy = depth / 2.0

    vertices = [
        (-hx, -hy, 0.0),
        (hx, -hy, 0.0),
        (hx, hy, 0.0),
        (-hx, hy, 0.0),
    ]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
    ]
    return Mesh(vertices=vertices, faces=faces)


def create_beam(length: float, profile_type: ProfileType) -> Mesh:
    """
    Create beam by profile specification.

    Supported profile types:
    - rect: width, height
      example: "rect:width=0.3,height=0.2"
    - circular: radius, segments(optional)
      example: "circular:radius=0.15,segments=32"
    - i: width, height, web_thickness, flange_thickness
      example: "i:width=0.25,height=0.4,web_thickness=0.02,flange_thickness=0.03"

    You can also pass mapping:
    {"type": "rect", "width": 0.3, "height": 0.2}
    """
    length = _require_positive(length, "length")
    spec = _coerce_profile(profile_type)
    profile_kind = str(spec["type"]).lower()

    if profile_kind in {"rect", "rectangle", "box"}:
        profile_width = _get_positive(spec, "width")
        profile_height = _get_positive(spec, "height")
        return create_box(width=length, height=profile_height, depth=profile_width)

    if profile_kind in {"circular", "circle", "round"}:
        radius = _get_positive(spec, "radius")
        segments = _get_segments(spec, default=24)
        return _create_cylinder_x(length=length, radius=radius, segments=segments)

    if profile_kind in {"i", "ibeam", "i-beam"}:
        profile_width = _get_positive(spec, "width")
        profile_height = _get_positive(spec, "height")
        web_thickness = _get_positive(spec, "web_thickness")
        flange_thickness = _get_positive(spec, "flange_thickness")

        if web_thickness > profile_width:
            raise ValueError("web_thickness cannot be greater than width.")
        if 2.0 * flange_thickness >= profile_height:
            raise ValueError("2 * flange_thickness must be < height.")

        flange_center_z = (profile_height - flange_thickness) / 2.0
        top_flange = create_box(
            width=length,
            height=flange_thickness,
            depth=profile_width,
        )
        _translate(top_flange, 0.0, 0.0, flange_center_z)

        bottom_flange = create_box(
            width=length,
            height=flange_thickness,
            depth=profile_width,
        )
        _translate(bottom_flange, 0.0, 0.0, -flange_center_z)

        web_height = profile_height - 2.0 * flange_thickness
        web = create_box(
            width=length,
            height=web_height,
            depth=web_thickness,
        )

        top_flange.merge(bottom_flange)
        top_flange.merge(web)
        return top_flange

    raise ValueError(
        "Unsupported profile type. Use one of: rect, circular, i."
    )
