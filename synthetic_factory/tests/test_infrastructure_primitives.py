from __future__ import annotations

import pytest

from synthetic_factory.geometry import Mesh
from synthetic_factory.parametric import (
    create_infra_cable_tray,
    create_infra_junction_node,
    create_infra_pipe_main,
    create_infra_tray_support,
    create_infra_vertical_drop,
)
from synthetic_factory.parametric.infrastructure_primitives import (
    create_cable_tray,
    create_junction_node,
    create_pipe_main,
    create_tray_support,
    create_vertical_drop,
)


def _vector_close(a: tuple[float, float, float], b: tuple[float, float, float], tol: float = 1e-6) -> bool:
    return (
        abs(a[0] - b[0]) <= tol
        and abs(a[1] - b[1]) <= tol
        and abs(a[2] - b[2]) <= tol
    )


def test_infrastructure_primitives_return_mesh() -> None:
    meshes = [
        create_cable_tray(width=0.35, height=0.12, length=2.5),
        create_tray_support(height=2.8, spacing=0.9),
        create_vertical_drop(length=1.9),
        create_junction_node(size=0.3),
        create_pipe_main(radius=0.08, length=3.2),
    ]

    for mesh in meshes:
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


def test_infrastructure_primitives_support_transform_and_connection_points() -> None:
    transform = (
        (1.0, 0.0, 0.0, 3.0),
        (0.0, 1.0, 0.0, -1.0),
        (0.0, 0.0, 1.0, 2.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    tray = create_cable_tray(width=0.4, height=0.14, length=2.0, transform=transform)
    xs = [x for x, _, _ in tray.vertices]
    ys = [y for _, y, _ in tray.vertices]
    zs = [z for _, _, z in tray.vertices]
    assert min(xs) > 1.9
    assert max(ys) < -0.7
    assert min(zs) > 1.9

    metadata = getattr(tray, "metadata", {})
    assert metadata.get("primitive") == "infrastructure_cable_tray"
    points = metadata.get("connection_points", {})
    assert isinstance(points, dict)
    assert "start" in points and "end" in points
    assert _vector_close(points["start"], (2.0, -1.0, 2.07))
    assert _vector_close(points["end"], (4.0, -1.0, 2.07))


def test_infrastructure_primitives_enable_segment_connection() -> None:
    tray_a = create_cable_tray(width=0.35, height=0.12, length=2.0)
    tray_b = create_cable_tray(
        width=0.35,
        height=0.12,
        length=2.0,
        transform=(
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    points_a = getattr(tray_a, "metadata", {}).get("connection_points", {})
    points_b = getattr(tray_b, "metadata", {}).get("connection_points", {})
    assert _vector_close(points_a["end"], points_b["start"])

    pipe_a = create_pipe_main(radius=0.09, length=3.0)
    pipe_b = create_pipe_main(
        radius=0.09,
        length=3.0,
        transform=(
            (1.0, 0.0, 0.0, 3.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    pipe_points_a = getattr(pipe_a, "metadata", {}).get("connection_points", {})
    pipe_points_b = getattr(pipe_b, "metadata", {}).get("connection_points", {})
    assert _vector_close(pipe_points_a["end"], pipe_points_b["start"])


def test_infrastructure_primitives_are_exported_via_parametric_package() -> None:
    assert isinstance(create_infra_cable_tray(width=0.35, height=0.12, length=1.6), Mesh)
    assert isinstance(create_infra_tray_support(height=2.4, spacing=0.8), Mesh)
    assert isinstance(create_infra_vertical_drop(length=1.2), Mesh)
    assert isinstance(create_infra_junction_node(size=0.25), Mesh)
    assert isinstance(create_infra_pipe_main(radius=0.07, length=2.2), Mesh)


def test_infrastructure_primitives_validate_inputs() -> None:
    with pytest.raises(ValueError):
        create_cable_tray(width=-0.3, height=0.1, length=2.0)
    with pytest.raises(ValueError):
        create_tray_support(height=2.0, spacing=0.0)
    with pytest.raises(ValueError):
        create_vertical_drop(length=0.0)
    with pytest.raises(ValueError):
        create_junction_node(size=-0.1)
    with pytest.raises(ValueError):
        create_pipe_main(radius=0.0, length=2.0)
