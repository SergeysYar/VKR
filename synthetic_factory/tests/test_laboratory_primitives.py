from __future__ import annotations

import pytest

from synthetic_factory.geometry import Mesh
from synthetic_factory.parametric import (
    create_bottle_cluster,
    create_cabinet,
    create_cable,
    create_equipment_unit,
    create_fume_hood,
    create_lab_bench,
    create_lab_pipe,
    create_light_fixture,
    create_shelf,
    create_small_instrument,
    create_sink,
    create_tube_connection,
    create_wall_mounted_unit,
    create_waste_container,
)


def test_laboratory_primitives_return_mesh() -> None:
    meshes = [
        create_lab_bench(width=1.8, depth=0.75, height=0.9),
        create_equipment_unit(width=0.65, height=0.95, type="generic"),
        create_fume_hood(width=1.25, height=2.3, depth=0.8),
        create_shelf(width=1.0, height=1.9, levels=4),
        create_cabinet(width=0.9, height=1.8, depth=0.55),
        create_sink(width=0.55, depth=0.45),
        create_lab_pipe(radius=0.02, length=1.4),
        create_cable(radius=0.01, length=1.4),
        create_light_fixture(size=0.7),
        create_bottle_cluster(bottle_radius=0.02, bottle_height=0.12, count=4),
        create_small_instrument(width=0.16, depth=0.12, height=0.1),
        create_tube_connection(radius=0.005, length=0.35, curvature=0.2),
        create_waste_container(width=0.28, depth=0.24, height=0.42),
        create_wall_mounted_unit(width=0.5, height=0.32, depth=0.16),
    ]

    for mesh in meshes:
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


def test_laboratory_primitives_support_transform_metadata_and_small_objects() -> None:
    transform = (
        (1.0, 0.0, 0.0, 1.4),
        (0.0, 1.0, 0.0, -0.8),
        (0.0, 0.0, 1.0, 0.6),
        (0.0, 0.0, 0.0, 1.0),
    )
    bench = create_lab_bench(width=1.8, depth=0.75, height=0.9, transform=transform)
    xs = [x for x, _, _ in bench.vertices]
    ys = [y for _, y, _ in bench.vertices]
    zs = [z for _, _, z in bench.vertices]
    assert min(xs) > 0.4
    assert max(ys) < 0.0
    assert min(zs) > 0.5

    tiny_pipe = create_lab_pipe(radius=0.0015, length=0.03)
    tiny_cable = create_cable(radius=0.001, length=0.02)
    assert len(tiny_pipe.vertices) > 0
    assert len(tiny_cable.vertices) > 0

    equipment = create_equipment_unit(width=0.6, height=0.9, type="generic")
    metadata = getattr(equipment, "metadata", {})
    assert metadata.get("primitive") == "equipment_unit"
    assert metadata.get("params", {}).get("type") == "generic"

    light = create_light_fixture(size=0.55)
    light_meta = getattr(light, "metadata", {})
    assert light_meta.get("primitive") == "light_fixture"

    bottles = create_bottle_cluster(bottle_radius=0.015, bottle_height=0.1, count=3)
    assert getattr(bottles, "metadata", {}).get("primitive") == "bottle_cluster"

    tube = create_tube_connection(radius=0.004, length=0.28, curvature=0.4)
    tube_meta = getattr(tube, "metadata", {})
    assert tube_meta.get("primitive") == "tube_connection"
    assert tube_meta.get("params", {}).get("curvature") == 0.4

    wall_unit = create_wall_mounted_unit(width=0.42, height=0.3, depth=0.14)
    assert getattr(wall_unit, "metadata", {}).get("primitive") == "wall_mounted_unit"


def test_laboratory_primitives_validate_inputs() -> None:
    with pytest.raises(ValueError):
        create_shelf(width=1.0, height=1.8, levels=0)

    with pytest.raises(ValueError):
        create_equipment_unit(width=0.6, height=0.9, type=" ")

    with pytest.raises(ValueError):
        create_sink(width=-0.5, depth=0.4)

    with pytest.raises(ValueError):
        create_bottle_cluster(bottle_radius=0.02, bottle_height=0.12, count=0)

    with pytest.raises(ValueError):
        create_waste_container(width=0.2, depth=0.25, height=-0.3)
