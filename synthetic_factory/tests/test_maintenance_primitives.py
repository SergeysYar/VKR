from __future__ import annotations

import pytest

from synthetic_factory.geometry import Mesh
from synthetic_factory.parametric import (
    create_cable_loose,
    create_crane_hook,
    create_machine_part,
    create_pallet,
    create_pipe_loose,
    create_spare_part,
    create_storage_shelf,
    create_tool_rack,
    create_toolbox,
    create_workbench,
)


def test_maintenance_primitives_return_mesh() -> None:
    meshes = [
        create_workbench(width=1.8, depth=0.9, height=0.95),
        create_tool_rack(width=1.0, height=2.1),
        create_toolbox(size=0.8),
        create_spare_part(type="gear", size=0.6),
        create_machine_part(size=0.9, complexity=3),
        create_crane_hook(height=2.4),
        create_pallet(width=1.2, depth=1.0),
        create_storage_shelf(width=1.0, height=2.0, levels=4),
        create_cable_loose(length=2.0, curvature=0.3),
        create_pipe_loose(length=2.2, curvature=-0.25),
    ]

    for mesh in meshes:
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


def test_maintenance_primitives_support_variability_and_scatter() -> None:
    gear = create_spare_part(type="gear", size=0.6, variation=0.1, seed=11)
    shaft = create_spare_part(type="shaft", size=0.6, variation=0.9, seed=11)
    assert getattr(gear, "metadata", {}).get("params", {}).get("type") == "gear"
    assert getattr(shaft, "metadata", {}).get("params", {}).get("type") == "shaft"
    assert len(gear.vertices) != len(shaft.vertices)

    simple = create_machine_part(size=0.9, complexity=1, variation=0.1, seed=5)
    complex_part = create_machine_part(size=0.9, complexity=5, variation=0.9, seed=5)
    assert len(complex_part.vertices) > len(simple.vertices)

    compact_toolbox = create_toolbox(size=0.8, variation=0.3, scatter=0.0, seed=3)
    scattered_toolbox = create_toolbox(size=0.8, variation=0.3, scatter=0.5, seed=3)
    assert compact_toolbox.vertices != scattered_toolbox.vertices

    cable_flat = create_cable_loose(length=2.0, curvature=0.0, variation=0.0, scatter=0.0, seed=1)
    cable_curved = create_cable_loose(length=2.0, curvature=0.5, variation=0.5, scatter=0.3, seed=1)
    assert cable_flat.vertices != cable_curved.vertices


def test_maintenance_primitives_support_transform_and_validate_inputs() -> None:
    transform = (
        (1.0, 0.0, 0.0, 2.0),
        (0.0, 1.0, 0.0, -1.0),
        (0.0, 0.0, 1.0, 0.5),
        (0.0, 0.0, 0.0, 1.0),
    )
    shelf = create_storage_shelf(width=1.0, height=2.0, levels=3, transform=transform)
    xs = [x for x, _, _ in shelf.vertices]
    ys = [y for _, y, _ in shelf.vertices]
    zs = [z for _, _, z in shelf.vertices]
    assert min(xs) > 1.0
    assert max(ys) < -0.4
    assert min(zs) > 0.4

    with pytest.raises(ValueError):
        create_workbench(width=-1.0, depth=0.9, height=0.95)
    with pytest.raises(ValueError):
        create_spare_part(type="unknown", size=0.4)
    with pytest.raises(ValueError):
        create_machine_part(size=0.8, complexity=0)
    with pytest.raises(ValueError):
        create_storage_shelf(width=1.0, height=2.0, levels=0)
