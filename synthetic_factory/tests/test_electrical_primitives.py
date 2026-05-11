from __future__ import annotations

import pytest

from synthetic_factory.geometry import Mesh
from synthetic_factory.parametric import (
    create_backup_battery,
    create_cable_bundle,
    create_ceiling_tray_support,
    create_cooling_unit,
    create_electrical_cabinet,
    create_electrical_cable_tray,
    create_floor_cable_entry,
    create_floor_tile,
    create_junction_box,
    create_small_control_panel,
    create_switch_panel,
    create_transformer_unit,
)


def test_electrical_primitives_return_mesh() -> None:
    meshes = [
        create_electrical_cabinet(width=0.9, depth=0.6, height=2.2, door_type="single"),
        create_electrical_cabinet(width=0.9, depth=0.6, height=2.2, door_type="double"),
        create_electrical_cable_tray(width=0.4, height=0.12, length=2.4),
        create_cable_bundle(radius=0.04, length=1.8, curvature=0.0),
        create_cable_bundle(radius=0.04, length=1.8, curvature=45.0),
        create_switch_panel(width=1.2, height=1.8),
        create_junction_box(size=0.28),
        create_floor_tile(size=0.6),
        create_floor_cable_entry(radius=0.045, height=0.08),
        create_transformer_unit(width=1.1, depth=0.8, height=1.4),
        create_backup_battery(width=0.75, depth=0.55, height=1.1),
        create_small_control_panel(width=0.5, height=0.9, depth=0.08),
        create_ceiling_tray_support(height=1.4, spacing=0.8),
        create_cooling_unit(width=0.9, height=2.0, airflow_direction="front_to_back"),
    ]

    for mesh in meshes:
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


def test_electrical_primitives_support_transform_and_metadata() -> None:
    transform = (
        (1.0, 0.0, 0.0, 2.0),
        (0.0, 1.0, 0.0, -1.0),
        (0.0, 0.0, 1.0, 0.8),
        (0.0, 0.0, 0.0, 1.0),
    )

    cabinet = create_electrical_cabinet(
        width=0.9,
        depth=0.6,
        height=2.2,
        door_type="single",
        transform=transform,
    )
    xs = [x for x, _, _ in cabinet.vertices]
    ys = [y for _, y, _ in cabinet.vertices]
    zs = [z for _, _, z in cabinet.vertices]
    assert min(xs) > 1.0
    assert max(ys) < 0.0
    assert min(zs) > 0.7

    metadata = getattr(cabinet, "metadata", {})
    assert metadata.get("primitive") == "electrical_cabinet"
    assert "instance_key" in metadata


def test_electrical_primitive_validates_enums() -> None:
    with pytest.raises(ValueError):
        create_electrical_cabinet(width=1.0, depth=0.6, height=2.0, door_type="invalid")

    with pytest.raises(ValueError):
        create_cooling_unit(width=0.9, height=1.8, airflow_direction="diagonal")
