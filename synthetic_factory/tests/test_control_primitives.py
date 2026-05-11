from __future__ import annotations

from synthetic_factory.geometry import Mesh
from synthetic_factory.parametric import (
    create_cable_tray,
    create_chair,
    create_console_block,
    create_console_cable_bundle,
    create_control_panel,
    create_desk,
    create_keyboard,
    create_light_panel,
    create_monitor,
    create_rack,
    create_small_screen,
    create_ups_unit,
    create_wall_display,
    create_wall_panel,
)


def test_control_primitives_return_mesh() -> None:
    meshes = [
        create_desk(width=1.4, depth=0.8, height=0.92),
        create_chair(seat_height=0.48, back_height=0.5),
        create_monitor(width=1.0, height=0.58, thickness=0.07),
        create_keyboard(width=0.5, depth=0.2, height=0.03),
        create_console_block(width=0.55, depth=0.3, height=0.12),
        create_small_screen(width=0.4, height=0.24, thickness=0.04),
        create_wall_display(width=1.6, height=0.9, thickness=0.03),
        create_ups_unit(width=0.22, depth=0.3, height=0.5),
        create_console_cable_bundle(length=0.8, radius=0.02),
        create_control_panel(width=1.6, height=1.2, tilt_angle=25.0),
        create_rack(height=1.45, width=0.52, depth=0.48),
        create_cable_tray(length=2.8, height=0.2),
        create_wall_panel(width=2.4, height=1.6),
        create_light_panel(size=0.6, intensity=420.0),
    ]

    for mesh in meshes:
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


def test_control_primitives_support_transforms_and_metadata() -> None:
    transform = (
        (1.0, 0.0, 0.0, 3.0),
        (0.0, 1.0, 0.0, -1.5),
        (0.0, 0.0, 1.0, 2.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    rack = create_rack(height=1.45, width=0.52, depth=0.48, transform=transform)
    xs = [x for x, _, _ in rack.vertices]
    ys = [y for _, y, _ in rack.vertices]
    zs = [z for _, _, z in rack.vertices]
    assert min(xs) > 2.0
    assert max(ys) < -0.5
    assert min(zs) > 1.0

    light = create_light_panel(size=0.6, intensity=700.0)
    metadata = getattr(light, "metadata", {})
    assert metadata.get("intensity") == 700.0
