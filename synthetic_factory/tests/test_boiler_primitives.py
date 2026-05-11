from __future__ import annotations

from synthetic_factory.geometry import Mesh
from synthetic_factory.parametric import (
    create_boiler_unit,
    create_bunker,
    create_chimney,
    create_control_box,
    create_heat_exchanger,
    create_ladder,
    create_pipe,
    create_pipe_support,
    create_pump,
    create_platform,
    create_stair,
    create_support_beam,
    create_tank,
    create_valve_cluster,
    create_valve,
)


def test_boiler_primitives_return_mesh() -> None:
    meshes = [
        create_boiler_unit(radius=0.6, height=3.2, segment_count=20),
        create_pipe(radius=0.12, length=4.0, bend_angle=0.0),
        create_pipe(radius=0.12, length=4.0, bend_angle=90.0),
        create_valve(radius=0.18, handle_size=0.55),
        create_platform(width=2.4, depth=1.6, height=0.18),
        create_ladder(height=2.8, step_count=8),
        create_stair(height=2.8, step_count=8),
        create_support_beam(length=3.6, profile_type={"type": "i", "width": 0.2, "height": 0.3, "web_thickness": 0.02, "flange_thickness": 0.03}),
        create_tank(radius=0.9, height=3.0),
        create_bunker(width=2.6, height=3.8, outlet_radius=0.22),
        create_pump(size=1.2, orientation="x"),
        create_heat_exchanger(length=3.2, radius=0.35),
        create_valve_cluster(count=4, spacing=0.8),
        create_pipe_support(height=2.6, spacing=1.2),
        create_control_box(size=0.9),
        create_chimney(height=7.5, radius=0.4),
    ]

    for mesh in meshes:
        assert isinstance(mesh, Mesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0


def test_boiler_primitives_support_transforms() -> None:
    transform = (
        (1.0, 0.0, 0.0, 5.0),
        (0.0, 1.0, 0.0, -2.0),
        (0.0, 0.0, 1.0, 3.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    tank = create_tank(radius=0.5, height=2.0, transform=transform)
    xs = [x for x, _, _ in tank.vertices]
    ys = [y for _, y, _ in tank.vertices]
    zs = [z for _, _, z in tank.vertices]

    assert min(xs) > 4.0
    assert max(ys) < -1.0
    assert min(zs) > 1.5

    pump = create_pump(size=1.0, orientation="z", transform=transform)
    px = [x for x, _, _ in pump.vertices]
    py = [y for _, y, _ in pump.vertices]
    pz = [z for _, _, z in pump.vertices]
    assert min(px) > 4.0
    assert max(py) < -1.0
    assert min(pz) > 2.0
