from __future__ import annotations

from synthetic_factory.parametric.refinery_primitives import (
    create_distillation_column,
    create_heat_exchanger,
    create_junction_node,
    create_ladder,
    create_pipe,
    create_pipe_from_path,
    create_platform,
    create_process_vessel,
    create_pump,
    create_reactor,
    create_refinery_pipe,
    create_refinery_platform,
    create_refinery_support,
    create_stair,
    create_storage_tank,
    create_valve,
)


def test_create_distillation_column_has_geometry() -> None:
    mesh = create_distillation_column(radius=1.0, height=12.0, segment_count=24, stage_count=6)
    assert mesh.vertices
    assert mesh.faces
    z_values = [vertex[2] for vertex in mesh.vertices]
    assert max(z_values) - min(z_values) > 12.0


def test_create_process_vessel_has_geometry() -> None:
    mesh = create_process_vessel(radius=0.8, length=3.2)
    assert mesh.vertices
    assert mesh.faces


def test_new_refinery_core_primitives_have_geometry() -> None:
    reactor = create_reactor(radius=1.2, height=7.5)
    exchanger = create_heat_exchanger(length=4.2, radius=0.45)
    tank = create_storage_tank(radius=1.4, height=6.4)
    valve = create_valve(size=0.8)
    pump = create_pump(size=1.1)
    platform = create_platform(width=2.4, depth=1.6, height=1.2)
    ladder = create_ladder(height=3.0, step_count=8)
    stair = create_stair(height=3.0, step_count=8)
    assert reactor.vertices and reactor.faces
    assert exchanger.vertices and exchanger.faces
    assert tank.vertices and tank.faces
    assert valve.vertices and valve.faces
    assert pump.vertices and pump.faces
    assert platform.vertices and platform.faces
    assert ladder.vertices and ladder.faces
    assert stair.vertices and stair.faces


def test_create_refinery_pipe_supports_bend() -> None:
    straight = create_refinery_pipe(radius=0.2, length=4.0, bend_angle=0.0)
    bent = create_refinery_pipe(radius=0.2, length=4.0, bend_angle=35.0)
    assert straight.vertices
    assert bent.vertices
    assert len(bent.vertices) > len(straight.vertices)


def test_extended_pipe_supports_bend_radius_and_path_connections() -> None:
    straight = create_pipe(radius=0.18, length=4.0, bend_radius=0.0, segments=16)
    bent = create_pipe(radius=0.18, length=4.0, bend_radius=0.6, segments=16)
    complex_path = create_pipe_from_path(
        points=[(0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (1.2, 0.8, 0.4), (2.0, 1.0, 0.8)],
        radius=0.14,
        segments=14,
    )
    assert straight.vertices and straight.faces
    assert bent.vertices and bent.faces
    assert complex_path.vertices and complex_path.faces
    assert len(bent.vertices) > len(straight.vertices)


def test_create_platform_support_and_node() -> None:
    platform = create_refinery_platform(width=3.0, depth=2.0, thickness=0.15)
    support = create_refinery_support(height=2.8, spacing=1.5)
    node = create_junction_node(size=0.45)
    assert platform.vertices and platform.faces
    assert support.vertices and support.faces
    assert node.vertices and node.faces
