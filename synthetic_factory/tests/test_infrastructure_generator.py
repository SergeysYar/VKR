from __future__ import annotations

from synthetic_factory.generators import FactoryGenerator, InfrastructureGenerator
from synthetic_factory.parametric.primitives import create_box, create_floor
from synthetic_factory.scene.scene_graph import Scene, SceneObject, Transform


def _build_params(
    include_pipes: bool = True,
    infrastructure_enabled: bool = True,
) -> dict[str, object]:
    return {
        "factory_width": 40.0,
        "factory_depth": 28.0,
        "number_of_rooms": 4,
        "room_size_range": (10.0, 10.0),
        "corridor_width": 2.0,
        "layout_strategy": "grid",
        "noise": {"seed": 42},
        "biomes": {"enabled": False},
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {
            "enabled": False,
            "infrastructure": {
                "enabled": infrastructure_enabled,
                "include_pipes": include_pipes,
                "tray_height": 2.8,
                "tray_width": 0.3,
                "tray_thickness": 0.1,
                "room_link_width": 0.24,
                "parallel_channels": 3,
                "channel_spacing": 3.2,
                "cross_connections": True,
                "cross_spacing": 6.0,
                "junction_size": 0.24,
                "support_spacing": 3.0,
                "support_radius": 0.05,
                "secondary_branches": True,
                "secondary_branch_count": 2,
                "secondary_branch_ratio": 0.3,
            },
        },
        "seed": 7,
    }


def test_factory_generator_applies_global_infrastructure_network() -> None:
    scene = FactoryGenerator(_build_params()).generate_scene()
    network_root = scene.get_object("global_infrastructure")
    assert network_root is not None

    network_types = {obj.type for obj in network_root.traverse()}
    assert "infra_cable_tray_backbone" in network_types
    assert "infra_cable_tray_room_link" in network_types
    assert "infra_cable_tray_branch" in network_types
    assert "infra_cable_tray_cross" in network_types
    assert "infra_junction_node" in network_types
    assert "infra_vertical_drop" in network_types
    assert "infra_support" in network_types
    assert "infra_pipe_backbone" in network_types
    assert "infra_pipe_casing" in network_types
    assert "infra_cable_bundle" in network_types

    room_roots = [obj for obj in scene.objects if obj.type.startswith("room_")]
    assert room_roots
    assert len(scene.find_by_type("infra_cable_tray_room_link")) >= max(1, len(room_roots) - 1)
    assert len(scene.find_by_type("infra_vertical_drop")) >= len(room_roots)
    assert len(scene.find_by_type("infra_cable_tray_backbone")) >= 2
    assert len(scene.find_by_type("infra_junction_node")) >= 2
    assert len(scene.find_by_type("infra_support")) >= 4


def test_infrastructure_respects_optional_pipe_toggle() -> None:
    scene = FactoryGenerator(_build_params(include_pipes=False)).generate_scene()
    pipe_objects = [obj for obj in scene.traverse() if obj.type.startswith("infra_pipe_")]
    assert not pipe_objects


def test_infrastructure_methods_work_individually() -> None:
    params = _build_params(infrastructure_enabled=False)
    generator = FactoryGenerator(params)
    generator.generate_layout()
    scene = generator.instantiate_rooms()

    infrastructure = InfrastructureGenerator(
        {
            "enabled": True,
            "include_pipes": True,
            "tray_height": 2.6,
            "secondary_branch_count": 2,
            "secondary_branch_ratio": 0.3,
        }
    )

    network = infrastructure.generate_backbone(scene)
    rooms = [obj for obj in scene.objects if obj.type.startswith("room_")]
    assert rooms
    assert network.get("group") is not None

    infrastructure.connect_room(rooms[0], network)
    infrastructure.add_secondary_branches(rooms[0])

    assert len(scene.find_by_type("infra_cable_tray_room_link")) >= 1
    assert len(scene.find_by_type("infra_vertical_drop")) >= 1
    assert len(scene.find_by_type("infra_cable_tray_branch")) >= 1


def test_infrastructure_backbone_level_and_bounds_from_ceiling_settings() -> None:
    params = _build_params()
    params["machinery"]["infrastructure"] = {  # type: ignore[index]
        **params["machinery"]["infrastructure"],  # type: ignore[index]
        "enabled": True,
        "routing_level": "sub_ceiling",
        "ceiling_offset": 0.6,
        "top_clearance": 0.2,
        "tray_height": None,
        "parallel_channels": 2,
        "cross_connections": True,
    }

    generator = FactoryGenerator(params)
    layout = generator.generate_layout()
    min_room_height = min(generator._effective_room_height(spec) for spec in layout)  # noqa: SLF001
    scene = generator.generate_scene()

    trays = scene.find_by_type("infra_cable_tray_backbone")
    assert trays
    expected_z = min_room_height - (0.2 + 0.6) - 0.1 / 2.0
    assert all(abs(obj.transform.position[2] - expected_z) <= 1e-6 for obj in trays)

    room_roots = [obj for obj in scene.objects if obj.type.startswith("room_")]
    min_x = min(room.transform.position[0] - next(c for c in room.children if c.type == "floor").mesh.vertices[1][0] for room in room_roots if any(c.type == "floor" for c in room.children))  # type: ignore[union-attr]
    max_x = max(room.transform.position[0] + next(c for c in room.children if c.type == "floor").mesh.vertices[1][0] for room in room_roots if any(c.type == "floor" for c in room.children))  # type: ignore[union-attr]
    min_y = min(room.transform.position[1] - next(c for c in room.children if c.type == "floor").mesh.vertices[2][1] for room in room_roots if any(c.type == "floor" for c in room.children))  # type: ignore[union-attr]
    max_y = max(room.transform.position[1] + next(c for c in room.children if c.type == "floor").mesh.vertices[2][1] for room in room_roots if any(c.type == "floor" for c in room.children))  # type: ignore[union-attr]

    for tray in trays:
        x, y, _ = tray.transform.position
        assert min_x - 2.0 <= x <= max_x + 2.0
        assert min_y - 2.0 <= y <= max_y + 2.0


def _single_room_scene(room_type: str) -> tuple[Scene, SceneObject]:
    scene = Scene()
    room = SceneObject(
        id="room_1",
        type=room_type,
        transform=Transform(position=(0.0, 0.0, 0.0)),
    )
    room.add_child(
        SceneObject(
            id="room_1_floor",
            type="floor",
            transform=Transform(position=(0.0, 0.0, 0.0)),
            mesh=create_floor(width=12.0, depth=10.0),
        )
    )
    room.add_child(
        SceneObject(
            id="room_1_ceiling",
            type="ceiling",
            transform=Transform(position=(0.0, 0.0, 4.2)),
            mesh=create_floor(width=12.0, depth=10.0),
        )
    )
    scene.add_object(room)
    return scene, room


def test_laboratory_profile_connects_multiple_targets() -> None:
    scene, room = _single_room_scene("room_laboratory")
    room.add_child(
        SceneObject(
            id="bench_a",
            type="lab_bench",
            transform=Transform(position=(-2.2, 0.0, 0.9)),
            mesh=create_box(width=1.6, height=0.9, depth=0.8),
        )
    )
    room.add_child(
        SceneObject(
            id="bench_b",
            type="lab_bench",
            transform=Transform(position=(2.1, 0.6, 0.9)),
            mesh=create_box(width=1.5, height=0.9, depth=0.8),
        )
    )

    generator = InfrastructureGenerator({"enabled": True, "include_pipes": True})
    network = generator.generate_backbone(scene)
    generator.connect_room(room, network)

    assert len(scene.find_by_type("infra_vertical_drop")) >= 2


def test_maintenance_profile_adds_hanging_cables() -> None:
    scene, room = _single_room_scene("room_maintenance")
    generator = InfrastructureGenerator({"enabled": True, "include_pipes": True})
    network = generator.generate_backbone(scene)
    generator.connect_room(room, network)

    assert len(scene.find_by_type("infra_hanging_cable")) >= 1


def test_control_and_electrical_profiles_include_pipe_network() -> None:
    for room_type in ("room_control", "room_electrical"):
        scene, room = _single_room_scene(room_type)
        generator = InfrastructureGenerator({"enabled": True})
        network = generator.generate_backbone(scene)
        generator.connect_room(room, network)
        generator.add_secondary_branches(room)

        pipe_objects = [obj for obj in scene.traverse() if obj.type.startswith("infra_pipe_")]
        assert pipe_objects, f"Expected pipe infrastructure for {room_type}"


def test_infrastructure_density_scales_backbone_channels() -> None:
    low_params = _build_params()
    low_params["machinery"]["infrastructure"] = {  # type: ignore[index]
        **low_params["machinery"]["infrastructure"],  # type: ignore[index]
        "parallel_channels": 1,
        "cross_connections": False,
        "density": 0.5,
    }
    high_params = _build_params()
    high_params["machinery"]["infrastructure"] = {  # type: ignore[index]
        **high_params["machinery"]["infrastructure"],  # type: ignore[index]
        "parallel_channels": 1,
        "cross_connections": False,
        "density": 3.0,
    }

    low_scene = FactoryGenerator(low_params).generate_scene()
    high_scene = FactoryGenerator(high_params).generate_scene()

    low_count = len(low_scene.find_by_type("infra_cable_tray_backbone"))
    high_count = len(high_scene.find_by_type("infra_cable_tray_backbone"))
    assert low_count >= 1
    assert high_count > low_count


def test_branch_and_drop_frequency_scale_branches_and_drops() -> None:
    low_scene, low_room = _single_room_scene("room_laboratory")
    high_scene, high_room = _single_room_scene("room_laboratory")
    for scene_room in (low_room, high_room):
        scene_room.add_child(
            SceneObject(
                id=f"{scene_room.id}_bench_a",
                type="lab_bench",
                transform=Transform(position=(-2.2, 0.0, 0.9)),
                mesh=create_box(width=1.6, height=0.9, depth=0.8),
            )
        )
        scene_room.add_child(
            SceneObject(
                id=f"{scene_room.id}_bench_b",
                type="lab_bench",
                transform=Transform(position=(2.1, 0.6, 0.9)),
                mesh=create_box(width=1.5, height=0.9, depth=0.8),
            )
        )

    low_generator = InfrastructureGenerator(
        {
            "enabled": True,
            "branch_frequency": 0.5,
            "vertical": {"drop_frequency": 0.2},
            "secondary_branch_count": 2,
        }
    )
    high_generator = InfrastructureGenerator(
        {
            "enabled": True,
            "branch_frequency": 2.0,
            "vertical": {"drop_frequency": 1.5},
            "secondary_branch_count": 2,
        }
    )

    low_network = low_generator.generate_backbone(low_scene)
    high_network = high_generator.generate_backbone(high_scene)
    low_generator.connect_room(low_room, low_network)
    high_generator.connect_room(high_room, high_network)
    low_generator.add_secondary_branches(low_room)
    high_generator.add_secondary_branches(high_room)

    low_drops = len(low_scene.find_by_type("infra_vertical_drop"))
    high_drops = len(high_scene.find_by_type("infra_vertical_drop"))
    low_branches = len(low_scene.find_by_type("infra_cable_tray_branch"))
    high_branches = len(high_scene.find_by_type("infra_cable_tray_branch"))

    assert high_drops > low_drops
    assert high_branches > low_branches


def test_connectivity_rule_auto_fixes_room_connection() -> None:
    scene, room = _single_room_scene("room_workshop")
    generator = InfrastructureGenerator(
        {
            "enabled": True,
            "rules": {"auto_fix": True, "require_vertical_drop": True},
        }
    )
    network = generator.generate_backbone(scene)
    group = network.get("group")
    assert group is not None
    assert isinstance(group, SceneObject)

    group.children = [
        child
        for child in group.children
        if not child.type.startswith("infra_cable_tray_room_link")
        and not child.type.startswith("infra_vertical_drop")
        and child.type != "infra_cable_tray_entry"
    ]

    generator._apply_infrastructure_rules(scene, network, [room])  # noqa: SLF001

    assert len(scene.find_by_type("infra_vertical_drop")) >= 1
    assert len(scene.find_by_type("infra_cable_tray_room_link")) >= 1


def test_support_rule_adds_supports_for_long_segments() -> None:
    scene, room = _single_room_scene("room_workshop")
    generator = InfrastructureGenerator(
        {
            "enabled": True,
            "support_spacing": 12.0,
            "rules": {
                "auto_fix": True,
                "support_spacing": 1.2,
                "long_segment_length": 1.0,
            },
        }
    )
    network = generator.generate_backbone(scene)
    generator.connect_room(room, network)
    before = len(scene.find_by_type("infra_support"))

    generator._apply_infrastructure_rules(scene, network, [room])  # noqa: SLF001
    after = len(scene.find_by_type("infra_support"))

    assert after > before


def test_object_connections_create_drop_and_cable_or_pipe() -> None:
    scene, room = _single_room_scene("room_mixed")
    desk = SceneObject(
        id="desk_1",
        type="desk",
        transform=Transform(position=(-2.6, -1.2, 0.38)),
        mesh=create_box(width=1.6, height=0.76, depth=0.8),
    )
    cabinet = SceneObject(
        id="cabinet_1",
        type="electrical_cabinet",
        transform=Transform(position=(2.3, -0.8, 1.1)),
        mesh=create_box(width=0.9, height=2.2, depth=0.7),
    )
    boiler = SceneObject(
        id="boiler_1",
        type="boiler_unit",
        transform=Transform(position=(0.2, 2.0, 1.8)),
        mesh=create_box(width=1.8, height=3.6, depth=1.8),
    )
    room.add_child(desk)
    room.add_child(cabinet)
    room.add_child(boiler)

    generator = InfrastructureGenerator(
        {
            "enabled": True,
            "include_pipes": True,
            "object_connections": {"enabled": True},
        }
    )
    network = generator.generate_backbone(scene)
    generator.connect_room(room, network)
    generator.connect_objects(room, network)

    drops = scene.find_by_type("infra_vertical_drop_object_link")
    cables = scene.find_by_type("infra_cable_object_link")
    pipes = scene.find_by_type("infra_pipe_object_link")

    assert len(drops) >= 3
    assert len(cables) >= 2
    assert len(pipes) >= 1

    drop_points = [(obj.transform.position[0], obj.transform.position[1]) for obj in drops]
    targets = [desk, cabinet, boiler]
    for target in targets:
        tx, ty, _ = target.transform.position
        assert any((px - tx) ** 2 + (py - ty) ** 2 <= 0.45**2 for px, py in drop_points)
