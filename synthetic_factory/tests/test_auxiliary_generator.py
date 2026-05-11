from __future__ import annotations

from synthetic_factory.generators import AuxiliaryGenerator, FactoryGenerator
from synthetic_factory.parametric.primitives import create_box
from synthetic_factory.scene import AnchorPoint, Scene, SceneObject, Transform


def _local_aabb(obj: SceneObject) -> tuple[float, float, float, float, float, float]:
    assert obj.mesh is not None
    xs = [vertex[0] + obj.transform.position[0] for vertex in obj.mesh.vertices]
    ys = [vertex[1] + obj.transform.position[1] for vertex in obj.mesh.vertices]
    zs = [vertex[2] + obj.transform.position[2] for vertex in obj.mesh.vertices]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def test_scene_object_generates_default_anchor_points_from_mesh() -> None:
    obj = SceneObject(
        id="mesh_obj",
        type="test_object",
        mesh=create_box(width=2.0, height=1.2, depth=0.8),
    )

    assert obj.anchor_points
    anchor_types = {anchor.type for anchor in obj.anchor_points}
    assert anchor_types == {"top", "side", "bottom"}
    assert len(obj.anchors_by_type("side")) >= 4


def test_scene_object_keeps_custom_anchor_points() -> None:
    custom_anchor = AnchorPoint(position=(0.0, 0.0, 1.0), type="top", name="custom_top")
    obj = SceneObject(
        id="custom_anchor_obj",
        type="test_object",
        mesh=create_box(width=1.0, height=1.0, depth=1.0),
        anchor_points=[custom_anchor],
    )

    assert len(obj.anchor_points) == 1
    assert obj.anchor_points[0].name == "custom_top"


def test_auxiliary_generator_attaches_children_to_primary_objects() -> None:
    scene = Scene()
    room = SceneObject(id="room_1", type="room")
    bench = SceneObject(
        id="bench_1",
        type="lab_bench",
        transform=Transform(position=(0.0, 0.0, 0.0)),
        mesh=create_box(width=2.0, height=1.0, depth=1.0),
    )
    room.add_child(bench)
    scene.add_object(room)

    generator = AuxiliaryGenerator(
        {
            "density": 1.0,
            "random_variation": False,
            "max_children_per_object": 3,
            "margin": 0.01,
        }
    )
    generator.generate(scene, "laboratory")

    assert bench.children
    assert all(child.parent is bench for child in bench.children)
    assert all(child.type.startswith("aux_") for child in bench.children)

    parent_box = _local_aabb(
        SceneObject(
            id="bench_box",
            type="tmp",
            transform=Transform(),
            mesh=bench.mesh,
        )
    )
    for child in bench.children:
        child_box = _local_aabb(child)
        assert child_box[0] >= parent_box[0] - 1e-6
        assert child_box[1] <= parent_box[1] + 1e-6
        assert child_box[2] >= parent_box[2] - 1e-6
        assert child_box[3] <= parent_box[3] + 1e-6


def test_auxiliary_generator_supports_multiple_biomes() -> None:
    scene = Scene()
    room = SceneObject(id="room_1", type="room")
    console = SceneObject(id="console_1", type="console", mesh=create_box(width=1.4, height=0.8, depth=0.7))
    cabinet = SceneObject(id="cab_1", type="electrical_cabinet", mesh=create_box(width=1.0, height=2.0, depth=0.7))
    boiler = SceneObject(id="boiler_1", type="boiler_unit", mesh=create_box(width=1.8, height=4.0, depth=1.8))
    room.add_child(console)
    room.add_child(cabinet)
    room.add_child(boiler)
    scene.add_object(room)

    generator = AuxiliaryGenerator({"density": 1.0, "random_variation": False})
    generator.generate(scene, "control")
    generator.generate(scene, "electrical")
    generator.generate(scene, "boiler")

    assert any(child.type.startswith("aux_") for child in console.children)
    assert any(child.type.startswith("aux_") for child in cabinet.children)
    assert any(child.type.startswith("aux_") for child in boiler.children)


def test_auxiliary_generator_anchor_attachment_for_boiler_examples() -> None:
    scene = Scene()
    room = SceneObject(id="room_anchor", type="room")
    boiler = SceneObject(
        id="boiler_anchor",
        type="boiler_unit",
        mesh=create_box(width=2.0, height=4.0, depth=2.0),
    )
    pipe = SceneObject(
        id="pipe_anchor",
        type="pipe_segment",
        mesh=create_box(width=3.0, height=0.3, depth=0.3),
    )
    room.add_child(boiler)
    room.add_child(pipe)
    scene.add_object(room)

    generator = AuxiliaryGenerator(
        {
            "density": 1.0,
            "random_variation": False,
            "max_children_per_object": 5,
            "margin": 0.01,
            "boiler": {
                "bunker_probability": 1.0,
                "pump_count": 1,
            },
        }
    )
    generator.generate(scene, "boiler")

    pump = next((child for child in boiler.children if child.type == "aux_pump"), None)
    bunker = next((child for child in boiler.children if child.type == "aux_bunker"), None)
    valve = next((child for child in pipe.children if child.type == "aux_valve"), None)

    assert pump is not None
    assert bunker is not None
    assert valve is not None

    boiler_min_z = min(vertex[2] for vertex in boiler.mesh.vertices)  # type: ignore[union-attr]
    assert pump.transform.position[2] < boiler_min_z

    boiler_box = _local_aabb(SceneObject(id="boiler_box", type="tmp", mesh=boiler.mesh))
    bunker_box = _local_aabb(bunker)
    assert bunker_box[0] < boiler_box[0] or bunker_box[1] > boiler_box[1] or bunker_box[2] < boiler_box[2] or bunker_box[3] > boiler_box[3]

    pipe_box = _local_aabb(SceneObject(id="pipe_box", type="tmp", mesh=pipe.mesh))
    valve_box = _local_aabb(valve)
    assert valve_box[0] < pipe_box[0] or valve_box[1] > pipe_box[1] or valve_box[2] < pipe_box[2] or valve_box[3] > pipe_box[3]


def test_auxiliary_density_and_complexity_scale_quantity() -> None:
    low_scene = Scene()
    low_room = SceneObject(id="room_low", type="room")
    low_bench = SceneObject(id="bench_low", type="lab_bench", mesh=create_box(width=2.0, height=1.0, depth=1.0))
    low_room.add_child(low_bench)
    low_scene.add_object(low_room)

    high_scene = Scene()
    high_room = SceneObject(id="room_high", type="room")
    high_bench = SceneObject(id="bench_high", type="lab_bench", mesh=create_box(width=2.0, height=1.0, depth=1.0))
    high_room.add_child(high_bench)
    high_scene.add_object(high_room)

    AuxiliaryGenerator(
        {
            "density": 0.5,
            "complexity": 1,
            "random_variation": False,
            "max_children_per_object": 2,
            "laboratory": {"clutter_level": 0.5},
        }
    ).generate(low_scene, "laboratory")

    AuxiliaryGenerator(
        {
            "density": 3.0,
            "complexity": 5,
            "random_variation": False,
            "max_children_per_object": 2,
            "laboratory": {"clutter_level": 2.0},
        }
    ).generate(high_scene, "laboratory")

    assert len(high_bench.children) > len(low_bench.children)


def test_auxiliary_boiler_parameters_control_bunker_and_pumps() -> None:
    scene_no_bunker = Scene()
    room_no_bunker = SceneObject(id="room_no_bunker", type="room")
    boiler_no_bunker = SceneObject(
        id="boiler_no_bunker",
        type="boiler_unit",
        mesh=create_box(width=2.0, height=4.0, depth=2.0),
    )
    room_no_bunker.add_child(boiler_no_bunker)
    scene_no_bunker.add_object(room_no_bunker)

    AuxiliaryGenerator(
        {
            "density": 2.0,
            "complexity": 3,
            "random_variation": False,
            "max_children_per_object": 10,
            "boiler": {
                "bunker_probability": 0.0,
                "pump_count": 4,
            },
        }
    ).generate(scene_no_bunker, "boiler")

    no_bunker_children = boiler_no_bunker.children
    assert not any(child.type == "aux_bunker" for child in no_bunker_children)
    assert len([child for child in no_bunker_children if child.type == "aux_pump"]) >= 2

    scene_with_bunker = Scene()
    room_with_bunker = SceneObject(id="room_with_bunker", type="room")
    boiler_with_bunker = SceneObject(
        id="boiler_with_bunker",
        type="boiler_unit",
        mesh=create_box(width=2.0, height=4.0, depth=2.0),
    )
    room_with_bunker.add_child(boiler_with_bunker)
    scene_with_bunker.add_object(room_with_bunker)

    AuxiliaryGenerator(
        {
            "density": 2.0,
            "complexity": 3,
            "random_variation": False,
            "max_children_per_object": 10,
            "boiler": {
                "bunker_probability": 1.0,
                "pump_count": 1,
            },
        }
    ).generate(scene_with_bunker, "boiler")

    assert any(child.type == "aux_bunker" for child in boiler_with_bunker.children)


def test_auxiliary_electrical_transformer_probability_controls_generation() -> None:
    low_scene = Scene()
    low_room = SceneObject(id="room_e_low", type="room")
    low_cabinet = SceneObject(id="cab_low", type="electrical_cabinet", mesh=create_box(width=1.0, height=2.0, depth=0.7))
    low_room.add_child(low_cabinet)
    low_scene.add_object(low_room)

    high_scene = Scene()
    high_room = SceneObject(id="room_e_high", type="room")
    high_cabinet = SceneObject(id="cab_high", type="electrical_cabinet", mesh=create_box(width=1.0, height=2.0, depth=0.7))
    high_room.add_child(high_cabinet)
    high_scene.add_object(high_room)

    AuxiliaryGenerator(
        {
            "density": 2.0,
            "complexity": 2,
            "random_variation": False,
            "electrical": {"transformer_probability": 0.0},
        }
    ).generate(low_scene, "electrical")

    AuxiliaryGenerator(
        {
            "density": 2.0,
            "complexity": 2,
            "random_variation": False,
            "electrical": {"transformer_probability": 1.0},
        }
    ).generate(high_scene, "electrical")

    assert not any(child.type == "aux_transformer" for child in low_cabinet.children)
    assert any(child.type == "aux_transformer" for child in high_cabinet.children)


def test_auxiliary_laboratory_clutter_level_controls_clutter_objects() -> None:
    low_scene = Scene()
    low_room = SceneObject(id="room_l_low", type="room")
    low_bench = SceneObject(id="bench_l_low", type="lab_bench", mesh=create_box(width=2.0, height=1.0, depth=1.0))
    low_room.add_child(low_bench)
    low_scene.add_object(low_room)

    high_scene = Scene()
    high_room = SceneObject(id="room_l_high", type="room")
    high_bench = SceneObject(id="bench_l_high", type="lab_bench", mesh=create_box(width=2.0, height=1.0, depth=1.0))
    high_room.add_child(high_bench)
    high_scene.add_object(high_room)

    AuxiliaryGenerator(
        {
            "density": 3.0,
            "complexity": 1,
            "random_variation": False,
            "laboratory": {"clutter_level": 0.5},
        }
    ).generate(low_scene, "laboratory")

    AuxiliaryGenerator(
        {
            "density": 3.0,
            "complexity": 1,
            "random_variation": False,
            "laboratory": {"clutter_level": 2.0},
        }
    ).generate(high_scene, "laboratory")

    low_clutter = [child for child in low_bench.children if child.type == "aux_lab_clutter"]
    high_clutter = [child for child in high_bench.children if child.type == "aux_lab_clutter"]
    assert len(high_clutter) > len(low_clutter)


def test_factory_generator_applies_auxiliary_details_when_enabled() -> None:
    params = {
        "factory_width": 28.0,
        "factory_depth": 22.0,
        "number_of_rooms": 1,
        "room_size_range": (14.0, 14.0),
        "corridor_width": 1.0,
        "layout_strategy": "grid",
        "noise": {"seed": 42},
        "biomes": {
            "enabled": True,
            "workshop_biome": "laboratory",
            "cycle_order": ["laboratory"],
        },
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {
            "enabled": True,
            "auxiliary": {
                "enabled": True,
                "density": 1.0,
                "random_variation": False,
            },
            "laboratory": {
                "seed": 13,
                "random_variation": False,
                "stations": {"count": [4, 4], "spacing": [1.8, 1.8], "type_variability": [0.2, 0.2]},
                "equipment": {"density": [2.0, 2.0]},
            },
        },
        "seed": 42,
    }

    scene = FactoryGenerator(params).generate_scene()
    aux_objects = [obj for obj in scene.traverse() if obj.type.startswith("aux_")]
    assert aux_objects
    assert all(obj.parent is not None for obj in aux_objects)
    assert all(not obj.parent.type.startswith("aux_") for obj in aux_objects if obj.parent is not None)
