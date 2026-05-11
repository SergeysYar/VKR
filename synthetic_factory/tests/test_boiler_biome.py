from __future__ import annotations

from synthetic_factory.generators import FactoryGenerator


def test_boiler_biome_uses_vertical_connected_multilevel_layout() -> None:
    params = {
        "factory_width": 10.0,
        "factory_depth": 8.0,
        "number_of_rooms": 1,
        "room_size_range": (6.0, 8.0),
        "corridor_width": 1.0,
        "layout_strategy": "grid",
        "noise": {"seed": 42},
        "biomes": {
            "enabled": True,
            "workshop_biome": "boiler",
            "cycle_order": ["boiler"],
        },
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {
            "enabled": True,
            "boiler": {
                "random_variation": False,
                "min_walkway": 1.2,
                "service_clearance": 0.8,
                "boiler": {
                    "count": [3, 3],
                    "height": [8.0, 8.0],
                    "radius": [1.0, 1.0],
                },
                "pipes": {
                    "density": [1.5, 1.5],
                    "radius": [0.18, 0.18],
                },
                "platforms": {
                    "levels": [2, 2],
                    "width_factor": [1.5, 1.5],
                },
                "tanks": {
                    "count": [2, 2],
                },
                "structure": {
                    "beam_density": [0.8, 0.8],
                },
            },
        },
        "seed": 42,
    }

    generator = FactoryGenerator(params)
    layout = generator.generate_layout()
    room_spec = layout[0]
    room_height = generator._effective_room_height(room_spec)  # noqa: SLF001

    assert room_spec.biome == "boiler"
    assert room_height >= 9.6
    assert room_height > room_spec.width

    scene = generator.generate_scene()
    object_types = {obj.type for obj in scene.traverse()}

    assert "boiler_unit" in object_types
    assert "riser_pipe" in object_types
    assert "pipe_manifold_main" in object_types
    assert "pipe_branch" in object_types
    assert "pipe_valve" in object_types
    assert "pump" in object_types
    assert "valve_cluster" in object_types
    assert "bunker" in object_types
    assert "heat_exchanger" in object_types
    assert "chimney" in object_types
    assert "pipe_support" in object_types
    assert "control_box" in object_types
    assert "service_platform" in object_types
    assert "ladder" in object_types
    assert "tank" in object_types
    assert "support_beam" in object_types
    assert "service_zone" in object_types

    boiler_count = len(scene.find_by_type("boiler_unit"))
    assert 1 <= boiler_count <= 3

    branch_count = len(scene.find_by_type("pipe_branch"))
    assert branch_count >= 1
    assert branch_count <= max(1, int(round(boiler_count * 2.0)))
