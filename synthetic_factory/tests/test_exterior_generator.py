from __future__ import annotations

from synthetic_factory.generators import FactoryGenerator


def _base_params() -> dict[str, object]:
    return {
        "factory_width": 60.0,
        "factory_depth": 40.0,
        "number_of_rooms": 1,
        "room_size_range": (20.0, 20.0),
        "corridor_width": 2.0,
        "room_height": 6.0,
        "layout_strategy": "grid",
        "noise": {"seed": 42},
        "biomes": {"enabled": False},
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {"enabled": False},
        "seed": 42,
    }


def _object_world_aabb(obj) -> tuple[float, float, float, float, float, float] | None:
    if obj.mesh is None or not obj.mesh.vertices:
        return None
    px, py, pz = obj.transform.position
    xs = [vertex[0] + px for vertex in obj.mesh.vertices]
    ys = [vertex[1] + py for vertex in obj.mesh.vertices]
    zs = [vertex[2] + pz for vertex in obj.mesh.vertices]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def _overlap_3d(
    a: tuple[float, float, float, float, float, float],
    b: tuple[float, float, float, float, float, float],
    padding: float = 0.0,
) -> bool:
    return (
        a[1] + padding > b[0]
        and b[1] + padding > a[0]
        and a[3] + padding > b[2]
        and b[3] + padding > a[2]
        and a[5] + padding > b[4]
        and b[5] + padding > a[4]
    )


def test_exterior_group_is_generated_from_internal_dimensions() -> None:
    params = _base_params()
    params["exterior"] = {
        "enabled": True,
        "group_id": "exterior",
        "wall_thickness": 0.5,
        "wall_offset": 0.2,
        "wall_headroom": 0.3,
        "roof_thickness": 0.4,
        "roof_overhang": 0.6,
        "apron_width": 1.8,
        "roof_units": {"enabled": False},
    }

    scene = FactoryGenerator(params).generate_scene()

    exterior = scene.get_object("exterior")
    assert exterior is not None
    assert exterior.type == "exterior_group"

    walls = [child for child in exterior.children if child.type == "exterior_wall"]
    assert len(walls) >= 4

    south_wall = scene.get_object("exterior_wall_south")
    north_wall = scene.get_object("exterior_wall_north")
    roof = scene.get_object("exterior_roof")
    assert south_wall is not None
    assert north_wall is not None
    assert roof is not None

    expected_half_depth = params["factory_depth"] / 2.0 + 0.2 + 0.5 / 2.0  # type: ignore[operator]
    assert abs(south_wall.transform.position[1] + expected_half_depth) <= 1e-6
    assert abs(north_wall.transform.position[1] - expected_half_depth) <= 1e-6
    assert roof.transform.position[2] > south_wall.transform.position[2]


def test_exterior_respects_enabled_toggle() -> None:
    params = _base_params()
    params["exterior"] = {"enabled": False}

    scene = FactoryGenerator(params).generate_scene()

    assert scene.get_object("exterior") is None
    assert scene.get_object("exterior_wall_south") is None
    assert scene.get_object("exterior_roof") is None


def test_exterior_wall_height_tracks_room_height() -> None:
    params = _base_params()
    params["room_height"] = 8.0
    params["exterior"] = {
        "enabled": True,
        "wall_headroom": 0.6,
        "roof_units": {"enabled": False},
        "gates": {"enabled": False},
    }

    generator = FactoryGenerator(params)
    scene = generator.generate_scene()

    south_wall = scene.get_object("exterior_wall_south")
    assert south_wall is not None
    expected_wall_height = 8.0 + 0.6
    assert abs(south_wall.transform.position[2] * 2.0 - expected_wall_height) <= 1e-6


def test_exterior_shell_contains_openings_and_wall_cutouts() -> None:
    params = _base_params()
    params["exterior"] = {
        "enabled": True,
        "gates": {
            "enabled": True,
            "side": "south",
            "count": 1,
            "width": 4.0,
            "height": 4.0,
            "edge_margin": 1.5,
            "gap": 1.2,
        },
        "roof_units": {"enabled": False},
    }

    scene = FactoryGenerator(params).generate_scene()
    opening_objects = scene.find_by_type("exterior_opening")

    assert opening_objects
    assert any("gate" in obj.id for obj in opening_objects)

    south_wall = scene.get_object("exterior_wall_south")
    assert south_wall is not None
    assert south_wall.mesh is not None
    assert len(south_wall.mesh.vertices) > 8


def test_facade_is_split_into_panels_and_sections() -> None:
    params = _base_params()
    params["exterior"] = {
        "enabled": True,
        "roof_units": {"enabled": False},
        "facade": {
            "panel_width": 2.0,
            "section_every": 2,
        },
        "gates": {
            "enabled": True,
            "side": "south",
            "count": 1,
        },
    }

    scene = FactoryGenerator(params).generate_scene()
    panels = scene.find_by_type("exterior_facade_panel")
    sections = scene.find_by_type("exterior_facade_section")

    assert panels
    assert sections


def test_gabled_roof_type_is_supported() -> None:
    params = _base_params()
    params["exterior"] = {
        "enabled": True,
        "roof": {
            "type": "gabled",
            "ridge_axis": "x",
            "gabled_rise": 2.2,
        },
        "gates": {"enabled": False},
    }

    scene = FactoryGenerator(params).generate_scene()
    roof = scene.get_object("exterior_roof")
    assert roof is not None
    assert roof.mesh is not None
    assert len(roof.mesh.vertices) >= 12


def test_sawtooth_roof_type_is_supported() -> None:
    params = _base_params()
    params["exterior"] = {
        "enabled": True,
        "roof": {
            "type": "sawtooth",
            "sawtooth_axis": "x",
            "tooth_count": 4,
            "sawtooth_rise": 1.6,
        },
        "gates": {"enabled": False},
    }

    scene = FactoryGenerator(params).generate_scene()
    roof = scene.get_object("exterior_roof")
    assert roof is not None
    assert roof.mesh is not None
    assert len(roof.mesh.vertices) >= 16


def test_roof_equipment_adds_ventilation_and_boiler_exits() -> None:
    params = _base_params()
    params.update(
        {
            "number_of_rooms": 2,
            "room_size_range": (16.0, 16.0),
            "biomes": {
                "enabled": True,
                "ensure_all_types": True,
                "cycle_order": ["workshop", "boiler"],
                "workshop_area_ratio": 0.5,
            },
        }
    )
    params["exterior"] = {
        "enabled": True,
        "gates": {"enabled": False},
        "roof_units": {"enabled": False},
        "ventilation": {"enabled": True, "count": 2},
        "roof_pipes": {"enabled": True, "per_boiler_room": 1},
    }

    scene = FactoryGenerator(params).generate_scene()
    vents = scene.find_by_type("exterior_vent_shaft")
    pipes = scene.find_by_type("exterior_roof_pipe")

    assert len(vents) >= 2
    assert pipes


def test_storage_biome_adds_large_gate_even_without_global_gate_setting() -> None:
    params = _base_params()
    params["biomes"] = {
        "enabled": True,
        "workshop_biome": "storage",
        "cycle_order": ["storage"],
    }
    params["exterior"] = {
        "enabled": True,
        "gates": {"enabled": False},
        "roof_units": {"enabled": False},
    }

    scene = FactoryGenerator(params).generate_scene()
    gates = scene.find_by_type("exterior_gate")

    assert gates


def test_control_has_more_windows_than_boiler() -> None:
    control_params = _base_params()
    control_params["biomes"] = {
        "enabled": True,
        "workshop_biome": "control",
        "cycle_order": ["control"],
    }
    control_params["exterior"] = {
        "enabled": True,
        "gates": {"enabled": False},
        "roof_units": {"enabled": False},
    }

    boiler_params = _base_params()
    boiler_params["biomes"] = {
        "enabled": True,
        "workshop_biome": "boiler",
        "cycle_order": ["boiler"],
    }
    boiler_params["exterior"] = {
        "enabled": True,
        "gates": {"enabled": False},
        "roof_units": {"enabled": False},
    }

    control_scene = FactoryGenerator(control_params).generate_scene()
    boiler_scene = FactoryGenerator(boiler_params).generate_scene()

    control_windows = len(control_scene.find_by_type("exterior_window"))
    boiler_windows = len(boiler_scene.find_by_type("exterior_window"))

    assert control_windows > boiler_windows


def _single_biome_scene(biome: str, exterior_override: dict[str, object] | None = None):
    params = _base_params()
    params["biomes"] = {
        "enabled": True,
        "workshop_biome": biome,
        "cycle_order": [biome],
    }
    exterior = {
        "enabled": True,
        "gates": {"enabled": False},
        "roof_units": {"enabled": False},
    }
    if exterior_override:
        exterior.update(exterior_override)
    params["exterior"] = exterior
    return FactoryGenerator(params).generate_scene()


def test_boiler_biome_links_generate_external_pipes_and_chimneys() -> None:
    scene = _single_biome_scene("boiler")
    assert scene.find_by_type("exterior_boiler_pipe")
    assert scene.find_by_type("exterior_chimney")


def test_electrical_biome_links_generate_transformers_and_power_cables() -> None:
    scene = _single_biome_scene("electrical")
    assert scene.find_by_type("exterior_transformer")
    assert scene.find_by_type("exterior_power_cable")


def test_maintenance_biome_links_generate_gates_and_repair_pads() -> None:
    scene = _single_biome_scene("maintenance")
    assert scene.find_by_type("exterior_gate")
    assert scene.find_by_type("exterior_repair_pad")


def test_laboratory_biome_links_generate_vents_and_technical_blocks() -> None:
    scene = _single_biome_scene(
        "laboratory",
        exterior_override={
            "ventilation": {"enabled": False},
        },
    )
    assert scene.find_by_type("exterior_lab_vent")
    assert scene.find_by_type("exterior_lab_technical_block")


def test_control_biome_keeps_exterior_clean_except_windows() -> None:
    scene = _single_biome_scene("control")
    assert scene.find_by_type("exterior_window")
    assert not scene.find_by_type("exterior_chimney")
    assert not scene.find_by_type("exterior_transformer")
    assert not scene.find_by_type("exterior_repair_pad")


def test_building_margin_and_height_variation_affect_shell_geometry() -> None:
    params = _base_params()
    params["room_height"] = 6.0
    params["exterior"] = {
        "enabled": True,
        "building": {
            "margin": 4.0,
            "height_variation": 0.0,
        },
        "wall_thickness": 0.5,
        "wall_headroom": 0.4,
        "gates": {"enabled": False},
        "roof_units": {"enabled": False},
    }

    base_scene = FactoryGenerator(params).generate_scene()
    base_wall = base_scene.get_object("exterior_wall_south")
    assert base_wall is not None

    params["exterior"]["building"]["height_variation"] = 1.5  # type: ignore[index]
    scene = FactoryGenerator(params).generate_scene()
    south_wall = scene.get_object("exterior_wall_south")
    assert south_wall is not None

    expected_half_depth = params["factory_depth"] / 2.0 + 4.0 + 0.5 / 2.0  # type: ignore[operator]
    assert abs(abs(south_wall.transform.position[1]) - expected_half_depth) <= 1e-6
    added_height = south_wall.transform.position[2] * 2.0 - base_wall.transform.position[2] * 2.0
    assert abs(added_height - 1.5) <= 1e-6


def test_roof_type_supports_option_list() -> None:
    params = _base_params()
    params["exterior"] = {
        "enabled": True,
        "seed": 3,
        "roof": {
            "type": ["flat", "sawtooth", "gabled"],
        },
        "gates": {"enabled": False},
        "roof_units": {"enabled": False},
    }
    scene = FactoryGenerator(params).generate_scene()
    roof = scene.get_object("exterior_roof")
    assert roof is not None
    assert roof.mesh is not None


def test_window_density_controls_facade_windows() -> None:
    low = _base_params()
    low["biomes"] = {
        "enabled": True,
        "workshop_biome": "control",
        "cycle_order": ["control"],
    }
    low["exterior"] = {
        "enabled": True,
        "gates": {"enabled": False},
        "roof_units": {"enabled": False},
        "facade": {"window_density": 0.0},
    }

    high = _base_params()
    high["biomes"] = {
        "enabled": True,
        "workshop_biome": "control",
        "cycle_order": ["control"],
    }
    high["exterior"] = {
        "enabled": True,
        "gates": {"enabled": False},
        "roof_units": {"enabled": False},
        "facade": {"window_density": 0.5},
    }

    low_scene = FactoryGenerator(low).generate_scene()
    high_scene = FactoryGenerator(high).generate_scene()

    assert len(low_scene.find_by_type("exterior_window")) == 0
    assert len(high_scene.find_by_type("exterior_window")) > 0


def test_equipment_density_increases_external_equipment_count() -> None:
    low = _base_params()
    low["exterior"] = {
        "enabled": True,
        "equipment_density": 0.5,
        "gates": {"enabled": False},
        "roof_units": {"enabled": True, "count": 4},
        "ventilation": {"enabled": True, "count": 4},
        "roof_pipes": {"enabled": False},
    }

    high = _base_params()
    high["exterior"] = {
        "enabled": True,
        "equipment_density": 2.0,
        "gates": {"enabled": False},
        "roof_units": {"enabled": True, "count": 4},
        "ventilation": {"enabled": True, "count": 4},
        "roof_pipes": {"enabled": False},
    }

    low_scene = FactoryGenerator(low).generate_scene()
    high_scene = FactoryGenerator(high).generate_scene()

    low_count = len(low_scene.find_by_type("exterior_technical_block")) + len(
        low_scene.find_by_type("exterior_vent_shaft")
    )
    high_count = len(high_scene.find_by_type("exterior_technical_block")) + len(
        high_scene.find_by_type("exterior_vent_shaft")
    )
    assert high_count > low_count


def test_shell_containment_rule_expands_shell_when_margin_is_high() -> None:
    params = _base_params()
    params["biomes"] = {
        "enabled": True,
        "workshop_biome": "workshop",
        "cycle_order": ["workshop"],
    }
    params["exterior"] = {
        "enabled": True,
        "wall_thickness": 0.4,
        "building": {
            "margin": 0.1,
        },
        "rules": {
            "enabled": True,
            "auto_fix": True,
            "shell_margin": 0.0,
        },
        "gates": {"enabled": False},
        "roof_units": {"enabled": False},
    }
    base_scene = FactoryGenerator(params).generate_scene()
    base_wall = base_scene.get_object("exterior_wall_south")
    assert base_wall is not None

    params["exterior"]["rules"]["shell_margin"] = 12.0  # type: ignore[index]
    scene = FactoryGenerator(params).generate_scene()
    south_wall = scene.get_object("exterior_wall_south")
    assert south_wall is not None
    assert abs(south_wall.transform.position[1]) > abs(base_wall.transform.position[1]) + 3.0


def test_access_rule_autofix_adds_entrance_if_all_openings_disabled() -> None:
    params = _base_params()
    params["exterior"] = {
        "enabled": True,
        "gates": {"enabled": False},
        "facade": {"functional": {"enabled": False}},
        "rules": {
            "enabled": True,
            "auto_fix": True,
            "min_entrances": 1,
        },
    }
    scene = FactoryGenerator(params).generate_scene()
    assert scene.find_by_type("exterior_door") or scene.find_by_type("exterior_gate")


def test_pipe_continuity_rule_autofix_builds_pipe_links() -> None:
    scene = _single_biome_scene(
        "boiler",
        exterior_override={
            "rules": {
                "enabled": True,
                "auto_fix": True,
                "pipe_tolerance": 0.05,
            }
        },
    )
    assert scene.find_by_type("exterior_pipe_link")


def test_ground_contact_rule_keeps_transformers_on_ground() -> None:
    scene = _single_biome_scene("electrical")
    transformers = scene.find_by_type("exterior_transformer")
    assert transformers
    for transformer in transformers:
        bounds = _object_world_aabb(transformer)
        assert bounds is not None
        assert abs(bounds[4]) <= 1e-6


def test_no_collision_rule_resolves_equipment_overlaps() -> None:
    params = _base_params()
    params["exterior"] = {
        "enabled": True,
        "equipment_density": 2.0,
        "roof_units": {"enabled": True, "count": 12, "width": 2.8, "depth": 2.2},
        "ventilation": {"enabled": True, "count": 10, "shaft_width": 1.4, "shaft_depth": 1.4},
        "gates": {"enabled": False},
        "rules": {
            "enabled": True,
            "auto_fix": True,
            "collision_padding": 0.02,
            "collision_step": 0.5,
        },
    }
    scene = FactoryGenerator(params).generate_scene()
    equipment = scene.find_by_type("exterior_technical_block") + scene.find_by_type("exterior_vent_shaft")
    bounds = [b for b in (_object_world_aabb(obj) for obj in equipment) if b is not None]
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            assert not _overlap_3d(bounds[i], bounds[j], padding=0.02)
