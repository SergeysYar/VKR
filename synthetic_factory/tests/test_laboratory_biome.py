from __future__ import annotations

from copy import deepcopy

from synthetic_factory.generators import FactoryGenerator


def _build_params() -> dict[str, object]:
    return {
        "factory_width": 34.0,
        "factory_depth": 26.0,
        "number_of_rooms": 1,
        "room_size_range": (18.0, 18.0),
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
            "laboratory": {
                "seed": 17,
                "random_variation": False,
                "room": {"width": [8.0, 30.0], "depth": [8.0, 30.0], "height": [2.5, 4.5]},
                "stations": {"count": [2, 12], "spacing": [1.5, 3.0], "type_variability": [0.2, 0.6]},
                "equipment": {"density": [2.0, 2.0]},
                "fumehood": {"probability": [0.25, 0.25]},
                "storage": {"shelves": [3, 3], "cabinets": [2, 2]},
                "infrastructure": {"cable_density": [1.0, 1.0], "pipe_density": [0.6, 0.6]},
                "lighting": {"intensity": [1.2, 1.2]},
                "module_width": [2.2, 2.2],
                "module_depth": [1.6, 1.6],
                "walkway_width": 1.3,
                "perimeter_walkway": 0.9,
                "access_depth": 0.85,
                "equipment_per_station": [2, 2],
                "sink_probability": 0.4,
            },
        },
        "seed": 42,
    }


def test_laboratory_generates_modular_stations_with_access() -> None:
    params = _build_params()
    scene = FactoryGenerator(params).generate_scene()

    modules = scene.find_by_type("lab_station_module")
    benches = scene.find_by_type("lab_bench")
    equipment = scene.find_by_type("lab_equipment_unit")
    access = scene.find_by_type("lab_access_zone")
    walkways = scene.find_by_type("lab_walkway")
    perimeter = scene.find_by_type("lab_perimeter_walkway")
    shelves = scene.find_by_type("lab_shelf")
    cabinets = scene.find_by_type("lab_cabinet")
    lights = scene.find_by_type("lab_light_fixture")
    station_lights = scene.find_by_type("lab_light_station")

    assert modules
    assert len(modules) == len(benches)
    assert len(access) == len(benches)
    assert len(equipment) >= len(benches)
    assert len(equipment) <= len(benches) * 5
    assert len(perimeter) == 4
    assert walkways
    assert shelves or cabinets
    assert lights
    assert station_lights

    hoods = scene.find_by_type("lab_fume_hood")
    assert hoods


def test_laboratory_connects_stations_to_infrastructure() -> None:
    params = _build_params()
    params["machinery"]["laboratory"].update(  # type: ignore[index]
        {
            "water_enabled": True,
            "gas_enabled": True,
        }
    )
    scene = FactoryGenerator(params).generate_scene()

    benches = scene.find_by_type("lab_bench")
    equipment = scene.find_by_type("lab_equipment_unit")
    cable_trunks = scene.find_by_type("lab_cable_trunk")
    cable_drops_up = scene.find_by_type("lab_cable_drop_up")
    cable_drops_down = scene.find_by_type("lab_cable_drop_down")
    pipe_main = scene.find_by_type("lab_pipe_main")
    pipe_ceiling = scene.find_by_type("lab_pipe_ceiling")
    pipe_branches = scene.find_by_type("lab_pipe_branch")
    pipe_drops = scene.find_by_type("lab_pipe_drop")
    sinks = scene.find_by_type("lab_sink")

    assert benches
    assert equipment
    assert cable_trunks
    assert len(cable_drops_up) + len(cable_drops_down) >= len(equipment)
    assert len(pipe_main) == 2
    assert len(pipe_ceiling) == 1
    assert sinks
    assert pipe_branches
    assert pipe_drops


def test_laboratory_secondary_elements_add_local_detail() -> None:
    params = _build_params()
    params["machinery"]["laboratory"].update(  # type: ignore[index]
        {
            "seed": 29,
            "random_variation": True,
            "water_enabled": True,
            "pattern": "research_lab",
            "secondary": {
                "detail_density": 1.0,
                "tube_curvature_max": 0.35,
                "wall_units_enabled": True,
            },
        }
    )
    scene = FactoryGenerator(params).generate_scene()

    benches = scene.find_by_type("lab_bench")
    shelves = scene.find_by_type("lab_shelf")
    bottles = scene.find_by_type("lab_bottle_cluster")
    instruments = scene.find_by_type("lab_small_instrument")
    tubes = scene.find_by_type("lab_tube_connection")
    wastes = scene.find_by_type("lab_waste_container")
    wall_units = scene.find_by_type("lab_wall_mounted_unit")

    assert benches
    assert bottles
    assert instruments
    assert tubes
    assert wastes
    assert wall_units
    assert len(bottles) >= len(benches)
    assert len(instruments) >= len(benches)
    assert len(wastes) == len(benches)
    if shelves:
        assert len(bottles) > len(benches)


def test_laboratory_station_generation_uses_equipment_types_and_cable_modes() -> None:
    params = _build_params()
    params["machinery"]["laboratory"].update(  # type: ignore[index]
        {
            "seed": 41,
            "random_variation": True,
            "pattern": "research_lab",
            "equipment_cable_mode": "mixed",
            "equipment_types": ["analyzer", "mixer", "pump", "controller"],
            "stations": {"count": [8, 8], "spacing": [1.8, 1.8], "type_variability": [0.3, 0.3]},
            "equipment": {"density": [2.5, 2.5]},
        }
    )
    scene = FactoryGenerator(params).generate_scene()

    equipment = scene.find_by_type("lab_equipment_unit")
    assert equipment

    metadata_types = {
        getattr(getattr(obj, "mesh", None), "metadata", {}).get("params", {}).get("type")
        for obj in equipment
    }
    metadata_types.discard(None)
    assert len(metadata_types) >= 2

    cable_up = scene.find_by_type("lab_cable_drop_up")
    cable_down = scene.find_by_type("lab_cable_drop_down")
    assert cable_up
    assert cable_down
    assert len(cable_up) + len(cable_down) >= len(equipment)


def _station_snapshot(scene: object) -> list[tuple[float, float, float]]:
    modules = scene.find_by_type("lab_station_module")
    return sorted(
        (
            round(obj.transform.position[0], 4),
            round(obj.transform.position[1], 4),
            round(obj.transform.position[2], 4),
        )
        for obj in modules
    )


def test_laboratory_seed_is_reproducible() -> None:
    params_a = _build_params()
    params_a["machinery"]["laboratory"].update(  # type: ignore[index]
        {
            "seed": 17,
            "random_variation": True,
            "stations": {"count": [4, 10], "spacing": [1.6, 2.6], "type_variability": [0.25, 0.75]},
            "equipment": {"density": [1.2, 3.2]},
            "fumehood": {"probability": [0.1, 0.45]},
            "lighting": {"intensity": [0.7, 1.8]},
            "station_type_weights": {"chemistry": 0.4, "analysis": 0.4, "preparation": 0.2},
        }
    )
    params_b = _build_params()
    params_b["machinery"]["laboratory"] = dict(params_a["machinery"]["laboratory"])  # type: ignore[index]

    scene_a = FactoryGenerator(params_a).generate_scene()
    scene_b = FactoryGenerator(params_b).generate_scene()
    assert _station_snapshot(scene_a) == _station_snapshot(scene_b)


def test_laboratory_pattern_wet_vs_dry_changes_infrastructure_density() -> None:
    wet_params = _build_params()
    wet_params["machinery"]["laboratory"].update(  # type: ignore[index]
        {
            "seed": 23,
            "random_variation": True,
            "pattern": "wet_lab",
            "water_enabled": False,
            "stations": {"count": [9, 9], "spacing": [1.8, 1.8], "type_variability": [0.3, 0.3]},
        }
    )

    dry_params = deepcopy(wet_params)
    dry_params["machinery"]["laboratory"].update(  # type: ignore[index]
        {
            "pattern": "dry_lab",
        }
    )

    wet_scene = FactoryGenerator(wet_params).generate_scene()
    dry_scene = FactoryGenerator(dry_params).generate_scene()

    wet_sinks = wet_scene.find_by_type("lab_sink")
    dry_sinks = dry_scene.find_by_type("lab_sink")
    wet_pipe_features = len(wet_scene.find_by_type("lab_pipe_drop")) + len(wet_scene.find_by_type("lab_pipe_row"))
    dry_pipe_features = len(dry_scene.find_by_type("lab_pipe_drop")) + len(dry_scene.find_by_type("lab_pipe_row"))

    assert len(wet_sinks) > len(dry_sinks)
    assert wet_pipe_features > dry_pipe_features


def test_laboratory_pattern_analytical_reduces_stations_and_increases_equipment() -> None:
    analytical_params = _build_params()
    analytical_params["machinery"]["laboratory"].update(  # type: ignore[index]
        {
            "seed": 31,
            "pattern": "analytical_lab",
            "random_variation": False,
            "stations": {"count": [10, 10], "spacing": [1.8, 1.8], "type_variability": [0.2, 0.2]},
            "equipment": {"density": [2.0, 2.0]},
        }
    )

    teaching_params = deepcopy(analytical_params)
    teaching_params["machinery"]["laboratory"].update(  # type: ignore[index]
        {
            "pattern": "teaching_lab",
        }
    )

    analytical_scene = FactoryGenerator(analytical_params).generate_scene()
    teaching_scene = FactoryGenerator(teaching_params).generate_scene()

    analytical_modules = analytical_scene.find_by_type("lab_station_module")
    teaching_modules = teaching_scene.find_by_type("lab_station_module")
    analytical_equipment = analytical_scene.find_by_type("lab_equipment_unit")
    teaching_equipment = teaching_scene.find_by_type("lab_equipment_unit")

    assert len(analytical_modules) < len(teaching_modules)
    assert (len(analytical_equipment) / len(analytical_modules)) > (len(teaching_equipment) / len(teaching_modules))


def test_laboratory_pattern_teaching_vs_research_orientation_behavior() -> None:
    teaching_params = _build_params()
    teaching_params["machinery"]["laboratory"].update(  # type: ignore[index]
        {
            "seed": 37,
            "pattern": "teaching_lab",
            "random_variation": True,
            "stations": {"count": [8, 8], "spacing": [2.0, 2.0], "type_variability": [0.2, 0.2]},
        }
    )

    research_params = deepcopy(teaching_params)
    research_params["machinery"]["laboratory"].update(  # type: ignore[index]
        {
            "pattern": "research_lab",
        }
    )

    teaching_scene = FactoryGenerator(teaching_params).generate_scene()
    research_scene = FactoryGenerator(research_params).generate_scene()

    teaching_yaws = {round(obj.transform.rotation[2], 4) for obj in teaching_scene.find_by_type("lab_bench")}
    research_yaws = {round(obj.transform.rotation[2], 4) for obj in research_scene.find_by_type("lab_bench")}

    assert teaching_yaws.issubset({0.0, 180.0})
    assert any(abs(yaw) > 0.01 and abs(abs(yaw) - 180.0) > 0.01 for yaw in research_yaws)


def test_laboratory_rules_autofix_spacing_and_access() -> None:
    params = _build_params()
    params["factory_width"] = 20.0
    params["factory_depth"] = 20.0
    params["room_size_range"] = (12.0, 12.0)
    params["machinery"]["laboratory"].update(  # type: ignore[index]
        {
            "seed": 53,
            "pattern": "research_lab",
            "random_variation": True,
            "auto_fix": True,
            "stations": {"count": [12, 12], "spacing": [1.0, 1.0], "type_variability": [0.6, 0.6]},
            "module_width": [2.6, 2.6],
            "module_depth": [2.0, 2.0],
            "access_depth": 1.1,
            "walkway_width": 0.6,
            "perimeter_walkway": 0.9,
            "rules": {"min_spacing": 1.1, "auto_fix": True},
        }
    )

    scene = FactoryGenerator(params).generate_scene()
    modules = scene.find_by_type("lab_station_module")
    access = scene.find_by_type("lab_access_zone")

    assert modules
    assert len(access) == len(modules)

    min_dx = 2.6 + 1.1
    min_dy = 2.0 + 1.1
    for idx, left in enumerate(modules):
        lx, ly, _ = left.transform.position
        for right in modules[idx + 1 :]:
            rx, ry, _ = right.transform.position
            assert abs(lx - rx) >= min_dx - 1e-6 or abs(ly - ry) >= min_dy - 1e-6

    half_d = 12.0 / 2.0
    perimeter_walkway = float(params["machinery"]["laboratory"]["perimeter_walkway"])  # type: ignore[index]
    for zone in access:
        assert abs(zone.transform.position[1]) <= half_d - perimeter_walkway / 2.0 + 1e-6


def test_laboratory_rules_equipment_infrastructure_and_safety_autofix() -> None:
    params = _build_params()
    params["machinery"]["laboratory"].update(  # type: ignore[index]
        {
            "seed": 71,
            "pattern": "wet_lab",
            "random_variation": True,
            "auto_fix": True,
            "water_enabled": True,
            "stations": {"count": [10, 10], "spacing": [1.4, 1.4], "type_variability": [0.4, 0.4]},
            "module_width": [2.2, 2.2],
            "module_depth": [2.0, 2.0],
            "walkway_width": 1.6,
            "fumehood": {"probability": [1.0, 1.0]},
            "rules": {
                "equipment_near_distance": 0.2,
                "safety_walkway_clearance": 0.08,
                "auto_fix": True,
            },
        }
    )

    scene = FactoryGenerator(params).generate_scene()
    benches = scene.find_by_type("lab_bench")
    equipment = scene.find_by_type("lab_equipment_unit")
    cable_drops_up = scene.find_by_type("lab_cable_drop_up")
    cable_drops_down = scene.find_by_type("lab_cable_drop_down")
    walkways = scene.find_by_type("lab_walkway")
    fumehoods = scene.find_by_type("lab_fume_hood")

    assert benches
    assert equipment
    assert len(cable_drops_up) + len(cable_drops_down) >= len(equipment)
    assert walkways
    assert fumehoods

    near_distance = 0.2
    span_x = (2.2 * 0.72) / 2.0 + near_distance
    span_y = (2.0 * 0.48) / 2.0 + near_distance
    bench_xy = [(b.transform.position[0], b.transform.position[1]) for b in benches]
    for eq in equipment:
        ex, ey, _ = eq.transform.position
        assert any(abs(ex - bx) <= span_x + 1e-6 and abs(ey - by) <= span_y + 1e-6 for bx, by in bench_xy)

    walkway_half = 1.6 / 2.0
    for hood in fumehoods:
        hy = hood.transform.position[1]
        for walkway in walkways:
            wy = walkway.transform.position[1]
            assert abs(hy - wy) >= walkway_half + 0.4
