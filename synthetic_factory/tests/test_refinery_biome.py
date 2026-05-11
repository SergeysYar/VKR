from __future__ import annotations

from synthetic_factory.generators import FactoryGenerator
from synthetic_factory.generators.biomes import refinery


def _build_params() -> dict[str, object]:
    return {
        "factory_width": 36.0,
        "factory_depth": 28.0,
        "number_of_rooms": 1,
        "room_size_range": (20.0, 20.0),
        "corridor_width": 1.0,
        "layout_strategy": "grid",
        "noise": {"seed": 42},
        "biomes": {
            "enabled": True,
            "workshop_biome": "refinery",
            "cycle_order": ["refinery"],
        },
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {
            "enabled": True,
            "refinery": {
                "seed": 7,
                "random_variation": False,
                "pattern": "dense_refinery",
                "walkway_width": 1.4,
                "main_columns": {
                    "count": [3, 3],
                    "height": [14.0, 14.0],
                    "radius": [1.1, 1.1],
                },
                "secondary": {
                    "count": [8, 8],
                    "radius": [0.55, 0.55],
                    "length": [3.2, 3.2],
                },
                "pipes": {
                    "density": [2.0, 2.0],
                    "radius": [0.22, 0.22],
                    "levels": [3, 3],
                    "support_spacing": [3.0, 3.0],
                },
                "platforms": {
                    "levels": [2, 2],
                    "width_factor": [1.8, 1.8],
                    "guardrails": {
                        "enabled": True,
                        "height": 1.0,
                        "thickness": 0.08,
                    },
                },
                "structure": {
                    "edge_offset": 1.1,
                    "min_clearance": 1.0,
                },
                "rules": {
                    "auto_fix": True,
                    "min_clearance": 1.0,
                    "pipe_collision_margin": 0.12,
                    "pipe_vertical_step": 0.28,
                    "support_spacing": 3.0,
                },
            },
        },
        "seed": 42,
    }


def test_refinery_biome_generates_vertical_dense_connected_structure() -> None:
    scene = FactoryGenerator(_build_params()).generate_scene()

    main_columns = scene.find_by_type("refinery_main_column")
    vessels = scene.find_by_type("refinery_secondary_vessel")
    backbones = scene.find_by_type("refinery_pipe_backbone")
    risers = scene.find_by_type("refinery_pipe_riser")
    links = scene.find_by_type("refinery_pipe_link")
    vertical_links = scene.find_by_type("refinery_pipe_vertical_link")
    node_links = scene.find_by_type("refinery_pipe_node_link")
    nodes = scene.find_by_type("refinery_junction_node")
    platforms = scene.find_by_type("refinery_platform")
    supports = scene.find_by_type("refinery_pipe_support")
    access = scene.find_by_type("refinery_access_zone")
    process_pipes = scene.find_by_type("refinery_process_pipe")
    process_valves = scene.find_by_type("refinery_process_valve")
    process_pumps = scene.find_by_type("refinery_process_pump")
    process_supports = scene.find_by_type("refinery_process_pipe_support")
    tanks = scene.find_by_type("refinery_storage_tank")
    service_walkways = scene.find_by_type("refinery_service_walkway")
    ladders = scene.find_by_type("refinery_ladder")
    guardrails = scene.find_by_type("refinery_guardrail")
    valve_clusters = scene.find_by_type("refinery_valve_cluster")
    secondary_pipe_supports = scene.find_by_type("refinery_secondary_pipe_support")
    control_boxes = scene.find_by_type("refinery_small_control_box")
    sensors = scene.find_by_type("refinery_sensor_unit")

    assert 3 <= len(main_columns) <= 6
    assert len(vessels) >= 5
    assert len(backbones) >= 3
    assert len(risers) >= len(main_columns)
    assert len(links) >= len(vessels) * 0.7
    assert len(vertical_links) >= len(vessels) * 0.7
    assert len(nodes) >= 3
    assert len(node_links) >= len(nodes) * 0.7
    assert len(platforms) >= len(main_columns)
    assert supports
    assert len(access) == 1
    assert len(tanks) >= 1
    assert len(process_pipes) >= max(5, len(main_columns))
    assert len(process_valves) >= len(process_pipes) * 0.7
    assert len(process_pumps) >= len(process_pipes) * 0.4
    assert len(process_supports) >= len(process_pipes)
    assert len(service_walkways) >= len(main_columns)
    assert len(ladders) >= len(main_columns)
    assert len(guardrails) >= len(platforms)
    assert len(valve_clusters) >= max(3, int(len(process_pipes) * 0.25))
    assert len(secondary_pipe_supports) >= len(valve_clusters) * 0.8
    assert len(control_boxes) >= len(main_columns)
    assert len(sensors) >= len(nodes)


def test_refinery_rules_include_autofix_constraints() -> None:
    rule_names = refinery.RULES.get("rules", [])
    assert "ConnectivityRule" in rule_names
    assert "NoPipeCollisionRule" in rule_names
    assert "SupportRule" in rule_names
    assert "AccessRule" in rule_names
    assert "HeightRule" in rule_names
    assert refinery.RULES.get("auto_fix") is True


def test_refinery_seed_is_reproducible() -> None:
    params_a = _build_params()
    params_a["machinery"]["refinery"].update(  # type: ignore[index]
        {
            "seed": 99,
            "random_variation": True,
            "main_columns": {"count": [2, 5], "height": [11.0, 18.0], "radius": [0.8, 1.6]},
            "secondary": {"count": [4, 12], "radius": [0.4, 1.0], "length": [2.2, 4.8]},
            "pipes": {"density": [1.0, 2.6], "radius": [0.14, 0.34], "levels": [2, 4]},
        }
    )
    params_b = _build_params()
    params_b["machinery"]["refinery"] = dict(params_a["machinery"]["refinery"])  # type: ignore[index]

    scene_a = FactoryGenerator(params_a).generate_scene()
    scene_b = FactoryGenerator(params_b).generate_scene()

    def snapshot(scene: object) -> list[tuple[float, float, float]]:
        columns = scene.find_by_type("refinery_main_column")
        return sorted(
            (
                round(obj.transform.position[0], 4),
                round(obj.transform.position[1], 4),
                round(obj.transform.position[2], 4),
            )
            for obj in columns
        )

    assert snapshot(scene_a) == snapshot(scene_b)
