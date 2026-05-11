from __future__ import annotations

from synthetic_factory.generators import FactoryGenerator


def _build_params(pattern: str) -> dict[str, object]:
    return {
        "factory_width": 24.0,
        "factory_depth": 20.0,
        "number_of_rooms": 1,
        "room_size_range": (14.0, 14.0),
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
                "pattern": pattern,
                "random_variation": False,
                "min_walkway": 1.2,
                "service_clearance": 0.8,
                "edge_offset": 1.0,
                "boiler": {
                    "count": [2, 2],
                    "height": [8.0, 8.0],
                    "radius": [1.0, 1.0],
                },
                "pipes": {
                    "density": [0.5, 0.5],
                    "radius": [0.2, 0.2],
                },
                "platforms": {
                    "levels": [2, 2],
                    "width_factor": [1.5, 1.5],
                },
                "tanks": {
                    "count": [1, 1],
                },
                "structure": {
                    "beam_density": [0.6, 0.6],
                },
            },
        },
        "seed": 42,
    }


def _generate_scene(pattern: str):
    return FactoryGenerator(_build_params(pattern)).generate_scene()


def test_clustered_pattern_creates_cluster_pipe_system() -> None:
    scene = _generate_scene("clustered_boilers")
    assert len(scene.find_by_type("boiler_unit")) >= 2
    assert len(scene.find_by_type("pipe_manifold_cluster")) >= 1


def test_linear_pattern_keeps_sequential_logic_without_extra_dense_manifolds() -> None:
    scene = _generate_scene("linear_boilers")
    assert len(scene.find_by_type("boiler_unit")) >= 2
    assert len(scene.find_by_type("pipe_manifold_cluster")) == 0
    assert len(scene.find_by_type("pipe_manifold_side")) == 0


def test_central_tower_alias_creates_single_dominant_boiler() -> None:
    scene = _generate_scene("central")
    assert len(scene.find_by_type("boiler_unit")) == 1
    assert len(scene.find_by_type("aux_module")) >= 1


def test_dense_industrial_adds_side_manifolds_and_more_branches_than_linear() -> None:
    linear_scene = _generate_scene("linear_boilers")
    dense_scene = _generate_scene("dense_industrial")

    assert len(dense_scene.find_by_type("pipe_manifold_side")) >= 2
    assert len(dense_scene.find_by_type("pipe_branch")) > len(
        linear_scene.find_by_type("pipe_branch")
    )

