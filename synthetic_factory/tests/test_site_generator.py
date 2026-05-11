from __future__ import annotations

from synthetic_factory.generators import FactoryGenerator


def _base_params() -> dict[str, object]:
    return {
        "factory_width": 72.0,
        "factory_depth": 54.0,
        "number_of_rooms": 4,
        "room_size_range": (12.0, 12.0),
        "corridor_width": 3.0,
        "room_height": 6.5,
        "layout_strategy": "grid",
        "noise": {"seed": 7},
        "biomes": {"enabled": False},
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {"enabled": False},
        "seed": 7,
    }


def test_site_group_generates_territory_and_outdoor_industrial_objects() -> None:
    params = _base_params()
    params["exterior"] = {"enabled": True}
    params["site"] = {
        "enabled": True,
        "terrain_margin": 16.0,
        "road": {"enabled": True},
        "fence": {"enabled": True},
        "logistics": {"enabled": True, "dock_count": 2},
        "industrial": {
            "enabled": True,
            "tanks": {"count": 3},
            "transformers": {"count": 2},
        },
    }

    scene = FactoryGenerator(params).generate_scene()
    site = scene.get_object("site")
    assert site is not None
    assert site.type == "site_group"

    assert len(scene.find_by_type("site_ground")) >= 1
    assert len(scene.find_by_type("site_road")) >= 4
    assert len(scene.find_by_type("site_fence")) >= 4
    assert len(scene.find_by_type("site_loading_dock")) >= 2
    assert len(scene.find_by_type("site_tank")) >= 3
    assert len(scene.find_by_type("site_transformer")) >= 2


def test_site_generator_respects_enabled_toggle() -> None:
    params = _base_params()
    params["site"] = {"enabled": False}
    scene = FactoryGenerator(params).generate_scene()

    assert scene.get_object("site") is None
    assert not scene.find_by_type("site_ground")
    assert not scene.find_by_type("site_road")


def test_site_dimensions_scale_with_factory_footprint() -> None:
    small = _base_params()
    large = _base_params()
    large["factory_width"] = 132.0
    large["factory_depth"] = 96.0

    for payload in (small, large):
        payload["site"] = {
            "enabled": True,
            "terrain_margin": 20.0,
            "industrial": {"enabled": False},
            "logistics": {"enabled": False},
            "fence": {"enabled": False},
            "road": {"enabled": False},
        }

    small_scene = FactoryGenerator(small).generate_scene()
    large_scene = FactoryGenerator(large).generate_scene()

    small_ground = small_scene.get_object("site_ground")
    large_ground = large_scene.get_object("site_ground")
    assert small_ground is not None
    assert large_ground is not None
    assert small_ground.mesh is not None
    assert large_ground.mesh is not None

    def mesh_span_xy(obj) -> tuple[float, float]:
        assert obj.mesh is not None
        xs = [vertex[0] for vertex in obj.mesh.vertices]
        ys = [vertex[1] for vertex in obj.mesh.vertices]
        return (max(xs) - min(xs), max(ys) - min(ys))

    small_w, small_d = mesh_span_xy(small_ground)
    large_w, large_d = mesh_span_xy(large_ground)
    assert large_w > small_w
    assert large_d > small_d


def test_site_footprint_matches_exterior_building_margin() -> None:
    params = _base_params()
    params["factory_width"] = 60.0
    params["factory_depth"] = 40.0
    params["room_height"] = 6.0
    params["exterior"] = {
        "enabled": True,
        "wall_thickness": 0.5,
        "wall_offset": 0.0,
        "wall_headroom": 0.4,
        "building": {
            "margin": 4.0,
            "height_variation": 1.5,
        },
        "gates": {"enabled": False},
        "roof_units": {"enabled": False},
    }
    params["site"] = {
        "enabled": True,
        "terrain_margin": 0.0,
        "terrain": {
            "size": 0.0,
        },
        "industrial": {"enabled": False},
        "logistics": {"enabled": False},
        "fence": {"enabled": False},
        "road": {"enabled": False},
    }

    scene = FactoryGenerator(params).generate_scene()
    ground = scene.get_object("site_ground")
    assert ground is not None
    assert ground.mesh is not None

    xs = [vertex[0] for vertex in ground.mesh.vertices]
    ys = [vertex[1] for vertex in ground.mesh.vertices]
    width = max(xs) - min(xs)
    depth = max(ys) - min(ys)

    expected_width = 60.0 + 2.0 * (4.0 + 0.5)
    expected_depth = 40.0 + 2.0 * (4.0 + 0.5)
    assert abs(width - expected_width) <= 1e-6
    assert abs(depth - expected_depth) <= 1e-6


def test_site_adds_access_roads_to_building_entries() -> None:
    params = _base_params()
    params["biomes"] = {
        "enabled": True,
        "workshop_biome": "storage",
        "cycle_order": ["storage"],
    }
    params["exterior"] = {
        "enabled": True,
        "gates": {"enabled": True, "count": 2},
    }
    params["site"] = {
        "enabled": True,
        "road": {"enabled": True, "connect_to_building": True},
        "industrial": {"enabled": False},
        "logistics": {"enabled": False},
        "fence": {"enabled": False},
    }

    scene = FactoryGenerator(params).generate_scene()
    access_roads = scene.find_by_type("site_access_road")
    assert access_roads


def test_site_supports_parking_zones_and_terrain_variation() -> None:
    params = _base_params()
    params["site"] = {
        "enabled": True,
        "terrain": {
            "thickness": 0.2,
            "height_noise": {
                "enabled": True,
                "amplitude": 0.05,
                "cells_x": 5,
                "cells_y": 4,
            },
        },
        "parking": {"enabled": True},
        "zones": {"enabled": True},
    }

    scene = FactoryGenerator(params).generate_scene()

    assert scene.find_by_type("site_ground_patch")
    assert scene.find_by_type("site_parking")
    assert scene.find_by_type("site_parking_line")
    assert scene.find_by_type("site_zone")


def test_site_terrain_size_and_road_density_parameters_are_applied() -> None:
    params = _base_params()
    params["factory_width"] = 60.0
    params["factory_depth"] = 40.0
    params["site"] = {
        "enabled": True,
        "terrain_margin": 4.0,
        "terrain": {
            "size": [120.0, 120.0],
            "road_density": 2.0,
        },
        "industrial": {"enabled": False},
        "logistics": {"enabled": False},
        "fence": {"enabled": False},
        "road": {"enabled": True},
    }

    scene = FactoryGenerator(params).generate_scene()
    ground = scene.get_object("site_ground")
    assert ground is not None
    assert ground.mesh is not None
    xs = [vertex[0] for vertex in ground.mesh.vertices]
    ys = [vertex[1] for vertex in ground.mesh.vertices]
    width = max(xs) - min(xs)
    depth = max(ys) - min(ys)
    assert width >= 120.0 - 1e-6
    assert depth >= 120.0 - 1e-6
    assert scene.find_by_type("site_road_secondary")


def test_site_equipment_density_increases_outdoor_object_count() -> None:
    low = _base_params()
    low["factory_width"] = 120.0
    low["factory_depth"] = 80.0
    low["site"] = {
        "enabled": True,
        "equipment_density": 0.5,
        "industrial": {
            "enabled": True,
            "tanks": {"count": 4},
            "transformers": {"count": 4},
        },
        "logistics": {"enabled": True, "dock_count": 4},
    }

    high = _base_params()
    high["factory_width"] = 120.0
    high["factory_depth"] = 80.0
    high["site"] = {
        "enabled": True,
        "equipment_density": 2.0,
        "industrial": {
            "enabled": True,
            "tanks": {"count": 4},
            "transformers": {"count": 4},
        },
        "logistics": {"enabled": True, "dock_count": 4},
    }

    low_scene = FactoryGenerator(low).generate_scene()
    high_scene = FactoryGenerator(high).generate_scene()
    low_count = len(low_scene.find_by_type("site_tank")) + len(low_scene.find_by_type("site_transformer")) + len(
        low_scene.find_by_type("site_loading_dock")
    )
    high_count = len(high_scene.find_by_type("site_tank")) + len(high_scene.find_by_type("site_transformer")) + len(
        high_scene.find_by_type("site_loading_dock")
    )
    assert high_count > low_count
