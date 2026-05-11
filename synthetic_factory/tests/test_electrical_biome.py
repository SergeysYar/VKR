from __future__ import annotations

from collections import defaultdict

from synthetic_factory.generators import FactoryGenerator


def _build_params() -> dict[str, object]:
    return {
        "factory_width": 28.0,
        "factory_depth": 20.0,
        "number_of_rooms": 1,
        "room_size_range": (16.0, 16.0),
        "corridor_width": 1.0,
        "layout_strategy": "grid",
        "noise": {"seed": 42},
        "biomes": {
            "enabled": True,
            "workshop_biome": "electrical",
            "cycle_order": ["electrical"],
        },
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {
            "enabled": True,
            "electrical": {
                "auto_fix": True,
                "module_size": 0.2,
                "orientation": "y+",
                "cabinet_width": 0.9,
                "cabinet_depth": 0.6,
                "cabinet_height": 2.2,
                "cabinet_door_type": "double",
                "rows": 3,
                "cabinets_per_row": 5,
                "cabinet_step": 1.6,
                "row_step": 3.0,
                "row_aisle_width": 1.4,
                "perimeter_aisle_width": 1.0,
                "min_clearance": 0.3,
                "access_depth": 0.8,
                "tray_mode": "overhead",
                "tray_width": 0.35,
                "tray_height": 0.12,
                "cross_tray_count": 2,
                "cable_radius": 0.03,
                "junction_mode": "walls",
                "junction_count": 4,
                "junction_box_size": 0.3,
                "cooling_count": 2,
                "cooling_width": 0.9,
                "cooling_height": 2.0,
                "cooling_airflow_direction": "front_to_back",
            },
        },
        "seed": 42,
    }


def test_electrical_biome_places_cabinets_in_rows_with_uniform_orientation() -> None:
    params = _build_params()
    scene = FactoryGenerator(params).generate_scene()
    cabinets = scene.find_by_type("electrical_cabinet")
    assert len(cabinets) >= 6

    rotations = {obj.transform.rotation for obj in cabinets}
    assert len(rotations) == 1

    row_to_x: dict[float, list[float]] = defaultdict(list)
    for cabinet in cabinets:
        row_to_x[round(cabinet.transform.position[1], 3)].append(cabinet.transform.position[0])
    assert len(row_to_x) >= 2

    min_allowed_spacing = (
        float(params["machinery"]["electrical"]["cabinet_width"])  # type: ignore[index]
        + float(params["machinery"]["electrical"]["min_clearance"])  # type: ignore[index]
    )
    for xs in row_to_x.values():
        sorted_x = sorted(xs)
        for idx in range(len(sorted_x) - 1):
            assert sorted_x[idx + 1] - sorted_x[idx] + 1e-6 >= min_allowed_spacing


def test_electrical_biome_builds_aisles_and_cable_infrastructure() -> None:
    params = _build_params()
    scene = FactoryGenerator(params).generate_scene()

    cabinets = scene.find_by_type("electrical_cabinet")
    aisles = scene.find_by_type("electrical_aisle")
    perimeter = scene.find_by_type("electrical_perimeter_aisle")
    row_trays = scene.find_by_type("cable_tray_row")
    cross_trays = scene.find_by_type("cable_tray_cross")
    drops = scene.find_by_type("cable_drop")

    assert len(aisles) >= 1
    assert len(perimeter) == 4
    assert len(row_trays) >= 2
    assert len(cross_trays) >= 1
    assert len(drops) == len(cabinets)

    cabinet_xy = {
        (round(obj.transform.position[0], 3), round(obj.transform.position[1], 3))
        for obj in cabinets
    }
    drop_xy = {
        (round(obj.transform.position[0], 3), round(obj.transform.position[1], 3))
        for obj in drops
    }
    assert drop_xy == cabinet_xy

    aisle_centers = [obj.transform.position[1] for obj in aisles]
    aisle_half_width = float(params["machinery"]["electrical"]["row_aisle_width"]) / 2.0  # type: ignore[index]
    for drop in drops:
        for center_y in aisle_centers:
            assert abs(drop.transform.position[1] - center_y) > aisle_half_width - 1e-6


def test_electrical_biome_keeps_access_and_edge_cooling_units() -> None:
    params = _build_params()
    scene = FactoryGenerator(params).generate_scene()

    cabinets = scene.find_by_type("electrical_cabinet")
    access = scene.find_by_type("electrical_access_zone")
    junction = scene.find_by_type("junction_box")
    cooling = scene.find_by_type("cooling_unit")

    assert len(access) >= int(len(cabinets) * 0.8)
    assert len(junction) >= 2
    assert len(cooling) == 2

    half_w = float(params["room_size_range"][1]) / 2.0  # type: ignore[index]
    perimeter_aisle = float(params["machinery"]["electrical"]["perimeter_aisle_width"])  # type: ignore[index]
    edge_threshold = half_w - perimeter_aisle - 0.3
    for unit in cooling:
        x, _, _ = unit.transform.position
        assert abs(x) >= edge_threshold


def _cabinet_snapshot(scene: object) -> list[tuple[float, float, float]]:
    cabinets = scene.find_by_type("electrical_cabinet")
    return sorted(
        (
            round(obj.transform.position[0], 4),
            round(obj.transform.position[1], 4),
            round(obj.transform.position[2], 4),
        )
        for obj in cabinets
    )


def test_electrical_seed_makes_generation_reproducible() -> None:
    params_a = _build_params()
    params_a["machinery"]["electrical"].update(  # type: ignore[index]
        {
            "seed": 123,
            "random_variation": True,
            "room": {"width": [12.0, 15.0], "depth": [12.0, 15.0], "height": [3.0, 5.0]},
            "cabinets": {"rows": [2, 5], "per_row": [4, 10], "spacing": [0.9, 1.4]},
            "walkways": {"width": [0.9, 1.8]},
            "cable_trays": {"height": [2.3, 3.2], "density": [0.8, 1.8]},
            "cables": {"density": [0.8, 2.2]},
            "cooling": {"units": [0, 4]},
        }
    )
    params_b = _build_params()
    params_b["machinery"]["electrical"] = dict(params_a["machinery"]["electrical"])  # type: ignore[index]

    scene_a = FactoryGenerator(params_a).generate_scene()
    scene_b = FactoryGenerator(params_b).generate_scene()
    assert _cabinet_snapshot(scene_a) == _cabinet_snapshot(scene_b)


def test_electrical_grid_snapping_applies_fixed_step() -> None:
    params = _build_params()
    step = 0.25
    params["machinery"]["electrical"].update(  # type: ignore[index]
        {
            "random_variation": False,
            "grid_snapping": True,
            "fixed_grid_step": step,
            "cabinets": {"rows": [3, 3], "per_row": [5, 5], "spacing": [1.13, 1.13]},
            "walkways": {"width": [1.21, 1.21]},
        }
    )

    scene = FactoryGenerator(params).generate_scene()
    cabinets = scene.find_by_type("electrical_cabinet")
    assert cabinets

    for cabinet in cabinets:
        x, y, _ = cabinet.transform.position
        assert abs((x / step) - round(x / step)) < 1e-6
        assert abs((y / step) - round(y / step)) < 1e-6


def test_electrical_dependency_constraints_are_enforced() -> None:
    params = _build_params()
    params["machinery"]["electrical"].update(  # type: ignore[index]
        {
            "random_variation": False,
            "room": {"width": [10.0, 10.0], "depth": [10.0, 10.0], "height": [4.0, 4.0]},
            "cabinets": {"rows": [6, 6], "per_row": [20, 20], "spacing": [1.2, 1.2]},
            "walkways": {"width": [1.5, 1.5]},
            "cabinet_depth": 0.6,
            "cabinet_width": 0.9,
        }
    )

    scene = FactoryGenerator(params).generate_scene()
    cabinets = scene.find_by_type("electrical_cabinet")
    assert cabinets

    rows = defaultdict(list)
    for cabinet in cabinets:
        rows[round(cabinet.transform.position[1], 3)].append(cabinet)

    row_count = len(rows)
    max_per_row = max(len(values) for values in rows.values())
    width_limit = 10.0
    depth_limit = 10.0
    expected_max_rows = int(width_limit // (0.6 + 1.5))
    expected_max_per_row = int(depth_limit // 0.9)

    assert row_count <= max(1, expected_max_rows)
    assert max_per_row <= max(1, expected_max_per_row)


def test_electrical_cable_density_scales_with_cabinet_count() -> None:
    params = _build_params()
    params["machinery"]["electrical"].update(  # type: ignore[index]
        {
            "random_variation": False,
            "cables": {"density": [2.0, 2.0]},
        }
    )

    scene = FactoryGenerator(params).generate_scene()
    cabinets = scene.find_by_type("electrical_cabinet")
    drops = scene.find_by_type("cable_drop")
    assert cabinets
    assert len(drops) == len(cabinets) * 2


def test_electrical_cable_network_builds_row_bundles_and_backbone() -> None:
    params = _build_params()
    params["machinery"]["electrical"].update(  # type: ignore[index]
        {
            "random_variation": False,
            "cables": {"density": [2.0, 2.0], "optimize_bundles": True, "variation": 0.06},
        }
    )

    scene = FactoryGenerator(params).generate_scene()
    row_bundles = scene.find_by_type("cable_bundle_row")
    backbone = scene.find_by_type("cable_bundle_backbone")
    trays = scene.find_by_type("cable_tray_row")

    assert row_bundles
    assert backbone
    tray_rows = {round(obj.transform.position[1], 3) for obj in trays}
    for bundle in row_bundles:
        assert round(bundle.transform.position[1], 3) in tray_rows


def test_electrical_single_row_pattern_places_one_row() -> None:
    params = _build_params()
    params["machinery"]["electrical"].update(  # type: ignore[index]
        {
            "random_variation": False,
            "pattern": "single_row",
            "cabinets": {"rows": [5, 5], "per_row": [8, 8], "spacing": [1.1, 1.1]},
        }
    )

    scene = FactoryGenerator(params).generate_scene()
    cabinets = scene.find_by_type("electrical_cabinet")
    assert cabinets
    unique_rows = {round(obj.transform.position[1], 3) for obj in cabinets}
    assert len(unique_rows) == 1


def test_electrical_back_to_back_pattern_creates_two_opposed_rows() -> None:
    params = _build_params()
    params["machinery"]["electrical"].update(  # type: ignore[index]
        {
            "random_variation": False,
            "pattern": "back_to_back",
            "cabinets": {"rows": [4, 4], "per_row": [6, 6], "spacing": [1.0, 1.0]},
            "walkways": {"width": [1.0, 1.0]},
        }
    )

    scene = FactoryGenerator(params).generate_scene()
    cabinets = scene.find_by_type("electrical_cabinet")
    assert cabinets

    unique_rows = sorted({round(obj.transform.position[1], 3) for obj in cabinets})
    assert 1 <= len(unique_rows) <= 2
    if len(unique_rows) == 2:
        assert abs(unique_rows[0] + unique_rows[1]) < 1e-6

    rotations = {round(obj.transform.rotation[2], 3) for obj in cabinets}
    assert len(rotations) >= 2


def test_electrical_dense_grid_has_higher_fill_than_sparse_technical() -> None:
    base = _build_params()
    base["machinery"]["electrical"].update(  # type: ignore[index]
        {
            "random_variation": False,
            "room": {"width": [16.0, 16.0], "depth": [16.0, 16.0], "height": [4.5, 4.5]},
            "cabinets": {"rows": [5, 5], "per_row": [10, 10], "spacing": [0.9, 1.2]},
            "walkways": {"width": [0.9, 1.8]},
        }
    )

    dense = _build_params()
    dense["machinery"]["electrical"] = dict(base["machinery"]["electrical"])  # type: ignore[index]
    dense["machinery"]["electrical"]["pattern"] = "dense_grid"  # type: ignore[index]

    sparse = _build_params()
    sparse["machinery"]["electrical"] = dict(base["machinery"]["electrical"])  # type: ignore[index]
    sparse["machinery"]["electrical"]["pattern"] = "sparse_technical"  # type: ignore[index]

    dense_scene = FactoryGenerator(dense).generate_scene()
    sparse_scene = FactoryGenerator(sparse).generate_scene()

    dense_count = len(dense_scene.find_by_type("electrical_cabinet"))
    sparse_count = len(sparse_scene.find_by_type("electrical_cabinet"))
    assert dense_count >= sparse_count
    assert dense_count > 0


def test_electrical_rules_clearance_and_access_autofix() -> None:
    params = _build_params()
    params["machinery"]["electrical"].update(  # type: ignore[index]
        {
            "random_variation": False,
            "pattern": "single_row",
            "room": {"width": [20.0, 20.0], "depth": [20.0, 20.0], "height": [4.0, 4.0]},
            "cabinets": {"rows": [1, 1], "per_row": [6, 6], "spacing": [1.2, 1.2]},
            "access_depth": 0.2,
            "rules": {"clearance_front": 1.4},
            "orientation": "y+",
        }
    )

    scene = FactoryGenerator(params).generate_scene()
    cabinets = scene.find_by_type("electrical_cabinet")
    access_zones = scene.find_by_type("electrical_access_zone")
    assert cabinets
    assert len(access_zones) == len(cabinets)

    cabinet_depth = float(params["machinery"]["electrical"]["cabinet_depth"])  # type: ignore[index]
    required_clearance = 1.4
    min_forward = cabinet_depth / 2.0 + required_clearance / 2.0
    for cabinet in cabinets:
        cx, cy, _ = cabinet.transform.position
        nearest = min(
            access_zones,
            key=lambda zone: (zone.transform.position[0] - cx) ** 2 + (zone.transform.position[1] - cy) ** 2,
        )
        ax, ay, _ = nearest.transform.position
        assert ay - cy + 1e-6 >= min_forward


def test_electrical_rules_cable_routing_and_height_autofix() -> None:
    params = _build_params()
    params["machinery"]["electrical"].update(  # type: ignore[index]
        {
            "random_variation": False,
            "pattern": "parallel_rows",
            "cables": {"density": [2.0, 2.0]},
            "rules": {"enforce_cable_routing": True, "tray_height_clearance": 0.6},
            "tray_height": 0.12,
            "cabinet_height": 2.2,
        }
    )

    scene = FactoryGenerator(params).generate_scene()
    trays = scene.find_by_type("cable_tray_row")
    drops = scene.find_by_type("cable_drop")
    assert trays
    assert drops

    tray_rows = {round(obj.transform.position[1], 3) for obj in trays}
    for drop in drops:
        assert round(drop.transform.position[1], 3) in tray_rows

    tray_height = float(params["machinery"]["electrical"]["tray_height"])  # type: ignore[index]
    cabinet_height = float(params["machinery"]["electrical"]["cabinet_height"])  # type: ignore[index]
    required_gap = 0.6
    for tray in trays:
        tz = tray.transform.position[2]
        assert tz + 1e-6 >= cabinet_height + required_gap + tray_height / 2.0


def test_electrical_rules_cooling_access_autofix() -> None:
    params = _build_params()
    params["machinery"]["electrical"].update(  # type: ignore[index]
        {
            "random_variation": False,
            "pattern": "parallel_rows",
            "room": {"width": [18.0, 18.0], "depth": [18.0, 18.0], "height": [4.0, 4.0]},
            "cabinets": {"rows": [3, 3], "per_row": [6, 6], "spacing": [1.2, 1.2]},
            "cooling": {"units": [4, 4]},
            "rules": {"cooling_access_clearance": 1.2},
        }
    )

    scene = FactoryGenerator(params).generate_scene()
    cabinets = scene.find_by_type("electrical_cabinet")
    cooling = scene.find_by_type("cooling_unit")
    assert cabinets
    assert cooling

    cooling_width = float(params["machinery"]["electrical"]["cooling_width"])  # type: ignore[index]
    cabinet_width = float(params["machinery"]["electrical"]["cabinet_width"])  # type: ignore[index]
    cabinet_depth = float(params["machinery"]["electrical"]["cabinet_depth"])  # type: ignore[index]
    required = cooling_width / 2.0 + max(cabinet_width, cabinet_depth) / 2.0 + 1.2
    required_sq = required * required

    for cu in cooling:
        x, y, _ = cu.transform.position
        best_sq = min((x - cab.transform.position[0]) ** 2 + (y - cab.transform.position[1]) ** 2 for cab in cabinets)
        assert best_sq + 1e-6 >= required_sq


def test_electrical_secondary_elements_are_generated() -> None:
    params = _build_params()
    params["machinery"]["electrical"].update(  # type: ignore[index]
        {
            "random_variation": False,
            "secondary": {"enabled": True},
        }
    )

    scene = FactoryGenerator(params).generate_scene()
    cabinets = scene.find_by_type("electrical_cabinet")
    transformers = scene.find_by_type("transformer_unit")
    floor_entries = scene.find_by_type("floor_cable_entry")
    batteries = scene.find_by_type("backup_battery")
    panels = scene.find_by_type("small_control_panel")
    bundles = scene.find_by_type("cable_bundle")

    assert cabinets
    assert transformers
    assert floor_entries
    assert batteries
    assert panels
    assert bundles

    assert len(floor_entries) == len(cabinets)
