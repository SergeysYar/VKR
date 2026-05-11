from __future__ import annotations

from math import sqrt

from synthetic_factory.generators import FactoryGenerator
from synthetic_factory.scene.scene_graph import SceneObject


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
            "workshop_biome": "maintenance",
            "cycle_order": ["maintenance"],
        },
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {
            "enabled": True,
            "maintenance": {
                "seed": 101,
                "random_variation": False,
                "density": [1.2, 1.2],
                "clutter": [0.9, 0.9],
                "min_walkway_width": 1.4,
                "edge_margin": 0.8,
                "repair_zone_count": [6, 6],
                "storage_rack_count": [4, 4],
                "major_spacing_x": 2.2,
                "major_row_spacing": 2.5,
                "workbench": {
                    "width": 1.8,
                    "depth": 0.9,
                    "height": 0.95,
                },
                "tool_rack": {
                    "width": 1.0,
                    "height": 2.1,
                },
                "rules": {
                    "min_clearance": 0.6,
                    "access_depth": 1.0,
                },
            },
        },
        "seed": 42,
    }


def _bounds_xy(obj: SceneObject) -> tuple[float, float, float, float]:
    assert obj.mesh is not None
    px, py, _ = obj.transform.position
    xs = [vertex[0] + px for vertex in obj.mesh.vertices]
    ys = [vertex[1] + py for vertex in obj.mesh.vertices]
    return (min(xs), max(xs), min(ys), max(ys))


def _intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[1] and a[1] > b[0] and a[2] < b[3] and a[3] > b[2]


def _distance_to_bounds_xy(x: float, y: float, bounds: tuple[float, float, float, float]) -> float:
    nx = min(max(x, bounds[0]), bounds[1])
    ny = min(max(y, bounds[2]), bounds[3])
    return sqrt((x - nx) ** 2 + (y - ny) ** 2)


def _bottom_z(obj: SceneObject) -> float:
    assert obj.mesh is not None
    pz = obj.transform.position[2]
    return min(vertex[2] + pz for vertex in obj.mesh.vertices)


def test_maintenance_biome_generates_functional_zones() -> None:
    scene = FactoryGenerator(_build_params()).generate_scene()
    object_types = {obj.type for obj in scene.traverse()}

    expected = {
        "maintenance_walkway",
        "maintenance_storage_zone",
        "maintenance_repair_zone",
        "workbench",
        "machine_part",
        "toolbox",
        "tool_rack",
        "pallet",
        "spare_part",
        "crane_hook",
    }
    assert expected.issubset(object_types)
    assert len(scene.find_by_type("workbench")) >= 2


def test_maintenance_biome_keeps_central_walkway_clear_for_major_objects() -> None:
    scene = FactoryGenerator(_build_params()).generate_scene()
    walkway = scene.find_by_type("maintenance_walkway")[0]
    walkway_bounds = _bounds_xy(walkway)

    for object_type in ("workbench", "tool_rack", "pallet"):
        for obj in scene.find_by_type(object_type):
            assert obj.mesh is not None
            assert not _intersects(_bounds_xy(obj), walkway_bounds)


def test_maintenance_storage_does_not_overlap_repair_zones() -> None:
    scene = FactoryGenerator(_build_params()).generate_scene()
    repair_bounds = [_bounds_xy(obj) for obj in scene.find_by_type("maintenance_repair_zone")]
    for pallet in scene.find_by_type("pallet"):
        pallet_bounds = _bounds_xy(pallet)
        assert all(not _intersects(pallet_bounds, repair_zone) for repair_zone in repair_bounds)


def test_maintenance_biome_is_seed_reproducible_with_variation_enabled() -> None:
    params = _build_params()
    params["machinery"]["maintenance"]["random_variation"] = True  # type: ignore[index]
    params["machinery"]["maintenance"]["repair_zone_count"] = [4, 8]  # type: ignore[index]
    params["machinery"]["maintenance"]["storage_rack_count"] = [2, 6]  # type: ignore[index]
    params["machinery"]["maintenance"]["density"] = [0.9, 1.4]  # type: ignore[index]
    params["machinery"]["maintenance"]["clutter"] = [0.4, 1.2]  # type: ignore[index]

    scene_a = FactoryGenerator(params).generate_scene()
    scene_b = FactoryGenerator(params).generate_scene()

    snapshot_types = {
        "workbench",
        "machine_part",
        "toolbox",
        "tool_rack",
        "pallet",
        "spare_part",
        "crane_hook",
    }

    def snapshot(scene: object) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
        result: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = []
        for obj in scene.traverse():  # type: ignore[attr-defined]
            if obj.type in snapshot_types:
                result.append((obj.type, obj.transform.position, obj.transform.rotation))
        return sorted(result, key=lambda item: (item[0], item[1], item[2]))

    assert snapshot(scene_a) == snapshot(scene_b)


def test_maintenance_pattern_heavy_forces_crane_and_extra_machine_parts() -> None:
    params = _build_params()
    params["machinery"]["maintenance"]["pattern"] = "Heavy Maintenance"  # type: ignore[index]
    params["machinery"]["maintenance"]["random_variation"] = False  # type: ignore[index]
    params["machinery"]["maintenance"]["crane"] = {"enabled": [False, False]}  # type: ignore[index]
    params["machinery"]["maintenance"]["parts"] = {  # type: ignore[index]
        "large_parts": [4, 4],
        "loose_parts": [0, 0],
    }

    scene = FactoryGenerator(params).generate_scene()
    workbenches = scene.find_by_type("workbench")
    machines = scene.find_by_type("machine_part")
    hooks = scene.find_by_type("crane_hook")

    assert workbenches
    assert len(hooks) >= len(workbenches)
    assert len(machines) > len(workbenches)


def test_maintenance_pattern_storage_dominant_increases_storage_presence() -> None:
    clean_params = _build_params()
    clean_params["machinery"]["maintenance"]["pattern"] = "clean_workshop"  # type: ignore[index]
    clean_params["machinery"]["maintenance"]["random_variation"] = False  # type: ignore[index]
    clean_params["machinery"]["maintenance"]["zones"] = {"storage": [2, 2]}  # type: ignore[index]

    storage_params = _build_params()
    storage_params["machinery"]["maintenance"]["pattern"] = "Storage Dominant"  # type: ignore[index]
    storage_params["machinery"]["maintenance"]["random_variation"] = False  # type: ignore[index]
    storage_params["machinery"]["maintenance"]["zones"] = {"storage": [2, 2]}  # type: ignore[index]

    clean_scene = FactoryGenerator(clean_params).generate_scene()
    storage_scene = FactoryGenerator(storage_params).generate_scene()

    assert len(storage_scene.find_by_type("maintenance_storage_zone")) > len(
        clean_scene.find_by_type("maintenance_storage_zone")
    )
    assert len(storage_scene.find_by_type("pallet")) >= len(clean_scene.find_by_type("pallet"))
    assert len(storage_scene.find_by_type("tool_rack")) >= len(clean_scene.find_by_type("tool_rack"))


def test_maintenance_pattern_chaotic_increases_parts_and_disassembly() -> None:
    clean_params = _build_params()
    clean_params["machinery"]["maintenance"]["pattern"] = "clean workshop"  # type: ignore[index]
    clean_params["machinery"]["maintenance"]["random_variation"] = True  # type: ignore[index]
    clean_params["machinery"]["maintenance"]["parts"] = {  # type: ignore[index]
        "large_parts": [2, 2],
        "loose_parts": [14, 14],
    }

    chaotic_params = _build_params()
    chaotic_params["machinery"]["maintenance"]["pattern"] = "Chaotic Workshop"  # type: ignore[index]
    chaotic_params["machinery"]["maintenance"]["random_variation"] = True  # type: ignore[index]
    chaotic_params["machinery"]["maintenance"]["parts"] = {  # type: ignore[index]
        "large_parts": [2, 2],
        "loose_parts": [14, 14],
    }

    clean_scene = FactoryGenerator(clean_params).generate_scene()
    chaotic_scene = FactoryGenerator(chaotic_params).generate_scene()

    assert len(chaotic_scene.find_by_type("spare_part")) > len(clean_scene.find_by_type("spare_part"))
    assert len(chaotic_scene.find_by_type("machine_part")) > len(clean_scene.find_by_type("machine_part"))


def test_maintenance_rules_stability_and_tool_placement() -> None:
    params = _build_params()
    params["machinery"]["maintenance"]["pattern"] = "active_repair"  # type: ignore[index]
    params["machinery"]["maintenance"]["random_variation"] = True  # type: ignore[index]
    params["machinery"]["maintenance"]["rules"] = {  # type: ignore[index]
        "min_clearance": 0.6,
        "access_depth": 1.0,
        "tool_max_distance": 2.2,
    }

    scene = FactoryGenerator(params).generate_scene()
    repair_bounds = [_bounds_xy(obj) for obj in scene.find_by_type("maintenance_repair_zone")]
    assert repair_bounds

    for obj_type in ("workbench", "tool_rack", "pallet", "machine_part", "toolbox", "spare_part"):
        for obj in scene.find_by_type(obj_type):
            assert _bottom_z(obj) >= -1e-6

    for obj_type in ("toolbox", "tool_rack"):
        for obj in scene.find_by_type(obj_type):
            x, y, _ = obj.transform.position
            nearest = min(_distance_to_bounds_xy(x, y, bounds) for bounds in repair_bounds)
            assert nearest <= 2.2 + 1e-6


def test_maintenance_rules_part_support_and_crane_clearance() -> None:
    params = _build_params()
    params["machinery"]["maintenance"]["pattern"] = "heavy_maintenance"  # type: ignore[index]
    params["machinery"]["maintenance"]["random_variation"] = False  # type: ignore[index]
    params["machinery"]["maintenance"]["parts"] = {  # type: ignore[index]
        "large_parts": [6, 6],
        "loose_parts": [0, 0],
    }
    params["machinery"]["maintenance"]["rules"] = {  # type: ignore[index]
        "min_clearance": 0.6,
        "access_depth": 1.0,
        "crane_clearance_radius": 0.55,
    }

    scene = FactoryGenerator(params).generate_scene()
    pallets = scene.find_by_type("pallet")
    pallet_bounds = [_bounds_xy(obj) for obj in pallets]
    pallet_tops = [max(vertex[2] + obj.transform.position[2] for vertex in obj.mesh.vertices) for obj in pallets if obj.mesh]  # type: ignore[union-attr]
    floor_tol = 0.08
    pallet_tol = 0.12

    supported_candidates = 0
    for part in scene.find_by_type("machine_part"):
        bottom = _bottom_z(part)
        x, y, _ = part.transform.position
        if bottom <= 0.35:
            supported_candidates += 1
            on_floor = bottom <= floor_tol + 1e-6
            on_pallet = False
            for bounds, top in zip(pallet_bounds, pallet_tops):
                inside_xy = bounds[0] - pallet_tol <= x <= bounds[1] + pallet_tol and bounds[2] - pallet_tol <= y <= bounds[3] + pallet_tol
                if inside_xy and abs(bottom - top) <= pallet_tol + 0.05:
                    on_pallet = True
                    break
            assert on_floor or on_pallet
    assert supported_candidates >= 1

    clearance_radius = 0.55
    blocked_types = ("workbench", "tool_rack", "pallet")
    blocked_bounds = [_bounds_xy(obj) for t in blocked_types for obj in scene.find_by_type(t)]
    for hook in scene.find_by_type("crane_hook"):
        hx, hy, _ = hook.transform.position
        for bounds in blocked_bounds:
            assert _distance_to_bounds_xy(hx, hy, bounds) >= clearance_radius - 1e-6


def test_maintenance_repair_scenario_splits_components_and_adds_disconnected_lines() -> None:
    params = _build_params()
    params["machinery"]["maintenance"]["random_variation"] = False  # type: ignore[index]
    params["machinery"]["maintenance"]["repair_scenario"] = {  # type: ignore[index]
        "enabled": True,
        "name": "repair",
        "count": [2, 2],
        "subcomponents": [4, 4],
        "tools_per_target": [2, 2],
        "disconnected_lines": True,
        "cable_probability": [1.0, 1.0],
        "pipe_probability": [1.0, 1.0],
    }

    scene = FactoryGenerator(params).generate_scene()
    components = scene.find_by_type("repair_component")
    assert len(components) >= 8
    assert len(scene.find_by_type("maintenance_cable_loose")) >= 1
    assert len(scene.find_by_type("maintenance_pipe_loose")) >= 1

    workbenches = scene.find_by_type("workbench")
    pallets = scene.find_by_type("pallet")
    assert workbenches
    assert pallets

    workbench_bounds = [_bounds_xy(obj) for obj in workbenches]
    pallet_bounds = [_bounds_xy(obj) for obj in pallets]
    floor_tol = 0.1
    on_floor = False
    on_bench = False
    on_pallet = False
    for component in components:
        bottom = _bottom_z(component)
        x, y, _ = component.transform.position
        if bottom <= floor_tol + 1e-6:
            on_floor = True
        for bounds in workbench_bounds:
            inside = bounds[0] - 0.12 <= x <= bounds[1] + 0.12 and bounds[2] - 0.12 <= y <= bounds[3] + 0.12
            if inside and bottom > floor_tol + 0.08:
                on_bench = True
                break
        for bounds in pallet_bounds:
            inside = bounds[0] - 0.12 <= x <= bounds[1] + 0.12 and bounds[2] - 0.12 <= y <= bounds[3] + 0.12
            if inside and bottom > floor_tol + 0.02:
                on_pallet = True
                break

    assert on_floor
    assert on_bench
    assert on_pallet
    assert len(scene.find_by_type("repair_toolbox")) >= 2
