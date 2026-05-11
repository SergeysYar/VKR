from __future__ import annotations

from collections import defaultdict
from math import asin, atan2, degrees

from synthetic_factory.generators import FactoryGenerator


def _build_params() -> dict[str, object]:
    return {
        "factory_width": 24.0,
        "factory_depth": 18.0,
        "number_of_rooms": 1,
        "room_size_range": (12.0, 12.0),
        "corridor_width": 1.0,
        "layout_strategy": "grid",
        "noise": {"seed": 42},
        "biomes": {
            "enabled": True,
            "workshop_biome": "control",
            "cycle_order": ["control"],
        },
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {
            "enabled": True,
            "control": {
                "seed": 101,
                "random_variation": False,
                "pattern": "linear_control_room",
                "auto_fix": True,
                "density_profile": "medium",
                "panel_layout": "wall",
                "rack_layout": "walls",
                "room": {
                    "width": [12.0, 12.0],
                    "depth": [12.0, 12.0],
                    "height": [3.4, 3.4],
                },
                "desks": {
                    "rows": [2, 2],
                    "cols": [4, 4],
                    "spacing_x": [1.6, 1.6],
                    "spacing_y": [1.9, 1.9],
                },
                "monitors": {
                    "per_desk": [1, 1],
                },
                "panels": {
                    "type": ["flat"],
                    "height": [1.8, 1.8],
                },
                "racks": {
                    "count": [4, 4],
                },
                "lighting": {
                    "grid_density": [1.0, 1.0],
                },
                "wall_margin": 0.8,
                "desk_width": 1.4,
                "desk_depth": 0.8,
                "desk_height": 0.92,
                "seat_width": 0.55,
                "seat_depth": 0.55,
                "seat_height": 0.95,
                "min_desk_clearance": 0.9,
                "main_aisle_width": 1.35,
                "side_aisle_width": 0.9,
                "front_clearance": 1.0,
                "panel_width": 1.5,
                "panel_height": 1.6,
                "panel_depth": 0.22,
                "panel_zone_depth": 2.2,
                "panel_tilt_angle": 24.0,
                "monitor_width": 0.62,
                "monitor_height": 0.36,
                "monitor_thickness": 0.06,
                "sight_corridor_width": 0.9,
                "visibility_blocker_height": 1.1,
                "monitor_view_angle_deg": 40.0,
                "rack_width": 0.52,
                "rack_depth": 0.48,
                "rack_height": 1.45,
                "cable_tray_height": 0.16,
                "cable_tray_elevation_ratio": 0.82,
                "ceiling_min": 2.8,
                "ceiling_max": 4.2,
                "ceiling_thickness": 0.06,
                "zone_thickness": 0.02,
                "alignment_grid_step": 0.1,
                "ergonomic_desk_height_min": 0.72,
                "ergonomic_desk_height_max": 1.15,
                "ergonomic_seat_height_min": 0.38,
                "ergonomic_seat_height_max": 0.58,
                "ergonomic_monitor_center_min": 0.22,
                "ergonomic_monitor_center_max": 0.65,
            },
        },
        "seed": 42,
    }


def test_control_biome_generates_grid_oriented_workstations_and_zones() -> None:
    params = _build_params()
    scene = FactoryGenerator(params).generate_scene()

    consoles = scene.find_by_type("console")
    desk_monitors = scene.find_by_type("desk_monitor")
    control_panels = scene.find_by_type("control_panel")
    seats = scene.find_by_type("operator_seat")
    assert len(consoles) >= 2
    assert len(seats) >= 1
    assert len(desk_monitors) == len(consoles)
    assert len(control_panels) >= 2
    assert len(scene.find_by_type("cable_tray")) >= 1
    assert len(scene.find_by_type("panel_zone")) >= 1
    assert len(scene.find_by_type("operator_zone")) >= 1
    assert len(scene.find_by_type("aisle_zone")) >= 1

    rotations = {obj.transform.rotation for obj in consoles}
    assert len(rotations) == 1
    monitor_rotations = {obj.transform.rotation for obj in desk_monitors}
    assert monitor_rotations == rotations
    seat_rotations = {obj.transform.rotation for obj in seats}
    assert seat_rotations == rotations
    panel_rotations = {obj.transform.rotation for obj in control_panels}
    assert panel_rotations == rotations

    desk_width = float(params["machinery"]["control"]["desk_width"])  # type: ignore[index]
    min_clearance = float(params["machinery"]["control"]["min_desk_clearance"])  # type: ignore[index]
    rows: dict[float, list[float]] = defaultdict(list)
    for console in consoles:
        rows[round(console.transform.position[1], 3)].append(console.transform.position[0])

    for row_x in rows.values():
        row_sorted = sorted(row_x)
        for idx in range(len(row_sorted) - 1):
            assert row_sorted[idx + 1] - row_sorted[idx] + 1e-6 >= desk_width + min_clearance


def test_control_biome_preserves_central_aisle_and_visibility() -> None:
    params = _build_params()
    scene = FactoryGenerator(params).generate_scene()

    consoles = scene.find_by_type("console")
    monitor_wall = scene.find_by_type("monitor_wall")[0]
    racks = scene.find_by_type("control_rack")

    main_aisle_width = float(params["machinery"]["control"]["main_aisle_width"])  # type: ignore[index]
    desk_width = float(params["machinery"]["control"]["desk_width"])  # type: ignore[index]
    sight_corridor_width = float(params["machinery"]["control"]["sight_corridor_width"])  # type: ignore[index]
    rack_width = float(params["machinery"]["control"]["rack_width"])  # type: ignore[index]
    rack_depth = float(params["machinery"]["control"]["rack_depth"])  # type: ignore[index]

    for console in consoles:
        assert abs(console.transform.position[0]) >= (main_aisle_width / 2.0 + desk_width / 2.0) - 1e-6
        assert console.transform.position[1] < monitor_wall.transform.position[1]

    for rack in racks:
        for console in consoles:
            cx, cy, _ = console.transform.position
            rx, ry, _ = rack.transform.position
            y_min = min(cy, monitor_wall.transform.position[1])
            y_max = max(cy, monitor_wall.transform.position[1])
            y_overlap = (ry + rack_depth / 2.0) >= y_min and (ry - rack_depth / 2.0) <= y_max
            x_overlap = abs(rx - cx) <= (sight_corridor_width / 2.0 + rack_width / 2.0)
            assert not (y_overlap and x_overlap)


def test_control_biome_uses_low_visual_ceiling() -> None:
    params = _build_params()
    generator = FactoryGenerator(params)
    layout = generator.generate_layout()
    room_height = generator._effective_room_height(layout[0])  # noqa: SLF001

    scene = generator.generate_scene()
    ceiling = scene.find_by_type("suspended_ceiling")
    assert len(ceiling) == 1
    assert ceiling[0].transform.position[2] < room_height * 0.8


def test_control_biome_supports_semicircle_panel_layout() -> None:
    params = _build_params()
    params["machinery"]["control"]["panel_layout"] = "semicircle"  # type: ignore[index]

    scene = FactoryGenerator(params).generate_scene()
    panels = scene.find_by_type("control_panel")
    assert len(panels) >= 3

    x_values = [panel.transform.position[0] for panel in panels]
    y_values = [panel.transform.position[1] for panel in panels]
    assert max(x_values) - min(x_values) > 0.5
    assert len({round(y, 3) for y in y_values}) > 1


def test_control_biome_supports_seeded_sampling() -> None:
    params = _build_params()
    params["machinery"]["control"]["random_variation"] = True  # type: ignore[index]
    params["machinery"]["control"]["seed"] = 77  # type: ignore[index]
    params["machinery"]["control"]["desks"] = {  # type: ignore[index]
        "rows": [1, 4],
        "cols": [2, 8],
        "spacing_x": [1.2, 1.9],
        "spacing_y": [1.5, 2.6],
    }
    params["machinery"]["control"]["monitors"] = {"per_desk": [1, 3]}  # type: ignore[index]
    params["machinery"]["control"]["panels"] = {  # type: ignore[index]
        "type": ["flat", "curved"],
        "height": [1.5, 2.6],
    }

    scene_a = FactoryGenerator(params).generate_scene()
    scene_b = FactoryGenerator(params).generate_scene()

    consoles_a = [(o.transform.position, o.transform.rotation) for o in scene_a.find_by_type("console")]
    consoles_b = [(o.transform.position, o.transform.rotation) for o in scene_b.find_by_type("console")]
    panels_a = [o.transform.position for o in scene_a.find_by_type("control_panel")]
    panels_b = [o.transform.position for o in scene_b.find_by_type("control_panel")]

    assert consoles_a == consoles_b
    assert panels_a == panels_b


def test_control_biome_respects_monitor_count_per_desk() -> None:
    params = _build_params()
    params["machinery"]["control"]["monitors"] = {"per_desk": [3, 3]}  # type: ignore[index]
    params["machinery"]["control"]["random_variation"] = False  # type: ignore[index]

    scene = FactoryGenerator(params).generate_scene()
    console_count = len(scene.find_by_type("console"))
    monitor_count = len(scene.find_by_type("desk_monitor"))
    assert monitor_count == console_count * 3


def test_control_biome_enforces_room_spacing_dependencies() -> None:
    params = _build_params()
    params["factory_width"] = 12.0
    params["factory_depth"] = 10.0
    params["room_size_range"] = (8.0, 8.0)
    params["machinery"]["control"]["random_variation"] = False  # type: ignore[index]
    params["machinery"]["control"]["desks"] = {  # type: ignore[index]
        "rows": [5, 5],
        "cols": [10, 10],
        "spacing_x": [2.0, 2.0],
        "spacing_y": [3.0, 3.0],
    }

    scene = FactoryGenerator(params).generate_scene()
    consoles = scene.find_by_type("console")

    cfg = params["machinery"]["control"]  # type: ignore[index]
    wall_margin = float(cfg["wall_margin"])  # type: ignore[index]
    side_aisle = float(cfg["side_aisle_width"])  # type: ignore[index]
    spacing_x = 2.0
    spacing_y = 3.0

    width_limit = max(params["room_size_range"][1] - 2.0 * (wall_margin + side_aisle), spacing_x)  # type: ignore[index]
    depth_limit = max(params["room_size_range"][1] - 2.0 * wall_margin, spacing_y)  # type: ignore[index]
    max_cols = max(1, int(width_limit // spacing_x))
    max_rows = max(1, int(depth_limit // spacing_y))

    assert len(consoles) <= max_cols * max_rows


def test_control_biome_pattern_switches_layout_algorithm() -> None:
    linear_params = _build_params()
    linear_params["machinery"]["control"]["pattern"] = "linear_control_room"  # type: ignore[index]
    semi_params = _build_params()
    semi_params["machinery"]["control"]["pattern"] = "Semi-Circular Control"  # type: ignore[index]
    cluster_params = _build_params()
    cluster_params["machinery"]["control"]["pattern"] = "clustered_workstations"  # type: ignore[index]

    linear_scene = FactoryGenerator(linear_params).generate_scene()
    semi_scene = FactoryGenerator(semi_params).generate_scene()
    cluster_scene = FactoryGenerator(cluster_params).generate_scene()

    linear_panels = linear_scene.find_by_type("control_panel")
    semi_panels = semi_scene.find_by_type("control_panel")
    cluster_consoles = cluster_scene.find_by_type("console")

    assert len({round(obj.transform.position[1], 3) for obj in linear_panels}) == 1
    assert len({round(obj.transform.position[1], 3) for obj in semi_panels}) > 1

    cluster_x = {round(obj.transform.position[0], 2) for obj in cluster_consoles}
    assert len(cluster_x) >= 2
    assert len(cluster_consoles) >= 2


def test_control_biome_pattern_changes_density() -> None:
    linear_params = _build_params()
    linear_params["machinery"]["control"]["pattern"] = "linear_control_room"  # type: ignore[index]
    minimal_params = _build_params()
    minimal_params["machinery"]["control"]["pattern"] = "minimal_control"  # type: ignore[index]
    dense_params = _build_params()
    dense_params["machinery"]["control"]["pattern"] = "high density"  # type: ignore[index]

    linear_scene = FactoryGenerator(linear_params).generate_scene()
    minimal_scene = FactoryGenerator(minimal_params).generate_scene()
    dense_scene = FactoryGenerator(dense_params).generate_scene()

    linear_desks = len(linear_scene.find_by_type("console"))
    minimal_desks = len(minimal_scene.find_by_type("console"))
    dense_desks = len(dense_scene.find_by_type("console"))

    assert minimal_desks < linear_desks
    assert dense_desks > linear_desks
    assert len(minimal_scene.find_by_type("control_rack")) <= len(linear_scene.find_by_type("control_rack"))
    assert len(dense_scene.find_by_type("desk_monitor")) >= len(linear_scene.find_by_type("desk_monitor"))


def test_control_biome_view_direction_aligns_all_workstations() -> None:
    params = _build_params()
    params["machinery"]["control"]["pattern"] = "semi_circular_control"  # type: ignore[index]

    scene = FactoryGenerator(params).generate_scene()
    panels = scene.find_by_type("control_panel")
    consoles = scene.find_by_type("console")
    monitors = scene.find_by_type("desk_monitor")
    chairs = scene.find_by_type("operator_seat")
    operator_zone = scene.find_by_type("operator_zone")[0]

    assert panels
    assert consoles

    panel_center_x = sum(obj.transform.position[0] for obj in panels) / len(panels)
    panel_center_y = sum(obj.transform.position[1] for obj in panels) / len(panels)
    _, operator_center_y, _ = operator_zone.transform.position
    vx = panel_center_x
    vy = panel_center_y - operator_center_y
    expected_yaw = degrees(atan2(-vx, vy))

    for desk in consoles:
        assert abs(desk.transform.rotation[2] - expected_yaw) <= 1e-6
    for monitor in monitors:
        assert abs(monitor.transform.rotation[2] - expected_yaw) <= 1e-6
    for chair in chairs:
        assert abs(chair.transform.rotation[2] - expected_yaw) <= 1e-6


def test_control_biome_alignment_and_ergonomics_rules_auto_fix() -> None:
    params = _build_params()
    params["machinery"]["control"]["auto_fix"] = True  # type: ignore[index]
    params["machinery"]["control"]["alignment_grid_step"] = 0.25  # type: ignore[index]
    params["machinery"]["control"]["desk_height"] = 1.8  # type: ignore[index]
    params["machinery"]["control"]["ergonomic_desk_height_max"] = 1.0  # type: ignore[index]

    scene = FactoryGenerator(params).generate_scene()
    consoles = scene.find_by_type("console")
    assert consoles

    for desk in consoles:
        x, y, z = desk.transform.position
        assert abs((x / 0.25) - round(x / 0.25)) < 1e-6
        assert abs((y / 0.25) - round(y / 0.25)) < 1e-6
        assert z <= 0.55


def test_control_biome_monitor_visibility_rule_auto_fix() -> None:
    params = _build_params()
    params["machinery"]["control"]["auto_fix"] = True  # type: ignore[index]
    params["machinery"]["control"]["monitor_view_angle_deg"] = 20.0  # type: ignore[index]
    params["machinery"]["control"]["monitors"] = {"per_desk": [4, 4]}  # type: ignore[index]
    params["machinery"]["control"]["desk_width"] = 2.0  # type: ignore[index]

    scene = FactoryGenerator(params).generate_scene()
    monitors = scene.find_by_type("desk_monitor")
    chairs = scene.find_by_type("operator_seat")
    assert monitors
    assert chairs

    max_angle = 20.0
    for monitor in monitors:
        mx, my, _ = monitor.transform.position
        candidates = [
            chair
            for chair in chairs
            if abs(chair.transform.position[0] - mx) <= 0.35 and chair.transform.position[1] < my
        ]
        if not candidates:
            candidates = [chair for chair in chairs if chair.transform.position[1] < my]
        nearest_chair = min(
            candidates,
            key=lambda chair: abs(chair.transform.position[0] - mx) + abs(chair.transform.position[1] - my),
        )
        cx, cy, _ = nearest_chair.transform.position
        dy = my - cy
        assert dy > 0.05
        dx = abs(mx - cx)
        ratio = min(dx / max(dy, 1e-6), 1.0)
        angle = degrees(asin(ratio))
        assert angle <= max_angle + 1e-6


def test_control_biome_secondary_elements_are_generated() -> None:
    params = _build_params()
    params["machinery"]["control"]["secondary"] = {  # type: ignore[index]
        "enabled": True,
        "small_screen_per_desk": 1,
    }

    scene = FactoryGenerator(params).generate_scene()
    consoles = scene.find_by_type("console")
    keyboards = scene.find_by_type("keyboard")
    console_blocks = scene.find_by_type("console_block")
    small_screens = scene.find_by_type("small_screen")
    cable_bundles = scene.find_by_type("desk_cable_bundle")
    wall_displays = scene.find_by_type("wall_display")
    ups_units = scene.find_by_type("ups_unit")

    assert consoles
    assert len(keyboards) == len(consoles)
    assert len(console_blocks) == len(consoles)
    assert len(small_screens) == len(consoles)
    assert len(cable_bundles) == len(consoles)
    assert len(ups_units) == len(consoles)
    assert len(wall_displays) >= 2
