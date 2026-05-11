from __future__ import annotations

import math
from pathlib import Path

from synthetic_factory.parametric.parameters import ParameterSet
from synthetic_factory.parametric.primitives import create_box, create_floor
from synthetic_factory.pipeline import build_default_scene_pipeline
from synthetic_factory.scene.scene_graph import Scene, SceneObject, Transform
from synthetic_factory.sensing import LidarSurveyGenerator


def _simple_lidar_scene() -> Scene:
    scene = Scene()
    room = SceneObject(
        id="room_test_1",
        type="room_test",
        transform=Transform(position=(0.0, 0.0, 0.0)),
    )
    room.add_child(
        SceneObject(
            id="room_floor",
            type="floor",
            mesh=create_floor(width=12.0, depth=10.0),
        )
    )
    room.add_child(
        SceneObject(
            id="room_ceiling",
            type="ceiling",
            transform=Transform(position=(0.0, 0.0, 4.0)),
            mesh=create_floor(width=12.0, depth=10.0),
        )
    )
    room.add_child(
        SceneObject(
            id="pipe_segment_1",
            type="pipe_segment",
            transform=Transform(position=(-2.0, 1.2, 1.0)),
            mesh=create_box(width=1.8, height=2.0, depth=0.4),
        )
    )
    room.add_child(
        SceneObject(
            id="wire_tray_1",
            type="infra_cable_tray_branch",
            transform=Transform(position=(2.4, -1.0, 1.6)),
            mesh=create_box(width=2.0, height=0.24, depth=0.4),
        )
    )
    room.add_child(
        SceneObject(
            id="machine_1",
            type="machine",
            transform=Transform(position=(0.0, 2.6, 1.1)),
            mesh=create_box(width=1.2, height=2.2, depth=1.0),
        )
    )
    scene.add_object(room)
    return scene


def _exterior_only_lidar_scene() -> Scene:
    scene = Scene()
    factory_room = SceneObject(
        id="__factory__",
        type="room_factory",
        transform=Transform(position=(0.0, 0.0, 0.0)),
    )
    factory_room.add_child(
        SceneObject(
            id="ext_wall_north",
            type="exterior_wall",
            transform=Transform(position=(0.0, 6.0, 2.5)),
            mesh=create_box(width=12.0, height=5.0, depth=0.3),
        )
    )
    factory_room.add_child(
        SceneObject(
            id="ext_wall_south",
            type="exterior_wall",
            transform=Transform(position=(0.0, -6.0, 2.5)),
            mesh=create_box(width=12.0, height=5.0, depth=0.3),
        )
    )
    factory_room.add_child(
        SceneObject(
            id="ext_wall_west",
            type="exterior_wall",
            transform=Transform(position=(-6.0, 0.0, 2.5)),
            mesh=create_box(width=0.3, height=5.0, depth=12.0),
        )
    )
    factory_room.add_child(
        SceneObject(
            id="ext_wall_east",
            type="exterior_wall",
            transform=Transform(position=(6.0, 0.0, 2.5)),
            mesh=create_box(width=0.3, height=5.0, depth=12.0),
        )
    )
    factory_room.add_child(
        SceneObject(
            id="ext_roof",
            type="exterior_roof",
            transform=Transform(position=(0.0, 0.0, 5.1)),
            mesh=create_box(width=12.0, height=0.3, depth=12.0),
        )
    )
    scene.add_object(factory_room)
    return scene


def _exterior_semantic_scene() -> Scene:
    scene = Scene()
    factory_room = SceneObject(
        id="__factory__",
        type="room_factory",
        transform=Transform(position=(0.0, 0.0, 0.0)),
    )
    factory_room.add_child(
        SceneObject(
            id="ext_roof_sem",
            type="exterior_roof",
            transform=Transform(position=(0.0, 0.0, 5.6)),
            mesh=create_box(width=18.0, height=0.35, depth=18.0),
        )
    )
    factory_room.add_child(
        SceneObject(
            id="ext_window_sem",
            type="exterior_window",
            transform=Transform(position=(0.0, 7.0, 2.2)),
            mesh=create_box(width=5.0, height=2.0, depth=0.4),
        )
    )
    factory_room.add_child(
        SceneObject(
            id="ext_gate_sem",
            type="exterior_gate",
            transform=Transform(position=(0.0, -7.0, 2.2)),
            mesh=create_box(width=6.0, height=3.4, depth=0.4),
        )
    )
    factory_room.add_child(
        SceneObject(
            id="site_road_sem",
            type="site_road",
            transform=Transform(position=(0.0, -9.0, 0.12)),
            mesh=create_box(width=16.0, height=0.24, depth=4.0),
        )
    )
    scene.add_object(factory_room)
    return scene


def test_lidar_generator_returns_stations_and_labeled_circle_points() -> None:
    scene = _simple_lidar_scene()
    generator = LidarSurveyGenerator(
        {
            "scan_range": 16.0,
            "station_spacing": 3.0,
            "angular_resolution_deg": 5.0,
            "sensor_height": 1.6,
        }
    )

    stations = generator.plan_stations(scene)
    circles = generator.generate_scans(scene, stations)

    assert stations
    assert circles
    points = [point for circle in circles for point in circle.points]
    assert points
    assert all(point.label >= 0 for point in points)
    assert any(point.class_name in {"pipe", "wire", "machine"} for point in points)


def test_lidar_exterior_density_factor_reduces_exterior_points() -> None:
    scene = _exterior_only_lidar_scene()
    stations = [(-0.2, 0.0, 1.6)]

    dense_generator = LidarSurveyGenerator(
        {
            "scan_range": 20.0,
            "angular_resolution_deg": 2.0,
            "vertical_resolution_deg": 2.0,
            "vertical_fov_up_deg": 35.0,
            "vertical_fov_down_deg": 55.0,
            "sensor_height": 1.6,
            "min_range": 0.35,
            "include_structural": True,
            "point_multiplier": 1,
            "points_per_station": 6000,
            "exterior_point_density_factor": 1.0,
            "factory_room_id": "__factory__",
            "include_factory_room": True,
        }
    )
    sparse_generator = LidarSurveyGenerator(
        {
            "scan_range": 20.0,
            "angular_resolution_deg": 2.0,
            "vertical_resolution_deg": 2.0,
            "vertical_fov_up_deg": 35.0,
            "vertical_fov_down_deg": 55.0,
            "sensor_height": 1.6,
            "min_range": 0.35,
            "include_structural": True,
            "point_multiplier": 1,
            "points_per_station": 6000,
            "exterior_point_density_factor": 0.25,
            "factory_room_id": "__factory__",
            "include_factory_room": True,
        }
    )

    dense_stations = dense_generator.stations_from_coordinates(room_id="__factory__", coordinates=stations)
    sparse_stations = sparse_generator.stations_from_coordinates(room_id="__factory__", coordinates=stations)

    dense_points = [
        point
        for circle in dense_generator.generate_scans(scene, dense_stations)
        for point in circle.points
        if point.domain == "exterior"
    ]
    sparse_points = [
        point
        for circle in sparse_generator.generate_scans(scene, sparse_stations)
        for point in circle.points
        if point.domain == "exterior"
    ]

    assert dense_points
    assert sparse_points
    assert len(sparse_points) < len(dense_points)
    assert len(sparse_points) <= int(len(dense_points) * 0.35)
    assert len(sparse_points) >= max(1, int(len(dense_points) * 0.15))


def test_lidar_exterior_types_are_mapped_to_semantic_classes() -> None:
    scene = _exterior_semantic_scene()
    generator = LidarSurveyGenerator(
        {
            "scan_range": 24.0,
            "angular_resolution_deg": 2.4,
            "vertical_resolution_deg": 2.0,
            "vertical_fov_up_deg": 38.0,
            "vertical_fov_down_deg": 60.0,
            "sensor_height": 1.6,
            "min_range": 0.35,
            "include_structural": True,
            "point_multiplier": 1,
            "points_per_station": 5000,
            "factory_room_id": "__factory__",
            "include_factory_room": True,
        }
    )

    stations = generator.stations_from_coordinates(
        room_id="__factory__",
        coordinates=[(0.0, 0.0, 1.6)],
    )
    circles = generator.generate_scans(scene, stations)
    points = [point for circle in circles for point in circle.points if point.domain == "exterior"]

    assert points
    classes = {point.class_name for point in points}
    assert {"roof", "window", "gate", "terrain"} <= classes


def test_lidar_core_infrastructure_labels_are_stable() -> None:
    generator = LidarSurveyGenerator()
    labels = generator.label_map

    assert labels["pipe"] == 1
    assert labels["wire"] == 2
    assert labels["wall"] == 3
    assert labels["floor"] == 4
    assert labels["ceiling"] == 5
    assert labels["infrastructure"] == 12

    class_name_pipe, label_pipe = generator._class_for_object_type("infra_pipe_casing")  # noqa: SLF001
    class_name_wire, label_wire = generator._class_for_object_type("infra_cable_bundle")  # noqa: SLF001
    assert class_name_pipe == "pipe"
    assert class_name_wire == "wire"
    assert label_pipe == labels["pipe"]
    assert label_wire == labels["wire"]


def test_blind_spot_is_present_per_station_and_covered_in_stitched_cloud() -> None:
    scene = _simple_lidar_scene()
    blind_spot_radius = 0.75
    generator = LidarSurveyGenerator(
        {
            "scan_range": 16.0,
            "angular_resolution_deg": 4.0,
            "vertical_resolution_deg": 4.0,
            "vertical_fov_up_deg": 35.0,
            "vertical_fov_down_deg": 65.0,
            "sensor_height": 1.6,
            "min_range": 0.35,
            "blind_spot_radius": blind_spot_radius,
            "include_structural": True,
        }
    )
    stations = generator.stations_from_coordinates(
        room_id="room_test_1",
        coordinates=[(-1.8, 0.0, 1.6), (1.8, 0.0, 1.6)],
    )
    circles = generator.generate_scans(scene, stations)
    all_points = [point for circle in circles for point in circle.points]
    assert all_points

    for circle in circles:
        cx, cy, cz = circle.center
        nearby = [
            point
            for point in circle.points
            if point.z <= cz + 1e-6
            and math.hypot(point.x - cx, point.y - cy) < blind_spot_radius * 0.95
        ]
        assert not nearby

    for station in stations:
        covered_by_other = any(
            point.station_id != station.id
            and point.z < station.z
            and math.hypot(point.x - station.x, point.y - station.y) < blind_spot_radius * 0.8
            for point in all_points
        )
        assert covered_by_other


def test_pipeline_generates_lidar_dataset_after_obj_export(tmp_path: Path) -> None:
    seed = 11
    output_obj = tmp_path / "factory.obj"
    output_lidar = tmp_path / "factory_lidar.json"
    output_lidar_dir = tmp_path / "factory_lidar"
    output_stitched = output_lidar_dir / "factory_stitched.ply"

    parameter_set = ParameterSet(
        {
            "factory_width": 28.0,
            "factory_depth": 22.0,
            "number_of_rooms": 1,
            "corridor_width": 1.0,
            "room_height": 4.0,
            "room_size_min": 14.0,
            "room_size_max": 14.0,
        },
        seed=seed,
    )
    base_parameters = {
        "factory_width": 28.0,
        "factory_depth": 22.0,
        "number_of_rooms": 1,
        "corridor_width": 1.0,
        "room_height": 4.0,
        "layout_strategy": "grid",
        "room_size_min": 14.0,
        "room_size_max": 14.0,
        "room_count_range": [1, 1],
        "noise": {"seed": seed},
        "biomes": {
            "enabled": True,
            "workshop_biome": "workshop",
            "cycle_order": ["workshop"],
        },
        "columns": {"enabled": False},
        "beams": {"enabled": False},
        "machinery": {
            "enabled": True,
            "conveyors_per_room": 2,
            "machines_per_room": 2,
            "auxiliary": {"enabled": False},
            "infrastructure": {"enabled": False},
        },
        "lidar": {
            "enabled": True,
            "output_path": str(output_lidar),
            "output_dir": str(output_lidar_dir),
            "stitched_output_path": str(output_stitched),
            "scan_range": 14.0,
            "angular_resolution_deg": 10.0,
            "station_spacing": 3.0,
            "sensor_height": 1.6,
        },
        "seed": seed,
    }

    pipeline = build_default_scene_pipeline(
        parameter_set=parameter_set,
        export_path=str(output_obj),
        seed=seed,
        base_parameters=base_parameters,
    )
    result = pipeline.run()

    assert output_obj.exists()
    assert output_lidar.exists()
    assert output_lidar_dir.exists()
    assert output_stitched.exists()
    assert result.lidar_station_records
    assert result.lidar_circle_records
    assert result.lidar_labels
    assert result.lidar_circle_file_paths
    assert result.lidar_stitched_cloud_path == str(output_stitched)
    assert len(result.lidar_circle_file_paths) == len(result.lidar_circle_records)
