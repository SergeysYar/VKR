from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from .obj_loader import load_scene_from_obj
from .sensing import LidarPoint, LidarSurveyGenerator, LidarSurveyResult


def _to_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


@dataclass(frozen=True)
class ScannerRunResult:
    input_obj: str
    output_dir: str
    metadata_json: str | None
    circle_files: list[str]
    stitched_combined: str | None
    stitched_interior: str | None
    stitched_exterior: str | None
    station_count: int
    circle_count: int
    point_count: int
    label_map: dict[str, int]
    run_summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_obj": self.input_obj,
            "output_dir": self.output_dir,
            "metadata_json": self.metadata_json,
            "circle_files": self.circle_files,
            "stitched_combined": self.stitched_combined,
            "stitched_interior": self.stitched_interior,
            "stitched_exterior": self.stitched_exterior,
            "station_count": self.station_count,
            "circle_count": self.circle_count,
            "point_count": self.point_count,
            "label_map": dict(sorted(self.label_map.items(), key=lambda item: item[1])),
            "run_summary_path": self.run_summary_path,
        }


def run_scan_pipeline(config: Mapping[str, Any]) -> ScannerRunResult:
    paths_cfg = dict(config.get("paths", {}))
    output_cfg = dict(config.get("output", {}))
    lidar_cfg = dict(config.get("lidar", {}))

    input_obj = Path(str(paths_cfg.get("input_obj", "out/factory.obj"))).resolve()
    scene = load_scene_from_obj(str(input_obj))

    run_dir = _resolve_output_dir(paths_cfg, output_cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    streaming_enabled = _to_bool(lidar_cfg.get("streaming_enabled", True), default=True)
    if streaming_enabled and int(lidar_cfg.get("points_per_station", 0)) <= 0:
        # Unlimited points per station can still OOM inside one station before flush.
        # Keep it very high by default, but finite for stability.
        lidar_cfg["points_per_station"] = int(lidar_cfg.get("streaming_station_point_cap", 250000))

    survey_generator = LidarSurveyGenerator(lidar_cfg)
    stations = survey_generator.plan_stations(scene)
    streaming_batch_size = max(1, int(lidar_cfg.get("streaming_batch_size", 12)))

    if not streaming_enabled:
        circles = survey_generator.generate_scans(scene, stations)
        survey_result = LidarSurveyResult(
            stations=stations,
            circles=circles,
            label_map=survey_generator.label_map,
            source_obj_path=str(input_obj),
        )

        circle_files: list[str] = []
        if _to_bool(output_cfg.get("export_individual_circles", True), default=True):
            circle_dir = run_dir / "circles"
            prefix = str(paths_cfg.get("circle_file_prefix", "scan_circle"))
            circle_files = survey_result.write_circle_ply_files(
                output_dir=str(circle_dir),
                file_prefix=prefix,
            )

        stitched_combined: str | None = None
        stitched_interior: str | None = None
        stitched_exterior: str | None = None
        if _to_bool(output_cfg.get("export_stitched_cloud", True), default=True):
            stitched_name = str(paths_cfg.get("stitched_filename", "factory_stitched.ply"))
            stitched_target = run_dir / "combined" / stitched_name
            stitched_paths = survey_result.write_split_stitched_ply(
                output_root_dir=str(run_dir),
                combined_path=str(stitched_target),
                blind_spot_radius=float(lidar_cfg.get("blind_spot_radius", 0.0)),
                fill_blind_spots=_to_bool(output_cfg.get("stitched_fill_blind_spots", True), default=True),
            )
            stitched_combined = stitched_paths["combined"]
            stitched_interior = stitched_paths["interior"]
            stitched_exterior = stitched_paths["exterior"]

        metadata_json: str | None = None
        if _to_bool(output_cfg.get("export_metadata_json", True), default=True):
            metadata_name = str(paths_cfg.get("metadata_filename", "scan_dataset.json"))
            metadata_json = survey_result.write_json(str(run_dir / metadata_name))

        point_count = sum(len(circle.points) for circle in circles)
        summary = ScannerRunResult(
            input_obj=str(input_obj),
            output_dir=str(run_dir),
            metadata_json=metadata_json,
            circle_files=circle_files,
            stitched_combined=stitched_combined,
            stitched_interior=stitched_interior,
            stitched_exterior=stitched_exterior,
            station_count=len(stations),
            circle_count=len(circles),
            point_count=point_count,
            label_map=dict(survey_result.label_map),
            run_summary_path=str(run_dir / "run_summary.json"),
        )
        Path(summary.run_summary_path).write_text(
            json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary

    # Streaming mode: process station batches and persist immediately to avoid RAM growth.
    label_map = dict(survey_generator.label_map)
    station_lookup = {station.id: idx for idx, station in enumerate(stations, start=1)}
    circle_files: list[str] = []
    circle_summaries: list[dict[str, Any]] = []
    point_count = 0

    export_circles = _to_bool(output_cfg.get("export_individual_circles", True), default=True)
    export_stitched = _to_bool(output_cfg.get("export_stitched_cloud", True), default=True)
    export_metadata = _to_bool(output_cfg.get("export_metadata_json", True), default=True)

    circle_dir = run_dir / "circles"
    prefix = str(paths_cfg.get("circle_file_prefix", "scan_circle"))
    if export_circles:
        circle_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = run_dir / "_tmp_stream"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    combined_tmp = tmp_dir / "combined_points.tmp"
    interior_tmp = tmp_dir / "interior_points.tmp"
    exterior_tmp = tmp_dir / "exterior_points.tmp"
    combined_handle = combined_tmp.open("w", encoding="utf-8", newline="\n")
    interior_handle = interior_tmp.open("w", encoding="utf-8", newline="\n")
    exterior_handle = exterior_tmp.open("w", encoding="utf-8", newline="\n")
    combined_count = 0
    interior_count = 0
    exterior_count = 0

    circle_index = 0
    for start in range(0, len(stations), streaming_batch_size):
        chunk = stations[start : start + streaming_batch_size]
        chunk_circles = survey_generator.generate_scans(scene, chunk)
        for circle in chunk_circles:
            circle_index += 1
            if export_circles:
                room_token = _sanitize_filename_token(circle.room_id)
                station_token = _sanitize_filename_token(circle.station_id)
                filename = f"{_sanitize_filename_token(prefix)}_{circle_index:04d}_{room_token}_{station_token}.ply"
                circle_path = _next_available_path(circle_dir / filename)
                _write_points_ply(
                    path=circle_path,
                    points=circle.points,
                    label_map=label_map,
                    station_lookup=station_lookup,
                    circle_index=circle_index,
                )
                circle_files.append(str(circle_path))

            local_point_count = len(circle.points)
            point_count += local_point_count
            circle_summaries.append(
                {
                    "station_id": circle.station_id,
                    "room_id": circle.room_id,
                    "center": list(circle.center),
                    "radius": circle.radius,
                    "point_count": local_point_count,
                }
            )

            if export_stitched:
                for point in circle.points:
                    line = _point_to_ply_vertex_line(
                        point=point,
                        label_map=label_map,
                        station_lookup=station_lookup,
                        circle_index=-1,
                    )
                    combined_handle.write(line)
                    combined_count += 1
                    if point.domain == "interior":
                        interior_handle.write(line.replace(" -1 ", " -2 ", 1))
                        interior_count += 1
                    elif point.domain == "exterior":
                        exterior_handle.write(line.replace(" -1 ", " -3 ", 1))
                        exterior_count += 1

    stitched_combined: str | None = None
    stitched_interior: str | None = None
    stitched_exterior: str | None = None
    combined_handle.close()
    interior_handle.close()
    exterior_handle.close()
    if export_stitched:
        stitched_name = str(paths_cfg.get("stitched_filename", "factory_stitched.ply"))
        stitched_combined_path = _next_available_path(run_dir / "combined" / stitched_name)
        stitched_interior_path = _next_available_path(run_dir / "interior" / "factory_stitched_interior.ply")
        stitched_exterior_path = _next_available_path(run_dir / "exterior" / "factory_stitched_exterior.ply")
        _write_ply_from_temp(stitched_combined_path, combined_tmp, combined_count, label_map)
        _write_ply_from_temp(stitched_interior_path, interior_tmp, interior_count, label_map)
        _write_ply_from_temp(stitched_exterior_path, exterior_tmp, exterior_count, label_map)
        stitched_combined = str(stitched_combined_path)
        stitched_interior = str(stitched_interior_path)
        stitched_exterior = str(stitched_exterior_path)
    # best effort cleanup
    for tmp_path in (combined_tmp, interior_tmp, exterior_tmp):
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    metadata_json: str | None = None
    if export_metadata:
        metadata_name = str(paths_cfg.get("metadata_filename", "scan_dataset.json"))
        metadata_path = run_dir / metadata_name
        metadata_payload = {
            "source_obj_path": str(input_obj),
            "station_count": len(stations),
            "circle_count": len(circle_summaries),
            "labels": dict(sorted(label_map.items(), key=lambda item: item[1])),
            "stations": [station.to_dict() for station in stations],
            "circles": circle_summaries,
            "streaming_enabled": True,
            "streaming_batch_size": streaming_batch_size,
        }
        metadata_path.write_text(
            json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        metadata_json = str(metadata_path)

    summary = ScannerRunResult(
        input_obj=str(input_obj),
        output_dir=str(run_dir),
        metadata_json=metadata_json,
        circle_files=circle_files,
        stitched_combined=stitched_combined,
        stitched_interior=stitched_interior,
        stitched_exterior=stitched_exterior,
        station_count=len(stations),
        circle_count=len(circle_summaries),
        point_count=point_count,
        label_map=label_map,
        run_summary_path=str(run_dir / "run_summary.json"),
    )
    Path(summary.run_summary_path).write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = "".join(path.suffixes)
    stem = path.name[:-len(suffix)] if suffix else path.name
    index = 1
    while True:
        candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _sanitize_filename_token(value: str) -> str:
    cleaned = "".join(char if (char.isalnum() or char in {"_", "-"}) else "_" for char in value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def _label_to_rgb(label: int) -> tuple[int, int, int]:
    palette: tuple[tuple[int, int, int], ...] = (
        (160, 160, 160),
        (220, 76, 70),
        (70, 150, 235),
        (170, 170, 170),
        (122, 88, 58),
        (209, 209, 209),
        (250, 184, 20),
        (40, 181, 120),
        (150, 88, 220),
        (244, 120, 52),
        (242, 212, 84),
        (96, 112, 126),
        (54, 184, 191),
    )
    if label < 0:
        return (160, 160, 160)
    return palette[label % len(palette)]


def _point_to_ply_vertex_line(
    point: LidarPoint,
    label_map: Mapping[str, int],
    station_lookup: Mapping[str, int],
    circle_index: int,
) -> str:
    red, green, blue = _label_to_rgb(point.label)
    station_index = station_lookup.get(point.station_id, -1)
    return (
        f"{point.x:.6f} {point.y:.6f} {point.z:.6f} "
        f"{point.label} {red} {green} {blue} "
        f"{station_index} {circle_index} {point.elevation_deg:.6f}\n"
    )


def _write_ply_header(handle: object, vertex_count: int, label_map: Mapping[str, int]) -> None:
    sorted_labels = sorted(label_map.items(), key=lambda item: item[1])
    handle.write("ply\n")
    handle.write("format ascii 1.0\n")
    handle.write("comment generated_by pointcloud_scanner\n")
    handle.write("comment semantic_label_property label\n")
    for class_name, label_id in sorted_labels:
        handle.write(f"comment label {label_id} {class_name}\n")
    handle.write(f"element vertex {vertex_count}\n")
    handle.write("property float x\n")
    handle.write("property float y\n")
    handle.write("property float z\n")
    handle.write("property int label\n")
    handle.write("property uchar red\n")
    handle.write("property uchar green\n")
    handle.write("property uchar blue\n")
    handle.write("property int station_index\n")
    handle.write("property int circle_index\n")
    handle.write("property float elevation_deg\n")
    handle.write("end_header\n")


def _write_points_ply(
    path: Path,
    points: list[LidarPoint],
    label_map: Mapping[str, int],
    station_lookup: Mapping[str, int],
    circle_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        _write_ply_header(handle, len(points), label_map)
        for point in points:
            handle.write(
                _point_to_ply_vertex_line(
                    point=point,
                    label_map=label_map,
                    station_lookup=station_lookup,
                    circle_index=circle_index,
                )
            )


def _write_ply_from_temp(
    path: Path,
    temp_points_path: Path,
    vertex_count: int,
    label_map: Mapping[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        _write_ply_header(handle, vertex_count, label_map)
        if temp_points_path.exists():
            with temp_points_path.open("r", encoding="utf-8", newline="\n") as source:
                for line in source:
                    handle.write(line)


def _resolve_output_dir(paths_cfg: Mapping[str, Any], output_cfg: Mapping[str, Any]) -> Path:
    output_dir = Path(str(paths_cfg.get("output_dir", "out/pointcloud_scans")))
    date_folder_enabled = _to_bool(output_cfg.get("date_folder_enabled", True), default=True)
    run_folder_enabled = _to_bool(output_cfg.get("run_folder_enabled", True), default=True)
    date_name = str(output_cfg.get("date_folder_name", date.today().isoformat()))
    run_name = str(output_cfg.get("run_folder_name", _timestamp_token()))
    unique_outputs = _to_bool(output_cfg.get("unique_outputs", True), default=True)

    result = output_dir
    if date_folder_enabled:
        result = result / date_name
    if run_folder_enabled:
        result = result / run_name
    if unique_outputs and not run_folder_enabled:
        result = _next_available_dir(result)
    return result


def _next_available_dir(path: Path) -> Path:
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}_{index:03d}")
        if not candidate.exists():
            return candidate
        index += 1
