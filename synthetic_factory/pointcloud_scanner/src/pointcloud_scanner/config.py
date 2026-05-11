from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "input_obj": "out/factory.obj",
        "output_dir": "out/pointcloud_scans",
        "metadata_filename": "scan_dataset.json",
        "stitched_filename": "factory_stitched.ply",
        "circle_file_prefix": "scan_circle",
    },
    "output": {
        "date_folder_enabled": True,
        "run_folder_enabled": True,
        "unique_outputs": True,
        "export_individual_circles": True,
        "export_stitched_cloud": True,
        "export_metadata_json": True,
    },
    "lidar": {
        "factory_room_id": "__factory__",
        "include_factory_room": True,
        "scan_range": 24.0,
        "angular_resolution_deg": 0.5,
        "vertical_resolution_deg": 0.8,
        "vertical_fov_up_deg": 88.0,
        "vertical_fov_down_deg": 88.0,
        "sensor_height": 1.6,
        "min_range": 0.35,
        "point_multiplier": 40,
        "point_jitter": 0.0075,
        "exterior_point_density_factor": 0.25,
        "points_per_station": 0,
        "blind_spot_radius": 0.55,
        "ensure_blind_spot_coverage": True,
        "stitched_fill_blind_spots": True,
        "global_coverage": False,
        "station_spacing": 4.0,
        "station_margin": 0.9,
        "obstacle_clearance": 0.35,
        "include_structural": True,
        "max_stations_per_room": 64,
        "max_stations": 96,
        "label_classes": {
            "unknown": 0,
            "pipe": 1,
            "wire": 2,
            "wall": 3,
            "floor": 4,
            "ceiling": 5,
            "machine": 6,
            "desk": 7,
            "rack": 8,
            "boiler": 9,
            "conveyor": 10,
            "structure": 11,
            "infrastructure": 12,
        },
    },
}


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            base[key] = deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def load_config(path: str | None = None) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Config file not found: {source}")
    data = _load_mapping(source)
    return deep_merge(cfg, data)


def save_default_config(path: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = _dump_yaml(DEFAULT_CONFIG)
        target.write_text(payload, encoding="utf-8")
        return target
    target.write_text(
        json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsn"}:
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            fallback_json = path.with_suffix(".json")
            if fallback_json.exists():
                data = json.loads(fallback_json.read_text(encoding="utf-8"))
            else:
                raise ImportError(
                    "PyYAML is required to load YAML config files."
                ) from exc
        else:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            data = loaded if loaded is not None else {}
    else:
        raise ValueError("Config format must be JSON or YAML.")
    if not isinstance(data, Mapping):
        raise TypeError("Top-level config must be a mapping.")
    return {str(key): value for key, value in data.items()}


def _dump_yaml(data: Mapping[str, Any]) -> str:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML is required to dump YAML config files.") from exc
    return yaml.safe_dump(
        dict(data),
        sort_keys=False,
        allow_unicode=True,
    )
