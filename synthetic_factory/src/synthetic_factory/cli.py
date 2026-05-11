from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from .config import Config
from .parametric.parameters import ParameterSet
from .pipeline import build_default_scene_pipeline


def _parse_scalar(value: str) -> object:
    raw = value.strip()
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        number = float(raw)
    except ValueError:
        return raw
    if number.is_integer() and "." not in raw and "e" not in lowered:
        return int(number)
    return number


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    direct_overrides: list[tuple[str, object | None]] = [
        ("factory.width", args.factory_width),
        ("factory.depth", args.factory_depth),
        ("layout.number_of_rooms", args.number_of_rooms),
        ("rooms.min_size", args.room_min_size),
        ("rooms.max_size", args.room_max_size),
        ("rooms.height", args.room_height),
        ("layout.corridor_width", args.corridor_width),
        ("lidar.point_multiplier", args.lidar_point_multiplier),
        ("lidar.points_per_station", args.lidar_points_per_station),
        ("lidar.station_spacing", args.lidar_station_spacing),
        ("lidar.max_stations_per_room", args.lidar_max_stations_per_room),
        ("lidar.max_stations", args.lidar_max_stations),
        ("lidar.compute_backend", args.lidar_compute_backend),
    ]
    for path, value in direct_overrides:
        if value is not None:
            config.set(path, value)

    for entry in args.set_override or []:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --set value '{entry}'. Use format: path=value"
            )
        path, raw_value = entry.split("=", 1)
        normalized_path = path.strip()
        if not normalized_path:
            raise ValueError(f"Invalid --set value '{entry}': empty path.")
        config.set(normalized_path, _parse_scalar(raw_value))
    return config


def _build_base_parameters(config: Config) -> dict[str, object]:
    room_count_range = config.get("layout.room_count_range", [8, 10])
    return {
        "factory_width": float(config.get("factory.width")),
        "factory_depth": float(config.get("factory.depth")),
        "number_of_rooms": int(config.get("layout.number_of_rooms")),
        "corridor_width": float(config.get("layout.corridor_width")),
        "room_height": float(config.get("rooms.height")),
        "layout_strategy": str(config.get("layout.strategy", "grid")),
        "room_size_min": float(config.get("rooms.min_size")),
        "room_size_max": float(config.get("rooms.max_size")),
        "room_count_range": room_count_range,
        "noise": config.get("noise", {}),
        "biomes": config.get("biomes", {}),
        "columns": config.get("columns", {}),
        "beams": config.get("beams", {}),
        "machinery": config.get("machinery", {}),
        "exterior": config.get("exterior", {}),
        "site": config.get("site", {}),
        "lidar": config.get("lidar", {}),
        "seed": int(config.get("random.seed", 0)),
    }


def _load_config(path: str) -> Config:
    config_path = Path(path)
    try:
        return Config.from_file(str(config_path))
    except ImportError:
        # Fallback for environments without PyYAML: try adjacent JSON config.
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            json_candidate = config_path.with_suffix(".json")
            if json_candidate.exists():
                return Config.from_file(str(json_candidate))
        raise


def _build_parameter_set(config: Config, seed: int | None) -> ParameterSet:
    raw_parameters = config.get("parameters", {})
    if isinstance(raw_parameters, Mapping) and raw_parameters:
        return ParameterSet(raw_parameters, seed=seed)

    return ParameterSet(
        {
            "factory_width": config.get("factory.width"),
            "factory_depth": config.get("factory.depth"),
            "number_of_rooms": config.get("layout.number_of_rooms"),
            "corridor_width": config.get("layout.corridor_width"),
            "room_height": config.get("rooms.height"),
            "room_size_min": config.get("rooms.min_size"),
            "room_size_max": config.get("rooms.max_size"),
        },
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthetic-factory",
        description=(
            "Generate synthetic factory scene and export OBJ/point clouds. "
            "Use --set path=value for full config control."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/presets/scene.yaml",
        help="Path to config file (.yaml/.yml/.json/.toml).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override export output path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR.",
    )
    parser.add_argument(
        "--factory-width",
        type=float,
        default=None,
        help="Override factory width (meters).",
    )
    parser.add_argument(
        "--factory-depth",
        type=float,
        default=None,
        help="Override factory depth (meters).",
    )
    parser.add_argument(
        "--number-of-rooms",
        type=int,
        default=None,
        help="Override number of rooms (biome rooms).",
    )
    parser.add_argument(
        "--room-min-size",
        type=float,
        default=None,
        help="Override rooms.min_size.",
    )
    parser.add_argument(
        "--room-max-size",
        type=float,
        default=None,
        help="Override rooms.max_size.",
    )
    parser.add_argument(
        "--room-height",
        type=float,
        default=None,
        help="Override rooms.height.",
    )
    parser.add_argument(
        "--corridor-width",
        type=float,
        default=None,
        help="Override layout.corridor_width.",
    )
    parser.add_argument(
        "--lidar-point-multiplier",
        type=int,
        default=None,
        help="Override lidar.point_multiplier.",
    )
    parser.add_argument(
        "--lidar-points-per-station",
        type=int,
        default=None,
        help="Override lidar.points_per_station.",
    )
    parser.add_argument(
        "--lidar-station-spacing",
        type=float,
        default=None,
        help="Override lidar.station_spacing.",
    )
    parser.add_argument(
        "--lidar-max-stations-per-room",
        type=int,
        default=None,
        help="Override lidar.max_stations_per_room (0 = unlimited).",
    )
    parser.add_argument(
        "--lidar-max-stations",
        type=int,
        default=None,
        help="Override lidar.max_stations (0 = unlimited).",
    )
    parser.add_argument(
        "--lidar-compute-backend",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override lidar.compute_backend.",
    )
    parser.add_argument(
        "--set",
        dest="set_override",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "Generic config override, repeatable. "
            "Example: --set biomes.workshop_area_ratio=0.8 "
            "--set lidar.include_factory_room=true"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(message)s")
    logger = logging.getLogger("synthetic_factory")

    config = _load_config(args.config)
    config = _apply_overrides(config, args)

    default_seed = int(config.get("random.seed", 0))
    seed = args.seed if args.seed is not None else default_seed

    export_path = args.output or str(config.get("export.output_path", "out/factory.obj"))
    parameter_set = _build_parameter_set(config, seed=seed)
    base_parameters = _build_base_parameters(config)
    base_parameters["seed"] = seed
    noise_cfg = base_parameters.get("noise")
    if isinstance(noise_cfg, Mapping):
        merged_noise = dict(noise_cfg)
        merged_noise["seed"] = seed
        base_parameters["noise"] = merged_noise

    pipeline = build_default_scene_pipeline(
        parameter_set=parameter_set,
        export_path=export_path,
        seed=seed,
        base_parameters=base_parameters,
        logger=logger,
    )
    result = pipeline.run()

    logger.info("OBJ exported to: %s", result.export_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
