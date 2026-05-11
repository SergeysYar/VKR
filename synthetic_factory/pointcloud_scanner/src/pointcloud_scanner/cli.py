from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config, save_default_config
from .pipeline import run_scan_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pointcloud-scanner",
        description="Standalone LiDAR point-cloud scanner from OBJ scene.",
    )
    parser.add_argument(
        "--config",
        default="configs/scanner.default.yaml",
        help="Path to scanner config (.yaml/.yml/.json).",
    )
    parser.add_argument(
        "--input-obj",
        default=None,
        help="Override path to input OBJ scene.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory root.",
    )
    parser.add_argument(
        "--write-default-config",
        default=None,
        help="Write default config to this path and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(message)s")
    logger = logging.getLogger("pointcloud_scanner")

    if args.write_default_config:
        target = save_default_config(args.write_default_config)
        logger.info("Default config written: %s", target)
        return 0

    config_path = Path(args.config)
    config = load_config(str(config_path))
    _resolve_config_relative_paths(config, config_path.parent)

    if args.input_obj is not None:
        config.setdefault("paths", {})
        config["paths"]["input_obj"] = args.input_obj
    if args.output_dir is not None:
        config.setdefault("paths", {})
        config["paths"]["output_dir"] = args.output_dir

    result = run_scan_pipeline(config)
    logger.info("Input OBJ: %s", result.input_obj)
    logger.info("Output dir: %s", result.output_dir)
    logger.info("Stations: %d", result.station_count)
    logger.info("Circles: %d", result.circle_count)
    logger.info("Points: %d", result.point_count)
    logger.info("Metadata JSON: %s", result.metadata_json)
    logger.info("Stitched cloud: %s", result.stitched_combined)
    logger.info("Run summary: %s", result.run_summary_path)
    return 0


def _resolve_config_relative_paths(config: dict[str, object], config_dir: Path) -> None:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        return
    for key in ("input_obj", "output_dir"):
        value = paths.get(key)
        if value is None:
            continue
        raw = Path(str(value))
        if raw.is_absolute():
            continue
        paths[key] = str((config_dir / raw).resolve())
