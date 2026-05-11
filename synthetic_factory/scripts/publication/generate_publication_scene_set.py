from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Unable to locate project root from script path.")


# Allow direct execution:
# uv run python scripts/publication/generate_publication_scene_set.py
# or legacy wrapper:
# uv run python scripts/generate_publication_scene_set.py
ROOT_DIR = _find_project_root(Path(__file__).resolve())
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from synthetic_factory.config import Config
from synthetic_factory.parametric.parameters import ParameterSet
from synthetic_factory.pipeline import PipelineContext, build_default_scene_pipeline


@dataclass(frozen=True)
class SceneSpec:
    slug: str
    title: str
    description: str
    key_differences: tuple[str, ...]
    seed: int
    overrides: Mapping[str, Any]


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def _build_base_parameters(cfg: Mapping[str, Any], seed: int) -> dict[str, object]:
    room_count_range = cfg.get("layout", {}).get("room_count_range", [8, 10])
    base = {
        "factory_width": float(cfg["factory"]["width"]),
        "factory_depth": float(cfg["factory"]["depth"]),
        "number_of_rooms": int(cfg["layout"]["number_of_rooms"]),
        "corridor_width": float(cfg["layout"]["corridor_width"]),
        "room_height": float(cfg["rooms"]["height"]),
        "layout_strategy": str(cfg["layout"].get("strategy", "grid")),
        "room_size_min": float(cfg["rooms"]["min_size"]),
        "room_size_max": float(cfg["rooms"]["max_size"]),
        "room_count_range": room_count_range,
        "noise": dict(cfg.get("noise", {})),
        "biomes": dict(cfg.get("biomes", {})),
        "columns": dict(cfg.get("columns", {})),
        "beams": dict(cfg.get("beams", {})),
        "machinery": dict(cfg.get("machinery", {})),
        "exterior": dict(cfg.get("exterior", {})),
        "site": dict(cfg.get("site", {})),
        "lidar": dict(cfg.get("lidar", {})),
        "seed": int(seed),
    }
    noise_cfg = base.get("noise")
    if isinstance(noise_cfg, dict):
        noise_cfg["seed"] = int(seed)
    return base


def _build_fixed_parameter_set(base: Mapping[str, object], seed: int) -> ParameterSet:
    return ParameterSet(
        {
            "factory_width": float(base["factory_width"]),
            "factory_depth": float(base["factory_depth"]),
            "number_of_rooms": int(base["number_of_rooms"]),
            "corridor_width": float(base["corridor_width"]),
            "room_height": float(base["room_height"]),
            "room_size_min": float(base["room_size_min"]),
            "room_size_max": float(base["room_size_max"]),
        },
        seed=seed,
    )


def _normalize_scene_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    random_cfg = dict(config.get("random", {}))
    random_cfg["seed"] = int(seed)
    random_cfg["deterministic"] = True
    config["random"] = random_cfg

    lidar_cfg = dict(config.get("lidar", {}))
    lidar_cfg["enabled"] = False
    config["lidar"] = lidar_cfg

    machinery_cfg = dict(config.get("machinery", {}))
    for key in ("boiler", "control", "electrical", "laboratory", "maintenance", "refinery"):
        biome_cfg = machinery_cfg.get(key)
        if isinstance(biome_cfg, Mapping):
            merged = dict(biome_cfg)
            merged["seed"] = int(seed)
            machinery_cfg[key] = merged

    aux_cfg = machinery_cfg.get("auxiliary")
    if isinstance(aux_cfg, Mapping):
        merged_aux = dict(aux_cfg)
        merged_aux["seed"] = int(seed)
        machinery_cfg["auxiliary"] = merged_aux

    infra_cfg = machinery_cfg.get("infrastructure")
    if isinstance(infra_cfg, Mapping):
        merged_infra = dict(infra_cfg)
        merged_infra["seed"] = int(seed)
        machinery_cfg["infrastructure"] = merged_infra

    config["machinery"] = machinery_cfg

    exterior_cfg = dict(config.get("exterior", {}))
    exterior_cfg["seed"] = int(seed)
    config["exterior"] = exterior_cfg

    site_cfg = dict(config.get("site", {}))
    site_cfg["seed"] = int(seed)
    config["site"] = site_cfg
    return config


def _count_prefix(counter: Counter[str], prefix: str) -> int:
    return sum(count for key, count in counter.items() if key.startswith(prefix))


def _object_density(total_objects: int, layout_records: list[dict[str, Any]]) -> float:
    area = 0.0
    for room in layout_records:
        area += float(room["width"]) * float(room["depth"])
    if area <= 1e-9:
        return 0.0
    return total_objects / area


def _scenario_specs_part_a() -> list[SceneSpec]:
    return [
        SceneSpec(
            slug="minimal_configuration",
            title="Minimal Configuration",
            description=(
                "Small room count with low object density and simplified utility backbone."
            ),
            key_differences=(
                "5 rooms on fixed scale footprint",
                "Minimal machinery and sparse infrastructure",
                "Auxiliary detail disabled for cleaner baseline",
            ),
            seed=1101,
            overrides={
                "layout": {"strategy": "grid", "number_of_rooms": 5, "room_count_range": [5, 5], "corridor_width": 4.4},
                "rooms": {"min_size": 8.0, "max_size": 14.0, "height": 5.1},
                "biomes": {
                    "workshop_biome": "control",
                    "workshop_area_ratio": 0.42,
                    "ensure_all_types": False,
                    "allow_corridorless_links": False,
                    "cycle_order": ["control", "office", "maintenance"],
                },
                "machinery": {
                    "density": 0.025,
                    "conveyors_per_room": 1,
                    "machines_per_room": 2,
                    "auxiliary": {"enabled": False},
                    "infrastructure": {
                        "density": [0.55, 0.55],
                        "branch_frequency": [0.55, 0.55],
                        "parallel_channels": 1,
                    },
                },
                "columns": {"enabled": False},
                "beams": {"enabled": False},
                "exterior": {"equipment_density": [0.5, 0.5]},
            },
        ),
        SceneSpec(
            slug="balanced_factory",
            title="Balanced Factory",
            description=(
                "Balanced distribution between boiler, control, and electrical zones."
            ),
            key_differences=(
                "Biome mix constrained to boiler/control/electrical",
                "Medium object and infrastructure density",
                "Comparable room sizes with mixed connectivity",
            ),
            seed=2202,
            overrides={
                "layout": {"strategy": "perlin", "number_of_rooms": 9, "room_count_range": [9, 9], "corridor_width": 3.4},
                "rooms": {"min_size": 6.0, "max_size": 15.0, "height": 6.0},
                "noise": {
                    "full_occupancy": False,
                    "slot_multiplier": 2.0,
                    "connectivity_bias": 0.8,
                    "threshold": 0.34,
                    "sampling_scale": 0.082,
                    "min_distance": 8.5,
                },
                "biomes": {
                    "workshop_biome": "boiler",
                    "workshop_area_ratio": 0.34,
                    "ensure_all_types": False,
                    "cycle_order": [
                        "boiler",
                        "control",
                        "electrical",
                        "boiler",
                        "control",
                        "electrical",
                    ],
                },
                "machinery": {
                    "density": 0.065,
                    "conveyors_per_room": 4,
                    "machines_per_room": 6,
                    "boiler": {"pattern": "linear_boilers"},
                    "control": {"pattern": "linear_control_room"},
                    "electrical": {"pattern": "parallel_rows"},
                    "auxiliary": {"enabled": True, "density": [1.0, 1.0], "complexity": [3, 3]},
                    "infrastructure": {
                        "density": [1.25, 1.25],
                        "branch_frequency": [1.0, 1.0],
                        "parallel_channels": 2,
                    },
                },
            },
        ),
        SceneSpec(
            slug="high_density_industrial",
            title="High Density Industrial",
            description=(
                "High room count and dense utility network with minimal clearances."
            ),
            key_differences=(
                "18 rooms with compact corridors",
                "Maximum machinery and auxiliary saturation",
                "Aggressive backbone branching and multi-channel routing",
            ),
            seed=3303,
            overrides={
                "layout": {"strategy": "perlin", "number_of_rooms": 18, "room_count_range": [18, 18], "corridor_width": 2.2},
                "rooms": {"min_size": 4.2, "max_size": 9.8, "height": 6.8},
                "noise": {
                    "full_occupancy": False,
                    "slot_multiplier": 2.6,
                    "connectivity_bias": 0.96,
                    "threshold": 0.27,
                    "sampling_scale": 0.093,
                    "min_distance": 5.8,
                    "aspect_strength": 0.4,
                },
                "biomes": {
                    "workshop_biome": "workshop",
                    "workshop_area_ratio": 0.8,
                    "ensure_all_types": False,
                    "allow_corridorless_links": True,
                    "corridorless_link_ratio": 0.55,
                    "corridorless_height_delta": 2.4,
                    "cycle_order": [
                        "workshop",
                        "boiler",
                        "electrical",
                        "maintenance",
                        "refinery",
                        "workshop",
                        "boiler",
                        "electrical",
                    ],
                },
                "machinery": {
                    "density": 0.14,
                    "clearance": 0.8,
                    "conveyors_per_room": 8,
                    "machines_per_room": 12,
                    "auxiliary": {"enabled": True, "density": [2.6, 2.6], "complexity": [5, 5]},
                    "infrastructure": {
                        "density": [2.8, 2.8],
                        "branch_frequency": [1.95, 1.95],
                        "vertical": {"drop_frequency": [1.35, 1.35]},
                        "supports": {"spacing": [1.25, 1.25]},
                        "parallel_channels": 3,
                        "tray_width": [0.7, 0.7],
                    },
                },
                "columns": {"enabled": True, "spacing": 5.0},
                "beams": {"enabled": True, "spacing": 4.8, "elevation": 6.2},
            },
        ),
        SceneSpec(
            slug="boiler_dominated",
            title="Boiler-Dominated",
            description=(
                "Boiler-centric configuration with dense vertical process equipment."
            ),
            key_differences=(
                "Boiler biome dominates room allocation",
                "Dense-industrial boiler pattern with tall units",
                "Higher probability of bunkers, pumps, and heavy pipe routing",
            ),
            seed=4404,
            overrides={
                "layout": {"strategy": "perlin", "number_of_rooms": 12, "room_count_range": [12, 12], "corridor_width": 2.8},
                "rooms": {"min_size": 5.5, "max_size": 16.0, "height": 8.2},
                "noise": {
                    "full_occupancy": False,
                    "slot_multiplier": 2.3,
                    "connectivity_bias": 0.88,
                    "threshold": 0.32,
                    "sampling_scale": 0.085,
                    "min_distance": 7.2,
                },
                "biomes": {
                    "workshop_biome": "boiler",
                    "workshop_area_ratio": 0.86,
                    "ensure_all_types": False,
                    "cycle_order": ["boiler", "boiler", "boiler", "maintenance", "electrical"],
                    "allow_corridorless_links": True,
                    "corridorless_link_ratio": 0.35,
                },
                "machinery": {
                    "density": 0.12,
                    "conveyors_per_room": 6,
                    "machines_per_room": 8,
                    "boiler": {
                        "pattern": "dense_industrial",
                        "boiler": {"count": [2, 3], "height": [14.0, 24.0], "radius": [1.6, 2.8]},
                        "pipes": {"density": [1.8, 2.0]},
                        "tanks": {"count": [3, 5]},
                        "structure": {"beam_density": [0.8, 1.0]},
                    },
                    "auxiliary": {
                        "enabled": True,
                        "density": [2.2, 2.2],
                        "complexity": [4, 5],
                        "boiler": {
                            "bunker_probability": [0.75, 0.85],
                            "pump_count": [3, 4],
                        },
                    },
                    "infrastructure": {
                        "biome_type": "boiler",
                        "density": [2.4, 2.4],
                        "branch_frequency": [1.7, 1.7],
                        "include_pipes": True,
                        "pipe_radius": 0.18,
                        "parallel_channels": 3,
                    },
                },
            },
        ),
    ]


def _scenario_specs_part_b() -> list[SceneSpec]:
    return [
        SceneSpec(
            slug="control_centric",
            title="Control-Centric",
            description=(
                "Control-room-focused layout with regular organization and lighter industry load."
            ),
            key_differences=(
                "Control biome dominates floor area",
                "Regular desk-panel orientation and linear arrangement",
                "Reduced industrial machinery and simplified infrastructure",
            ),
            seed=5505,
            overrides={
                "layout": {"strategy": "grid", "number_of_rooms": 10, "room_count_range": [10, 10], "corridor_width": 3.7},
                "rooms": {"min_size": 6.5, "max_size": 14.0, "height": 4.6},
                "biomes": {
                    "workshop_biome": "control",
                    "workshop_area_ratio": 0.74,
                    "ensure_all_types": False,
                    "allow_corridorless_links": False,
                    "cycle_order": ["control", "control", "office", "electrical"],
                },
                "machinery": {
                    "density": 0.04,
                    "conveyors_per_room": 2,
                    "machines_per_room": 3,
                    "control": {
                        "pattern": "linear_control_room",
                        "density_profile": "medium",
                    },
                    "auxiliary": {"enabled": True, "density": [0.75, 0.75], "complexity": [2, 2]},
                    "infrastructure": {
                        "biome_type": "control",
                        "density": [0.8, 0.8],
                        "branch_frequency": [0.7, 0.7],
                        "parallel_channels": 1,
                        "include_pipes": False,
                    },
                },
                "columns": {"enabled": False},
                "beams": {"enabled": False},
            },
        ),
        SceneSpec(
            slug="electrical_grid",
            title="Electrical Grid",
            description=(
                "Electrical-biome dominant scene with strict cabinet rows and tray-heavy infrastructure."
            ),
            key_differences=(
                "Electrical rooms dominate allocation",
                "Dense-grid cabinet pattern with strict row regularity",
                "High cable tray density and branch count",
            ),
            seed=6606,
            overrides={
                "layout": {"strategy": "grid", "number_of_rooms": 12, "room_count_range": [12, 12], "corridor_width": 3.0},
                "rooms": {"min_size": 5.6, "max_size": 13.5, "height": 5.2},
                "biomes": {
                    "workshop_biome": "electrical",
                    "workshop_area_ratio": 0.8,
                    "ensure_all_types": False,
                    "cycle_order": ["electrical", "electrical", "control", "maintenance"],
                },
                "machinery": {
                    "density": 0.08,
                    "conveyors_per_room": 3,
                    "machines_per_room": 4,
                    "electrical": {
                        "pattern": "dense_grid",
                        "cabinets": {"rows": [4, 6], "per_row": [12, 18], "spacing": [0.8, 1.0]},
                        "walkways": {"width": [1.0, 1.2]},
                        "cable_trays": {"density": [1.8, 2.0]},
                        "cables": {"density": [2.3, 2.8]},
                    },
                    "auxiliary": {
                        "enabled": True,
                        "density": [1.8, 1.8],
                        "complexity": [3, 4],
                        "electrical": {"transformer_probability": [0.6, 0.7]},
                    },
                    "infrastructure": {
                        "biome_type": "electrical",
                        "density": [2.2, 2.2],
                        "branch_frequency": [1.6, 1.6],
                        "parallel_channels": 3,
                        "tray_width": [0.65, 0.65],
                    },
                },
            },
        ),
        SceneSpec(
            slug="mixed_complex",
            title="Mixed Complex",
            description=(
                "Complex multi-biome factory with balanced high-complexity infrastructure."
            ),
            key_differences=(
                "Includes boiler, control, electrical, laboratory, and maintenance biomes",
                "High cross-biome infrastructure branching",
                "Rich auxiliary detail across all room types",
            ),
            seed=7707,
            overrides={
                "layout": {"strategy": "perlin", "number_of_rooms": 16, "room_count_range": [16, 16], "corridor_width": 2.8},
                "rooms": {"min_size": 5.0, "max_size": 16.5, "height": 6.5},
                "noise": {
                    "full_occupancy": False,
                    "slot_multiplier": 2.4,
                    "connectivity_bias": 0.85,
                    "threshold": 0.33,
                    "sampling_scale": 0.087,
                    "min_distance": 7.0,
                },
                "biomes": {
                    "workshop_biome": "boiler",
                    "workshop_area_ratio": 0.52,
                    "ensure_all_types": False,
                    "allow_corridorless_links": True,
                    "corridorless_link_ratio": 0.3,
                    "cycle_order": [
                        "boiler",
                        "control",
                        "electrical",
                        "laboratory",
                        "maintenance",
                        "boiler",
                        "control",
                        "electrical",
                        "laboratory",
                        "maintenance",
                    ],
                },
                "machinery": {
                    "density": 0.11,
                    "conveyors_per_room": 7,
                    "machines_per_room": 10,
                    "boiler": {"pattern": "clustered_boilers"},
                    "control": {"pattern": "clustered_workstations"},
                    "electrical": {"pattern": "parallel_rows"},
                    "laboratory": {"pattern": "research_lab"},
                    "maintenance": {"pattern": "active_repair"},
                    "auxiliary": {"enabled": True, "density": [2.0, 2.0], "complexity": [4, 4]},
                    "infrastructure": {
                        "density": [2.0, 2.0],
                        "branch_frequency": [1.5, 1.5],
                        "parallel_channels": 3,
                        "include_pipes": True,
                    },
                },
            },
        ),
        SceneSpec(
            slug="refinery_complex",
            title="Refinery Complex",
            description=(
                "Refinery-heavy scene with multi-level process graph and dense pipe network."
            ),
            key_differences=(
                "Refinery biome dominates with tower and reactor hierarchy",
                "High pipe graph complexity with multiple routing levels",
                "Maximum auxiliary and infrastructure complexity",
            ),
            seed=8808,
            overrides={
                "layout": {"strategy": "perlin", "number_of_rooms": 14, "room_count_range": [14, 14], "corridor_width": 2.6},
                "rooms": {"min_size": 6.0, "max_size": 18.0, "height": 8.8},
                "noise": {
                    "full_occupancy": False,
                    "slot_multiplier": 2.5,
                    "connectivity_bias": 0.9,
                    "threshold": 0.31,
                    "sampling_scale": 0.086,
                    "min_distance": 7.5,
                },
                "biomes": {
                    "workshop_biome": "refinery",
                    "workshop_area_ratio": 0.9,
                    "ensure_all_types": False,
                    "allow_corridorless_links": True,
                    "corridorless_link_ratio": 0.4,
                    "cycle_order": [
                        "refinery",
                        "refinery",
                        "boiler",
                        "maintenance",
                        "electrical",
                        "refinery",
                    ],
                },
                "machinery": {
                    "density": 0.15,
                    "conveyors_per_room": 5,
                    "machines_per_room": 8,
                    "refinery": {
                        "pattern": "dense_refinery",
                        "main_columns": {"count": [3, 4], "height": [22.0, 30.0], "radius": [1.2, 2.0]},
                        "secondary": {"count": [8, 12], "radius": [0.5, 1.2], "length": [2.5, 5.2]},
                        "pipes": {"density": [2.3, 2.5], "radius": [0.2, 0.5], "levels": [3, 4], "support_spacing": [2.5, 3.2]},
                        "platforms": {"levels": [3, 4], "width_factor": [1.7, 2.3]},
                        "secondary_elements": {"enabled": True, "density": 1.5, "sensor_density": 1.6},
                        "structure": {"min_clearance": 0.8},
                        "rules": {"auto_fix": True, "min_clearance": 0.8, "pipe_collision_margin": 0.14},
                    },
                    "auxiliary": {"enabled": True, "density": [2.4, 2.4], "complexity": [5, 5]},
                    "infrastructure": {
                        "density": [2.7, 2.7],
                        "branch_frequency": [1.95, 1.95],
                        "parallel_channels": 3,
                        "include_pipes": True,
                        "pipe_radius": 0.22,
                    },
                },
                "exterior": {"roof": {"type": ["sawtooth"]}},
            },
        ),
    ]


def _scenario_specs() -> list[SceneSpec]:
    return _scenario_specs_part_a() + _scenario_specs_part_b()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    output_root = ROOT_DIR / "out" / "publication" / "scene_set" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    default_cfg = Config.from_default().to_dict()
    common_overrides: dict[str, Any] = {
        "factory": {"width": 130.0, "depth": 95.0},
        "layout": {"corridor_width": 3.0},
        "rooms": {"min_size": 5.0, "max_size": 16.0, "height": 6.0},
        "noise": {
            "full_occupancy": True,
            "slot_multiplier": 2.0,
            "connectivity_bias": 0.82,
            "threshold": 0.35,
            "sampling_scale": 0.085,
            "min_distance": 8.0,
        },
        "lidar": {"enabled": False},
        "columns": {"enabled": True, "spacing": 6.0, "radius": 0.3},
        "beams": {"enabled": True, "spacing": 6.0, "elevation": 5.5},
        "biomes": {
            "enabled": True,
            "ensure_all_types": False,
            "workshop_biome": "workshop",
            "workshop_area_ratio": 0.58,
            "allow_corridorless_links": True,
            "corridorless_link_ratio": 0.2,
            "corridorless_height_delta": 1.8,
            "clustered_assignment": True,
            "height_grouping": True,
            "cycle_order": [
                "workshop",
                "boiler",
                "control",
                "electrical",
                "maintenance",
                "laboratory",
                "office",
                "storage",
                "refinery",
            ],
        },
        "machinery": {
            "enabled": True,
            "density": 0.07,
            "conveyors_per_room": 4,
            "machines_per_room": 6,
            "auxiliary": {"enabled": True, "density": [1.1, 1.1], "complexity": [3, 3]},
            "infrastructure": {
                "enabled": True,
                "density": [1.3, 1.3],
                "branch_frequency": [1.0, 1.0],
                "vertical": {"drop_frequency": [0.85, 0.85]},
                "supports": {"spacing": [2.0, 2.0]},
                "parallel_channels": 2,
                "include_pipes": True,
            },
        },
        "exterior": {"enabled": True, "equipment_density": [1.0, 1.0]},
        "site": {"enabled": True},
    }

    dataset_records: list[dict[str, Any]] = []
    for index, spec in enumerate(_scenario_specs(), start=1):
        scenario_dir = output_root / f"{index:02d}_{spec.slug}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        cfg = _deep_merge(deepcopy(default_cfg), common_overrides)
        cfg = _deep_merge(cfg, spec.overrides)
        cfg = _normalize_scene_config(cfg, seed=spec.seed)

        base_parameters = _build_base_parameters(cfg, seed=spec.seed)
        parameter_set = _build_fixed_parameter_set(base_parameters, seed=spec.seed)
        export_hint = str(scenario_dir / f"{spec.slug}.obj")

        print(f"[{spec.slug}] generating ...")
        pipeline = build_default_scene_pipeline(
            parameter_set=parameter_set,
            export_path=export_hint,
            seed=spec.seed,
            base_parameters=base_parameters,
            logger=None,
        )
        context: PipelineContext = pipeline.run()

        generated_obj = Path(str(context.export_path)) if context.export_path else None
        if generated_obj is None or not generated_obj.exists():
            raise RuntimeError(f"OBJ export missing for scenario '{spec.slug}'.")
        canonical_obj = scenario_dir / f"{spec.slug}.obj"
        if generated_obj.resolve() != canonical_obj.resolve():
            shutil.copy2(generated_obj, canonical_obj)

        scene = context.scene
        if scene is None:
            raise RuntimeError(f"Scene was not produced for scenario '{spec.slug}'.")
        traversed_objects = list(scene.traverse())
        object_counter = Counter(obj.type for obj in traversed_objects)
        layout_records = [
            {
                "room_id": room.room_id,
                "biome": room.biome,
                "row": room.row,
                "col": room.col,
                "center_x": room.center_x,
                "center_y": room.center_y,
                "width": room.width,
                "depth": room.depth,
            }
            for room in context.layout
        ]
        biome_counter = Counter(room["biome"] for room in layout_records)

        stats = {
            "seed": spec.seed,
            "room_count": len(layout_records),
            "scene_object_count": len(traversed_objects),
            "root_object_count": len(scene.objects),
            "infrastructure_object_count": _count_prefix(object_counter, "infra_"),
            "auxiliary_object_count": _count_prefix(object_counter, "aux_"),
            "exterior_object_count": _count_prefix(object_counter, "exterior_"),
            "site_object_count": _count_prefix(object_counter, "site_"),
            "object_density_per_m2": _object_density(len(traversed_objects), layout_records),
            "biome_counts": dict(sorted(biome_counter.items())),
            "top_object_types": object_counter.most_common(30),
        }

        scene_payload = {
            "name": spec.title,
            "slug": spec.slug,
            "description": spec.description,
            "key_differences": list(spec.key_differences),
            "artifacts": {
                "obj": str(canonical_obj),
                "generated_obj_raw": str(generated_obj),
            },
            "stats": stats,
            "layout": layout_records,
            "parameters": base_parameters,
            "overrides": {
                "common": common_overrides,
                "scenario": dict(spec.overrides),
            },
        }

        scenario_json = scenario_dir / f"{spec.slug}_scene.json"
        scenario_json.write_text(
            json.dumps(scene_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        dataset_records.append(
            {
                "index": index,
                "slug": spec.slug,
                "title": spec.title,
                "description": spec.description,
                "key_differences": list(spec.key_differences),
                "seed": spec.seed,
                "rooms": stats["room_count"],
                "objects": stats["scene_object_count"],
                "infrastructure_objects": stats["infrastructure_object_count"],
                "auxiliary_objects": stats["auxiliary_object_count"],
                "object_density_per_m2": stats["object_density_per_m2"],
                "biome_counts": stats["biome_counts"],
                "obj_path": str(canonical_obj),
                "json_path": str(scenario_json),
            }
        )

    manifest_json = output_root / "scene_set_manifest.json"
    manifest_json.write_text(
        json.dumps(dataset_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest_md_lines = [
        "# Demonstration 3D Scene Set",
        "",
        "Caption:",
        '"Examples of generated scenes with different room configurations and infrastructure patterns."',
        "",
        "| # | Scene | Rooms | Objects | Infra objects | Aux objects | Density (obj/m2) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in dataset_records:
        manifest_md_lines.append(
            "| "
            f"{record['index']} | {record['title']} | {record['rooms']} | {record['objects']} | "
            f"{record['infrastructure_objects']} | {record['auxiliary_objects']} | "
            f"{float(record['object_density_per_m2']):.3f} |"
        )
    manifest_md_lines.append("")
    manifest_md_lines.append("## Scene Notes")
    manifest_md_lines.append("")
    for record in dataset_records:
        manifest_md_lines.append(f"### {record['index']}. {record['title']}")
        manifest_md_lines.append(record["description"])
        for diff in record["key_differences"]:
            manifest_md_lines.append(f"- {diff}")
        manifest_md_lines.append(f"- OBJ: `{record['obj_path']}`")
        manifest_md_lines.append(f"- JSON: `{record['json_path']}`")
        manifest_md_lines.append("")

    manifest_md = output_root / "scene_set_manifest.md"
    manifest_md.write_text("\n".join(manifest_md_lines) + "\n", encoding="utf-8")

    csv_path = output_root / "scene_set_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "slug",
                "title",
                "seed",
                "rooms",
                "objects",
                "infrastructure_objects",
                "auxiliary_objects",
                "object_density_per_m2",
                "obj_path",
                "json_path",
            ]
        )
        for record in dataset_records:
            writer.writerow(
                [
                    record["index"],
                    record["slug"],
                    record["title"],
                    record["seed"],
                    record["rooms"],
                    record["objects"],
                    record["infrastructure_objects"],
                    record["auxiliary_objects"],
                    f"{float(record['object_density_per_m2']):.6f}",
                    record["obj_path"],
                    record["json_path"],
                ]
            )

    caption_path = output_root / "figure_caption_ru.txt"
    caption_path.write_text(
        "Примеры сгенерированных сцен с различной конфигурацией помещений и инфраструктуры.\n",
        encoding="utf-8",
    )

    print(f"Scene set generated in: {output_root}")
    print(f"Manifest JSON: {manifest_json}")
    print(f"Manifest Markdown: {manifest_md}")
    print(f"Manifest CSV: {csv_path}")
    print(f"Caption: {caption_path}")


if __name__ == "__main__":
    main()
