from __future__ import annotations

import json
import math
import shutil
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Unable to locate project root from script path.")


# Allow direct execution:
# uv run python scripts/publication/generate_single_scene_multiview_dataset.py
# or legacy wrapper:
# uv run python scripts/generate_single_scene_multiview_dataset.py
ROOT_DIR = _find_project_root(Path(__file__).resolve())
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from synthetic_factory.config import Config
from synthetic_factory.exporters.obj_exporter import ObjExporter
from synthetic_factory.parametric.parameters import ParameterSet
from synthetic_factory.pipeline import PipelineContext, build_default_scene_pipeline
from synthetic_factory.scene.scene_graph import SceneObject


TARGET_BIOMES = ("boiler", "electrical", "control", "laboratory", "maintenance")


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
    return config


def _matrix_mul_point(matrix: list[list[float]], x: float, y: float, z: float) -> tuple[float, float, float]:
    tx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3]
    ty = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3]
    tz = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3]
    tw = matrix[3][0] * x + matrix[3][1] * y + matrix[3][2] * z + matrix[3][3]
    if abs(tw) > 1e-12:
        return (tx / tw, ty / tw, tz / tw)
    return (tx, ty, tz)


def _coerce_room_biome_map(layout_records: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for room in layout_records:
        room_id = str(room["room_id"])
        biome = str(room["biome"]).strip().lower()
        mapping[room_id] = biome
    return mapping


def _infer_biome_for_object(
    obj: SceneObject,
    room_biomes: Mapping[str, str],
) -> str:
    current: SceneObject | None = obj
    while current is not None:
        direct = room_biomes.get(current.id)
        if direct is not None:
            return direct
        current = current.parent

    obj_id = obj.id
    for room_id, biome in room_biomes.items():
        if obj_id.startswith(f"{room_id}_"):
            return biome

    if obj.type.startswith("room_"):
        suffix = obj.type[5:].strip().lower()
        if suffix:
            return suffix
    return "empty"


def _safe_bbox_from_vertices(vertices: list[tuple[float, float, float]]) -> dict[str, list[float]]:
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    zs = [vertex[2] for vertex in vertices]
    return {
        "min": [min(xs), min(ys), min(zs)],
        "max": [max(xs), max(ys), max(zs)],
    }


def _bbox_volume(bbox: Mapping[str, list[float]]) -> float:
    minimum = bbox["min"]
    maximum = bbox["max"]
    dx = max(0.0, float(maximum[0]) - float(minimum[0]))
    dy = max(0.0, float(maximum[1]) - float(minimum[1]))
    dz = max(0.0, float(maximum[2]) - float(minimum[2]))
    return dx * dy * dz


def _global_bounds_from_layout(layout_records: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    if not layout_records:
        return (-1.0, 1.0, -1.0, 1.0)
    min_x = min(float(room["center_x"]) - float(room["width"]) / 2.0 for room in layout_records)
    max_x = max(float(room["center_x"]) + float(room["width"]) / 2.0 for room in layout_records)
    min_y = min(float(room["center_y"]) - float(room["depth"]) / 2.0 for room in layout_records)
    max_y = max(float(room["center_y"]) + float(room["depth"]) / 2.0 for room in layout_records)
    return (min_x, max_x, min_y, max_y)


def _gaussian_kernel_1d(sigma: float) -> list[float]:
    if sigma <= 0.0:
        return [1.0]
    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel: list[float] = []
    denom = 2.0 * sigma * sigma
    for offset in range(-radius, radius + 1):
        kernel.append(math.exp(-(offset * offset) / denom))
    total = sum(kernel)
    if total <= 0.0:
        return [1.0]
    return [value / total for value in kernel]


def _apply_gaussian_blur(grid: list[list[float]], sigma: float) -> list[list[float]]:
    if sigma <= 0.0:
        return [row[:] for row in grid]
    kernel = _gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    tmp = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            acc = 0.0
            for k, weight in enumerate(kernel):
                dx = k - radius
                sx = x + dx
                if sx < 0:
                    sx = 0
                elif sx >= cols:
                    sx = cols - 1
                acc += grid[y][sx] * weight
            tmp[y][x] = acc

    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            acc = 0.0
            for k, weight in enumerate(kernel):
                dy = k - radius
                sy = y + dy
                if sy < 0:
                    sy = 0
                elif sy >= rows:
                    sy = rows - 1
                acc += tmp[sy][x] * weight
            out[y][x] = acc
    return out


def _normalize_grid(grid: list[list[float]]) -> tuple[list[list[float]], float]:
    max_value = 0.0
    for row in grid:
        for value in row:
            if value > max_value:
                max_value = value
    if max_value <= 1e-12:
        return ([[0.0 for _ in row] for row in grid], 0.0)
    normalized = [[value / max_value for value in row] for row in grid]
    return (normalized, max_value)


def _histogram(values: list[float], bins: int = 10) -> list[dict[str, float]]:
    if not values:
        return []
    bins = max(1, bins)
    counts = [0 for _ in range(bins)]
    for value in values:
        clamped = min(max(value, 0.0), 1.0)
        index = min(bins - 1, int(clamped * bins))
        counts[index] += 1
    total = len(values)
    result: list[dict[str, float]] = []
    for idx, count in enumerate(counts):
        low = idx / bins
        high = (idx + 1) / bins
        result.append(
            {
                "bin_start": low,
                "bin_end": high,
                "count": float(count),
                "ratio": float(count) / float(total),
            }
        )
    return result


def _generate_dataset() -> dict[str, Any]:
    seed = 20260410
    grid_size = 100
    blur_sigma = 1.1

    default_cfg = Config.from_default().to_dict()
    overrides: dict[str, Any] = {
        "factory": {"width": 122.0, "depth": 94.0},
        "layout": {"strategy": "perlin", "number_of_rooms": 12, "room_count_range": [12, 12], "corridor_width": 3.0},
        "rooms": {"min_size": 5.5, "max_size": 16.0, "height": 6.8},
        "noise": {
            "full_occupancy": False,
            "slot_multiplier": 2.3,
            "connectivity_bias": 0.86,
            "threshold": 0.33,
            "sampling_scale": 0.086,
            "min_distance": 7.0,
        },
        "biomes": {
            "enabled": True,
            "ensure_all_types": False,
            "workshop_biome": "boiler",
            "workshop_area_ratio": 0.48,
            "allow_corridorless_links": True,
            "corridorless_link_ratio": 0.28,
            "clustered_assignment": True,
            "height_grouping": True,
            "cycle_order": [
                "boiler",
                "electrical",
                "control",
                "laboratory",
                "maintenance",
                "boiler",
                "electrical",
                "control",
                "laboratory",
                "maintenance",
            ],
        },
        "machinery": {
            "enabled": True,
            "density": 0.11,
            "conveyors_per_room": 6,
            "machines_per_room": 9,
            "auxiliary": {"enabled": True, "density": [1.8, 1.8], "complexity": [4, 4]},
            "infrastructure": {
                "enabled": True,
                "density": [1.8, 1.8],
                "branch_frequency": [1.4, 1.4],
                "vertical": {"drop_frequency": [1.1, 1.1]},
                "supports": {"spacing": [2.0, 2.0]},
                "parallel_channels": 3,
                "include_pipes": True,
            },
        },
        "columns": {"enabled": True, "spacing": 6.0, "radius": 0.32},
        "beams": {"enabled": True, "spacing": 6.0, "elevation": 5.9},
        "exterior": {"enabled": False},
        "site": {"enabled": False},
        "lidar": {"enabled": False},
    }

    cfg = _deep_merge(deepcopy(default_cfg), overrides)
    cfg = _normalize_scene_config(cfg, seed=seed)
    base_parameters = _build_base_parameters(cfg, seed=seed)
    parameter_set = _build_fixed_parameter_set(base_parameters, seed=seed)

    output_root = ROOT_DIR / "out" / "publication" / "single_scene_multiview" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    export_hint = str(output_root / "raw_pipeline_scene.obj")
    pipeline = build_default_scene_pipeline(
        parameter_set=parameter_set,
        export_path=export_hint,
        seed=seed,
        base_parameters=base_parameters,
        logger=None,
    )
    context: PipelineContext = pipeline.run()
    if context.scene is None:
        raise RuntimeError("Scene was not produced by the pipeline.")

    generated_obj = Path(str(context.export_path)) if context.export_path else None
    if generated_obj is None or not generated_obj.exists():
        raise RuntimeError("OBJ export missing after pipeline execution.")

    final_obj = output_root / "scene.obj"
    shutil.copy2(generated_obj, final_obj)

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
    room_biomes = _coerce_room_biome_map(layout_records)

    exporter = ObjExporter()
    object_records: list[dict[str, Any]] = []
    mesh_object_records: list[dict[str, Any]] = []

    for obj, world_matrix, group_name in exporter._iter_objects(context.scene):
        world_position = _matrix_mul_point(world_matrix, 0.0, 0.0, 0.0)
        biome = _infer_biome_for_object(obj, room_biomes)
        record: dict[str, Any] = {
            "id": obj.id,
            "type": obj.type,
            "group": group_name,
            "biome": biome,
            "position": {"x": world_position[0], "y": world_position[1], "z": world_position[2]},
            "has_mesh": obj.mesh is not None,
            "bounding_box": None,
            "bbox_volume": 0.0,
        }
        if obj.mesh is not None and obj.mesh.vertices:
            world_vertices = [_matrix_mul_point(world_matrix, x, y, z) for x, y, z in obj.mesh.vertices]
            bbox = _safe_bbox_from_vertices(world_vertices)
            bbox_volume = _bbox_volume(bbox)
            record["bounding_box"] = bbox
            record["bbox_volume"] = bbox_volume
            mesh_object_records.append(record)
        object_records.append(record)

    if mesh_object_records:
        min_x = min(float(record["bounding_box"]["min"][0]) for record in mesh_object_records)
        max_x = max(float(record["bounding_box"]["max"][0]) for record in mesh_object_records)
        min_y = min(float(record["bounding_box"]["min"][1]) for record in mesh_object_records)
        max_y = max(float(record["bounding_box"]["max"][1]) for record in mesh_object_records)
    else:
        min_x, max_x, min_y, max_y = _global_bounds_from_layout(layout_records)

    if max_x - min_x < 1e-6:
        max_x += 1.0
        min_x -= 1.0
    if max_y - min_y < 1e-6:
        max_y += 1.0
        min_y -= 1.0

    padding = 0.5
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding

    width = max_x - min_x
    depth = max_y - min_y
    cell_w = width / float(grid_size)
    cell_h = depth / float(grid_size)

    biome_scores: list[list[dict[str, float]]] = [
        [defaultdict(float) for _ in range(grid_size)] for _ in range(grid_size)
    ]
    density_raw: list[list[float]] = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]

    def to_index_x(x: float) -> int:
        return min(grid_size - 1, max(0, int(math.floor((x - min_x) / cell_w))))

    def to_index_y(y: float) -> int:
        return min(grid_size - 1, max(0, int(math.floor((y - min_y) / cell_h))))

    for record in mesh_object_records:
        bbox = record["bounding_box"]
        assert bbox is not None
        bx0 = float(bbox["min"][0])
        bx1 = float(bbox["max"][0])
        by0 = float(bbox["min"][1])
        by1 = float(bbox["max"][1])

        ix0 = to_index_x(bx0)
        ix1 = to_index_x(bx1)
        iy0 = to_index_y(by0)
        iy1 = to_index_y(by1)
        touched = max(1, (ix1 - ix0 + 1) * (iy1 - iy0 + 1))
        volume = float(record["bbox_volume"])
        contribution = volume / float(touched) if volume > 0.0 else 1.0 / float(touched)

        biome = str(record["biome"]).strip().lower()
        biome_key = biome if biome in TARGET_BIOMES else "empty"
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                biome_scores[iy][ix][biome_key] += contribution
                density_raw[iy][ix] += contribution

    biome_map: list[list[str]] = []
    for row in biome_scores:
        biome_row: list[str] = []
        for cell_scores in row:
            if not cell_scores:
                biome_row.append("empty")
                continue
            best_label = "empty"
            best_score = -1.0
            for label, score in cell_scores.items():
                if score > best_score:
                    best_label = label
                    best_score = score
            biome_row.append(best_label)
        biome_map.append(biome_row)

    density_blurred = _apply_gaussian_blur(density_raw, sigma=blur_sigma)
    density_map, density_max_before_norm = _normalize_grid(density_blurred)

    biome_areas: dict[str, int] = {biome: 0 for biome in (*TARGET_BIOMES, "empty")}
    biome_density_values: dict[str, list[float]] = {biome: [] for biome in (*TARGET_BIOMES, "empty")}
    biome_boundary_cells: dict[str, list[list[int]]] = {biome: [] for biome in TARGET_BIOMES}

    for iy in range(grid_size):
        for ix in range(grid_size):
            biome = biome_map[iy][ix]
            biome_areas[biome] = biome_areas.get(biome, 0) + 1
            biome_density_values.setdefault(biome, []).append(density_map[iy][ix])
            if biome not in TARGET_BIOMES:
                continue
            neighbors = []
            if iy > 0:
                neighbors.append(biome_map[iy - 1][ix])
            if iy < grid_size - 1:
                neighbors.append(biome_map[iy + 1][ix])
            if ix > 0:
                neighbors.append(biome_map[iy][ix - 1])
            if ix < grid_size - 1:
                neighbors.append(biome_map[iy][ix + 1])
            if any(neighbor != biome for neighbor in neighbors):
                biome_boundary_cells[biome].append([ix, iy])

    biome_boundaries_world: dict[str, dict[str, float] | None] = {}
    for biome in TARGET_BIOMES:
        cells = [
            (ix, iy)
            for iy in range(grid_size)
            for ix in range(grid_size)
            if biome_map[iy][ix] == biome
        ]
        if not cells:
            biome_boundaries_world[biome] = None
            continue
        min_ix = min(ix for ix, _ in cells)
        max_ix = max(ix for ix, _ in cells)
        min_iy = min(iy for _, iy in cells)
        max_iy = max(iy for _, iy in cells)
        biome_boundaries_world[biome] = {
            "min_x": min_x + min_ix * cell_w,
            "max_x": min_x + (max_ix + 1) * cell_w,
            "min_y": min_y + min_iy * cell_h,
            "max_y": min_y + (max_iy + 1) * cell_h,
        }

    flat_density = [value for row in density_map for value in row]
    density_mean = sum(flat_density) / float(len(flat_density)) if flat_density else 0.0
    density_var = (
        sum((value - density_mean) ** 2 for value in flat_density) / float(len(flat_density))
        if flat_density
        else 0.0
    )
    density_std = math.sqrt(density_var)
    heterogeneity = density_std / density_mean if density_mean > 1e-12 else 0.0

    per_biome_metrics: dict[str, dict[str, float | int]] = {}
    for biome in TARGET_BIOMES:
        values = biome_density_values.get(biome, [])
        mean_value = sum(values) / float(len(values)) if values else 0.0
        max_value = max(values) if values else 0.0
        per_biome_metrics[biome] = {
            "area_cells": int(biome_areas.get(biome, 0)),
            "mean_density": mean_value,
            "max_density": max_value,
        }

    metadata = {
        "title": "Single Synthetic Industrial Scene (3-view dataset)",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "grid_size": grid_size,
        "target_biomes": list(TARGET_BIOMES),
        "parameters": {
            "base_parameters": base_parameters,
            "overrides": overrides,
        },
        "artifacts": {
            "scene_obj": str(final_obj),
            "scene_metadata": str(output_root / "scene_metadata.json"),
            "biome_map": str(output_root / "biome_map.json"),
            "density_map": str(output_root / "density_map.json"),
        },
        "coordinate_system": {
            "x_range": [min_x, max_x],
            "y_range": [min_y, max_y],
            "cell_size": [cell_w, cell_h],
            "grid_origin": [min_x, min_y],
            "note": "Same XY bounds/grid used for biome and density maps.",
        },
        "scene": {
            "room_layout": layout_records,
            "objects": object_records,
            "object_count_total": len(object_records),
            "object_count_mesh": len(mesh_object_records),
            "present_biomes_in_layout": sorted({str(room['biome']) for room in layout_records}),
        },
        "biome_map_stats": {
            "areas_cells": biome_areas,
            "boundaries_world": biome_boundaries_world,
            "boundary_cells": biome_boundary_cells,
            "per_biome_density": per_biome_metrics,
        },
        "density_stats": {
            "max_before_normalization": density_max_before_norm,
            "mean": density_mean,
            "max": max(flat_density) if flat_density else 0.0,
            "variance": density_var,
            "std": density_std,
            "heterogeneity_coefficient": heterogeneity,
            "histogram_10_bins": _histogram(flat_density, bins=10),
        },
    }

    biome_payload = {
        "grid_size": grid_size,
        "x_range": [min_x, max_x],
        "y_range": [min_y, max_y],
        "labels": [*TARGET_BIOMES, "empty"],
        "matrix": biome_map,
    }
    density_payload = {
        "grid_size": grid_size,
        "x_range": [min_x, max_x],
        "y_range": [min_y, max_y],
        "normalized": True,
        "blur_sigma": blur_sigma,
        "matrix": density_map,
    }

    (output_root / "scene_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "biome_map.json").write_text(
        json.dumps(biome_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "density_map.json").write_text(
        json.dumps(density_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "output_root": str(output_root),
        "scene_obj": str(final_obj),
        "scene_metadata": str(output_root / "scene_metadata.json"),
        "biome_map": str(output_root / "biome_map.json"),
        "density_map": str(output_root / "density_map.json"),
        "seed": seed,
        "grid_size": grid_size,
    }


def main() -> None:
    result = _generate_dataset()
    print(f"Generated dataset folder: {result['output_root']}")
    print(f"scene.obj: {result['scene_obj']}")
    print(f"scene_metadata.json: {result['scene_metadata']}")
    print(f"biome_map.json: {result['biome_map']}")
    print(f"density_map.json: {result['density_map']}")
    print(f"seed={result['seed']} grid={result['grid_size']}x{result['grid_size']}")


if __name__ == "__main__":
    main()
