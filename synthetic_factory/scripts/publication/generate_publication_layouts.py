from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Unable to locate project root from script path.")


# Allow direct execution:
# uv run python scripts/publication/generate_publication_layouts.py
# or legacy wrapper:
# uv run python scripts/generate_publication_layouts.py
ROOT_DIR = _find_project_root(Path(__file__).resolve())
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from synthetic_factory.config import Config
from synthetic_factory.exporters.obj_exporter import ObjExporter
from synthetic_factory.generators.factory_generator import FactoryGenerator, FactoryParams, RoomLayout


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    title: str
    description: str
    overrides: Mapping[str, Any]


@dataclass(frozen=True)
class TopologyEdge:
    source: str
    target: str
    edge_type: str  # "corridor" | "direct_link"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def _scenario_specs() -> list[ScenarioSpec]:
    return [
        ScenarioSpec(
            name="linear_spine",
            title="Linear Spine Topology",
            description="Elongated building with mostly serial corridor connectivity.",
            overrides={
                "factory": {"width": 176.0, "depth": 18.0},
                "layout": {"strategy": "grid", "number_of_rooms": 11, "room_count_range": [11, 11], "corridor_width": 2.2},
                "rooms": {"min_size": 6.0, "max_size": 13.2, "height": 4.8},
                "biomes": {
                    "enabled": True,
                    "ensure_all_types": True,
                    "allow_corridorless_links": False,
                    "corridorless_link_ratio": 0.0,
                    "connector_spurs": True,
                },
                "noise": {"seed": 41, "full_occupancy": True},
            },
        ),
        ScenarioSpec(
            name="dense_lattice",
            title="Dense Lattice / Hybrid Links",
            description="Dense near-square grid with many direct links in addition to corridors.",
            overrides={
                "factory": {"width": 96.0, "depth": 96.0},
                "layout": {"strategy": "grid", "number_of_rooms": 16, "room_count_range": [16, 16], "corridor_width": 2.8},
                "rooms": {"min_size": 5.6, "max_size": 14.0, "height": 5.0},
                "biomes": {
                    "enabled": True,
                    "ensure_all_types": True,
                    "allow_corridorless_links": True,
                    "corridorless_link_ratio": 0.82,
                    "corridorless_height_delta": 2.0,
                    "connector_spurs": True,
                },
                "noise": {"seed": 52},
            },
        ),
        ScenarioSpec(
            name="perlin_clustered_core",
            title="Perlin Clustered Core",
            description="Perlin occupancy with strong connectivity bias and compact core.",
            overrides={
                "factory": {"width": 102.0, "depth": 76.0},
                "layout": {"strategy": "perlin", "number_of_rooms": 14, "room_count_range": [14, 14], "corridor_width": 2.9},
                "rooms": {"min_size": 5.0, "max_size": 14.0, "height": 5.2},
                "noise": {
                    "seed": 1337,
                    "full_occupancy": False,
                    "slot_multiplier": 2.2,
                    "connectivity_bias": 1.0,
                    "threshold": 0.25,
                    "sampling_scale": 0.092,
                    "min_distance": 6.5,
                },
                "biomes": {
                    "enabled": True,
                    "ensure_all_types": False,
                    "allow_corridorless_links": True,
                    "corridorless_link_ratio": 0.38,
                    "connector_spurs": True,
                    "clustered_assignment": True,
                },
            },
        ),
        ScenarioSpec(
            name="perlin_branched",
            title="Perlin Branched Structure",
            description="Moderately sparse Perlin occupancy with branch-like connectivity.",
            overrides={
                "factory": {"width": 98.0, "depth": 72.0},
                "layout": {"strategy": "perlin", "number_of_rooms": 12, "room_count_range": [12, 12], "corridor_width": 3.1},
                "rooms": {"min_size": 4.6, "max_size": 12.2, "height": 4.9},
                "noise": {
                    "seed": 2024,
                    "full_occupancy": False,
                    "slot_multiplier": 2.8,
                    "connectivity_bias": 0.45,
                    "threshold": 0.54,
                    "sampling_scale": 0.075,
                    "min_distance": 10.0,
                },
                "biomes": {
                    "enabled": True,
                    "ensure_all_types": False,
                    "allow_corridorless_links": True,
                    "corridorless_link_ratio": 0.2,
                    "connector_spurs": False,
                    "clustered_assignment": False,
                },
            },
        ),
        ScenarioSpec(
            name="perlin_islands",
            title="Perlin Islands / Multi-Component",
            description="Sparse island-like topology with visibly separated connected components.",
            overrides={
                "factory": {"width": 106.0, "depth": 78.0},
                "layout": {"strategy": "perlin", "number_of_rooms": 12, "room_count_range": [12, 12], "corridor_width": 3.3},
                "rooms": {"min_size": 4.8, "max_size": 11.5, "height": 4.8},
                "noise": {
                    "seed": 2601,
                    "full_occupancy": False,
                    "slot_multiplier": 3.4,
                    "connectivity_bias": 0.0,
                    "threshold": 0.68,
                    "sampling_scale": 0.07,
                    "min_distance": 12.0,
                },
                "biomes": {
                    "enabled": True,
                    "ensure_all_types": False,
                    "allow_corridorless_links": False,
                    "corridorless_link_ratio": 0.0,
                    "connector_spurs": False,
                    "clustered_assignment": False,
                },
            },
        ),
    ]


def _to_factory_params(cfg: Mapping[str, Any]) -> FactoryParams:
    return FactoryParams(
        factory_width=float(cfg["factory"]["width"]),
        factory_depth=float(cfg["factory"]["depth"]),
        number_of_rooms=int(cfg["layout"]["number_of_rooms"]),
        room_size_range=(float(cfg["rooms"]["min_size"]), float(cfg["rooms"]["max_size"])),
        corridor_width=float(cfg["layout"]["corridor_width"]),
        room_height=float(cfg["rooms"]["height"]),
        layout_strategy=str(cfg["layout"]["strategy"]),
        noise=dict(cfg.get("noise", {})),
        biomes=dict(cfg.get("biomes", {})),
        columns={"enabled": False},
        beams={"enabled": False},
        machinery={"enabled": False},
        exterior={"enabled": False},
        site={"enabled": False},
        seed=int(cfg.get("random", {}).get("seed", 0)),
    )


def _extract_edges(layout: list[RoomLayout], scene: Any) -> list[TopologyEdge]:
    cell_to_room: dict[tuple[int, int], RoomLayout] = {(room.row, room.col): room for room in layout}

    corridor_group = scene.get_object("corridors")
    link_ids: set[str] = set()
    if corridor_group is not None:
        for obj in corridor_group.traverse():
            link_ids.add(obj.id)

    edges: list[TopologyEdge] = []
    for room in layout:
        right = cell_to_room.get((room.row, room.col + 1))
        if right is not None:
            link_id = f"room_link_v_{room.col + 1}_{room.row + 1}"
            edge_type = "direct_link" if link_id in link_ids else "corridor"
            edges.append(TopologyEdge(source=room.room_id, target=right.room_id, edge_type=edge_type))
        down = cell_to_room.get((room.row + 1, room.col))
        if down is not None:
            link_id = f"room_link_h_{room.row + 1}_{room.col + 1}"
            edge_type = "direct_link" if link_id in link_ids else "corridor"
            edges.append(TopologyEdge(source=room.room_id, target=down.room_id, edge_type=edge_type))
    return edges


def _connected_components(room_ids: Iterable[str], edges: list[TopologyEdge]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)

    unseen = set(room_ids)
    components: list[list[str]] = []
    while unseen:
        root = unseen.pop()
        queue = deque([root])
        component = [root]
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, set()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda item: (len(item), item))


def _build_topology_metrics(layout: list[RoomLayout], edges: list[TopologyEdge]) -> dict[str, Any]:
    room_ids = [room.room_id for room in layout]
    components = _connected_components(room_ids, edges)
    n = len(layout)
    e = len(edges)
    degree_sum = 2 * e
    avg_degree = degree_sum / n if n > 0 else 0.0
    density = (2 * e) / (n * (n - 1)) if n > 1 else 0.0
    direct_links = sum(1 for edge in edges if edge.edge_type == "direct_link")
    corridor_links = e - direct_links
    return {
        "room_count": n,
        "edge_count": e,
        "direct_link_edges": direct_links,
        "corridor_edges": corridor_links,
        "direct_link_ratio": (direct_links / e) if e > 0 else 0.0,
        "avg_degree": avg_degree,
        "density": density,
        "component_count": len(components),
        "components": components,
    }


def _draw_svg(
    layout: list[RoomLayout],
    edges: list[TopologyEdge],
    title: str,
    path: Path,
    metrics: Mapping[str, Any],
) -> None:
    biome_color = {
        "workshop": "#00A6FB",
        "office": "#00E676",
        "boiler": "#FF3D00",
        "refinery": "#FFB300",
        "electrical": "#7C4DFF",
        "maintenance": "#FF6D00",
        "laboratory": "#00E5FF",
        "control": "#D500F9",
        "storage": "#76FF03",
        "unassigned": "#90A4AE",
    }
    room_by_id = {room.room_id: room for room in layout}
    component_palette = ["#F50057", "#18FFFF", "#FFEA00", "#7C4DFF", "#69F0AE", "#FF6D00"]
    room_component: dict[str, int] = {}
    for idx, component in enumerate(metrics.get("components", [])):
        for room_id in component:
            room_component[str(room_id)] = idx
    min_x = min(room.center_x - room.width / 2.0 for room in layout)
    max_x = max(room.center_x + room.width / 2.0 for room in layout)
    min_y = min(room.center_y - room.depth / 2.0 for room in layout)
    max_y = max(room.center_y + room.depth / 2.0 for room in layout)

    margin = 64.0
    scale = 13.0
    canvas_w = int((max_x - min_x) * scale + margin * 2.0)
    canvas_h = int((max_y - min_y) * scale + margin * 2.0 + 76.0)

    def to_svg_x(x: float) -> float:
        return margin + (x - min_x) * scale

    def to_svg_y(y: float) -> float:
        return margin + (max_y - y) * scale

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">')
    lines.append("<defs>")
    lines.append('<linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">')
    lines.append('<stop offset="0%" stop-color="#020617"/>')
    lines.append('<stop offset="55%" stop-color="#0B1024"/>')
    lines.append('<stop offset="100%" stop-color="#101827"/>')
    lines.append("</linearGradient>")
    lines.append('<filter id="edgeGlow" x="-20%" y="-20%" width="140%" height="140%">')
    lines.append('<feDropShadow dx="0" dy="0" stdDeviation="1.5" flood-color="#FDE047" flood-opacity="0.55"/>')
    lines.append("</filter>")
    lines.append("</defs>")
    lines.append('<rect width="100%" height="100%" fill="url(#bgGrad)"/>')
    grid_step = 84
    for x in range(0, canvas_w + 1, grid_step):
        lines.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{canvas_h}" stroke="#1E293B" stroke-width="1" opacity="0.42"/>')
    for y in range(0, canvas_h + 1, grid_step):
        lines.append(f'<line x1="0" y1="{y}" x2="{canvas_w}" y2="{y}" stroke="#1E293B" stroke-width="1" opacity="0.42"/>')
    lines.append(
        f'<text x="{margin}" y="34" fill="#F8FAFC" font-size="24" font-weight="700" '
        'font-family="Segoe UI, Arial">'
        f"{title}</text>"
    )
    lines.append(
        f'<text x="{margin}" y="54" fill="#C4D4F4" font-size="12" font-family="Segoe UI, Arial">'
        "Room topology and connectivity overview</text>"
    )
    stats_text = (
        f"rooms={metrics.get('room_count', 0)}   "
        f"edges={metrics.get('edge_count', 0)}   "
        f"direct={metrics.get('direct_link_edges', 0)}   "
        f"components={metrics.get('component_count', 0)}   "
        f"avg_degree={float(metrics.get('avg_degree', 0.0)):.2f}"
    )
    lines.append(
        f'<text x="{margin}" y="72" fill="#E2E8F0" font-size="12" font-weight="600" font-family="Consolas, Segoe UI, Arial">'
        f"{stats_text}</text>"
    )

    for edge in edges:
        source = room_by_id.get(edge.source)
        target = room_by_id.get(edge.target)
        if source is None or target is None:
            continue
        x1 = to_svg_x(source.center_x)
        y1 = to_svg_y(source.center_y)
        x2 = to_svg_x(target.center_x)
        y2 = to_svg_y(target.center_y)
        if edge.edge_type == "direct_link":
            lines.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#FFD740" stroke-width="3.4" '
                'stroke-linecap="round" filter="url(#edgeGlow)"/>'
            )
        else:
            lines.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#80DEEA" stroke-width="2.5" '
                'stroke-dasharray="8 5" stroke-linecap="round" opacity="0.95"/>'
            )

    for room in layout:
        x = to_svg_x(room.center_x - room.width / 2.0)
        y = to_svg_y(room.center_y + room.depth / 2.0)
        w = room.width * scale
        h = room.depth * scale
        color = biome_color.get(room.biome, biome_color["unassigned"])
        cx = to_svg_x(room.center_x)
        cy = to_svg_y(room.center_y)
        component_idx = room_component.get(room.room_id, 0)
        component_color = component_palette[component_idx % len(component_palette)]
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{color}" fill-opacity="0.46" stroke="{component_color}" stroke-opacity="0.98" stroke-width="2.8" rx="5" ry="5"/>'
        )
        lines.append(
            f'<rect x="{x + 2:.2f}" y="{y + 2:.2f}" width="{max(w - 4, 1):.2f}" height="{max(h - 4, 1):.2f}" '
            f'fill="none" stroke="#FFFFFF" stroke-width="1.1" rx="4" ry="4" opacity="0.9"/>'
        )
        lines.append(
            f'<text x="{cx:.2f}" y="{cy - 4:.2f}" '
            'fill="#FFFFFF" font-size="10.5" font-weight="700" text-anchor="middle" '
            'stroke="#020617" stroke-width="1.6" paint-order="stroke" font-family="Segoe UI, Arial">'
            f'{room.room_id}</text>'
        )
        lines.append(
            f'<text x="{cx:.2f}" y="{cy + 11:.2f}" '
            'fill="#E3F2FD" font-size="9.5" text-anchor="middle" '
            'stroke="#020617" stroke-width="1.2" paint-order="stroke" font-family="Segoe UI, Arial">'
            f'{room.biome}</text>'
        )

    legend_w = 560
    legend_h = 32
    legend_x = margin
    legend_y = canvas_h - 30
    lines.append(
        f'<rect x="{legend_x - 12:.2f}" y="{legend_y - 22:.2f}" width="{legend_w}" height="{legend_h}" '
        'fill="#020617" fill-opacity="0.68" stroke="#1E293B" stroke-width="1.0" rx="6" ry="6"/>'
    )
    lines.append(
        f'<line x1="{margin}" y1="{legend_y}" x2="{margin + 44}" y2="{legend_y}" stroke="#80DEEA" stroke-width="2.5" stroke-dasharray="8 5"/>'
    )
    lines.append(
        f'<text x="{margin + 52}" y="{legend_y + 4}" fill="#B2EBF2" font-size="11" font-family="Segoe UI, Arial">corridor edge</text>'
    )
    lines.append(
        f'<line x1="{margin + 204}" y1="{legend_y}" x2="{margin + 248}" y2="{legend_y}" stroke="#FFD740" stroke-width="3.4"/>'
    )
    lines.append(
        f'<text x="{margin + 256}" y="{legend_y + 4}" fill="#FFE082" font-size="11" font-family="Segoe UI, Arial">direct link edge</text>'
    )
    lines.append(
        f'<line x1="{margin + 408}" y1="{legend_y}" x2="{margin + 452}" y2="{legend_y}" stroke="#F50057" stroke-width="3.0"/>'
    )
    lines.append(
        f'<text x="{margin + 460}" y="{legend_y + 4}" fill="#F8BBD0" font-size="11" font-family="Segoe UI, Arial">component border</text>'
    )

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Publication Layout Examples",
        "",
        "| Scenario | Rooms | Edges | Direct links | Components | Avg degree | Density |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['name']} | {row['room_count']} | {row['edge_count']} | "
            f"{row['direct_link_edges']} | {row['component_count']} | "
            f"{row['avg_degree']:.2f} | {row['density']:.3f} |"
        )
    lines.append("")
    lines.append("Legend: dashed cyan edge = corridor connection, solid yellow edge = direct room-to-room link.")
    lines.append("Room border color encodes connected component (for multi-component topologies).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_publication_package(output_root: Path, rows: list[dict[str, Any]]) -> None:
    publication_dir = output_root / "publication_ready"
    publication_dir.mkdir(parents=True, exist_ok=True)

    figure_index_lines = [
        "# Figure 1 Assets",
        "",
        "Sub-figures prepared for publication (topology/connectivity examples):",
        "",
    ]

    for index, row in enumerate(rows):
        letter = chr(ord("a") + index)
        scenario_name = str(row["name"])
        scenario_title = str(row.get("title", scenario_name))

        source_svg = Path(str(row["svg"]))
        source_json = Path(str(row["topology_json"]))
        source_obj = Path(str(row["obj"]))

        figure_svg = publication_dir / f"figure_1_{letter}_{scenario_name}.svg"
        topology_json = publication_dir / f"figure_1_{letter}_{scenario_name}_topology.json"
        layout_obj = publication_dir / f"figure_1_{letter}_{scenario_name}.obj"

        shutil.copy2(source_svg, figure_svg)
        shutil.copy2(source_json, topology_json)
        shutil.copy2(source_obj, layout_obj)

        figure_index_lines.append(
            f"- **Figure 1{letter}**: `{scenario_title}` -> `{figure_svg.name}`"
        )

    figure_index_path = publication_dir / "figure_1_index.md"
    figure_index_path.write_text("\n".join(figure_index_lines) + "\n", encoding="utf-8")

    csv_path = publication_dir / "table_1_topology_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "scenario",
                "title",
                "rooms",
                "edges",
                "direct_links",
                "corridor_links",
                "components",
                "avg_degree",
                "density",
                "direct_link_ratio",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["name"],
                    row.get("title", row["name"]),
                    row["room_count"],
                    row["edge_count"],
                    row["direct_link_edges"],
                    row["corridor_edges"],
                    row["component_count"],
                    f"{float(row['avg_degree']):.4f}",
                    f"{float(row['density']):.4f}",
                    f"{float(row['direct_link_ratio']):.4f}",
                ]
            )

    table_md_lines = [
        "# Table 1",
        "",
        "| Scenario | Rooms | Edges | Direct links | Corridor links | Components | Avg degree | Density |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_md_lines.append(
            "| "
            f"{row['name']} | {row['room_count']} | {row['edge_count']} | {row['direct_link_edges']} | "
            f"{row['corridor_edges']} | {row['component_count']} | {float(row['avg_degree']):.2f} | {float(row['density']):.3f} |"
        )
    table_md_lines.append("")
    table_md_lines.append(
        "Legend: corridor edges are dashed cyan; direct links are solid yellow."
    )
    table_md_path = publication_dir / "table_1_topology_metrics.md"
    table_md_path.write_text("\n".join(table_md_lines) + "\n", encoding="utf-8")

    insert_lines = [
        "# Publication Insert Template (RU)",
        "",
        "Рисунок 1 демонстрирует примеры сгенерированных планировок с различной топологией помещений и связностью.",
        "Подфигуры 1a-1e соответствуют линейной, решеточной, кластерной, ветвящейся и островной структурам.",
        "",
        "Таблица 1 содержит сравнительные графовые характеристики для тех же конфигураций",
        "(число комнат, число связей, доля прямых связей, число компонент связности, средняя степень, плотность).",
        "",
        "## Files",
        "- Figure assets: `publication_ready/figure_1_*.svg`",
        "- Topology metadata: `publication_ready/figure_1_*_topology.json`",
        "- Table CSV: `publication_ready/table_1_topology_metrics.csv`",
        "- Table Markdown: `publication_ready/table_1_topology_metrics.md`",
    ]
    insert_path = publication_dir / "publication_insert_template_ru.md"
    insert_path.write_text("\n".join(insert_lines) + "\n", encoding="utf-8")


def main() -> None:
    output_root = ROOT_DIR / "out" / "publication" / "layout_topologies" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    default_cfg = Config.from_default().to_dict()
    exporter = ObjExporter()

    summary_rows: list[dict[str, Any]] = []
    for spec in _scenario_specs():
        cfg = _deep_merge(deepcopy(default_cfg), spec.overrides)
        params = _to_factory_params(cfg)
        generator = FactoryGenerator(params)

        layout = generator.generate_layout()
        generator.instantiate_rooms()
        scene = generator.place_corridors()

        edges = _extract_edges(layout, scene)
        metrics = _build_topology_metrics(layout, edges)

        scenario_dir = output_root / spec.name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        obj_path = scenario_dir / f"{spec.name}.obj"
        svg_path = scenario_dir / f"{spec.name}.svg"
        json_path = scenario_dir / f"{spec.name}_topology.json"

        exporter.export(scene, str(obj_path))
        _draw_svg(layout, edges, spec.title, svg_path, metrics)

        payload = {
            "scenario": {
                "name": spec.name,
                "title": spec.title,
                "description": spec.description,
                "overrides": dict(spec.overrides),
            },
            "metrics": metrics,
            "layout": [
                {
                    "room_id": room.room_id,
                    "row": room.row,
                    "col": room.col,
                    "biome": room.biome,
                    "center_x": room.center_x,
                    "center_y": room.center_y,
                    "width": room.width,
                    "depth": room.depth,
                }
                for room in layout
            ],
            "edges": [
                {"source": edge.source, "target": edge.target, "edge_type": edge.edge_type}
                for edge in edges
            ],
            "artifacts": {
                "obj": str(obj_path),
                "svg": str(svg_path),
                "topology_json": str(json_path),
            },
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        summary_rows.append(
            {
                "name": spec.name,
                "title": spec.title,
                "description": spec.description,
                **metrics,
                "obj": str(obj_path),
                "svg": str(svg_path),
                "topology_json": str(json_path),
            }
        )

    summary_json_path = output_root / "summary.json"
    summary_md_path = output_root / "summary.md"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_summary_markdown(summary_md_path, summary_rows)
    _write_publication_package(output_root, summary_rows)

    print(f"Publication materials generated in: {output_root}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary Markdown: {summary_md_path}")


if __name__ == "__main__":
    main()
