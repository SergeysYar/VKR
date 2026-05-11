from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG: dict[str, Any] = {
    "factory": {
        "name": "synthetic_factory",
        "width": 100.0,
        "depth": 80.0,
        "height": 14.0,
        "levels": 1,
        "units": "meters",
        "ground_elevation": 0.0,
        "safety_clearance": 1.5,
    },
    "layout": {
        "strategy": "perlin",
        "number_of_rooms": 8,
        "room_count_range": [8, 10],
        "corridor_width": 4.0,
        "boundary_margin": 2.0,
        "snap_grid": 0.5,
        "allow_irregular_rooms": False,
    },
    "noise": {
        "seed": 42,
        "octaves": 4,
        "frequency": 1.0,
        "persistence": 0.5,
        "lacunarity": 2.0,
        "sampling_scale": 0.08,
        "full_occupancy": True,
        "slot_multiplier": 2.0,
        "threshold": 0.35,
        "connectivity_bias": 0.8,
        "min_distance": 10.0,
        "aspect_strength": 0.35,
        "room_margin_ratio": 0.92,
    },
    "rooms": {
        "min_size": 5.0,
        "max_size": 20.0,
        "height": 6.0,
        "wall_thickness": 0.2,
        "door_count": 1,
        "window_count": 2,
        "size_distribution": "uniform",
        "zoning": {
            "production_ratio": 0.65,
            "storage_ratio": 0.2,
            "service_ratio": 0.1,
            "office_ratio": 0.05,
        },
    },
    "biomes": {
        "enabled": True,
        "ensure_all_types": True,
        "unified_shell": True,
        "shell_ceiling_mode": "stepped",
        "corridor_ceiling_mode": "global_min_room",
        "shell_wall_height_mode": "ceiling",
        "connector_spurs": True,
        "connector_span": 2.4,
        "allow_corridorless_links": True,
        "corridorless_link_ratio": 0.18,
        "corridorless_height_delta": 1.4,
        "shell_wall_thickness": 0.22,
        "workshop_biome": "workshop",
        "workshop_area_ratio": 0.75,
        "clustered_assignment": True,
        "height_grouping": True,
        "height_group_axis": "auto",
        "height_group_high_first": False,
        "inter_room_doors": True,
        "door_height": 2.3,
        "industrial_size_bias": False,
        "industrial_biomes": [
            "workshop",
            "refinery",
            "boiler",
            "storage",
            "electrical",
            "maintenance",
            "laboratory",
        ],
        "industrial_size_multiplier": 1.35,
        "non_industrial_size_multiplier": 0.55,
        "industrial_min_size_ratio": 0.78,
        "industrial_max_size_ratio": 0.99,
        "non_industrial_min_size_ratio": 0.28,
        "non_industrial_max_size_ratio": 0.58,
        "cycle_order": [
            "workshop",
            "office",
            "boiler",
            "storage",
            "electrical",
            "maintenance",
            "laboratory",
            "control",
        ],
        "profiles": {
            "workshop": {
                "height_multiplier": 1.25,
                "size_multiplier": 1.22,
                "min_height": 6.0,
                "max_height": 12.0,
                "door_count": 2,
                "window_count": 2,
            },
            "office": {
                "height_multiplier": 0.85,
                "size_multiplier": 0.82,
                "min_height": 2.8,
                "max_height": 4.2,
                "door_count": 1,
                "window_count": 4,
            },
            "refinery": {
                "height_multiplier": 1.6,
                "size_multiplier": 1.28,
                "min_height": 12.0,
                "max_height": 20.0,
                "door_count": 1,
                "window_count": 0,
            },
            "boiler": {
                "height_multiplier": 1.5,
                "size_multiplier": 1.15,
                "min_height": 10.0,
                "door_count": 1,
                "window_count": 1,
            },
            "storage": {
                "height_multiplier": 1.0,
                "size_multiplier": 1.05,
                "min_height": 4.0,
                "max_height": 8.0,
                "door_count": 2,
                "window_count": 1,
            },
            "electrical": {
                "height_multiplier": 0.9,
                "size_multiplier": 0.95,
                "min_height": 3.2,
                "max_height": 5.2,
                "door_count": 1,
                "window_count": 0,
            },
            "maintenance": {
                "height_multiplier": 0.95,
                "size_multiplier": 0.9,
                "min_height": 3.4,
                "max_height": 6.0,
                "door_count": 1,
                "window_count": 1,
            },
            "laboratory": {
                "height_multiplier": 0.9,
                "size_multiplier": 1.0,
                "min_height": 3.0,
                "max_height": 5.0,
                "door_count": 1,
                "window_count": 2,
            },
            "control": {
                "height_multiplier": 0.85,
                "size_multiplier": 0.78,
                "min_height": 2.8,
                "max_height": 4.2,
                "door_count": 1,
                "window_count": 2,
            },
        },
    },
    "walls": {
        "thickness": 0.2,
        "material": "concrete",
        "fire_rating": "REI120",
        "insulation": {
            "enabled": True,
            "type": "mineral_wool",
            "thickness": 0.08,
        },
        "finish": {
            "inner": "paint",
            "outer": "plaster",
        },
    },
    "exterior": {
        "enabled": True,
        "group_id": "exterior",
        "seed": 0,
        "building": {
            "margin": [1.0, 5.0],
            "height_variation": [0.0, 2.0],
        },
        "equipment_density": [0.5, 2.0],
        "wall_thickness": 0.34,
        "wall_offset": 0.12,
        "wall_headroom": 0.25,
        "roof_thickness": 0.3,
        "roof_overhang": 0.55,
        "roof_lift": 0.0,
        "roof": {
            "type": ["flat", "sawtooth", "gabled"],
            "ridge_axis": "x",
            "gabled_rise": 1.8,
            "sawtooth_axis": "x",
            "sawtooth_rise": 1.5,
            "tooth_count": "auto",
            "tooth_width": 8.0,
            "clerestory_ratio": 0.28,
        },
        "foundation_thickness": 0.45,
        "foundation_border": 1.0,
        "apron_width": 2.2,
        "apron_thickness": 0.08,
        "parapet": {
            "enabled": True,
            "height": 0.9,
            "thickness": 0.24,
        },
        "gates": {
            "enabled": True,
            "side": "south",
            "count": "auto",
            "width": 3.8,
            "height": 4.2,
            "depth": 0.22,
            "edge_margin": 2.2,
            "gap": 1.6,
        },
        "opening_clearance": 0.18,
        "door_width": 1.25,
        "facade": {
            "section_width": 4.0,
            "panel_width": 4.2,
            "panel_thickness": 0.08,
            "panel_reveal": 0.05,
            "section_every": 3,
            "seam_width": 0.12,
            "min_panel_height": 2.6,
            "window_density": [0.0, 0.5],
            "functional": {
                "enabled": True,
                "min_gate_segment": 4.8,
            },
            "windows": {
                "width": 1.55,
                "height": 1.35,
                "thickness": 0.06,
                "sill_height": 1.3,
                "spacing": 1.8,
            },
            "doors": {
                "leaf_depth": 0.08,
            },
            "gates": {
                "leaf_depth": 0.1,
            },
            "gate_frame": {
                "thickness": 0.16,
                "depth": 0.2,
                "gap_from_wall": 0.04,
            },
        },
        "canopies": {
            "enabled": True,
            "depth": 2.4,
            "thickness": 0.14,
            "width_factor": 1.25,
            "z_offset": 0.25,
        },
        "ventilation": {
            "enabled": True,
            "count": "auto",
            "count_per_1000m2": 3.0,
            "shaft_width": 0.85,
            "shaft_depth": 0.85,
            "shaft_height": 1.6,
            "edge_margin": 2.4,
        },
        "roof_pipes": {
            "enabled": True,
            "radius": 0.24,
            "height": 2.4,
            "per_boiler_room": 2,
        },
        "roof_units": {
            "enabled": True,
            "count_per_1000m2": 2.0,
            "width": 1.8,
            "depth": 1.4,
            "height": 1.0,
            "edge_margin": 2.8,
        },
        "biome_links": {
            "enabled": True,
            "boiler": {
                "pipe_radius": 0.24,
                "pipe_length": 2.8,
                "pipes_per_room": "auto",
                "chimney_radius": 0.34,
                "chimney_height": 7.0,
            },
            "electrical": {
                "transformers_per_room": "auto",
                "transformer_width": 2.4,
                "transformer_depth": 1.8,
                "transformer_height": 2.2,
                "cable_height": 2.4,
                "cable_thickness": 0.14,
            },
            "maintenance": {
                "pad_depth": 5.2,
                "pad_thickness": 0.08,
            },
            "laboratory": {
                "vent_radius": 0.22,
                "vent_height": 1.6,
                "block_width": 1.1,
                "block_depth": 0.9,
                "block_height": 0.72,
            },
        },
        "services": {
            "enabled": True,
            "tray_height": 3.2,
            "tray_width": 0.45,
            "tray_thickness": 0.12,
            "tray_offset": 1.0,
            "support_spacing": 3.0,
            "support_width": 0.12,
            "support_depth": 0.12,
            "corner_margin": 1.4,
            "cable_height": 3.3,
            "cable_thickness": 0.09,
            "pipe_height": 3.7,
            "pipe_radius": 0.14,
            "pipe_casing_thickness": 0.34,
            "connectors_per_room": "auto",
            "pipe_biomes": [
                "boiler",
                "refinery",
                "workshop",
                "maintenance",
                "laboratory",
                "storage",
                "electrical",
            ],
            "wire_biomes": [
                "office",
                "control",
                "electrical",
                "laboratory",
                "maintenance",
                "workshop",
                "storage",
                "refinery",
                "boiler",
            ],
        },
        "rules": {
            "enabled": True,
            "auto_fix": True,
            "shell_margin": 0.05,
            "collision_padding": 0.06,
            "collision_step": 0.45,
            "min_entrances": 1,
            "access_clearance": 1.6,
            "pipe_tolerance": 0.35,
            "pipe_link_thickness": 0.18,
            "ground_tolerance": 0.03,
            "ground_contact_types": [
                "exterior_transformer",
                "exterior_repair_pad",
                "exterior_door",
                "exterior_gate",
            ],
        },
    },
    "site": {
        "enabled": True,
        "group_id": "site",
        "seed": 0,
        "terrain_margin": 22.0,
        "terrain": {
            "thickness": 0.24,
            "size": [50.0, 200.0],
            "road_density": [0.5, 2.0],
            "height_noise": {
                "enabled": True,
                "amplitude": 0.06,
                "cells_x": 10,
                "cells_y": 8,
                "patch_thickness": 0.1,
            },
        },
        "road": {
            "enabled": True,
            "width": 6.0,
            "thickness": 0.12,
            "offset_from_building": 5.5,
            "connect_to_building": True,
            "connect_gates": True,
            "connect_doors": True,
            "connector_width": 4.2,
            "min_connector_spacing": 2.8,
        },
        "parking": {
            "enabled": False,
            "side": "north",
            "width": 26.0,
            "depth": 15.0,
            "thickness": 0.08,
            "offset_from_road": 2.4,
            "slots": {
                "width": 2.6,
                "depth": 5.2,
                "aisle": 6.0,
                "line_width": 0.08,
                "line_height": 0.02,
            },
        },
        "zones": {
            "enabled": True,
            "thickness": 0.07,
            "margin": 1.1,
            "min_size": 2.2,
        },
        "logistics": {
            "enabled": True,
            "dock_count": "auto",
            "dock_width": 4.8,
            "dock_depth": 2.4,
            "dock_height": 1.2,
            "dock_gap": 1.6,
            "edge_margin": 2.8,
        },
        "fence": {
            "enabled": True,
            "inset": 2.0,
            "height": 2.6,
            "thickness": 0.16,
            "post_spacing": 6.0,
            "post_size": 0.18,
        },
        "industrial": {
            "enabled": True,
            "tanks": {
                "count": "auto",
                "radius": 1.15,
                "height": 3.4,
                "side_offset": 8.0,
                "span_margin": 4.0,
                "pipe_thickness": 0.22,
            },
            "transformers": {
                "count": 2,
                "width": 2.2,
                "depth": 1.8,
                "height": 2.0,
                "side_offset": 8.5,
            },
        },
    },
    "columns": {
        "enabled": True,
        "spacing": 6.0,
        "radius": 0.3,
        "edge_offset": 1.0,
        "height": 6.0,
        "count_x": 0,
        "count_y": 0,
    },
    "beams": {
        "enabled": True,
        "spacing": 6.0,
        "elevation": 5.5,
        "profile": {
            "type": "i",
            "width": 0.3,
            "height": 0.45,
            "web_thickness": 0.02,
            "flange_thickness": 0.03,
        },
    },
    "floors": {
        "thickness": 0.25,
        "material": "reinforced_concrete",
        "load_capacity_kpa": 15.0,
        "finish": "epoxy",
    },
    "openings": {
        "door": {
            "width": 1.2,
            "height": 2.4,
            "sill_height": 0.0,
            "lintel_height": 2.6,
        },
        "window": {
            "width": 1.6,
            "height": 1.4,
            "sill_height": 1.0,
        },
    },
    "machinery": {
        "enabled": True,
        "auxiliary": {
            "enabled": False,
            "seed": 0,
            "random_variation": True,
            "density": [0.5, 3.0],
            "complexity": [1, 5],
            "max_children_per_object": 2,
            "margin": 0.02,
            "allow_above_ratio": 0.24,
            "boiler": {
                "bunker_probability": [0.2, 0.8],
                "pump_count": [1, 4],
            },
            "electrical": {
                "transformer_probability": [0.3, 0.7],
            },
            "laboratory": {
                "clutter_level": [0.5, 2.0],
            },
        },
        "infrastructure": {
            "enabled": True,
            "group_id": "global_infrastructure",
            "biome_type": "auto",
            "biome_profiles": {
                "boiler": {
                    "include_pipes": True,
                    "pipe_radius_scale": 1.75,
                    "branch_count_scale": 1.8,
                    "branch_ratio_scale": 1.2,
                    "extra_pipe_lines": 2,
                },
                "electrical": {
                    "include_pipes": True,
                    "tray_width_scale": 1.2,
                    "room_link_width_scale": 1.25,
                    "branch_count_scale": 2.0,
                    "strict_linearity": True,
                },
                "control": {
                    "include_pipes": True,
                    "tray_width_scale": 0.72,
                    "room_link_width_scale": 0.68,
                    "branch_count_scale": 0.55,
                    "hidden_layout": True,
                },
                "laboratory": {
                    "include_pipes": True,
                    "pipe_radius_scale": 0.65,
                    "room_link_width_scale": 0.85,
                    "multi_target_connections": True,
                    "max_targets": 6,
                },
                "maintenance": {
                    "include_pipes": True,
                    "branch_count_scale": 0.85,
                    "hanging_cables": 3,
                    "hanging_ratio": 0.46,
                },
            },
            "routing_level": "ceiling",
            "ceiling_offset": 0.25,
            "tray_height": [2.5, 4.0],
            "tray_width": [0.2, 1.0],
            "tray_thickness": 0.12,
            "room_link_width": 0.28,
            "density": [0.5, 3.0],
            "branch_frequency": [0.5, 2.0],
            "vertical": {
                "drop_frequency": [0.2, 1.5],
            },
            "supports": {
                "spacing": [1.0, 3.0],
            },
            "parallel_channels": 2,
            "channel_spacing": 4.0,
            "cross_connections": True,
            "cross_spacing": 12.0,
            "junction_size": 0.28,
            "backbone_margin": 1.0,
            "wall_clearance": 0.25,
            "min_segment_length": 0.25,
            "include_pipes": True,
            "pipe_radius": 0.09,
            "pipe_drop": 0.28,
            "pipe_casing_enabled": True,
            "pipe_casing_thickness": 0.02,
            "pipe_casing_scale": 1.22,
            "tray_cable_bundle_enabled": True,
            "tray_cable_bundle_count": 2,
            "tray_cable_bundle_radius": 0.022,
            "tray_cable_bundle_max_offset": 0.08,
            "support_spacing": 2.0,
            "support_radius": 0.06,
            "drop_clearance": 0.25,
            "min_vertical_drop_length": 0.35,
            "secondary_branches": True,
            "secondary_branch_count": 2,
            "secondary_branch_ratio": 0.32,
            "top_clearance": 0.35,
            "rules": {
                "auto_fix": True,
                "clearance": 0.25,
                "min_height": 0.35,
                "top_margin": 0.2,
                "require_vertical_drop": True,
                "support_spacing": 2.0,
                "long_segment_length": 2.4,
                "support_tolerance": 0.22,
                "collision_spacing": 0.16,
                "collision_padding": 0.03,
            },
            "object_connections": {
                "enabled": True,
                "clearance": 0.16,
                "cable_radius": 0.03,
                "pipe_connection_radius": 0.1,
                "max_targets_per_room": 256,
            },
        },
        "density": 0.06,
        "clearance": 1.2,
        "types": ["cnc", "press", "conveyor", "robotic_cell"],
        "power_line_height": 4.5,
        "conveyors_per_room": 4,
        "machines_per_room": 6,
        "conveyor": {
            "width": 0.8,
            "height": 0.35,
            "elevation": 0.4,
        },
        "machine": {
            "width": 2.2,
            "depth": 1.4,
            "height": 1.8,
        },
        "boiler": {
            "random_variation": True,
            "pattern": "linear_boilers",
            "rules": {
                "auto_fix": True,
                "min_clearance": 1.2,
            },
            "min_walkway": 1.2,
            "service_clearance": 0.8,
            "edge_offset": 1.0,
            "boiler": {
                "count": [1, 3],
                "height": [8.0, 20.0],
                "radius": [1.0, 3.0],
            },
            "pipes": {
                "density": [0.5, 2.0],
                "radius": [0.1, 0.5],
            },
            "platforms": {
                "levels": [1, 3],
                "width_factor": [1.2, 2.0],
                "thickness": 0.18,
                "ladder_width": 0.75,
            },
            "tanks": {
                "count": [0, 5],
            },
            "structure": {
                "ceiling_height": "auto",
                "beam_density": [0.2, 1.0],
                "beam_offset": 1.6,
                "support_beam_profile": {
                    "type": "i",
                    "width": 0.22,
                    "height": 0.32,
                    "web_thickness": 0.016,
                    "flange_thickness": 0.022,
                },
            },
        },
        "refinery": {
            "seed": 0,
            "random_variation": True,
            "pattern": "tower_cluster",
            "walkway_width": 1.4,
            "main_columns": {
                "count": [2, 4],
                "height": [10.0, 24.0],
                "radius": [0.8, 1.8],
            },
            "secondary": {
                "count": [3, 10],
                "radius": [0.35, 0.95],
                "length": [1.8, 4.8],
            },
            "pipes": {
                "density": [1.0, 2.4],
                "radius": [0.12, 0.42],
                "levels": [2, 4],
                "support_spacing": [2.5, 4.2],
            },
            "platforms": {
                "levels": [2, 3],
                "width_factor": [1.4, 2.2],
                "guardrails": {
                    "enabled": False,
                    "height": 1.05,
                    "thickness": 0.08,
                },
            },
            "routing": {
                "smooth_bends": True,
                "corner_chamfer": 0.35,
                "pump_length_threshold": 7.0,
                "group_lane_spacing": 0.75,
            },
            "secondary_elements": {
                "enabled": True,
                "density": 1.0,
                "sensor_density": 1.0,
            },
            "structure": {
                "edge_offset": 1.0,
                "min_clearance": 0.9,
            },
            "rules": {
                "auto_fix": True,
                "min_clearance": 0.9,
                "pipe_collision_margin": 0.12,
                "pipe_vertical_step": 0.28,
                "support_spacing": 3.2,
            },
        },
        "electrical": {
            "seed": 0,
            "random_variation": True,
            "auto_fix": True,
            "pattern": "parallel_rows",
            "rules": {
                "clearance_front": 0.8,
                "min_spacing": 0.3,
                "enforce_cable_routing": True,
                "strict_alignment": True,
                "tray_height_clearance": 0.25,
                "cooling_access_clearance": 0.7,
            },
            "grid_snapping": True,
            "fixed_grid_step": 0.2,
            "orientation": "y+",
            "room": {
                "width": [10.0, 50.0],
                "depth": [10.0, 60.0],
                "height": [3.0, 6.0],
            },
            "cabinets": {
                "rows": [1, 6],
                "per_row": [3, 20],
                "spacing": [0.8, 1.5],
            },
            "walkways": {
                "width": [0.8, 2.0],
                "perimeter_width": 1.0,
            },
            "cable_trays": {
                "height": [2.2, 3.5],
                "density": [0.5, 2.0],
            },
            "cables": {
                "density": [0.5, 3.0],
                "optimize_bundles": True,
                "variation": 0.0,
            },
            "cooling": {
                "units": [0, 4],
            },
            "cabinet_width": 0.9,
            "cabinet_depth": 0.6,
            "cabinet_height": 2.2,
            "cabinet_door_type": "double",
            "row_step": 2.6,
            "min_clearance": 0.3,
            "access_depth": 0.8,
            "tray_mode": "overhead",
            "tray_width": 0.35,
            "tray_height": 0.12,
            "tray_support_spacing": 3.0,
            "cross_tray_count": 2,
            "cable_radius": 0.03,
            "junction_mode": "walls",
            "junction_count": 4,
            "junction_box_size": 0.3,
            "cooling_width": 0.9,
            "cooling_height": 2.0,
            "cooling_airflow_direction": "front_to_back",
        },
        "laboratory": {
            "seed": 0,
            "random_variation": True,
            "pattern": "research_lab",
            "auto_fix": True,
            "room": {
                "width": [8.0, 30.0],
                "depth": [8.0, 30.0],
                "height": [2.5, 4.5],
            },
            "stations": {
                "count": [2, 12],
                "spacing": [1.5, 3.0],
                "type_variability": [0.15, 0.65],
            },
            "equipment": {
                "density": [1.0, 5.0],
            },
            "fumehood": {
                "probability": [0.0, 0.5],
            },
            "storage": {
                "shelves": [0, 10],
                "cabinets": [0, 8],
            },
            "infrastructure": {
                "cable_density": [0.5, 2.0],
                "pipe_density": [0.2, 1.0],
            },
            "lighting": {
                "intensity": [0.5, 2.0],
            },
            "density": 0.55,
            "module_width": [1.9, 2.8],
            "module_depth": [1.4, 2.2],
            "equipment_per_station": [1, 5],
            "min_station_spacing": 0.95,
            "walkway_width": 1.3,
            "perimeter_walkway": 0.9,
            "access_depth": 0.85,
            "bench_height": 0.9,
            "fumehood_probability": 0.32,
            "sink_probability": 0.35,
            "near_station_storage_probability": 0.34,
            "cable_radius": 0.008,
            "pipe_radius": 0.01,
            "equipment_cable_mode": "mixed",
            "equipment_types": ["analyzer", "mixer", "centrifuge", "pump", "controller", "heater"],
            "equipment_type_map": {
                "chemistry": ["reactor", "mixer", "heater", "pump"],
                "analysis": ["analyzer", "spectrometer", "chromatograph", "sensor"],
                "preparation": ["balance", "dispenser", "stirrer", "controller"],
            },
            "station_pipe_probability": 0.45,
            "station_alignment_step": 0.04,
            "rules": {
                "auto_fix": True,
                "min_spacing": 0.95,
                "equipment_near_distance": 0.24,
                "equipment_min_spacing": 0.16,
                "safety_walkway_clearance": 0.06,
                "edge_margin": 0.04,
            },
            "light_grid_density": 1.0,
            "light_size": 0.52,
            "station_light_scale": 1.2,
            "water_enabled": False,
            "gas_enabled": False,
            "station_type_weights": {
                "chemistry": 0.34,
                "analysis": 0.38,
                "preparation": 0.28,
            },
        },
        "maintenance": {
            "seed": 0,
            "random_variation": True,
            "pattern": "active_repair",
            "density": [0.9, 1.35],
            "clutter": [0.4, 1.25],
            "organization": {
                "order_level": [0.0, 1.0],
            },
            "zones": {
                "repair": [1, 5],
                "storage": [1, 3],
            },
            "min_walkway_width": 1.2,
            "edge_margin": 0.8,
            "repair_zone_count": [4, 10],
            "storage_rack_count": [2, 8],
            "tools": {
                "density": [0.5, 3.0],
            },
            "parts": {
                "loose_parts": [0, 50],
                "large_parts": [0, 10],
            },
            "repair_scenario": {
                "enabled": False,
                "name": "repair",
                "count": [1, 2],
                "subcomponents": [3, 6],
                "tools_per_target": [1, 2],
                "tool_size_scale": [0.85, 1.15],
                "disconnected_lines": True,
                "cable_probability": [0.75, 1.0],
                "pipe_probability": [0.55, 0.95],
                "cable_length": [0.6, 1.6],
                "pipe_length": [0.55, 1.4],
                "cable_curvature": [0.15, 0.72],
                "pipe_curvature": [0.08, 0.58],
            },
            "major_spacing_x": 2.2,
            "major_row_spacing": 2.4,
            "workbench": {
                "count": [1, 10],
                "width": 1.8,
                "depth": 0.9,
                "height": 0.95,
            },
            "tool_rack": {
                "width": 1.0,
                "depth": 0.55,
                "height": 2.1,
            },
            "rules": {
                "min_clearance": 0.6,
                "access_depth": 1.0,
                "tool_max_distance": 2.2,
                "floor_tolerance": 0.08,
                "pallet_attach_tolerance": 0.12,
                "crane_clearance_radius": 0.55,
            },
            "crane": {
                "enabled": [True, False],
            },
        },
        "control": {
            "seed": 0,
            "random_variation": True,
            "pattern": "linear_control_room",
            "auto_fix": True,
            "density_profile": "medium",
            "panel_layout": "auto",
            "rack_layout": "walls",
            "room": {
                "width": [10.0, 40.0],
                "depth": [8.0, 30.0],
                "height": [2.5, 5.0],
            },
            "desks": {
                "rows": [1, 5],
                "cols": [2, 10],
                "spacing_x": [1.2, 2.0],
                "spacing_y": [1.5, 3.0],
            },
            "monitors": {
                "per_desk": [1, 4],
            },
            "panels": {
                "type": ["flat", "curved"],
                "height": [1.5, 3.0],
            },
            "racks": {
                "count": [0, 10],
            },
            "lighting": {
                "grid_density": [0.5, 2.0],
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
    "utilities": {
        "lighting": {
            "enabled": True,
            "spacing": 4.0,
            "mount_height": 5.8,
            "intensity_lux": 500,
        },
        "ventilation": {
            "enabled": True,
            "duct_height": 6.5,
            "airflow_m3h": 25000,
        },
        "sprinklers": {
            "enabled": True,
            "spacing": 3.0,
            "line_offset": 0.5,
        },
    },
    "materials": {
        "default_wall": "concrete",
        "default_floor": "epoxy",
        "default_column": "steel",
        "default_beam": "steel",
    },
    "export": {
        "format": "obj",
        "output_path": "out/factory.obj",
        "include_normals": True,
        "group_objects": True,
        "object_naming": "hierarchical",
    },
    "lidar": {
        "enabled": False,
        "output_path": "out/factory_lidar_scans.json",
        "output_dir": "out/factory_lidar_scans",
        "stitched_output_path": "out/factory_lidar_scans/factory_stitched.ply",
        "circle_file_prefix": "scan_circle",
        "export_individual_circles": True,
        "export_stitched_cloud": True,
        "export_stitched_by_room": False,
        "export_stitched_by_biome": False,
        "export_metadata_json": True,
        "retain_circle_records_in_context": False,
        "memory_safe_mode": True,
        "max_total_points_in_memory": 1200000,
        "scan_range": 24.0,
        "scan_pattern": "circular",
        "horizontal_fov_deg": 360.0,
        "angular_resolution_deg": 0.5,
        "vertical_resolution_deg": 0.8,
        "compute_backend": "auto",
        "cuda_ray_batch_size": 1024,
        "cuda_object_batch_size": 1024,
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
        "emit_no_hit_returns": False,
        "stitched_fill_blind_spots": True,
        "global_coverage": False,
        "factory_room_id": "__factory__",
        "include_factory_room": True,
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
            "roof": 13,
            "window": 14,
            "door": 15,
            "gate": 16,
            "terrain": 17,
            "facade": 18,
        },
    },
    "pipeline": {
        "enable_parameter_sampling": True,
        "enable_layout_generation": True,
        "enable_room_generation": True,
        "enable_scene_assembly": True,
        "enable_export": True,
        "strict_mode": True,
    },
    "random": {
        "seed": 42,
        "deterministic": True,
        "distribution_defaults": {
            "size": "uniform",
            "density": "normal",
        },
    },
    "validation": {
        "enforce_bounds": True,
        "fail_on_overlap": True,
        "min_walkway_width": 1.2,
        "check_opening_collisions": True,
    },
    "parameters": {
        "factory_width": {"value": 100.0, "min": 60.0, "max": 180.0, "distribution": "uniform"},
        "factory_depth": {"value": 80.0, "min": 50.0, "max": 150.0, "distribution": "uniform"},
        "number_of_rooms": {"value": 8, "min": 8, "max": 10, "distribution": "uniform"},
        "corridor_width": {"value": 4.0, "min": 2.0, "max": 8.0, "distribution": "uniform"},
        "room_size_min": {"value": 5.0, "min": 3.0, "max": 10.0, "distribution": "uniform"},
        "room_size_max": {"value": 20.0, "min": 8.0, "max": 30.0, "distribution": "uniform"},
        "room_height": {"value": "room_size_max * 0.3", "min": 4.0, "max": 10.0, "distribution": "normal"},
        "wall_thickness": {"value": "corridor_width * 0.05", "min": 0.12, "max": 0.45, "distribution": "uniform"},
        "column_spacing": {"value": 6.0, "min": 4.0, "max": 10.0, "distribution": "uniform"},
        "column_radius": {"value": 0.3, "min": 0.2, "max": 0.6, "distribution": "normal"},
    },
    "metadata": {
        "version": "1.0.0",
        "author": "synthetic_factory",
        "description": "Extended configuration for scene-oriented factory generation.",
    },
}

_MISSING = object()


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, override_value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(override_value, Mapping)
        ):
            base[key] = _deep_merge(dict(base[key]), override_value)
        else:
            base[key] = deepcopy(override_value)
    return base


def _ensure_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping/dict.")
    return value


@dataclass
class Config:
    """
    Extendable hierarchical config with path-based access.

    Paths support dot notation, for example:
    - `factory.width`
    - `rooms.min_size`
    """

    _data: dict[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Config:
        return cls(deepcopy(dict(data)))

    @classmethod
    def from_default(cls) -> Config:
        return cls.from_dict(DEFAULT_CONFIG)

    @classmethod
    def from_file(cls, path: str) -> Config:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Config file not found: {source}")

        suffix = source.suffix.lower()
        raw_text = source.read_text(encoding="utf-8")

        if suffix == ".json":
            loaded = json.loads(raw_text)
            return cls.from_dict(_ensure_mapping(loaded, "JSON root"))

        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "YAML config requires PyYAML. Install with: pip install pyyaml"
                ) from exc
            loaded = yaml.safe_load(raw_text)
            return cls.from_dict(_ensure_mapping(loaded, "YAML root"))

        if suffix == ".toml":
            try:
                import tomllib
            except ModuleNotFoundError as exc:
                raise ImportError("TOML parsing requires Python 3.11+ (tomllib).") from exc
            loaded = tomllib.loads(raw_text)
            return cls.from_dict(_ensure_mapping(loaded, "TOML root"))

        raise ValueError(
            "Unsupported config format. Use .json, .yaml/.yml, or .toml"
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def save_json(self, path: str, indent: int = 2) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self._data, indent=indent, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def save_yaml(self, path: str, sort_keys: bool = False) -> None:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "YAML export requires PyYAML. Install with: pip install pyyaml"
            ) from exc

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(
            self._data,
            allow_unicode=True,
            sort_keys=sort_keys,
            default_flow_style=False,
        )
        target.write_text(content, encoding="utf-8")

    def get(self, path: str, default: Any = _MISSING) -> Any:
        cursor: Any = self._data
        for chunk in self._split_path(path):
            if not isinstance(cursor, Mapping) or chunk not in cursor:
                if default is _MISSING:
                    raise KeyError(f"Config path not found: {path}")
                return default
            cursor = cursor[chunk]
        return cursor

    def require(self, path: str) -> Any:
        return self.get(path)

    def set(self, path: str, value: Any) -> Config:
        chunks = self._split_path(path)
        if not chunks:
            raise ValueError("Config path cannot be empty.")

        cursor: dict[str, Any] = self._data
        for chunk in chunks[:-1]:
            next_value = cursor.get(chunk)
            if not isinstance(next_value, dict):
                next_value = {}
                cursor[chunk] = next_value
            cursor = next_value
        cursor[chunks[-1]] = value
        return self

    def override(self, values: Mapping[str, Any]) -> Config:
        self._data = _deep_merge(dict(self._data), values)
        return self

    def section(self, path: str) -> Config:
        section = self.get(path)
        mapping = _ensure_mapping(section, f"Section '{path}'")
        return Config.from_dict(mapping)

    def __getitem__(self, path: str) -> Any:
        return self.get(path)

    def __contains__(self, path: str) -> bool:
        try:
            self.get(path)
            return True
        except KeyError:
            return False

    @staticmethod
    def _split_path(path: str) -> list[str]:
        raw = path.strip()
        if not raw:
            return []
        return [chunk.strip() for chunk in raw.split(".") if chunk.strip()]


def create_default_config() -> Config:
    return Config.from_default()
