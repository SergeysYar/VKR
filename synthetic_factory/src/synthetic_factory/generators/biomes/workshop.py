from __future__ import annotations

from math import ceil
from typing import Mapping

from ...parametric.primitives import create_beam, create_box, create_column
from .shared import (
    AddObjectFn,
    BiomeRoom,
    room_area_scale,
    symmetric_positions,
    to_mapping,
    to_non_negative_int,
    to_positive_float,
)

PRIMITIVES = ("beam", "box", "column")
RULES: dict[str, object] = {
    "largest_room_preferred": True,
    "flow_pattern": "cross conveyors + mirrored machine lanes",
    "overhead_crane": True,
}


def populate(room: BiomeRoom, add_object: AddObjectFn, settings: Mapping[str, object]) -> None:
    conveyors_per_room_base = to_non_negative_int(
        settings.get("conveyors_per_room"),
        3,
        "machinery.conveyors_per_room",
    )
    machines_per_room_base = to_non_negative_int(
        settings.get("machines_per_room"),
        4,
        "machinery.machines_per_room",
    )

    # Scale content with room area so large workshops are denser.
    area_scale = room_area_scale(room, reference_area=320.0, min_scale=0.8, max_scale=3.2)
    conveyors_per_room = max(1, int(round(conveyors_per_room_base * area_scale)))
    machines_per_room = max(1, int(round(machines_per_room_base * area_scale)))

    conveyor_cfg = to_mapping(settings.get("conveyor"))
    conveyor_width = to_positive_float(
        conveyor_cfg.get("width"),
        0.8,
        "machinery.conveyor.width",
    )
    conveyor_height = to_positive_float(
        conveyor_cfg.get("height"),
        0.35,
        "machinery.conveyor.height",
    )
    conveyor_elevation = to_positive_float(
        conveyor_cfg.get("elevation"),
        0.4,
        "machinery.conveyor.elevation",
    )

    machine_cfg = to_mapping(settings.get("machine"))
    machine_width_default = to_positive_float(
        machine_cfg.get("width"),
        2.2,
        "machinery.machine.width",
    )
    machine_depth_default = to_positive_float(
        machine_cfg.get("depth"),
        1.4,
        "machinery.machine.depth",
    )
    machine_height_default = to_positive_float(
        machine_cfg.get("height"),
        1.8,
        "machinery.machine.height",
    )

    max_x = max(room.width / 2.0 - 1.0, 0.0)
    max_y = max(room.depth / 2.0 - 1.0, 0.0)

    # Physical upper bounds avoid impossible packing in narrow rooms.
    max_conveyors_by_space = max(2, int(max(room.width, room.depth) // max(conveyor_width * 2.2, 1.6)))
    max_machines_by_space = max(2, int((room.width * room.depth) // max(machine_width_default * machine_depth_default * 2.0, 2.5)))
    conveyors_per_room = min(conveyors_per_room, max_conveyors_by_space)
    machines_per_room = min(machines_per_room, max_machines_by_space)

    conveyor_x_count = (conveyors_per_room + 1) // 2
    conveyor_y_count = conveyors_per_room // 2
    y_offsets = symmetric_positions(conveyor_x_count, max(max_y * 0.55, 0.0))
    x_offsets = symmetric_positions(conveyor_y_count, max(max_x * 0.55, 0.0))

    conveyor_length_x = max(room.width - 2.0, 1.0)
    conveyor_length_y = max(room.depth - 2.0, 1.0)

    conveyor_index = 1
    for offset in y_offsets:
        add_object(
            "conveyor",
            conveyor_index,
            create_beam(
                length=conveyor_length_x,
                profile_type={
                    "type": "rect",
                    "width": conveyor_width,
                    "height": conveyor_height,
                },
            ),
            (0.0, offset, conveyor_elevation + conveyor_height / 2.0),
            (0.0, 0.0, 0.0),
        )
        conveyor_index += 1

    for offset in x_offsets:
        add_object(
            "conveyor",
            conveyor_index,
            create_beam(
                length=conveyor_length_y,
                profile_type={
                    "type": "rect",
                    "width": conveyor_width,
                    "height": conveyor_height,
                },
            ),
            (offset, 0.0, conveyor_elevation + conveyor_height / 2.0),
            (0.0, 0.0, 90.0),
        )
        conveyor_index += 1

    side_slots = ceil(max(1, machines_per_room) / 2)
    machine_x_offsets = symmetric_positions(side_slots, max(max_x * 0.6, 0.0))
    machine_width = min(machine_width_default, max(room.width * 0.35, 0.8))
    machine_depth = min(machine_depth_default, max(room.depth * 0.25, 0.6))
    machine_height = machine_height_default

    for idx in range(machines_per_room):
        lane = idx // 2
        side = -1.0 if idx % 2 == 0 else 1.0
        x = (
            machine_x_offsets[min(lane, len(machine_x_offsets) - 1)]
            if machine_x_offsets
            else 0.0
        )
        y = side * max(max_y * 0.75, 0.0)
        add_object(
            "machine",
            idx + 1,
            create_box(
                width=machine_width,
                height=machine_height,
                depth=machine_depth,
            ),
            (x, y, machine_height / 2.0),
            (0.0, 0.0, 0.0),
        )

    crane_z = room.height * 0.82
    crane_profile = {
        "type": "i",
        "width": 0.22,
        "height": 0.32,
        "web_thickness": 0.015,
        "flange_thickness": 0.022,
    }
    rail_length = max(room.width - 1.5, 1.0)
    bridge_length = max(room.depth - 1.5, 1.0)
    rail_y = max(max_y * 0.9, 0.0)

    add_object(
        "crane_rail",
        1,
        create_beam(length=rail_length, profile_type=crane_profile),
        (0.0, -rail_y, crane_z),
        (0.0, 0.0, 0.0),
    )
    add_object(
        "crane_rail",
        2,
        create_beam(length=rail_length, profile_type=crane_profile),
        (0.0, rail_y, crane_z),
        (0.0, 0.0, 0.0),
    )
    add_object(
        "crane_bridge",
        1,
        create_beam(length=bridge_length, profile_type=crane_profile),
        (0.0, 0.0, crane_z),
        (0.0, 0.0, 90.0),
    )

    hook_height = max(room.height * 0.45, 1.0)
    add_object(
        "crane_hook",
        1,
        create_column(radius=0.07, height=hook_height, segments=16),
        (0.0, 0.0, crane_z - hook_height / 2.0),
        (0.0, 0.0, 0.0),
    )
