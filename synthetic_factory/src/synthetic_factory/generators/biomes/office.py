from __future__ import annotations

from typing import Mapping

from ...parametric.primitives import create_box
from .shared import AddObjectFn, BiomeRoom, room_area_scale, symmetric_positions

PRIMITIVES = ("box",)
RULES: dict[str, object] = {
    "layout": "desk grid + storage wall + meeting zone",
    "human_scale": True,
}


def populate(room: BiomeRoom, add_object: AddObjectFn, settings: Mapping[str, object]) -> None:
    _ = settings
    max_x = max(room.width / 2.0 - 0.9, 0.0)
    max_y = max(room.depth / 2.0 - 0.9, 0.0)
    area_scale = room_area_scale(room, reference_area=130.0, min_scale=0.75, max_scale=2.8)
    desk_rows = max(2, int(round(max(2.0, room.depth // 4.8) * min(area_scale, 2.0))))
    desk_cols = max(2, int(round(max(2.0, room.width // 3.0) * min(area_scale, 2.0))))
    desk_rows = min(desk_rows, max(2, int(room.depth // 2.2)))
    desk_cols = min(desk_cols, max(2, int(room.width // 2.2)))

    x_positions = symmetric_positions(desk_cols, max(max_x * 0.7, 0.0))
    y_positions = symmetric_positions(desk_rows, max(max_y * 0.35, 0.0))

    desk_idx = 1
    chair_idx = 1
    for y in y_positions:
        for x in x_positions:
            add_object(
                "desk",
                desk_idx,
                create_box(width=1.4, height=0.75, depth=0.7),
                (x, y, 0.375),
                (0.0, 0.0, 0.0),
            )
            add_object(
                "chair",
                chair_idx,
                create_box(width=0.5, height=0.9, depth=0.5),
                (x, y - 0.6, 0.45),
                (0.0, 0.0, 0.0),
            )
            desk_idx += 1
            chair_idx += 1

    cabinet_count = max(2, int(round(max(2.0, room.depth // 4.0) * min(area_scale, 2.4))))
    cabinet_count = min(cabinet_count, max(2, int(room.depth // 1.6)))
    cabinet_positions = symmetric_positions(cabinet_count, max(max_y * 0.9, 0.0))
    for idx, y in enumerate(cabinet_positions, start=1):
        add_object(
            "cabinet",
            idx,
            create_box(width=0.55, height=1.8, depth=0.45),
            (-max_x, y, 0.9),
            (0.0, 0.0, 0.0),
        )

    meeting_count = max(1, min(3, int(round(area_scale))))
    meeting_positions = symmetric_positions(meeting_count, max(max_x * 0.35, 0.0))
    for idx, x in enumerate(meeting_positions, start=1):
        add_object(
            "meeting_table",
            idx,
            create_box(width=max(2.0, room.width * 0.22), height=0.78, depth=1.2),
            (x, max_y * 0.82, 0.39),
            (0.0, 0.0, 0.0),
        )
