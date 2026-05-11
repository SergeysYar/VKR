from __future__ import annotations

from typing import Mapping

from ...parametric.primitives import create_box
from .shared import AddObjectFn, BiomeRoom, room_area_scale, symmetric_positions

PRIMITIVES = ("box",)
RULES: dict[str, object] = {
    "layout": "rack aisles + pallet front zone",
    "dense_storage": True,
}


def populate(room: BiomeRoom, add_object: AddObjectFn, settings: Mapping[str, object]) -> None:
    _ = settings
    max_x = max(room.width / 2.0 - 0.9, 0.0)
    max_y = max(room.depth / 2.0 - 0.9, 0.0)
    area_scale = room_area_scale(room, reference_area=170.0, min_scale=0.75, max_scale=3.0)

    rack_lines = max(2, int(round(max(2.0, room.depth // 4.0) * min(area_scale, 2.4))))
    rack_lines = min(rack_lines, max(2, int(room.depth // 2.0)))
    y_positions = symmetric_positions(rack_lines, max(max_y * 0.75, 0.0))
    for idx, y in enumerate(y_positions, start=1):
        add_object(
            "rack",
            idx,
            create_box(width=max(room.width * 0.65, 2.0), height=2.6, depth=0.7),
            (0.0, y, 1.3),
            (0.0, 0.0, 0.0),
        )

    pallet_cols = max(2, int(round(max(2.0, room.width // 4.0) * min(area_scale, 2.6))))
    pallet_cols = min(pallet_cols, max(2, int(room.width // 1.5)))
    x_positions = symmetric_positions(pallet_cols, max(max_x * 0.8, 0.0))
    pallet_rows = max(1, min(3, int(round(area_scale))))
    row_positions = symmetric_positions(pallet_rows, max(max_y * 0.25, 0.0))
    pallet_index = 1
    for row_y in row_positions:
        for x in x_positions:
            add_object(
                "pallet",
                pallet_index,
                create_box(width=1.0, height=0.25, depth=0.8),
                (x, row_y - max_y * 0.62, 0.125),
                (0.0, 0.0, 0.0),
            )
            pallet_index += 1
