from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from . import boiler, control, electrical, laboratory, maintenance, office, refinery, storage, workshop
from .shared import AddObjectFn, BiomeRoom

PopulateFn = Callable[[BiomeRoom, AddObjectFn, Mapping[str, object]], None]


@dataclass(frozen=True)
class BiomeDefinition:
    name: str
    primitives: tuple[str, ...]
    rules: Mapping[str, object]
    populate: PopulateFn


DEFAULT_BIOME_ORDER = [
    "workshop",
    "office",
    "boiler",
    "storage",
    "electrical",
    "maintenance",
    "laboratory",
    "control",
]

DEFAULT_BIOME_PROFILES: dict[str, dict[str, float | int]] = {
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
}

BIOME_REGISTRY: dict[str, BiomeDefinition] = {
    "workshop": BiomeDefinition(
        name="workshop",
        primitives=workshop.PRIMITIVES,
        rules=workshop.RULES,
        populate=workshop.populate,
    ),
    "office": BiomeDefinition(
        name="office",
        primitives=office.PRIMITIVES,
        rules=office.RULES,
        populate=office.populate,
    ),
    "boiler": BiomeDefinition(
        name="boiler",
        primitives=boiler.PRIMITIVES,
        rules=boiler.RULES,
        populate=boiler.populate,
    ),
    "refinery": BiomeDefinition(
        name="refinery",
        primitives=refinery.PRIMITIVES,
        rules=refinery.RULES,
        populate=refinery.populate,
    ),
    "storage": BiomeDefinition(
        name="storage",
        primitives=storage.PRIMITIVES,
        rules=storage.RULES,
        populate=storage.populate,
    ),
    "electrical": BiomeDefinition(
        name="electrical",
        primitives=electrical.PRIMITIVES,
        rules=electrical.RULES,
        populate=electrical.populate,
    ),
    "maintenance": BiomeDefinition(
        name="maintenance",
        primitives=maintenance.PRIMITIVES,
        rules=maintenance.RULES,
        populate=maintenance.populate,
    ),
    "laboratory": BiomeDefinition(
        name="laboratory",
        primitives=laboratory.PRIMITIVES,
        rules=laboratory.RULES,
        populate=laboratory.populate,
    ),
    "control": BiomeDefinition(
        name="control",
        primitives=control.PRIMITIVES,
        rules=control.RULES,
        populate=control.populate,
    ),
}


def get_biome_definition(name: str) -> BiomeDefinition:
    normalized = name.strip().lower()
    return BIOME_REGISTRY.get(normalized, BIOME_REGISTRY["workshop"])
