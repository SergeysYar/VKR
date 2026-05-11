from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from ...geometry.mesh import Mesh

Vector3 = tuple[float, float, float]
AddObjectFn = Callable[[str, int, Mesh, Vector3, Vector3], None]


def to_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def to_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping.")
    return {str(k): v for k, v in value.items()}


def to_positive_float(value: object, default: float, label: str) -> float:
    if value is None:
        return default
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{label} must be > 0.")
    return number


def to_non_negative_int(value: object, default: int, label: str) -> int:
    if value is None:
        return default
    number = int(value)
    if number < 0:
        raise ValueError(f"{label} must be >= 0.")
    return number


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def room_area_scale(
    room: "BiomeRoom",
    reference_area: float,
    min_scale: float = 0.7,
    max_scale: float = 3.0,
    exponent: float = 0.5,
) -> float:
    ref = max(float(reference_area), 1e-6)
    area = max(float(room.width) * float(room.depth), 1e-6)
    ratio = area / ref
    scale = ratio ** exponent
    return clamp(scale, min_scale, max_scale)


def symmetric_positions(count: int, max_offset: float) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    if max_offset <= 0.0:
        return [0.0 for _ in range(count)]

    step = (2.0 * max_offset) / (count - 1)
    return [(-max_offset + step * i) for i in range(count)]


@dataclass(frozen=True)
class BiomeRoom:
    room_id: str
    width: float
    depth: float
    height: float
    biome: str
