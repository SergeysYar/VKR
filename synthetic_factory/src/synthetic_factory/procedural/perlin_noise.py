from __future__ import annotations

from dataclasses import dataclass
from math import floor


def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _hash2d(x: int, y: int, seed: int) -> int:
    value = x * 374761393 + y * 668265263 + seed * 69069
    value = (value ^ (value >> 13)) * 1274126177
    value = value ^ (value >> 16)
    return value & 0xFFFFFFFF


def _gradient(x: int, y: int, seed: int) -> tuple[float, float]:
    hashed = _hash2d(x, y, seed) & 7
    gradients = (
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.7071067812, 0.7071067812),
        (-0.7071067812, 0.7071067812),
        (0.7071067812, -0.7071067812),
        (-0.7071067812, -0.7071067812),
    )
    return gradients[hashed]


@dataclass(frozen=True)
class PerlinNoise2D:
    """
    Deterministic 2D Perlin noise with octave support.

    `sample(x, y)` returns value in [-1, 1].
    `sample01(x, y)` returns value in [0, 1].
    """

    seed: int = 0
    octaves: int = 4
    frequency: float = 1.0
    persistence: float = 0.5
    lacunarity: float = 2.0

    def __post_init__(self) -> None:
        if self.octaves <= 0:
            raise ValueError("octaves must be > 0.")
        if self.frequency <= 0.0:
            raise ValueError("frequency must be > 0.")
        if self.persistence <= 0.0:
            raise ValueError("persistence must be > 0.")
        if self.lacunarity <= 0.0:
            raise ValueError("lacunarity must be > 0.")

    def sample(self, x: float, y: float) -> float:
        amplitude = 1.0
        frequency = self.frequency
        total = 0.0
        amplitude_sum = 0.0

        for octave in range(self.octaves):
            total += amplitude * self._single_perlin(
                x * frequency,
                y * frequency,
                octave_seed=self.seed + octave * 1013,
            )
            amplitude_sum += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity

        if amplitude_sum == 0.0:
            return 0.0
        return max(-1.0, min(1.0, total / amplitude_sum))

    def sample01(self, x: float, y: float) -> float:
        return (self.sample(x, y) + 1.0) * 0.5

    def _single_perlin(self, x: float, y: float, octave_seed: int) -> float:
        x0 = floor(x)
        y0 = floor(y)
        x1 = x0 + 1
        y1 = y0 + 1

        sx = x - x0
        sy = y - y0

        g00 = _gradient(x0, y0, octave_seed)
        g10 = _gradient(x1, y0, octave_seed)
        g01 = _gradient(x0, y1, octave_seed)
        g11 = _gradient(x1, y1, octave_seed)

        n00 = g00[0] * (x - x0) + g00[1] * (y - y0)
        n10 = g10[0] * (x - x1) + g10[1] * (y - y0)
        n01 = g01[0] * (x - x0) + g01[1] * (y - y1)
        n11 = g11[0] * (x - x1) + g11[1] * (y - y1)

        u = _fade(sx)
        v = _fade(sy)

        nx0 = _lerp(n00, n10, u)
        nx1 = _lerp(n01, n11, u)
        return _lerp(nx0, nx1, v)
