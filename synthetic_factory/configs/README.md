# Config Interface

This directory is the external configuration interface for generation.

## Presets

- `presets/scene.yaml` - balanced default scene preset.
- `presets/scene_essential.yaml` - essential key-parameter preset (with JSON fallback).
- `presets/scene_user_basic_ru.yaml` - user-friendly base preset with Russian comments.
- `presets/showcase.yaml` - high-detail showcase preset.
- `presets/connected_showcase.yaml` - connected plant showcase preset.
- `presets/scene_lidar_working.yaml` - working LiDAR preset.
- `presets/scene_lidar_ultra_dense.yaml` - max-density LiDAR preset.
- `presets/scene_lidar_portable_max.yaml` - portable compact preset (all biomes + memory-safe LiDAR density).
- `presets/scene_lidar_full_factory_4x.json` - full-scale factory preset for strong machines (4x denser LiDAR clouds).
- `presets/scene_lidar_test_small.yaml` - small fast LiDAR test preset.

Linux helper for the heavy preset:
- `scripts/run_lidar_full_factory_4x.sh`

## Usage

Run generator with any preset:

```powershell
uv run -m synthetic_factory --config configs/presets/scene.yaml
```

Override seed/output at runtime:

```powershell
uv run -m synthetic_factory --config configs/presets/showcase.yaml --seed 2026 --output out/custom.obj
```

Override any config path at runtime:

```powershell
uv run -m synthetic_factory `
  --config configs/presets/scene_lidar_test_small.yaml `
  --set factory.width=84 `
  --set layout.number_of_rooms=6 `
  --set lidar.point_multiplier=32
```

LiDAR output layout:

- outputs are grouped by run date in `output_dir/<YYYY-MM-DD>/`
- merged clouds are split into:
  - `interior/factory_stitched_interior.ply`
  - `exterior/factory_stitched_exterior.ply`
  - `combined/factory_stitched.ply`

## Notes

- YAML is the primary format for external setup.
- Keep values human-editable and document new blocks near related presets.
- `biomes.workshop_area_ratio` controls workshop territory share (for example `0.75`).
- `biomes.clustered_assignment: true` keeps workshop cells contiguous.
- `biomes.inter_room_doors: true` places door openings on walls facing neighbor rooms.
- `biomes.door_height` sets door opening height (keeps lintel above doors).
- `biomes.industrial_size_bias` with multipliers makes industrial rooms much larger than office/control.
- `noise.connectivity_bias: 0.8` makes Perlin room placement more connected.
- `noise.full_occupancy: true` removes empty layout slots.
- `lidar.exterior_point_density_factor` controls outdoor scan decimation (`0.25` means 4x less dense exterior points).
- exterior semantic classes are available in LiDAR labels: `roof`, `window`, `door`, `gate`, `terrain`, `facade`.
- global room infrastructure now uses stable cross-biome object types: `infra_cable_tray_*`, `infra_cable_bundle`, `infra_pipe_*`, `infra_pipe_casing` (consistent LiDAR labels across runs).
- for full setup/run/config instructions see: `docs/FULL_GUIDE_RU.md`.
