# Synthetic Factory

Scene-oriented synthetic factory generator with parametric control and OBJ export.

## Quick Start (Windows PowerShell)

```powershell
cd synthetic_factory
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

Result OBJ path by default:

`out/factory.obj`

## Architecture Text (RU)

Concept and algorithm description document:

- `docs/PROJECT_CONCEPT_AND_ALGORITHMS_RU.md`
- full run/install/config guide (Windows + Linux):
  - `docs/FULL_GUIDE_RU.md`

## Config Interface

External YAML presets:

- `configs/presets/scene.yaml`
- `configs/presets/scene_essential.yaml` (essential key-parameter preset)
- `configs/presets/scene_user_basic_ru.yaml` (user preset with Russian comments)
- `configs/presets/showcase.yaml`
- `configs/presets/connected_showcase.yaml`
- `configs/presets/scene_lidar_working.yaml`
- `configs/presets/scene_lidar_ultra_dense.yaml`

Config notes:

- `configs/README.md`

## Manual Run

```powershell
cd synthetic_factory
uv sync --all-extras
uv run -m synthetic_factory --config configs/presets/scene.yaml
```

## Showcase Run (Best Demo)

Ready-to-use high-detail preset:

- config: `configs/presets/showcase.yaml`
- output: `out/factory_showcase.obj`

Run:

```powershell
cd synthetic_factory
powershell -ExecutionPolicy Bypass -File .\scripts\run_showcase.ps1
```

## Connected Plant Showcase (Unified Factory)

Preset focused on a single logical plant layout:

- connected Perlin room cluster
- central production workshop with explicit factory flow spine
- linked sections around the main shop floor

Files:

- config: `configs/presets/connected_showcase.yaml`
- script: `scripts/run_connected_showcase.ps1`
- output: `out/factory_connected_showcase.obj`

Run:

```powershell
cd synthetic_factory
powershell -ExecutionPolicy Bypass -File .\scripts\run_connected_showcase.ps1
```

If Python is missing:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-python.ps1
```

## CLI Options

```powershell
uv run -m synthetic_factory --help
```

Key options:

- `--config` config file path (`.yaml/.json/.toml`)
- `--seed` override random seed
- `--output` override OBJ output path
- `--log-level` `DEBUG|INFO|WARNING|ERROR`
- fast overrides: `--factory-width`, `--factory-depth`, `--number-of-rooms`,
  `--room-min-size`, `--room-max-size`, `--room-height`, `--corridor-width`
- LiDAR overrides: `--lidar-point-multiplier`, `--lidar-points-per-station`,
  `--lidar-station-spacing`, `--lidar-max-stations-per-room`, `--lidar-max-stations`
- universal override: `--set path=value` (repeatable), for example:
  `--set biomes.workshop_area_ratio=0.8 --set lidar.include_factory_room=true`

## Configurable Test Preset

Small and fast test preset for debugging/tuning:

- config: `configs/presets/scene_lidar_test_small.yaml`
- script: `scripts/run_configurable_test.ps1`

Run with defaults:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_configurable_test.ps1
```

Run with custom factory/LiDAR settings:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_configurable_test.ps1 `
  -FactoryWidth 84 -FactoryDepth 62 -Rooms 6 `
  -LidarPointMultiplier 32 -LidarPointsPerStation 320000 `
  -Set "biomes.workshop_area_ratio=0.72" `
  -Set "noise.frequency=1.2"
```

## LiDAR Presets

Two dedicated presets for scan density:

- `configs/presets/scene_lidar_working.yaml`  
  Faster workflow preset (good for iteration).
- `configs/presets/scene_lidar_ultra_dense.yaml`  
  Maximum density preset (heavy runtime / very large point clouds).
- `configs/presets/scene_lidar_portable_max.yaml`  
  Portable compact preset for another PC:
  reduced factory footprint, all biomes enabled, and memory-safe LiDAR density.
- `configs/presets/scene_lidar_full_factory_4x.json`  
  Full-scale factory preset for strong machines with 4x denser LiDAR clouds.

Run working preset:

```powershell
uv run -m synthetic_factory --config configs/presets/scene_lidar_working.yaml --seed 42
```

or:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_lidar_working.ps1
```

Run ultra-dense preset:

```powershell
uv run -m synthetic_factory --config configs/presets/scene_lidar_ultra_dense.yaml --seed 42
```

or:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_lidar_ultra_dense.ps1
```

Run portable max preset (auto-bootstrap via `uv sync` if needed):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_portable_max.ps1
```

Run full factory 4x preset (strong machine):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_lidar_full_factory_4x.ps1
```

Linux:

```bash
bash ./scripts/run_lidar_full_factory_4x.sh
```

Linux with custom seed:

```bash
SEED=2026 bash ./scripts/run_lidar_full_factory_4x.sh
```

Direct `uv` command (cross-platform):

```bash
uv sync --all-extras
uv run -m synthetic_factory --config configs/presets/scene_lidar_full_factory_4x.json --seed 42
```

## Biome Dataset (Linux, uv)

Generate a dataset with dense circular scans for each biome:

- 30 runs per biome
- 8x denser clouds
- each run saved into its own folder
- circles + stitched outputs (combined / by room / by biome)

Run:

```bash
cd /path/to/synthetic_factory
uv sync --all-extras
uv run bash scripts/generate_biome_dataset_linux.sh
```

Optional: custom biome list (example):

```bash
uv run bash scripts/generate_biome_dataset_linux.sh boiler electrical control
```

Optional env overrides:

```bash
COUNT=30 DENSITY_MULTIPLIER=8 POINTS_PER_STATION=240000 uv run bash scripts/generate_biome_dataset_linux.sh
```

Output structure:

- `out/datasets/biome_scans/<biome>/run_###_seed_<seed>/circles/*.ply`
- `out/datasets/biome_scans/<biome>/run_###_seed_<seed>/combined/*.ply`
- `out/datasets/biome_scans/<biome>/run_###_seed_<seed>/rooms/*.ply`
- `out/datasets/biome_scans/<biome>/run_###_seed_<seed>/biomes/*.ply`
- `out/datasets/biome_scans/<biome>/run_###_seed_<seed>/*.json`

LiDAR guarantees for dataset scripts:

- full horizontal sweep: `360°` (`lidar.horizontal_fov_deg=360`)
- semantic label per point: `label`
- instance segmentation per point: `instance_id` (unique object instance ID)

## Room-Only Biome Dataset (Linux, uv)

Scenario for powerful machines:

- no corridor scanning
- one biome-room per run
- 5 runs per biome by default
- ultra-dense 360-degree scans (high station count + high angular/vertical sampling)

Run:

```bash
cd /path/to/synthetic_factory
uv run bash scripts/generate_biome_room_only_linux.sh
```

Optional overrides:

```bash
COUNT=5 DENSITY_MULTIPLIER=12 POINTS_PER_STATION=500000 uv run bash scripts/generate_biome_room_only_linux.sh
```

Output root:

- `out/datasets/biome_room_only_scans/<biome>/run_###_seed_<seed>/...`

LiDAR outputs are now written into a date folder (for example `2026-04-04`) and split by capture domain:

- `.../<date>/interior/factory_stitched_interior.ply`
- `.../<date>/exterior/factory_stitched_exterior.ply`
- `.../<date>/combined/factory_stitched.ply`
- circle scans in `.../<date>/circles/`

Outdoor scan density can be reduced independently via `lidar.exterior_point_density_factor` (`0.25` = 4x less dense than interior).

### NVIDIA GPU Acceleration (LiDAR)

LiDAR ray-casting supports optional CUDA acceleration for NVIDIA GPUs.

- config key: `lidar.compute_backend`
  - `auto` (default): use CUDA if available, otherwise CPU fallback
  - `cpu`: force CPU mode
  - `cuda`: force CUDA mode (raises error if CUDA is unavailable)
- batch controls:
  - `lidar.cuda_ray_batch_size`
  - `lidar.cuda_object_batch_size`

Example override:

```bash
uv run -m synthetic_factory --config configs/presets/scene_lidar_full_factory_4x.json --seed 42 --lidar-compute-backend cuda
```

Note: CUDA mode requires a PyTorch build with CUDA support installed in your environment.

## Perlin (Minecraft-style) Scene

`configs/presets/scene.yaml` already uses Perlin layout:

- `layout.strategy: perlin`
- `layout.room_count_range: [8, 10]`
- `noise.*` controls octaves/frequency/threshold and room placement pattern
- `biomes.*` controls room roles (workshop/office/boiler/storage/etc.)

Largest generated room is assigned as main `workshop` and filled with conveyors,
machines and overhead crane elements.

## Biome Modules

Biome generation is modular and split into dedicated files:

- `src/synthetic_factory/generators/biomes/workshop.py`
- `src/synthetic_factory/generators/biomes/office.py`
- `src/synthetic_factory/generators/biomes/boiler.py`
- `src/synthetic_factory/generators/biomes/storage.py`
- `src/synthetic_factory/generators/biomes/electrical.py`
- `src/synthetic_factory/generators/biomes/maintenance.py`
- `src/synthetic_factory/generators/biomes/laboratory.py`
- `src/synthetic_factory/generators/biomes/control.py`
- `src/synthetic_factory/generators/auxiliary_generator.py` (secondary detail pass)

Each module contains:

- primitive usage (`create_box`, `create_beam`, `create_column`, ...)
- generation rules (`RULES`)
- room content placement logic (`populate(...)`)

Auxiliary detail pass (`AuxiliaryGenerator`):

- consumes `Scene` + biome and adds secondary objects (`aux_*`) attached to primary biome objects
- uses `generate_for_object(obj, context)` and `attach_to_parent(child, parent)`
- keeps child placement within parent bounds and avoids overlaps with existing child details
- can be enabled in config:

```yaml
machinery:
  auxiliary:
    enabled: true
    seed: 0
    density: [0.5, 3.0]
    complexity: [1, 5]
    max_children_per_object: 2
    boiler:
      bunker_probability: [0.2, 0.8]
      pump_count: [1, 4]
    electrical:
      transformer_probability: [0.3, 0.7]
    laboratory:
      clutter_level: [0.5, 2.0]
```

`complexity` increases the number of generated secondary candidates, and `density` scales spawn quantity/probability.

Registry and defaults are centralized in:

- `src/synthetic_factory/generators/biomes/registry.py`

Boiler biome now includes deterministic industrial rules:

- central axis detection and 1-3 evenly spaced boilers
- vertical risers + horizontal links + seeded random branches
- platforms on 1/3 and 2/3 height with connecting ladders
- perimeter tanks and ceiling support beams for pipe network
- floor-footprint collision checks and access walkway preservation

Boiler parameter system:

```yaml
machinery:
  boiler:
    random_variation: true
    pattern: linear_boilers # clustered_boilers | linear_boilers | central_tower | dense_industrial
    rules:
      auto_fix: true
      min_clearance: 1.2
    boiler:
      count: [1, 3]
      height: [8.0, 20.0]
      radius: [1.0, 3.0]
    pipes:
      density: [0.5, 2.0]
      radius: [0.1, 0.5]
    platforms:
      levels: [1, 3]
      width_factor: [1.2, 2.0]
    tanks:
      count: [0, 5]
    structure:
      ceiling_height: auto
      beam_density: [0.2, 1.0]
```

Dependencies applied in code:

- room height for boiler biome is forced to at least `max(boiler.height) * 1.2`
- generated branch pipe count is proportional to boiler count (`~ boiler_count * pipes.density`)
- platform levels are distributed along boiler height

Semantic boiler patterns:

- `clustered_boilers`: grouped boilers with shared cluster manifold
- `linear_boilers`: boilers in sequence along main axis
- `central_tower`: one dominant boiler + auxiliary modules around it
- `dense_industrial`: extra side manifolds and increased pipe/equipment density

Boiler rule engine (with auto-fix):

- `MinClearanceRule`: enforces minimal spacing for boiler placements.
- `AccessibilityRule`: guarantees access corridor and service zones.
- `HeightConstraint`: clamps all major object placements to room height.
- `PipeConnectivityRule`: auto-adds missing risers/manifold/links.
- `PlatformSupportRule`: auto-adds support columns under each platform.

Control biome uses rule-driven dispatcher layout:

- horizontal dominance with low suspended visual ceiling
- pattern-driven layout algorithm (`linear_control_room`, `semi_circular_control`, `clustered_workstations`, `minimal_control`, `high_density`)
- explicit functional zones: `operator_zone`, `panel_zone`, `aisle_zone`
- `control_panel` layout modes: `wall`, `semicircle`, or `auto` (derived from `panels.type`)
- `desk_monitor` on every workstation, facing panel direction
- global `view direction` is derived from `control_panel` placement
- every `console` (desk) gets orientation, and `desk_monitor`/`operator_seat` inherit it
- `cable_tray` network linking panel line and rack placements
- automatic aisle and visibility preservation to `monitor_wall`
- rule set with auto-fix: `VisibilityRule`, `MinSpacingRule`, `WalkwayRule`, `AlignmentRule`, `ErgonomicsRule`

Control parameter ranges and seeded sampling:

```yaml
machinery:
  control:
    seed: 0
    random_variation: true
    pattern: linear_control_room
    auto_fix: true
    monitor_view_angle_deg: 40.0
    alignment_grid_step: 0.1
    room:
      width: [10.0, 40.0]
      depth: [8.0, 30.0]
      height: [2.5, 5.0]
    desks:
      rows: [1, 5]
      cols: [2, 10]
      spacing_x: [1.2, 2.0]
      spacing_y: [1.5, 3.0]
    monitors:
      per_desk: [1, 4]
    panels:
      type: [flat, curved]
      height: [1.5, 3.0]
    racks:
      count: [0, 10]
    lighting:
      grid_density: [0.5, 2.0]
    ergonomic_desk_height_min: 0.72
    ergonomic_desk_height_max: 1.15
    ergonomic_monitor_center_min: 0.22
    ergonomic_monitor_center_max: 0.65
```

Dependencies applied in generator:

- desk count is `rows * cols` after room-fit auto-fix
- `room.width >= cols * spacing_x` and `room.depth >= rows * spacing_y` are enforced
- panel arrangement scales with room width
- sampling is deterministic for each room via `seed + stable(room_id)` when `random_variation: true`
- pattern changes workstation placement strategy and density behavior:
  - `linear_control_room`: row grid + panel wall
  - `semi_circular_control`: arc panel + concentric desk arcs
  - `clustered_workstations`: island groups of desks
  - `minimal_control`: sparse desks and reduced equipment
  - `high_density`: maximized workstation fill

Electrical biome uses deterministic row-based engineering layout:

- linear rows of `electrical_cabinet` with shared orientation
- strict aisles (`electrical_aisle`, `electrical_perimeter_aisle`)
- overhead/underground `cable_tray` backbone and vertical `cable_drop` lines
- tray-level cable network (`cable_bundle_row`, `cable_bundle_backbone`, `cable_bundle_branch`)
- optional edge `cooling_unit` placement
- deterministic randomization via `seed + stable(room_id)`
- optional fixed-step grid snapping for modular placement
- layout patterns via `pattern`:
  - `single_row` (small rooms)
  - `parallel_rows` (default multi-row)
  - `back_to_back` (paired mirrored rows)
  - `dense_grid` (max fill)
  - `sparse_technical` (more free space)
- rule-based auto-fix:
  - `ClearanceRule`
  - `AccessRule`
  - `CableRoutingRule`
  - `AlignmentRule`
  - `HeightRule`
  - `CoolingAccessRule`

Electrical parameter ranges:

```yaml
machinery:
  electrical:
    seed: 0
    random_variation: true
    auto_fix: true
    pattern: parallel_rows
    rules:
      clearance_front: 0.8
      min_spacing: 0.3
      enforce_cable_routing: true
      strict_alignment: true
      tray_height_clearance: 0.25
      cooling_access_clearance: 0.7
    grid_snapping: true
    fixed_grid_step: 0.2
    room:
      width: [10, 50]
      depth: [10, 60]
      height: [3, 6]
    cabinets:
      rows: [1, 6]
      per_row: [3, 20]
      spacing: [0.8, 1.5]
    walkways:
      width: [0.8, 2.0]
    cable_trays:
      height: [2.2, 3.5]
      density: [0.5, 2.0]
    cables:
      density: [0.5, 3.0]
      optimize_bundles: true
      variation: 0.0
    cooling:
      units: [0, 4]
```

Electrical dependencies and constraints:

- `room.width >= rows * (cabinet_depth + walkway_width)` is enforced via row auto-fit.
- `room.depth >= per_row * cabinet_width` is enforced via per-row auto-fit.
- cable drops scale with cabinet count (`base + extra ~ (density - 1) * cabinet_count`).
- each cabinet creates a deterministic connection point and rises vertically to nearest tray row.
- tray cabling is merged into bundle trunks/backbone when `cables.optimize_bundles: true`.
- `cables.variation` adds small seeded offsets/curvature for less repetitive cable topology.
- with `grid_snapping: true`, cabinet/row positions are snapped to `fixed_grid_step`.

Laboratory biome now uses modular station logic:

- room is split into station modules with seeded occupancy sampling
- station types: `chemistry`, `analysis`, `preparation`
- each station is generated as a functional unit:
  `lab_bench` + typed `lab_equipment_unit` (1..5) + per-equipment cable link (`up`/`down`) + wall pipe service + optional `lab_fume_hood`/`lab_sink`
- walkways: perimeter + row passages + `lab_access_zone` per station
- infrastructure: `lab_cable_*` (up/down/trunks) + `lab_pipe_*` (wall/ceiling/branches/drops)
- storage: `lab_shelf` / `lab_cabinet` on walls and near selected stations
- sinks: `lab_sink` in subset of stations (especially chemistry, if water enabled)
- medium-density target with spacing and accessibility constraints

Laboratory parameter block:

```yaml
machinery:
  laboratory:
    seed: 0
    random_variation: true
    pattern: research_lab # wet_lab | dry_lab | analytical_lab | teaching_lab | research_lab
    auto_fix: true
    room:
      width: [8, 30]
      depth: [8, 30]
      height: [2.5, 4.5]
    stations:
      count: [2, 12]
      spacing: [1.5, 3.0]
      type_variability: [0.15, 0.65]
      type_mix:
        chemistry: 0.34
        analysis: 0.38
        preparation: 0.28
    equipment:
      density: [1, 5]
    fumehood:
      probability: [0.0, 0.5]
    storage:
      shelves: [0, 10]
      cabinets: [0, 8]
    infrastructure:
      cable_density: [0.5, 2.0]
      pipe_density: [0.2, 1.0]
    lighting:
      intensity: [0.5, 2.0]
    equipment_cable_mode: mixed # up | down | mixed
    equipment_types: [analyzer, mixer, centrifuge, pump, controller, heater]
    equipment_type_map:
      chemistry: [reactor, mixer, heater, pump]
      analysis: [analyzer, spectrometer, chromatograph, sensor]
      preparation: [balance, dispenser, stirrer, controller]
    station_pipe_probability: 0.45
    station_alignment_step: 0.04
    rules:
      auto_fix: true
      min_spacing: 0.95
      equipment_near_distance: 0.24
      equipment_min_spacing: 0.16
      safety_walkway_clearance: 0.06
      edge_margin: 0.04
```

Dependencies in generator:

- station count is clamped by available room area and spacing.
- total equipment count scales with station count (`~ stations * equipment.density`).
- fixture count/size scales with station count and `lighting.intensity`.
- `pattern` switches layout logic:
  - `wet_lab`: more sinks and pipe-heavy infrastructure.
  - `dry_lab`: no water sinks, stronger electronics/cable bias.
  - `analytical_lab`: fewer stations with more equipment per station.
  - `teaching_lab`: strict grid and uniform station setup.
  - `research_lab`: chaotic jittered layout and high variation.
- rule-based auto-fix is enabled by default:
  - `AccessRule`
  - `MinSpacingRule`
  - `EquipmentPlacementRule`
  - `InfrastructureRule`
  - `SafetyRule`

Laboratory-specific parametric primitives are available in:

- `src/synthetic_factory/parametric/laboratory_primitives.py`

Provided primitives:

- `create_lab_bench`
- `create_equipment_unit`
- `create_fume_hood`
- `create_shelf`
- `create_cabinet`
- `create_sink`
- `create_pipe` (module-local) / `create_lab_pipe` (package export)
- `create_cable`
- `create_light_fixture`

Boiler-specific parametric primitives are available in:

- `src/synthetic_factory/parametric/boiler_primitives.py`

Provided primitives:

- `create_boiler_unit`
- `create_pipe` (supports `bend_angle`)
- `create_valve`
- `create_platform`
- `create_ladder` / `create_stair`
- `create_support_beam`
- `create_tank`

Control-specific parametric primitives are available in:

- `src/synthetic_factory/parametric/control_primitives.py`

Provided primitives:

- `create_desk`
- `create_chair`
- `create_monitor`
- `create_control_panel`
- `create_rack`
- `create_cable_tray`
- `create_wall_panel`
- `create_light_panel` (stores `intensity` in mesh metadata)

Change seed for a different deterministic factory:

```powershell
uv run -m synthetic_factory --config configs/presets/scene.yaml --seed 7 --output out/factory_seed7.obj
```
