#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${CONFIG_PATH:-configs/presets/scene_lidar_test_small.yaml}"
COUNT="${COUNT:-30}"
SEED_START="${SEED_START:-1000}"
DENSITY_MULTIPLIER="${DENSITY_MULTIPLIER:-8}"
POINTS_PER_STATION="${POINTS_PER_STATION:-240000}"

if [[ $# -gt 0 ]]; then
  BIOMES=("$@")
else
  BIOMES=(boiler electrical control laboratory maintenance refinery workshop storage office)
fi

uv sync --all-extras

BASE_POINT_MULTIPLIER=2
EFFECTIVE_POINT_MULTIPLIER=$(( BASE_POINT_MULTIPLIER * DENSITY_MULTIPLIER ))
if [[ "$EFFECTIVE_POINT_MULTIPLIER" -lt 1 ]]; then
  EFFECTIVE_POINT_MULTIPLIER=1
fi

for biome_index in "${!BIOMES[@]}"; do
  biome="${BIOMES[$biome_index]}"
  dataset_root="out/datasets/biome_scans/${biome}"

  for ((i=0; i<COUNT; i++)); do
    seed=$(( SEED_START + biome_index * 100000 + i ))
    run_num=$(( i + 1 ))
    run_folder=$(printf "run_%03d_seed_%d" "$run_num" "$seed")

    echo "[BiomeDataset] ${run_num}/${COUNT} biome=${biome} seed=${seed} density_x=${DENSITY_MULTIPLIER}"

    uv run -m synthetic_factory \
      --config "$CONFIG_PATH" \
      --log-level INFO \
      --seed "$seed" \
      --set layout.number_of_rooms=1 \
      --set layout.room_count_range=[1,1] \
      --set number_of_rooms=1 \
      --set room_count_range=[1,1] \
      --set biomes.ensure_all_types=false \
      --set biomes.workshop_area_ratio=1.0 \
      --set biomes.workshop_biome="$biome" \
      --set lidar.scan_pattern=circular \
      --set lidar.horizontal_fov_deg=360 \
      --set lidar.include_structural=true \
      --set lidar.include_factory_room=false \
      --set lidar.optimize_raycasts=true \
      --set lidar.emit_no_hit_returns=false \
      --set lidar.memory_safe_mode=false \
      --set lidar.global_coverage=true \
      --set lidar.max_stations=0 \
      --set lidar.max_stations_per_room=12 \
      --set lidar.coverage_grid_step=1.2 \
      --set lidar.coverage_wall_step=1.6 \
      --set lidar.coverage_max_targets=5000 \
      --set lidar.scan_range=32.0 \
      --set lidar.angular_resolution_deg=0.35 \
      --set lidar.vertical_resolution_deg=0.45 \
      --set lidar.points_per_station="$POINTS_PER_STATION" \
      --set lidar.point_multiplier="$EFFECTIVE_POINT_MULTIPLIER" \
      --set lidar.point_jitter=0.0 \
      --set lidar.date_folder_enabled=false \
      --set lidar.run_folder_enabled=true \
      --set lidar.unique_outputs=true \
      --set lidar.run_folder_name="$run_folder" \
      --set lidar.export_stitched_by_room=true \
      --set lidar.export_stitched_by_biome=true \
      --set lidar.output_dir="$dataset_root" \
      --set machinery.enabled=true \
      --set machinery.density=0.2 \
      --set machinery.machines_per_room=20 \
      --set machinery.conveyors_per_room=12 \
      --set machinery.auxiliary.enabled=true \
      --set machinery.auxiliary.density=[2.4,3.0] \
      --set machinery.auxiliary.complexity=[4,5] \
      --set machinery.infrastructure.enabled=true \
      --set machinery.infrastructure.include_pipes=true \
      --set machinery.infrastructure.density=[2.4,3.0] \
      --set machinery.infrastructure.branch_frequency=[2.0,2.5] \
      --set machinery.infrastructure.parallel_channels=4
  done

  echo "[BiomeDataset] completed biome=${biome}, count=${COUNT}, output_root=${dataset_root}"
done
