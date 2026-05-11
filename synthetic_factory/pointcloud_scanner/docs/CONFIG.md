# Config Reference

## `paths`

- `input_obj`: path to source OBJ scene
- `output_dir`: root directory for scan artifacts
- `metadata_filename`: metadata JSON filename
- `stitched_filename`: stitched PLY filename
- `circle_file_prefix`: prefix for per-circle PLY files

## `output`

- `date_folder_enabled`: create date folder (`YYYY-MM-DD`)
- `run_folder_enabled`: create run timestamp subfolder
- `unique_outputs`: avoid overwrite when run folder disabled
- `export_individual_circles`: write per-station clouds
- `export_stitched_cloud`: write stitched clouds
- `export_metadata_json`: write metadata JSON

## `lidar`

Geometry and sampling:
- `scan_range`
- `angular_resolution_deg`
- `vertical_resolution_deg`
- `vertical_fov_up_deg`
- `vertical_fov_down_deg`
- `sensor_height`
- `min_range`
- `point_multiplier`
- `point_jitter`
- `exterior_point_density_factor`
- `points_per_station` (`0` = auto)

Station planning:
- `global_coverage`
- `station_spacing`
- `station_margin`
- `obstacle_clearance`
- `max_stations_per_room`
- `max_stations`

Blind-spot handling:
- `blind_spot_radius`
- `ensure_blind_spot_coverage`
- `stitched_fill_blind_spots`

Scene filtering:
- `include_structural`

Semantic labels:
- `label_classes`: mapping `class_name -> integer label`
