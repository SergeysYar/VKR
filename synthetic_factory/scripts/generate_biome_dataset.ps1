param(
    [string[]]$Biomes = @("boiler", "electrical", "control", "laboratory", "maintenance", "refinery", "workshop", "storage", "office"),
    [int]$Count = 30,
    [int]$SeedStart = 1000,
    [int]$DensityMultiplier = 8,
    [int]$PointsPerStation = 240000,
    [switch]$AutoBootstrap = $true
)

$ErrorActionPreference = "Stop"

$helperPath = Join-Path $PSScriptRoot "_uv_common.ps1"
. $helperPath

$projectRoot = Get-ProjectRoot
$configPath = Join-Path $projectRoot "configs\presets\scene_lidar_test_small.yaml"

if (-not (Test-Path $configPath)) {
    Write-Error "Config not found: $configPath"
    exit 1
}

Ensure-UvEnvironment -ProjectRoot $projectRoot -AutoBootstrap:$AutoBootstrap

$basePointMultiplier = 2
$effectivePointMultiplier = [Math]::Max(1, $basePointMultiplier * $DensityMultiplier)

$commonArgs = @(
    "--config", $configPath,
    "--log-level", "INFO",
    "--set", "layout.number_of_rooms=1",
    "--set", "layout.room_count_range=[1,1]",
    "--set", "number_of_rooms=1",
    "--set", "room_count_range=[1,1]",
    "--set", "biomes.ensure_all_types=false",
    "--set", "biomes.workshop_area_ratio=1.0",
    "--set", "lidar.scan_pattern=circular",
    "--set", "lidar.horizontal_fov_deg=360",
    "--set", "lidar.include_structural=true",
    "--set", "lidar.include_factory_room=false",
    "--set", "lidar.optimize_raycasts=true",
    "--set", "lidar.emit_no_hit_returns=false",
    "--set", "lidar.memory_safe_mode=false",
    "--set", "lidar.global_coverage=true",
    "--set", "lidar.max_stations=0",
    "--set", "lidar.max_stations_per_room=12",
    "--set", "lidar.coverage_grid_step=1.2",
    "--set", "lidar.coverage_wall_step=1.6",
    "--set", "lidar.coverage_max_targets=5000",
    "--set", "lidar.scan_range=32.0",
    "--set", "lidar.angular_resolution_deg=0.35",
    "--set", "lidar.vertical_resolution_deg=0.45",
    "--set", "lidar.points_per_station=$PointsPerStation",
    "--set", "lidar.point_multiplier=$effectivePointMultiplier",
    "--set", "lidar.point_jitter=0.0",
    "--set", "lidar.date_folder_enabled=false",
    "--set", "lidar.run_folder_enabled=true",
    "--set", "lidar.unique_outputs=true",
    "--set", "lidar.export_stitched_by_room=true",
    "--set", "lidar.export_stitched_by_biome=true",
    "--set", "machinery.enabled=true",
    "--set", "machinery.density=0.2",
    "--set", "machinery.machines_per_room=20",
    "--set", "machinery.conveyors_per_room=12",
    "--set", "machinery.auxiliary.enabled=true",
    "--set", "machinery.auxiliary.density=[2.4,3.0]",
    "--set", "machinery.auxiliary.complexity=[4,5]",
    "--set", "machinery.infrastructure.enabled=true",
    "--set", "machinery.infrastructure.include_pipes=true",
    "--set", "machinery.infrastructure.density=[2.4,3.0]",
    "--set", "machinery.infrastructure.branch_frequency=[2.0,2.5]",
    "--set", "machinery.infrastructure.parallel_channels=4"
)

$biomeIndex = 0
foreach ($Biome in $Biomes) {
    $datasetRoot = "out/datasets/biome_scans/$Biome"
    $biomeArgs = @(
        "--set", "biomes.workshop_biome=$Biome",
        "--set", "lidar.output_dir=$datasetRoot"
    )

    for ($i = 0; $i -lt $Count; $i++) {
        $seed = $SeedStart + ($biomeIndex * 100000) + $i
        $runFolder = ("run_{0:D3}_seed_{1}" -f ($i + 1), $seed)
        Write-Host ("[BiomeDataset] {0}/{1} biome={2} seed={3} density_x={4}" -f ($i + 1), $Count, $Biome, $seed, $DensityMultiplier)
        Invoke-FactoryCli -ProjectRoot $projectRoot -CliArgs ($commonArgs + $biomeArgs + @("--set", "lidar.run_folder_name=$runFolder", "--seed", "$seed"))
    }

    Write-Host ("[BiomeDataset] completed biome={0}, count={1}, output_root={2}" -f $Biome, $Count, $datasetRoot)
    $biomeIndex++
}
