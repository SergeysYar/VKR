param(
    [int]$Seed = 42,
    [double]$FactoryWidth = 70.0,
    [double]$FactoryDepth = 52.0,
    [int]$Rooms = 5,
    [double]$RoomMinSize = 4.0,
    [double]$RoomMaxSize = 12.0,
    [double]$RoomHeight = 4.5,
    [double]$CorridorWidth = 3.2,
    [int]$LidarPointMultiplier = 24,
    [int]$LidarPointsPerStation = 220000,
    [double]$LidarStationSpacing = 2.8,
    [int]$LidarMaxStationsPerRoom = 8,
    [int]$LidarMaxStations = 20,
    [string]$OutputObj = "out/factory_lidar_test_small.obj",
    [string[]]$Set = @()
)

$ErrorActionPreference = "Stop"

$helperPath = Join-Path $PSScriptRoot "_uv_common.ps1"
. $helperPath

$projectRoot = Get-ProjectRoot

Ensure-UvEnvironment -ProjectRoot $projectRoot -AutoBootstrap

$configPath = Join-Path $projectRoot "configs\presets\scene_lidar_test_small.yaml"
if (-not (Test-Path $configPath)) {
    Write-Error "LiDAR test config not found: $configPath"
    exit 1
}

$argsList = @(
    "--config", $configPath,
    "--seed", "$Seed",
    "--log-level", "INFO",
    "--output", $OutputObj,
    "--factory-width", "$FactoryWidth",
    "--factory-depth", "$FactoryDepth",
    "--number-of-rooms", "$Rooms",
    "--room-min-size", "$RoomMinSize",
    "--room-max-size", "$RoomMaxSize",
    "--room-height", "$RoomHeight",
    "--corridor-width", "$CorridorWidth",
    "--lidar-point-multiplier", "$LidarPointMultiplier",
    "--lidar-points-per-station", "$LidarPointsPerStation",
    "--lidar-station-spacing", "$LidarStationSpacing",
    "--lidar-max-stations-per-room", "$LidarMaxStationsPerRoom",
    "--lidar-max-stations", "$LidarMaxStations"
)

foreach ($entry in $Set) {
    $argsList += "--set"
    $argsList += $entry
}

Write-Host "Running configurable test preset..."
Write-Host ("Config: " + $configPath)
Write-Host ("Seed: " + $Seed)
Write-Host ("Output OBJ: " + $OutputObj)
if ($Set.Count -gt 0) {
    Write-Host ("Custom --set overrides: " + ($Set -join ", "))
}

Invoke-FactoryCli -ProjectRoot $projectRoot -CliArgs $argsList
