$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_uv_common.ps1")

$projectRoot = Get-ProjectRoot
$configPath = Join-Path $projectRoot "configs\presets\scene_lidar_ultra_dense.yaml"

if (-not (Test-Path $configPath)) {
    Write-Error "LiDAR ultra-dense config not found: $configPath"
    exit 1
}

Ensure-UvEnvironment -ProjectRoot $projectRoot -AutoBootstrap
Invoke-FactoryCli -ProjectRoot $projectRoot -CliArgs @("--config", $configPath, "--seed", "42", "--log-level", "INFO")
