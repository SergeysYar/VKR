param(
    [int]$Seed = 42,
    [switch]$AutoBootstrap = $true
)

$ErrorActionPreference = "Stop"

$helperPath = Join-Path $PSScriptRoot "_uv_common.ps1"
. $helperPath

$projectRoot = Get-ProjectRoot
$configPath = Join-Path $projectRoot "configs\presets\scene_lidar_portable_max.yaml"

if (-not (Test-Path $configPath)) {
    Write-Error "Portable max config not found: $configPath"
    exit 1
}

Ensure-UvEnvironment -ProjectRoot $projectRoot -AutoBootstrap:$AutoBootstrap
Invoke-FactoryCli -ProjectRoot $projectRoot -CliArgs @("--config", $configPath, "--seed", "$Seed", "--log-level", "INFO")
