$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_uv_common.ps1")

$projectRoot = Get-ProjectRoot
$configPath = Join-Path $projectRoot "configs\presets\connected_showcase.yaml"

if (-not (Test-Path $configPath)) {
    Write-Error "Connected showcase config not found: $configPath"
    exit 1
}

Ensure-UvEnvironment -ProjectRoot $projectRoot -AutoBootstrap
Invoke-FactoryCli -ProjectRoot $projectRoot -CliArgs @("--config", $configPath, "--seed", "2026", "--log-level", "INFO")
