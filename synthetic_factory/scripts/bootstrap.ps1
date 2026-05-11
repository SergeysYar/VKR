$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_uv_common.ps1")

$projectRoot = Get-ProjectRoot
Ensure-UvEnvironment -ProjectRoot $projectRoot -AutoBootstrap -AlwaysSync

Write-Host ""
Write-Host "Environment is ready (uv)."
Write-Host "Run:"
Write-Host "  uv run --project `"$projectRoot`" -m synthetic_factory --config `"$projectRoot\configs\presets\scene.yaml`""
