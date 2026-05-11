$ErrorActionPreference = "Stop"

function Get-ProjectRoot {
    return (Split-Path -Parent $PSScriptRoot)
}

function Initialize-UvRuntime {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot
    )

    if ([string]::IsNullOrWhiteSpace($env:UV_CACHE_DIR)) {
        $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"
    }

    if (-not (Test-Path $env:UV_CACHE_DIR)) {
        New-Item -ItemType Directory -Path $env:UV_CACHE_DIR -Force | Out-Null
    }
}

function Ensure-UvAvailable {
    param(
        [switch]$AutoInstall = $true
    )

    if (Get-Command uv -ErrorAction SilentlyContinue) {
        return
    }

    if ($AutoInstall -and (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "uv not found. Installing uv via winget..."
        winget install -e --id astral-sh.uv --accept-package-agreements --accept-source-agreements
    }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error "uv is not installed. Install uv and rerun the script."
        exit 1
    }
}

function Ensure-UvEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [switch]$AutoBootstrap = $true,
        [switch]$AlwaysSync = $true
    )

    Initialize-UvRuntime -ProjectRoot $ProjectRoot
    Ensure-UvAvailable -AutoInstall:$AutoBootstrap
    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

    if ((-not $AutoBootstrap) -and (-not (Test-Path $venvPython))) {
        Write-Error "Environment not found. Run scripts/bootstrap.ps1 first."
        exit 1
    }

    if ($AlwaysSync -or (-not (Test-Path $venvPython))) {
        Write-Host "Syncing dependencies with uv..."
        & uv sync --project $ProjectRoot --all-extras
        if ($LASTEXITCODE -ne 0) {
            Write-Error "uv sync failed."
            exit 1
        }
    }
}

function Invoke-FactoryCli {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [string[]]$CliArgs
    )

    Initialize-UvRuntime -ProjectRoot $ProjectRoot
    & uv run --project $ProjectRoot -m synthetic_factory @CliArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "synthetic_factory CLI failed."
        exit 1
    }
}
