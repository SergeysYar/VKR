param(
    [switch]$SkipSync,
    [int]$Port = 8501,
    [ValidateSet("tabs", "showcase", "builder", "analysis")]
    [string]$App = "tabs"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $root ".venv"
$env:UV_CACHE_DIR = Join-Path $root ".uv-cache"
switch ($App) {
    "builder" {
        $appPath = Join-Path $root "lesson5\factory_synthetic_builder_app.py"
        $appTitle = "builder app"
    }
    "analysis" {
        $appPath = Join-Path $root "lesson5\factory_analysis_app.py"
        $appTitle = "analysis app"
    }
    "tabs" {
        $appPath = Join-Path $root "lesson5\factory_tabs_app.py"
        $appTitle = "tabs app"
    }
    default {
        $appPath = Join-Path $root "lesson5\factory_showcase_app.py"
        $appTitle = "showcase app"
    }
}

if (-not (Test-Path -LiteralPath $venvPath)) {
    Write-Host "Creating uv virtual environment at $venvPath ..."
    uv venv $venvPath
}

if (-not $SkipSync) {
    Write-Host "Installing dependencies with uv sync ..."
    uv sync --project $root
}

Write-Host "Starting Streamlit $appTitle ..."
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
uv run --project $root streamlit run $appPath --server.headless true --server.port $Port
