$ErrorActionPreference = "Stop"

if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Error "winget is not available. Install Python manually from https://www.python.org/downloads/"
    exit 1
}

winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements

if ($LASTEXITCODE -ne 0) {
    Write-Error "Python installation failed."
    exit 1
}

Write-Host "Python installation completed."
Write-Host "Next step:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\\scripts\\bootstrap.ps1"
