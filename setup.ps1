# One-time setup for twin-sync-sorter on Windows.
#
# Run from the cloned repo:
#
#     .\setup.ps1
#
# It installs Python dependencies (flask, opencv-python, numpy, requests, plus
# the dev tools pytest / pytest-cov / ruff) and registers the repo path in your
# user PYTHONPATH so Isaac's Script Editor can `import robot_motion` without
# any path boilerplate.
#
# Re-running is safe: pip is idempotent, and the PYTHONPATH write is a single
# registry update.

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Definition

Write-Host ""
Write-Host "twin-sync-sorter setup" -ForegroundColor Cyan
Write-Host "----------------------" -ForegroundColor Cyan
Write-Host "Project root: $here"
Write-Host ""

Write-Host "[1/2] Installing Python dependencies (pip install -e .[dev])..." -ForegroundColor Yellow
& python -m pip install -e "$here[dev]"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "pip install failed. If the system drive is full, redirect pip's temp/cache:" -ForegroundColor Red
    Write-Host '    $env:TMP = "D:\pip-tmp"; $env:TEMP = "D:\pip-tmp"' -ForegroundColor Red
    Write-Host '    & python -m pip install --cache-dir="D:\pip-cache" -e ".[dev]"' -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[2/2] Registering project on user PYTHONPATH..." -ForegroundColor Yellow
[System.Environment]::SetEnvironmentVariable('PYTHONPATH', $here, 'User')
Write-Host "PYTHONPATH (user) = $here"

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps (one-time per Isaac session):" -ForegroundColor Cyan
Write-Host "  1. Restart Isaac Sim so the new PYTHONPATH is picked up."
Write-Host "  2. In Isaac: File -> Open -> $here\assets\robot_cv_scene.usd, press Play."
Write-Host "  3. In a separate terminal: python `"$here\dashboard_server.py`""
Write-Host "  4. In Isaac's Script Editor:"
Write-Host "       import robot_motion; robot_motion.main()" -ForegroundColor White
Write-Host ""
