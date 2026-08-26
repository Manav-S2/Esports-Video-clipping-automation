<#
.SYNOPSIS
    One-command setup for the highlight pipeline on Windows.

.DESCRIPTION
    Creates an isolated virtual environment from the committed uv.lock (or
    requirements-lock.txt as a fallback), then verifies that ffmpeg and
    streamlink are reachable. Run from the repository root:

        powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1

.PARAMETER SkipToolCheck
    Skip the ffmpeg/streamlink PATH verification.
#>
[CmdletBinding()]
param(
    [switch]$SkipToolCheck
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "==> Setting up environment in $repoRoot" -ForegroundColor Cyan

$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) {
    Write-Host "==> uv found; syncing locked environment (uv.lock)"
    uv sync --frozen
    $python = Join-Path $repoRoot ".venv\Scripts\python.exe"
} else {
    Write-Host "==> uv not found; falling back to venv + requirements-lock.txt"
    if (-not (Test-Path ".venv")) {
        python -m venv .venv
    }
    $python = Join-Path $repoRoot ".venv\Scripts\python.exe"
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements-lock.txt
}

Write-Host "==> Verifying the test suite runs"
& $python -m pytest -q

if (-not $SkipToolCheck) {
    Write-Host "==> Checking external media tools"
    foreach ($tool in @("ffmpeg", "streamlink")) {
        $found = Get-Command $tool -ErrorAction SilentlyContinue
        if ($found) {
            Write-Host ("    OK      {0}: {1}" -f $tool, $found.Source) -ForegroundColor Green
        } else {
            Write-Host ("    MISSING {0} (needed for live capture; see docs/SETUP.md)" -f $tool) -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "Setup complete. Next steps:" -ForegroundColor Cyan
Write-Host "  1. copy .env.example .env                                     # fill in API keys"
Write-Host "  2. copy live_pipeline_config.example.json live_pipeline_config.json"
Write-Host "  3. .\.venv\Scripts\python.exe live_stream_highlight_pipeline.py --config .\live_pipeline_config.json"
