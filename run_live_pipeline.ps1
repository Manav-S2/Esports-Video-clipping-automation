# Run live pipeline using Windows .venv from setup_windows_env.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "Missing .venv. Run .\setup_windows_env.ps1 once."
    exit 1
}

$env:PYTHONUNBUFFERED = "1"
try {
    $ca = & $Py -c "import certifi; print(certifi.where())" 2>$null
    if ($ca) {
        $env:SSL_CERT_FILE = $ca.Trim()
    }
}
catch {}

& $Py live_stream_highlight_pipeline.py --config live_pipeline_config.json @args
