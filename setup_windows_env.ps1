# Creates a standard Windows .venv (CPython wheels for numpy, pip, SSL).
# Avoid MSYS Python here — PyPI has no numpy wheels for that tag.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Find-PythonExe {
    $tryPy = {
        param([string[]]$ArgBeforeC)
        $exe = & py @ArgBeforeC -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe -match "\.exe") {
            return ($exe | Out-String).Trim()
        }
        return $null
    }
    foreach ($args in @(@(), @("-3.14"), @("-3.13"), @("-3.12"), @("-3.11"), @("-3.10"))) {
        $found = & $tryPy $args
        if ($found) {
            return $found
        }
    }
    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
    if ($cmd -and $cmd -notmatch "msys64") {
        return $cmd
    }
    return $null
}

$PyExe = Find-PythonExe
if (-not $PyExe) {
    Write-Host @"
No usable Windows CPython found.

  1) Install Python from https://www.python.org/downloads/windows/
  2) Enable **py launcher** and **Add python.exe to PATH** (run `py -0p` to verify).
  3) Re-run this script.

Do NOT use MSYS2 Python for pip installs — numpy has no wheels for it.
"@
    exit 1
}

Write-Host "Using interpreter: $PyExe"
& $PyExe -m venv ".venv"
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Error "venv creation failed (.venv\Scripts\python.exe missing)."
    exit 1
}

Write-Host "Upgrading pip..."
& $VenvPy -m pip install --upgrade pip setuptools wheel

Write-Host "Installing certifi (helps pip TLS on broken CA stores)..."
& $VenvPy -m pip install "certifi>=2024"

try {
    $ca = & $VenvPy -c "import certifi; print(certifi.where())"
    $env:SSL_CERT_FILE = $ca.Trim()
    Write-Host "SSL_CERT_FILE=$($env:SSL_CERT_FILE)"
}
catch {
    Write-Host "certifi path not available; pip uses default trust store."
}

Write-Host "Installing requirements.txt..."
& $VenvPy -m pip install -r (Join-Path $Root "requirements.txt")

Write-Host ""
Write-Host "Done. Run the pipeline:"
Write-Host "  .\run_live_pipeline.ps1"
Write-Host "Or:"
Write-Host "  .\.venv\Scripts\python.exe live_stream_highlight_pipeline.py --config live_pipeline_config.json"
