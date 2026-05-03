# Install requirements without using the broken global `pip.exe` shim (Python314 missing).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Find-PipPython {
    # 1) Existing venv from setup_windows_env.ps1
    $venvPy = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) {
        return $venvPy
    }
    # 2) py launcher: default tag first, then common versions (you may have 3.14, not 3.12)
    $tryPy = {
        param([string[]]$ArgBeforeC)
        $exe = & py @ArgBeforeC -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe -match "\.exe") {
            return ($exe | Out-String).Trim()
        }
        return $null
    }
    $candidates = @(
        @(),
        @("-3.14"), @("-3.13"), @("-3.12"), @("-3.11"), @("-3.10")
    )
    foreach ($args in $candidates) {
        $found = & $tryPy $args
        if ($found) {
            return $found
        }
    }
    # 3) python on PATH if not MSYS
    $which = Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
    if ($which -and $which -notmatch "msys64") {
        return $which
    }
    return $null
}

$Py = Find-PipPython
if (-not $Py) {
    Write-Host @"
No usable Python found for pip.

Fix options:
  A) Run .\setup_windows_env.ps1 (creates .venv + installs everything), or
  B) Install Python from https://www.python.org/downloads/windows/ and tick **py launcher** + **Add to PATH**.

See what the launcher sees:  py -0p

Your error happens because global `pip` may point at a removed Python. Use:  py -m pip install -r requirements.txt
"@
    exit 1
}

Write-Host "Using: $Py"
& $Py -m pip install --upgrade pip
try {
    $ca = & $Py -c "import certifi; print(certifi.where())" 2>$null
    if ($LASTEXITCODE -eq 0 -and $ca) {
        $env:SSL_CERT_FILE = $ca.Trim()
    }
}
catch {}
& $Py -m pip install -r (Join-Path $Root "requirements.txt")
Write-Host "Done."
