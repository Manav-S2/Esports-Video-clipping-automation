param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,

    [Parameter(Mandatory = $false)]
    [string]$OutputDir,

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 30)]
    [int]$SampleFps = 6,

    [Parameter(Mandatory = $false)]
    [ValidateRange(3, 120)]
    [int]$MinTextLength = 8
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $InputVideo)) {
    throw "Input video not found: $InputVideo"
}

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
if (-not $ffmpeg) { throw 'ffmpeg is not available in PATH.' }
if (-not $ffprobe) { throw 'ffprobe is not available in PATH.' }

$tesseractPath = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
if (-not (Test-Path -LiteralPath $tesseractPath)) {
    $tesseractCmd = Get-Command tesseract -ErrorAction SilentlyContinue
    if (-not $tesseractCmd) {
        throw 'Tesseract is not available. Install it or add to PATH.'
    }
    $tesseractPath = $tesseractCmd.Source
}

$inputItem = Get-Item -LiteralPath $InputVideo
if (-not $OutputDir) {
    $stamp = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
    $OutputDir = Join-Path $inputItem.DirectoryName ("killfeed_snapshots_$stamp")
}

$tempDir = Join-Path $env:TEMP ("killfeed_scan_" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

try {
    # OCR-friendly frame preprocessing while sampling frames.
    $vf = "fps=$SampleFps,hqdn3d=1.2:1.2:6:6,eq=contrast=1.55:brightness=0.02:gamma=0.95,unsharp=7:7:2.0:7:7:0.0,scale=1920:1080:flags=lanczos"

    $samplePattern = Join-Path $tempDir 'frame_%06d.jpg'

    & $ffmpeg.Source -hide_banner -nostdin -y -i $InputVideo -an -sn -vf $vf -q:v 3 $samplePattern | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg frame sampling failed with exit code $LASTEXITCODE"
    }

    $frames = Get-ChildItem -LiteralPath $tempDir -Filter 'frame_*.jpg' | Sort-Object Name
    if (-not $frames -or $frames.Count -eq 0) {
        throw 'No sampled frames were generated.'
    }

    function Test-KillfeedText {
        param([string]$Text, [int]$MinLen)

        if ([string]::IsNullOrWhiteSpace($Text)) { return $false }

        $t = $Text -replace "`r", ' ' -replace "`n", ' '
        $t = $t -replace '\s+', ' '
        $t = $t.Trim()

        if ($t.Length -lt $MinLen) { return $false }

        # Require multiple alphanumeric chunks to reduce false positives from tiny HUD marks.
        $tokens = [regex]::Matches($t, '[A-Za-z0-9]{2,}')
        return $tokens.Count -ge 2
    }

    $inEvent = $false
    $eventIndex = 0
    $oldNativeErrorPreference = $PSNativeCommandUseErrorActionPreference
    $PSNativeCommandUseErrorActionPreference = $false

    try {
        for ($i = 0; $i -lt $frames.Count; $i++) {
            $frame = $frames[$i]
            $ocrText = ''

            try {
                $ocrText = (& $tesseractPath $frame.FullName stdout --psm 6 2>$null) -join "`n"
            }
            catch {
                # Skip frames where OCR fails due to decoder/scaling edge cases.
                $ocrText = ''
            }

            if ($LASTEXITCODE -ne 0) {
                $ocrText = ''
            }

            $hasKillfeed = Test-KillfeedText -Text $ocrText -MinLen $MinTextLength

            if ($hasKillfeed) {
                if (-not $inEvent) {
                    $eventIndex++
                    $seconds = [math]::Round($i / [double]$SampleFps, 2)
                    $safeTime = ('{0:0.00}' -f $seconds).Replace('.', '_')
                    $outFile = Join-Path $OutputDir ("killfeed_{0:000}_t{1}s.jpg" -f $eventIndex, $safeTime)
                    Copy-Item -LiteralPath $frame.FullName -Destination $outFile -Force
                }
                $inEvent = $true
            }
            else {
                $inEvent = $false
            }
        }
    }
    finally {
        $PSNativeCommandUseErrorActionPreference = $oldNativeErrorPreference
    }

    Write-Host "Input video       : $InputVideo"
    Write-Host "Sampled FPS       : $SampleFps"
    Write-Host "Snapshots folder  : $OutputDir"
    Write-Host "Killfeed events   : $eventIndex"

    if ($eventIndex -eq 0) {
        Write-Host 'No killfeed events detected with current OCR threshold.'
        Write-Host 'Tip: lower -MinTextLength (for example 6) or increase -SampleFps (for example 10).'
    }
}
finally {
    if (Test-Path -LiteralPath $tempDir) {
        Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
