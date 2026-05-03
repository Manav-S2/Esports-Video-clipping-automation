param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,

    [Parameter(Mandatory = $false)]
    [string]$WorkingDir,

    [Parameter(Mandatory = $false)]
    [string]$OutputVideo,

    [Parameter(Mandatory = $false)]
    [ValidateSet('h264', 'h265')]
    [string]$Codec = 'h264',

    [Parameter(Mandatory = $false)]
    [ValidateRange(0, 51)]
    [int]$Crf = 14,

    [Parameter(Mandatory = $false)]
    [ValidateSet('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow')]
    [string]$Preset = 'slow',

    [Parameter(Mandatory = $false)]
    [ValidateRange(2, 31)]
    [int]$JpegQuality = 2,

    [Parameter(Mandatory = $false)]
    [switch]$UseNumpyPandasOptimization = $false,

    [Parameter(Mandatory = $false)]
    [switch]$SkipNumpyPandasOptimization,

    [Parameter(Mandatory = $false)]
    [string]$PythonExe,

    [Parameter(Mandatory = $false)]
    [ValidateRange(0.5, 3.0)]
    [double]$SharpenAmount = 1.4,

    [Parameter(Mandatory = $false)]
    [ValidateRange(0.8, 1.5)]
    [double]$SaturationBoost = 1.07
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $InputVideo)) {
    throw "Input video not found: $InputVideo"
}

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    throw 'ffmpeg is not available in PATH.'
}

$inputItem = Get-Item -LiteralPath $InputVideo
$inputBaseName = [System.IO.Path]::GetFileNameWithoutExtension($inputItem.Name)

if (-not $WorkingDir) {
    $stamp = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
    $WorkingDir = Join-Path $inputItem.DirectoryName ("every_second_$stamp")
}

if (-not $OutputVideo) {
    $extension = '.mp4'
    $OutputVideo = Join-Path $inputItem.DirectoryName ("$inputBaseName.every-second.8k$extension")
}

$extractDir = Join-Path $WorkingDir 'frames_1fps'
$optimizedDir = Join-Path $WorkingDir 'frames_optimized_np'
$upscaledDir = Join-Path $WorkingDir 'frames_8k'
$metricsCsv = Join-Path $WorkingDir 'frame_metrics.csv'

New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
New-Item -ItemType Directory -Force -Path $optimizedDir | Out-Null
New-Item -ItemType Directory -Force -Path $upscaledDir | Out-Null

$extractPattern = Join-Path $extractDir 'frame_%08d.jpg'
$optimizedPattern = Join-Path $optimizedDir 'frame_%08d.jpg'
$upscaledPattern = Join-Path $upscaledDir 'frame_%08d.jpg'

Write-Host 'Step 1/3: Extracting 1 frame per second...'
& $ffmpeg.Source -hide_banner -nostdin -y -i $InputVideo -an -sn -vf 'fps=1' -q:v $JpegQuality $extractPattern
if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg extraction failed with exit code $LASTEXITCODE"
}

$frameCount = (Get-ChildItem -LiteralPath $extractDir -Filter 'frame_*.jpg' | Measure-Object).Count
if ($frameCount -le 0) {
    throw 'No frames were extracted from input video.'
}

$sourcePatternForUpscale = $extractPattern

if ($UseNumpyPandasOptimization -and -not $SkipNumpyPandasOptimization) {
    Write-Host 'Step 2/4: Running NumPy + pandas frame optimization (color + sharpening)...'

    if (-not $PythonExe) {
        $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($pyLauncher) {
            $PythonExe = $pyLauncher.Source
        }
        else {
            $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
            if ($pythonCmd) {
                $PythonExe = $pythonCmd.Source
            }
            else {
                throw 'Python was not found. Install Python and dependencies: pip install numpy pandas pillow'
            }
        }
    }

    $optimizerScript = Join-Path $PSScriptRoot 'optimize_frames_numpy_pandas.py'
    if (-not (Test-Path -LiteralPath $optimizerScript)) {
        throw "Optimizer script not found: $optimizerScript"
    }

    $pyArgs = @(
        $optimizerScript,
        '--input-dir', $extractDir,
        '--output-dir', $optimizedDir,
        '--metrics-csv', $metricsCsv,
        '--sharpen-amount', $SharpenAmount,
        '--saturation', $SaturationBoost
    )

    & $PythonExe @pyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "NumPy+pandas optimization failed with exit code $LASTEXITCODE"
    }

    $optimizedCount = (Get-ChildItem -LiteralPath $optimizedDir -Filter 'frame_*.jpg' | Measure-Object).Count
    if ($optimizedCount -le 0) {
        throw 'No optimized frames were produced by Python stage.'
    }

    $sourcePatternForUpscale = $optimizedPattern
}

Write-Host 'Step 3/4: Upscaling frames to 8K (7680x4320) with extra sharpening...'
& $ffmpeg.Source -hide_banner -nostdin -y -framerate 1 -i $sourcePatternForUpscale -vf 'eq=saturation=1.05,unsharp=7:7:1.2:7:7:0.0,scale=7680:4320:flags=lanczos+accurate_rnd+full_chroma_int,unsharp=5:5:1.0:5:5:0.0' -q:v $JpegQuality $upscaledPattern
if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg 8K upscaling failed with exit code $LASTEXITCODE"
}

Write-Host 'Step 4/4: Combining 8K frames into a single video...'

$encodeArgs = @(
    '-hide_banner',
    '-nostdin',
    '-y',
    '-framerate', '1',
    '-i', $upscaledPattern
)

if ($Codec -eq 'h265') {
    $encodeArgs += @(
        '-c:v', 'libx265',
        '-crf', $Crf,
        '-preset', $Preset,
        '-pix_fmt', 'yuv420p',
        '-tag:v', 'hvc1',
        '-movflags', '+faststart'
    )
}
else {
    $encodeArgs += @(
        '-c:v', 'libx264',
        '-crf', $Crf,
        '-preset', $Preset,
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart'
    )
}

$encodeArgs += $OutputVideo

& $ffmpeg.Source @encodeArgs
if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg video combine failed with exit code $LASTEXITCODE"
}

Write-Host ''
Write-Host "Input video   : $InputVideo"
Write-Host "Frames (1fps) : $extractDir"
if ($UseNumpyPandasOptimization -and -not $SkipNumpyPandasOptimization) {
    Write-Host "Frames (NP)   : $optimizedDir"
    Write-Host "Metrics CSV   : $metricsCsv"
}
Write-Host "Frames (8K)   : $upscaledDir"
Write-Host "Frame count   : $frameCount"
Write-Host "Output video  : $OutputVideo"
Write-Host 'Done.'