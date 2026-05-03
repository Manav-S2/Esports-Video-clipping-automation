param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,

    [Parameter(Mandatory = $false)]
    [string]$OutputVideo,

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 4)]
    [int]$ScaleFactor = 2,

    [Parameter(Mandatory = $false)]
    [ValidateRange(0, 51)]
    [int]$Crf = 18,

    [Parameter(Mandatory = $false)]
    [ValidateSet('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow')]
    [string]$Preset = 'slow',

    [Parameter(Mandatory = $false)]
    [ValidateSet('h264', 'h265')]
    [string]$Codec = 'h264',

    [Parameter(Mandatory = $false)]
    [switch]$Lossless,

    [Parameter(Mandatory = $false)]
    [switch]$Binarize,

    [Parameter(Mandatory = $false)]
    [ValidateRange(0, 255)]
    [int]$Threshold = 150
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $InputVideo)) {
    throw "Input video not found: $InputVideo"
}

if (-not $OutputVideo) {
    $inputItem = Get-Item -LiteralPath $InputVideo
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($inputItem.Name)
    $outputDir = $inputItem.DirectoryName
    $suffix = if ($Lossless) { '.ocr-max.mkv' } else { '.ocr-max.mp4' }
    $OutputVideo = Join-Path $outputDir ("$baseName$suffix")
}

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    throw 'ffmpeg is not available in PATH.'
}

$vfParts = @(
    'hqdn3d=1.2:1.2:6:6',
    'eq=contrast=1.55:brightness=0.02:gamma=0.95',
    'unsharp=7:7:2.2:7:7:0.0',
    "scale=iw*${ScaleFactor}:ih*${ScaleFactor}:flags=lanczos+accurate_rnd+full_chroma_int",
    'unsharp=5:5:1.4:5:5:0.0'
)

if ($Binarize) {
    # Strong thresholding for high-contrast text; tune with -Threshold for your footage.
    $vfParts += "lutyuv=y='if(gte(val,$Threshold),255,0)'"
}

$vf = $vfParts -join ','

$args = @(
    '-hide_banner',
    '-y',
    '-i', $InputVideo,
    '-map', '0:v:0',
    '-an',
    '-sn',
    '-vf', $vf
)

if ($Lossless) {
    $args += @(
        '-c:v', 'ffv1',
        '-level', '3',
        '-coder', '1',
        '-context', '1',
        '-g', '1',
        '-slices', '24',
        '-slicecrc', '1'
    )
}
else {
    if ($Codec -eq 'h265') {
        $args += @(
            '-c:v', 'libx265',
            '-crf', $Crf,
            '-preset', $Preset,
            '-pix_fmt', 'yuv420p',
            '-tag:v', 'hvc1',
            '-movflags', '+faststart'
        )
    }
    else {
        $args += @(
            '-c:v', 'libx264',
            '-crf', $Crf,
            '-preset', $Preset,
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart'
        )
    }
}

$args += $OutputVideo

Write-Host "Input : $InputVideo"
Write-Host "Output: $OutputVideo"
Write-Host "Mode  : $(if ($Lossless) { 'lossless (FFV1)' } else { "compressed ($($Codec.ToUpper()), CRF=$Crf, preset=$Preset)" })"
Write-Host "Filter: $vf"
Write-Host ''

& $ffmpeg.Source @args
if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg failed with exit code $LASTEXITCODE"
}

Write-Host ''
Write-Host 'OCR-optimized video created successfully.'
