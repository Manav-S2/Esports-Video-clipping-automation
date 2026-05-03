param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,

    [Parameter(Mandatory = $false)]
    [string]$CaptionsFile,

    [Parameter(Mandatory = $false)]
    [string]$OutputVideo,

    [Parameter(Mandatory = $false)]
    [ValidateRange(720, 2160)]
    [int]$CanvasWidth = 1080,

    [Parameter(Mandatory = $false)]
    [ValidateRange(1280, 3840)]
    [int]$CanvasHeight = 1920,

    [Parameter(Mandatory = $false)]
    [ValidateRange(4, 80)]
    [int]$BlurStrength = 26,

    [Parameter(Mandatory = $false)]
    [ValidateRange(0.0, 0.8)]
    [double]$BackgroundDarken = 0.12,

    [Parameter(Mandatory = $false)]
    [ValidateRange(24, 120)]
    [int]$CaptionFontSize = 64,

    [Parameter(Mandatory = $false)]
    [string]$CaptionFont = 'Montserrat ExtraBold'
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
if (-not $OutputVideo) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($inputItem.Name)
    $OutputVideo = Join-Path $inputItem.DirectoryName ("$base.vertical.riverside.mp4")
}

# Vertical layout with explicit blur strips: blurred full-frame background + centered gameplay band.
$layoutFilter = "[0:v]split=2[bgsrc][fgsrc];[bgsrc]scale=$CanvasWidth`:$CanvasHeight,boxblur=$BlurStrength`:$BlurStrength,eq=brightness=-$BackgroundDarken[bg];[fgsrc]scale=$CanvasWidth`:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[vbase]"

$filterComplex = "$layoutFilter;[vbase]null[vout]"
if (-not [string]::IsNullOrWhiteSpace($CaptionsFile)) {
    if (-not (Test-Path -LiteralPath $CaptionsFile)) {
        throw "Captions file not found: $CaptionsFile"
    }

    $captionsPath = (Resolve-Path -LiteralPath $CaptionsFile).Path
    $captionsPath = $captionsPath -replace '\\', '/'
    $captionsPath = $captionsPath -replace ':', '\\:'

    # Top-centered bold subtitle style for short-form edits.
    $subtitleFilter = "[vbase]subtitles='$captionsPath':force_style='FontName=$CaptionFont,FontSize=$CaptionFontSize,Alignment=8,MarginV=120,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,Bold=1,BorderStyle=1,Outline=3,Shadow=0'[vout]"
    $filterComplex = "$layoutFilter;$subtitleFilter"
}

& $ffmpeg.Source -hide_banner -nostdin -y `
    -i $InputVideo `
    -filter_complex $filterComplex `
    -map '[vout]' `
    -map 0:a? -c:a copy `
    -c:v libx264 -crf 14 -preset slow -pix_fmt yuv420p -movflags +faststart `
    $OutputVideo

if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg edit failed with exit code $LASTEXITCODE"
}

Write-Host "Input video   : $InputVideo"
if (-not [string]::IsNullOrWhiteSpace($CaptionsFile)) {
    Write-Host "Captions file : $CaptionsFile"
}
Write-Host "Output video  : $OutputVideo"
Write-Host 'Done.'
