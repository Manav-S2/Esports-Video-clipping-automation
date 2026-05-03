param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,

    [Parameter(Mandatory = $false)]
    [string]$OutputDir,

    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 30)]
    [int]$SampleFps = 8,

    [Parameter(Mandatory = $false)]
    [ValidateRange(3, 120)]
    [int]$MinTextLength = 8,

    [Parameter(Mandatory = $false)]
    [ValidateRange(0, 255)]
    [int]$Threshold = 155,

    [Parameter(Mandatory = $false)]
    [ValidateRange(0.0, 0.95)]
    [double]$RoiX = 0.50,

    [Parameter(Mandatory = $false)]
    [ValidateRange(0.0, 0.80)]
    [double]$RoiY = 0.00,

    [Parameter(Mandatory = $false)]
    [ValidateRange(0.05, 0.50)]
    [double]$RoiW = 0.50,

    [Parameter(Mandatory = $false)]
    [ValidateRange(0.05, 0.60)]
    [double]$RoiH = 0.50,

    [Parameter(Mandatory = $false)]
    [ValidateRange(0.0, 10.0)]
    [double]$EventCooldownSeconds = 1.5
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $InputVideo)) {
    throw "Input video not found: $InputVideo"
}

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
if (-not $ffmpeg) { throw 'ffmpeg is not available in PATH.' }
if (-not $ffprobe) { throw 'ffprobe is not available in PATH.' }

if (($RoiX + $RoiW) -gt 1.0 -or ($RoiY + $RoiH) -gt 1.0) {
    throw 'Invalid ROI ratios. Ensure RoiX+RoiW <= 1 and RoiY+RoiH <= 1.'
}

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

$framesRawDir = Join-Path $OutputDir 'frames_raw'
$framesRoiDir = Join-Path $OutputDir 'frames_killfeed_roi'
$framesOcrDir = Join-Path $OutputDir 'frames_ocr_ready'
$snapshotsDir = Join-Path $OutputDir 'snapshots'
$metaDir = Join-Path $OutputDir 'meta'

$ignoreTerms = @(
    'polymarket',
    'refrag',
    'blacklyte',
    'republic',
    'gamers',
    'gaming',
    'nova',
    'wallet'
)

function Reset-GeneratedDirectory {
    param([string]$Path)

    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Reset-GeneratedDirectory -Path $framesRawDir
Reset-GeneratedDirectory -Path $framesRoiDir
Reset-GeneratedDirectory -Path $framesOcrDir
Reset-GeneratedDirectory -Path $snapshotsDir
Reset-GeneratedDirectory -Path $metaDir

try {
    $probeJson = & $ffprobe.Source -v error -select_streams v:0 -show_entries stream=width,height -of json $InputVideo
    if ($LASTEXITCODE -ne 0) {
        throw "ffprobe failed with exit code $LASTEXITCODE"
    }

    $probeObj = $probeJson | ConvertFrom-Json
    if (-not $probeObj.streams -or $probeObj.streams.Count -eq 0) {
        throw 'Unable to read video stream metadata.'
    }

    $videoWidth = [int]$probeObj.streams[0].width
    $videoHeight = [int]$probeObj.streams[0].height
    if ($videoWidth -le 0 -or $videoHeight -le 0) {
        throw 'Invalid video dimensions reported by ffprobe.'
    }

    $cropX = [int][math]::Floor($videoWidth * $RoiX)
    $cropY = [int][math]::Floor($videoHeight * $RoiY)
    $cropW = [int][math]::Floor($videoWidth * $RoiW)
    $cropH = [int][math]::Floor($videoHeight * $RoiH)

    if ($cropW % 2 -ne 0) { $cropW -= 1 }
    if ($cropH % 2 -ne 0) { $cropH -= 1 }

    $cropW = [math]::Max(32, $cropW)
    $cropH = [math]::Max(32, $cropH)

    if (($cropX + $cropW) -gt $videoWidth) {
        $cropX = [math]::Max(0, $videoWidth - $cropW)
    }
    if (($cropY + $cropH) -gt $videoHeight) {
        $cropY = [math]::Max(0, $videoHeight - $cropH)
    }

    $rawPattern = Join-Path $framesRawDir 'frame_%06d.png'
    $roiPattern = Join-Path $framesRoiDir 'frame_%06d.png'
    $ocrPattern = Join-Path $framesOcrDir 'frame_%06d.png'

    Write-Host 'Step 1/4: Extracting raw frames (lossless PNG)...'
    & $ffmpeg.Source -hide_banner -nostdin -y -i $InputVideo -an -sn -vf "fps=$SampleFps" $rawPattern | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg raw frame extraction failed with exit code $LASTEXITCODE"
    }

    Write-Host 'Step 2/4: Cropping killfeed ROI into dedicated frame set...'
    $cropFilter = "fps=$SampleFps,crop=$cropW`:$cropH`:$cropX`:$cropY"
    & $ffmpeg.Source -hide_banner -nostdin -y -i $InputVideo -an -sn -vf $cropFilter $roiPattern | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg ROI crop failed with exit code $LASTEXITCODE"
    }

    Write-Host 'Step 3/4: Building OCR-ready ROI frames...'
    $ocrFilter = "hqdn3d=1.1:1.1:4:4,eq=contrast=1.65:brightness=0.03:gamma=0.95,unsharp=5:5:1.6:5:5:0.0,scale=iw*2:ih*2:flags=lanczos,lutyuv=y='if(gte(val,$Threshold),255,0)'"
    & $ffmpeg.Source -hide_banner -nostdin -y -framerate $SampleFps -i $roiPattern -vf $ocrFilter $ocrPattern | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "ffmpeg OCR preprocessing failed with exit code $LASTEXITCODE"
    }

    $rawFrames = Get-ChildItem -LiteralPath $framesRawDir -Filter 'frame_*.png' | Sort-Object Name
    $ocrFrames = Get-ChildItem -LiteralPath $framesOcrDir -Filter 'frame_*.png' | Sort-Object Name

    if (-not $rawFrames -or $rawFrames.Count -eq 0) {
        throw 'No raw frames were generated.'
    }
    if (-not $ocrFrames -or $ocrFrames.Count -eq 0) {
        throw 'No OCR-ready frames were generated.'
    }

    $rawByName = @{}
    foreach ($rf in $rawFrames) {
        $rawByName[$rf.Name] = $rf.FullName
    }

    function Test-KillfeedText {
        param([string]$Text, [int]$MinLen, [string[]]$IgnoredTerms)

        if ([string]::IsNullOrWhiteSpace($Text)) { return $false }

        $t = $Text -replace "`r", ' ' -replace "`n", ' '
        $t = $t -replace '\s+', ' '
        $t = $t.Trim()

        if ($t.Length -lt $MinLen) { return $false }
        if ($t.Length -gt 48) { return $false }

        # Require multiple alphanumeric chunks to reduce false positives from tiny HUD marks.
        $tokens = [regex]::Matches($t, '[A-Za-z0-9]{2,}')
        if ($tokens.Count -lt 2) { return $false }

        $tokenValues = @($tokens | ForEach-Object { $_.Value.ToLowerInvariant() })
        $nonIgnored = @($tokenValues | Where-Object { $_ -notin $IgnoredTerms })
        if ($nonIgnored.Count -lt 2) { return $false }

        $longTokens = @($nonIgnored | Where-Object { $_.Length -ge 3 })
        return $longTokens.Count -ge 2
    }

    $inEvent = $false
    $eventIndex = 0
    $lastEventTime = -99999.0
    $events = New-Object System.Collections.Generic.List[object]
    $ocrRows = New-Object System.Collections.Generic.List[object]
    $oldNativeErrorPreference = $PSNativeCommandUseErrorActionPreference
    $PSNativeCommandUseErrorActionPreference = $false

    try {
        Write-Host 'Step 4/4: OCR scan and event snapshot creation...'

        for ($i = 0; $i -lt $ocrFrames.Count; $i++) {
            $frame = $ocrFrames[$i]
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

            $ocrClean = ($ocrText -replace "`r", ' ' -replace "`n", ' ' -replace '\s+', ' ').Trim()

            $hasKillfeed = Test-KillfeedText -Text $ocrClean -MinLen $MinTextLength -IgnoredTerms $ignoreTerms

            $ocrRows.Add([pscustomobject]@{
                frame = $frame.Name
                has_killfeed = $hasKillfeed
                text = $ocrClean
            })

            if ($hasKillfeed) {
                if (-not $inEvent) {
                    $frameNumberText = [System.IO.Path]::GetFileNameWithoutExtension($frame.Name).Replace('frame_', '')
                    $frameNumber = [int]$frameNumberText
                    $seconds = [math]::Round(($frameNumber - 1) / [double]$SampleFps, 2)
                    if (($seconds - $lastEventTime) -lt $EventCooldownSeconds) {
                        $inEvent = $true
                        continue
                    }

                    $eventIndex++
                    $safeTime = ('{0:0.00}' -f $seconds).Replace('.', '_')

                    $fullOut = Join-Path $snapshotsDir ("killfeed_{0:000}_t{1}s_full.png" -f $eventIndex, $safeTime)
                    $roiOut = Join-Path $snapshotsDir ("killfeed_{0:000}_t{1}s_roi.png" -f $eventIndex, $safeTime)

                    if ($rawByName.ContainsKey($frame.Name)) {
                        Copy-Item -LiteralPath $rawByName[$frame.Name] -Destination $fullOut -Force
                    }
                    Copy-Item -LiteralPath $frame.FullName -Destination $roiOut -Force

                    $events.Add([pscustomobject]@{
                        event = $eventIndex
                        frame = $frame.Name
                        time_seconds = $seconds
                        ocr_text = $ocrClean
                        full_snapshot = [System.IO.Path]::GetFileName($fullOut)
                        roi_snapshot = [System.IO.Path]::GetFileName($roiOut)
                    })

                    $lastEventTime = $seconds
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

    $scanCsv = Join-Path $metaDir 'ocr_scan.csv'
    $eventsCsv = Join-Path $metaDir 'killfeed_events.csv'
    $summaryTxt = Join-Path $metaDir 'killfeed_extracted_summary.txt'

    $ocrRows | Export-Csv -LiteralPath $scanCsv -NoTypeInformation -Encoding UTF8
    $events | Export-Csv -LiteralPath $eventsCsv -NoTypeInformation -Encoding UTF8

    $summaryLines = @(
        "Kill feed extracted from run folder: $OutputDir",
        "Source video: $InputVideo",
        "Method: ROI-first OCR with lossless PNG intermediates and thresholded killfeed crop.",
        "",
        "ROI ratios: x=$RoiX y=$RoiY w=$RoiW h=$RoiH",
        "Crop pixels: x=$cropX y=$cropY w=$cropW h=$cropH",
        "Sample FPS: $SampleFps",
        "Threshold: $Threshold",
        "Event cooldown (s): $EventCooldownSeconds",
        "",
        "Detected events: $eventIndex"
    )

    if ($eventIndex -gt 0) {
        $summaryLines += ''
        $summaryLines += 'Event list:'
        foreach ($e in $events) {
            $summaryLines += ("{0}. {1} at t={2:0.00}s" -f $e.event, $e.frame, [double]$e.time_seconds)
            if (-not [string]::IsNullOrWhiteSpace($e.ocr_text)) {
                $summaryLines += ("   OCR: {0}" -f $e.ocr_text)
            }
        }
    }

    Set-Content -LiteralPath $summaryTxt -Value $summaryLines -Encoding UTF8

    Write-Host "Input video       : $InputVideo"
    Write-Host "Sampled FPS       : $SampleFps"
    Write-Host "Run folder        : $OutputDir"
    Write-Host "Raw frames        : $framesRawDir"
    Write-Host "ROI frames        : $framesRoiDir"
    Write-Host "OCR-ready frames  : $framesOcrDir"
    Write-Host "Snapshots folder  : $snapshotsDir"
    Write-Host "Meta folder       : $metaDir"
    Write-Host "Killfeed events   : $eventIndex"

    if ($eventIndex -eq 0) {
        Write-Host 'No killfeed events detected with current settings.'
        Write-Host 'Tips: lower -MinTextLength, lower -Threshold, or adjust ROI ratios.'
    }
}
finally {
}
