# apply_phase.ps1
#
# One-time helper: put this in your project root (D:\sentrisense-phase0\sentrisense\).
# Whenever I hand you a new "sentrisense-phaseN.zip", just run:
#
#   .\apply_phase.ps1 -PhaseZip "sentrisense-phase3.zip"
#
# It searches common download locations for the zip, extracts it, merges it
# into the current folder, and cleans up — no manual path-hunting needed.

param(
    [Parameter(Mandatory = $true)]
    [string]$PhaseZip
)

$ErrorActionPreference = "Stop"

$searchRoots = @(
    "$env:USERPROFILE\Downloads",
    "C:\",
    "D:\"
)

$found = $null
foreach ($root in $searchRoots) {
    $matches = Get-ChildItem -Path $root -Filter $PhaseZip -Recurse -ErrorAction SilentlyContinue -File
    if ($matches) {
        $found = $matches | Select-Object -First 1
        break
    }
    # Also check for an already-auto-extracted folder with the same base name
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($PhaseZip)
    $folderMatches = Get-ChildItem -Path $root -Filter $baseName -Recurse -ErrorAction SilentlyContinue -Directory
    if ($folderMatches) {
        $found = $folderMatches | Select-Object -First 1
        break
    }
}

if (-not $found) {
    Write-Host "Could not find '$PhaseZip' anywhere under Downloads, C:\, or D:\." -ForegroundColor Red
    Write-Host "Check where your browser saved it and re-run with the full path instead:" -ForegroundColor Yellow
    Write-Host "  Expand-Archive -Path '<full path>' -DestinationPath `"`$env:TEMP\extract`" -Force" -ForegroundColor Yellow
    exit 1
}

Write-Host "Found: $($found.FullName)" -ForegroundColor Green

$projectRoot = Get-Location
$tempExtract = Join-Path $env:TEMP "phase_extract_$(Get-Random)"

if ($found.PSIsContainer) {
    # Already an extracted folder
    $sourceRoot = $found.FullName
} else {
    Expand-Archive -Path $found.FullName -DestinationPath $tempExtract -Force
    $sourceRoot = $tempExtract
}

# The zips always contain a top-level "sentrisense" folder — copy its
# CONTENTS into the project root, merging with what's already there.
$innerFolder = Join-Path $sourceRoot "sentrisense"
if (Test-Path $innerFolder) {
    Copy-Item "$innerFolder\*" -Destination $projectRoot -Recurse -Force
} else {
    Copy-Item "$sourceRoot\*" -Destination $projectRoot -Recurse -Force
}

if (Test-Path $tempExtract) {
    Remove-Item $tempExtract -Recurse -Force
}

Write-Host "Merged into $projectRoot" -ForegroundColor Green
Write-Host "Done. Run your usual verification commands next." -ForegroundColor Cyan
