#requires -Version 7
<#
.SYNOPSIS
    Benchmark nexa-sdk's Parakeet-NPU vs OpenWritr's CPU INT8 baseline.

.DESCRIPTION
    Runs `nexa infer NexaAI/parakeet-tdt-0.6b-v3-npu --input <wav>` N times
    per file, captures wall-clock per run. First run includes one-time model
    load + HTP graph compilation (~5-10 s expected); subsequent runs are
    steady-state (load is cached, the same nexa server is reused).

    Compares to the CPU INT8 reference times in calibration/cpu_ref.txt
    (~191 ms on test_en, ~197 ms on test_de) and the in-app log medians
    (~25x realtime).

.NOTES
    nexa.exe must be on PATH (installer puts it under %LOCALAPPDATA%\Programs\NexaCLI).
#>

[CmdletBinding()]
param(
    [string]$Model = 'NexaAI/parakeet-tdt-0.6b-v3-npu',
    [string[]]$Wavs = @(
        "$env:LOCALAPPDATA\OpenWritr\models\parakeet-tdt-0.6b-v3\test_en.wav",
        "$env:LOCALAPPDATA\OpenWritr\models\parakeet-tdt-0.6b-v3\test_de.wav"
    ),
    [int]$Runs = 5
)

$ErrorActionPreference = 'Stop'
$nexa = "$env:LOCALAPPDATA\Programs\NexaCLI\nexa.exe"
if (-not (Test-Path $nexa)) { throw "nexa.exe not found at $nexa" }

# Audio length in seconds via WAV header (sample_rate at 24..27, byte_count at 40..43)
function Get-WavSeconds([string]$path) {
    $b = [System.IO.File]::ReadAllBytes($path)
    $sr = [BitConverter]::ToInt32($b, 24)
    $byteCount = [BitConverter]::ToInt32($b, 40)
    $bytesPerSample = [BitConverter]::ToInt16($b, 34) / 8 * [BitConverter]::ToInt16($b, 22)
    return [math]::Round($byteCount / ($sr * $bytesPerSample), 3)
}

$rows = @()
foreach ($wav in $Wavs) {
    if (-not (Test-Path $wav)) { Write-Warning "missing: $wav"; continue }
    $secs = Get-WavSeconds $wav
    Write-Host ""
    Write-Host "=== $(Split-Path -Leaf $wav) ($secs s audio) ==="

    for ($i = 1; $i -le $Runs; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $out = & $nexa infer $Model --input $wav 2>&1
        $sw.Stop()
        $ms = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
        $rt = if ($secs -gt 0) { [math]::Round(($secs * 1000) / $ms, 1) } else { 0 }
        $transcript = ($out | Where-Object { $_ -and $_ -notmatch '^\s*$' -and $_ -notmatch 'INFO|WARN|DEBUG|cached|Loading|done' } | Select-Object -Last 3) -join ' / '
        Write-Host ("run {0}: {1,6:N1} ms ({2}x RT)  -> {3}" -f $i, $ms, $rt, $transcript.Substring(0, [math]::Min(80, $transcript.Length)))
        $rows += [PSCustomObject]@{ wav=Split-Path -Leaf $wav; audio_s=$secs; run=$i; wall_ms=$ms; xRT=$rt }
    }
}

Write-Host ""
Write-Host "=== Summary (median of runs 2..N, excludes first-run model load) ==="
$rows | Group-Object wav | ForEach-Object {
    $steady = $_.Group | Where-Object { $_.run -ge 2 } | Select-Object -ExpandProperty wall_ms | Sort-Object
    if ($steady.Count -eq 0) { return }
    $median = $steady[[math]::Floor($steady.Count / 2)]
    $secs = ($_.Group | Select-Object -First 1).audio_s
    $rt = [math]::Round(($secs * 1000) / $median, 1)
    Write-Host ("  {0,-20} {1,6} ms  ({2,5} x RT, {3} s audio)" -f $_.Name, $median, $rt, $secs)
}
