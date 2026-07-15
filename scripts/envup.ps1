param(
    [ValidateSet("arm64", "x64")]
    [string]$Arch = "arm64"
)

$ErrorActionPreference = "Stop"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$installation = $null
if (Test-Path $vswhere) {
    $installation = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if (-not $installation) {
    $installation = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $installation) {
    throw "Visual Studio 2022 C++ build tools were not found"
}

$vcvars = Join-Path $installation "VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $vcvars)) {
    throw "vcvarsall.bat was not found at $vcvars"
}

$toolPaths = @(
    "C:\Program Files\LLVM\bin",
    (Join-Path $installation "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"),
    (Split-Path $vswhere -Parent),
    "$env:USERPROFILE\.cargo\bin"
) | Where-Object { $_ -and (Test-Path $_) }
$env:Path = ($toolPaths + $env:Path) -join ";"

$temporary = [System.IO.Path]::GetTempFileName()
try {
    & $env:ComSpec /d /s /c "call `"$vcvars`" $Arch >nul && set" |
        Set-Content $temporary -Encoding ascii
    if ($LASTEXITCODE -ne 0) {
        throw "vcvarsall.bat failed for $Arch"
    }
    foreach ($line in Get-Content $temporary) {
        if ($line -match "^([^=]+)=(.*)$") {
            Set-Item "Env:$($Matches[1])" $Matches[2]
        }
    }
    $env:Path = ($toolPaths + $env:Path) -join ";"
} finally {
    Remove-Item $temporary -Force -ErrorAction SilentlyContinue
}
