# Build MSIX packages for the Microsoft Store (both architectures) and an
# optional .msixbundle for upload.
#
# Prerequisites:
#   - Both exes built:   cargo build --release --bin openwritr
#                        cargo build --release --target x86_64-pc-windows-msvc --bin openwritr
#   - QNN DLLs staged:   cargo run --release --bin package      (arm64 zip side effect)
#   - x64 ORT vendored:  python scripts/fetch_x64_ort.py
#   - Store assets:      python installer/make_icon.py
#
# Usage:
#   .\scripts\build_msix.ps1 -IdentityName "12345TorstenMahr.OpenWritr" `
#                            -Publisher "CN=A1B2C3D4-...." `
#                            [-Version 0.3.0.0]
#
# Without -IdentityName/-Publisher it builds with TEST placeholders — fine for
# local validation (makeappx succeeds, package installs after self-signing),
# but the Store upload requires the real Partner Center identity values.

param(
    [string]$IdentityName = "TEST.OpenWritr",
    [string]$Publisher = "CN=TEST",
    [string]$Version = "0.3.0.0"
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$sdkBin = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.*\arm64\makeappx.exe" |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty DirectoryName
if (-not $sdkBin) { throw "Windows SDK (makeappx.exe) not found" }
$makeappx = Join-Path $sdkBin "makeappx.exe"

$dist = Join-Path $root "target\dist"
New-Item -ItemType Directory -Force $dist | Out-Null

# Per-arch file sets. arm64 carries the QNN runtime; x64 is CPU-only.
$qnnFiles = @(
    "onnxruntime_providers_qnn.dll",
    "QnnHtp.dll", "QnnHtpPrepare.dll",
    "QnnHtpV73Stub.dll", "QnnHtpV81Stub.dll",
    "libQnnHtpV73Skel.so", "libQnnHtpV81Skel.so",
    "libqnnhtpv73.cat", "libqnnhtpv81.cat",
    "QnnSystem.dll", "QnnCpu.dll", "QnnGpu.dll", "QnnIr.dll", "Genie.dll"
)

$packages = @()
foreach ($arch in @("arm64", "x64")) {
    $stage = Join-Path $root "target\msix-$arch"
    Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force "$stage\Assets" | Out-Null

    if ($arch -eq "arm64") {
        $exeSrc = Join-Path $root "target\release"
        Copy-Item "$exeSrc\openwritr.exe" $stage
        Copy-Item "$exeSrc\onnxruntime.dll" $stage
        foreach ($f in $qnnFiles) {
            $p = Join-Path $exeSrc $f
            if (Test-Path $p) { Copy-Item $p $stage } else { Write-Warning "missing $f" }
        }
    } else {
        Copy-Item (Join-Path $root "target\x86_64-pc-windows-msvc\release\openwritr.exe") $stage
        Copy-Item (Join-Path $root "vendor\x64\onnxruntime.dll") $stage
    }

    Copy-Item (Join-Path $root "installer\store-assets\*") "$stage\Assets\"
    Copy-Item (Join-Path $root "LICENSE") $stage
    # Qualcomm + Microsoft third-party notices (arm64 bundles their DLLs).
    $venvQnn = Join-Path $root ".venv\Lib\site-packages\onnxruntime_qnn"
    if ($arch -eq "arm64" -and (Test-Path "$venvQnn\Qualcomm_LICENSE.pdf")) {
        New-Item -ItemType Directory -Force "$stage\third-party-licenses" | Out-Null
        Copy-Item "$venvQnn\Qualcomm_LICENSE.pdf"  "$stage\third-party-licenses\"
        Copy-Item "$venvQnn\ThirdPartyNotices.txt" "$stage\third-party-licenses\"
    }

    $manifest = Get-Content (Join-Path $root "installer\AppxManifest.template.xml") -Raw
    $manifest = $manifest.Replace("{IDENTITY_NAME}", $IdentityName).
                          Replace("{PUBLISHER}", $Publisher).
                          Replace("{VERSION}", $Version).
                          Replace("{ARCH}", $arch)
    Set-Content "$stage\AppxManifest.xml" $manifest -Encoding utf8

    $msix = Join-Path $dist "openwritr-$arch-$Version.msix"
    Remove-Item $msix -Force -ErrorAction SilentlyContinue
    & $makeappx pack /d $stage /p $msix /o | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0) { throw "makeappx failed for $arch" }
    $packages += $msix
    Write-Host "built $msix ($([math]::Round((Get-Item $msix).Length/1MB,1)) MB)"
}

# Bundle both arches for a single Store upload.
$bundleDir = Join-Path $root "target\msix-bundle"
Remove-Item $bundleDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $bundleDir | Out-Null
foreach ($p in $packages) { Copy-Item $p $bundleDir }
$bundle = Join-Path $dist "openwritr-$Version.msixbundle"
Remove-Item $bundle -Force -ErrorAction SilentlyContinue
& $makeappx bundle /d $bundleDir /p $bundle /o | Select-Object -Last 1
if ($LASTEXITCODE -ne 0) { throw "makeappx bundle failed" }
Write-Host "built $bundle ($([math]::Round((Get-Item $bundle).Length/1MB,1)) MB)"
Write-Host ""
Write-Host "Store upload: submit the .msixbundle in Partner Center → your app → Packages."
Write-Host "(The Store signs it during ingestion — no local signing needed for upload.)"
