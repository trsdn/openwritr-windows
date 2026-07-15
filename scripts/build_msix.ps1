# Build MSIX packages for the Microsoft Store (both architectures) and an
# optional .msixbundle for upload.
#
# Prerequisites:
#   - Both exes built:   cargo build --release --bin openwritr
#                        cargo build --release --target x86_64-pc-windows-msvc --bin openwritr
#   - Release staged:    python scripts/prepare_release.py --arch arm64
#                        python scripts/prepare_release.py --arch x64
#   - Store assets:      python installer/make_icon.py
#
# Usage:
#   .\scripts\build_msix.ps1 -IdentityName "12345TorstenMahr.OpenWritr" `
#                            -Publisher "CN=A1B2C3D4-...." `
#                            [-Version 0.4.0.0]
#
# Without -IdentityName/-Publisher it builds with TEST placeholders - fine for
# local validation (makeappx succeeds, package installs after self-signing),
# but the Store upload requires the real Partner Center identity values.

param(
    [string]$IdentityName = "TEST.OpenWritr",
    [string]$Publisher = "CN=TEST",
    [string]$Version = "0.4.0.0",
    [switch]$RequireStoreIdentity
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($IdentityName) -or
    [string]::IsNullOrWhiteSpace($Publisher)) {
    throw "MSIX identity name and publisher must both be provided"
}
$usesTestIdentity = $IdentityName -eq "TEST.OpenWritr" -or $Publisher -eq "CN=TEST"
if ($RequireStoreIdentity -and $usesTestIdentity) {
    throw "Store MSIX build requires the real Partner Center identity name and publisher"
}

$root = Split-Path $PSScriptRoot -Parent
$sdkBin = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\10.*\arm64\makeappx.exe" |
    Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty DirectoryName
if (-not $sdkBin) { throw "Windows SDK (makeappx.exe) not found" }
$makeappx = Join-Path $sdkBin "makeappx.exe"

$dist = Join-Path $root "target\dist"
New-Item -ItemType Directory -Force $dist | Out-Null

$packages = @()
foreach ($arch in @("arm64", "x64")) {
    & python (Join-Path $root "scripts\prepare_release.py") --arch $arch
    if ($LASTEXITCODE -ne 0) { throw "release staging failed for $arch" }

    $stage = Join-Path $root "target\msix-$arch"
    Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force "$stage\Assets" | Out-Null
    $releaseStage = Join-Path $root "target\stage\$arch"
    Copy-Item "$releaseStage\*" $stage -Recurse

    Copy-Item (Join-Path $root "installer\store-assets\*") "$stage\Assets\"

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
if ($usesTestIdentity) {
    Write-Warning "Built validation-only MSIX packages with TEST identity; do not publish them."
} else {
    Write-Host "Store upload: submit the .msixbundle in Partner Center > your app > Packages."
    Write-Host "(The Store signs it during ingestion; no local signing is needed for upload.)"
}
