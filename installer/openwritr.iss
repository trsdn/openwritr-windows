; Inno Setup script — produces a single-exe installer for OpenWritr.
;
; Build with:
;     "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" /Qp installer\openwritr.iss
;
; The Cargo bin `installer_build` (cargo run --release --bin installer_build)
; runs this automatically as part of the release flow.

#define AppName       "OpenWritr"
#define AppVersion    "0.3.0"
#define AppPublisher  "Torsten Mahr"
#define AppURL        "https://github.com/trsdn/openwritr-windows"
#define AppExeName    "openwritr.exe"

[Setup]
AppId={{2A8F4D3E-7C61-4B9F-A52B-3D7E0F88D911}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
; Per-user install — no UAC, no admin rights, no system-wide footprint.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={localappdata}\OpenWritr\app
DefaultGroupName={#AppName}
DisableProgramGroupPage=auto
ArchitecturesAllowed=arm64
ArchitecturesInstallIn64BitMode=arm64
OutputDir=..\target\dist
OutputBaseFilename=openwritr-windows-arm64-v{#AppVersion}-setup
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
CloseApplications=force
RestartApplications=no
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
LicenseFile=..\LICENSE

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"

[Tasks]
Name: "autostart"; \
      Description: "Start {#AppName} automatically when I log in"; \
      GroupDescription: "Startup:"
Name: "startmenuicon"; \
      Description: "Create a Start Menu shortcut"; \
      GroupDescription: "Shortcuts:"; \
      Flags: checkedonce
Name: "desktopicon"; \
      Description: "Create a desktop shortcut"; \
      GroupDescription: "Shortcuts:"; \
      Flags: unchecked

[Files]
; All exe + DLLs + skel + cat files staged in target/release by `cargo run --bin package`.
Source: "..\target\release\openwritr.exe";                  DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\onnxruntime.dll";                DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\onnxruntime_providers_qnn.dll";  DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\QnnHtp.dll";                     DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\QnnHtpPrepare.dll";              DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\QnnHtpV73Stub.dll";              DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\QnnHtpV81Stub.dll";              DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\libQnnHtpV73Skel.so";            DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\libQnnHtpV81Skel.so";            DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\libqnnhtpv73.cat";               DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\libqnnhtpv81.cat";               DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\QnnSystem.dll";                  DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\QnnCpu.dll";                     DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\QnnGpu.dll";                     DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\QnnIr.dll";                      DestDir: "{app}"; Flags: ignoreversion
Source: "..\target\release\Genie.dll";                      DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md";                                     DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE";                                       DestDir: "{app}"; Flags: ignoreversion
; Third-party license files (fed from the venv where they live alongside the DLLs).
Source: "..\.venv\Lib\site-packages\onnxruntime_qnn\Qualcomm_LICENSE.pdf";  DestDir: "{app}\third-party-licenses"; DestName: "Qualcomm_LICENSE.pdf";           Flags: ignoreversion
Source: "..\.venv\Lib\site-packages\onnxruntime_qnn\ThirdPartyNotices.txt"; DestDir: "{app}\third-party-licenses"; DestName: "ThirdPartyNotices.txt";          Flags: ignoreversion
Source: "..\.venv\Lib\site-packages\onnxruntime_qnn\LICENSE";               DestDir: "{app}\third-party-licenses"; DestName: "onnxruntime-qnn-LICENSE.txt";   Flags: ignoreversion
Source: "..\.venv\Lib\site-packages\onnxruntime_qnn\Privacy.md";            DestDir: "{app}\third-party-licenses"; DestName: "onnxruntime-qnn-Privacy.md";    Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}";       Filename: "{app}\{#AppExeName}"; Tasks: startmenuicon
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"; Tasks: startmenuicon
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: autostart

[Run]
; Optional: launch right after install. SkipIfSilent so headless installs don't pop a window.
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Stop any running instance before removing files.
Filename: "{cmd}"; Parameters: "/C taskkill /IM {#AppExeName} /F"; Flags: runhidden; RunOnceId: "KillOpenWritr"

[UninstallDelete]
; Leave user data (settings, models, logs) under %LOCALAPPDATA%\OpenWritr\ alone.
; Only the app/ subfolder is uninstalled. Add a custom message in the wizard.
Type: filesandordirs; Name: "{app}"
