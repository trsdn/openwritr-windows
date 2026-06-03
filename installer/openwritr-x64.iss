; Inno Setup script — Intel/AMD (x86_64) build of OpenWritr.
;
; CPU-only: Parakeet runs on the ONNX Runtime CPU EP. No Qualcomm QNN
; runtime (Hexagon is Snapdragon-only). Much smaller than the arm64 build.
;
; Build:
;   "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" /Qp installer\openwritr-x64.iss

#define AppName       "OpenWritr"
#define AppVersion    "0.3.0"
#define AppPublisher  "Torsten Mahr"
#define AppURL        "https://github.com/trsdn/openwritr-windows"
#define AppExeName    "openwritr.exe"
#define SrcDir        "..\target\x86_64-pc-windows-msvc\release"

[Setup]
AppId={{7F3C1A92-5E84-4D17-B6A9-1C2E4F77A083}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={localappdata}\OpenWritr\app
DefaultGroupName={#AppName}
DisableProgramGroupPage=auto
; x64 installer: runs on Intel/AMD Windows. (Also runs under x64 emulation on
; ARM64, but those users want the native arm64 build.)
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\target\dist
OutputBaseFilename=openwritr-windows-x64-v{#AppVersion}-setup
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
CloseApplications=force
RestartApplications=no
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
SetupIconFile=openwritr.ico
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
LicenseFile=..\LICENSE

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"

[Tasks]
Name: "autostart"; Description: "Start {#AppName} automatically when I log in"; GroupDescription: "Startup:"
Name: "startmenuicon"; Description: "Create a Start Menu shortcut"; GroupDescription: "Shortcuts:"; Flags: checkedonce
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
Source: "{#SrcDir}\openwritr.exe"; DestDir: "{app}"; Flags: ignoreversion
; CPU build of onnxruntime, vendored from the win_amd64 wheel (see
; scripts/fetch_x64_ort.py). No QNN/Hexagon DLLs on Intel.
Source: "..\vendor\x64\onnxruntime.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\.venv\Lib\site-packages\onnxruntime_qnn\ThirdPartyNotices.txt"; DestDir: "{app}\third-party-licenses"; DestName: "ThirdPartyNotices.txt"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: startmenuicon
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"; Tasks: startmenuicon
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: autostart

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{cmd}"; Parameters: "/C taskkill /IM {#AppExeName} /F"; Flags: runhidden; RunOnceId: "KillOpenWritr"

[UninstallDelete]
; Leave user data (settings, models, logs) under %LOCALAPPDATA%\OpenWritr\ alone.
Type: filesandordirs; Name: "{app}"
