; Inno Setup script — Intel/AMD (x86_64) build of OpenWritr.
;
; CPU-only: Parakeet runs on the ONNX Runtime CPU EP. No Qualcomm QNN
; runtime (Hexagon is Snapdragon-only). Much smaller than the arm64 build.
;
; Build:
;   "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" /Qp installer\openwritr-x64.iss

#define AppName       "OpenWritr"
#ifndef AppVersion
#define AppVersion    "0.6.1"
#endif
#define AppPublisher  "Torsten Mahr"
#define AppURL        "https://github.com/trsdn/openwritr-windows"
#define AppExeName    "openwritr.exe"
#define SrcDir        "..\target\stage\x64"

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
; The canonical release stage contains no Qualcomm-only files on x64.
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: startmenuicon
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"; Tasks: startmenuicon
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; Autostart is a HKCU\...\Run value so the app can read, toggle, and verify the
; exact same mechanism at runtime (Settings → Startup). uninsdeletevalue removes
; it on uninstall even if the task was unselected but later enabled in the app.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "OpenWritr"; ValueData: """{app}\{#AppExeName}"""; Flags: uninsdeletevalue; Tasks: autostart

[InstallDelete]
; Migrate away from the legacy startup-folder shortcut used by older installers,
; so autostart state never lives in two mechanisms at once. Gated on the autostart
; task so we only remove the shortcut when we actually write the Run value in its
; place. If the user deselects autostart on upgrade, the shortcut is left untouched
; and the app-side migration in autostart.rs remains able to pick it up later —
; a previously active autostart is never silently dropped.
Type: files; Name: "{userstartup}\{#AppName}.lnk"; Tasks: autostart

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{cmd}"; Parameters: "/C taskkill /IM {#AppExeName} /F"; Flags: runhidden; RunOnceId: "KillOpenWritr"
; Always remove the autostart Run value, even if the user enabled it from inside
; the app after installing without the autostart task (so uninsdeletevalue never
; recorded it). reg delete is a no-op with exit 1 if it is already gone.
Filename: "{cmd}"; Parameters: "/C reg delete ""HKCU\Software\Microsoft\Windows\CurrentVersion\Run"" /v OpenWritr /f"; Flags: runhidden; RunOnceId: "RemoveOpenWritrRun"

[UninstallDelete]
; Leave user data (settings, models, logs) under %LOCALAPPDATA%\OpenWritr\ alone.
Type: files; Name: "{userstartup}\{#AppName}.lnk"
Type: filesandordirs; Name: "{app}"

[Code]
{ Preselect the autostart task on upgrade when this user already had autostart
  enabled by an older install — either the legacy {userstartup} shortcut or the
  HKCU Run value. Inno remembers the previous task selection across upgrades, so
  a user who once deselected the task would otherwise get it deselected again by
  default; this restores it to checked when prior autostart state is detected. }
function LegacyAutostartEnabled(): Boolean;
begin
  Result := FileExists(ExpandConstant('{userstartup}\{#AppName}.lnk')) or
    RegValueExists(HKEY_CURRENT_USER,
      'Software\Microsoft\Windows\CurrentVersion\Run', 'OpenWritr');
end;

procedure InitializeWizard();
begin
  if LegacyAutostartEnabled() then
    WizardSelectTasks('autostart');
end;
