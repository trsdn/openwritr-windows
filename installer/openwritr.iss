; Inno Setup script — produces a single-exe installer for OpenWritr.
;
; Build with:
;     "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" /Qp installer\openwritr.iss
;
; The Cargo bin `installer_build` (cargo run --release --bin installer_build)
; runs this automatically as part of the release flow.

#define AppName       "OpenWritr"
#ifndef AppVersion
#define AppVersion    "0.6.1"
#endif
#define AppPublisher  "Torsten Mahr"
#define AppURL        "https://github.com/trsdn/openwritr-windows"
#define AppExeName    "openwritr.exe"
#define SrcDir        "..\target\stage\arm64"

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
; scripts/prepare_release.py creates this directory from release-manifest.json
; and fails before Inno starts if any required runtime file is absent or invalid.
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";       Filename: "{app}\{#AppExeName}"; Tasks: startmenuicon
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"; Tasks: startmenuicon
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; Autostart is a HKCU\...\Run value so the app can read, toggle, and verify the
; exact same mechanism at runtime (Settings → Startup). uninsdeletevalue removes
; it on uninstall even if the task was unselected but later enabled in the app.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
      ValueType: string; ValueName: "OpenWritr"; \
      ValueData: """{app}\{#AppExeName}"""; \
      Flags: uninsdeletevalue; Tasks: autostart

[InstallDelete]
; Migrate away from the legacy startup-folder shortcut used by older installers,
; so autostart state never lives in two mechanisms at once. Gated on the autostart
; task so we only remove the shortcut when we actually write the Run value in its
; place. If the user deselects autostart on upgrade, the shortcut is left untouched
; and the app-side migration in autostart.rs remains able to pick it up later —
; a previously active autostart is never silently dropped.
Type: files; Name: "{userstartup}\{#AppName}.lnk"; Tasks: autostart

[Run]
; Optional: launch right after install. SkipIfSilent so headless installs don't pop a window.
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Stop any running instance before removing files.
Filename: "{cmd}"; Parameters: "/C taskkill /IM {#AppExeName} /F"; Flags: runhidden; RunOnceId: "KillOpenWritr"
; Always remove the autostart Run value, even if the user enabled it from inside
; the app after installing without the autostart task (so uninsdeletevalue never
; recorded it). reg delete is a no-op with exit 1 if it is already gone.
Filename: "{cmd}"; Parameters: "/C reg delete ""HKCU\Software\Microsoft\Windows\CurrentVersion\Run"" /v OpenWritr /f"; Flags: runhidden; RunOnceId: "RemoveOpenWritrRun"

[UninstallDelete]
; Leave user data (settings, models, logs) under %LOCALAPPDATA%\OpenWritr\ alone.
; Only the app/ subfolder is uninstalled. Add a custom message in the wizard.
Type: files; Name: "{userstartup}\{#AppName}.lnk"
Type: filesandordirs; Name: "{app}"

[Code]
// Preselect the autostart task on upgrade when this user already had autostart
// enabled by an older install — either the legacy startup-folder shortcut or the
// HKCU Run value. Inno remembers the previous task selection across upgrades, so
// a user who once deselected the task would otherwise get it deselected again by
// default; this restores it to checked when prior autostart state is detected.
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
