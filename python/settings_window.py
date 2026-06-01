"""Native WPF settings window via pythonnet + .NET 8 WindowsDesktop runtime.

Uses XAML inline for the layout — real Win32/WPF window with native
controls, Fluent-style colours, proper DPI scaling. No bundled dependencies
beyond pythonnet (already installed) and the .NET 8 Desktop runtime that
ships on Windows on ARM.

The dialog is launched as a subprocess from openwritr.py so each open
gets a clean WPF Application + Dispatcher.
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
from pathlib import Path

APPDATA = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "OpenWritr"
SETTINGS_PATH = APPDATA / "settings.json"

DEFAULTS = {
    "hotkey_modifiers": ["ctrl", "win"],
    "auto_paste": True,
    "overlay": True,
    "sounds": True,
    "min_record_seconds": 0.25,
    "max_record_seconds": 60,
    "enhance": {"provider": "off", "base_url": "https://api.openai.com/v1",
                "api_key": "", "model": "gpt-4o-mini"},
}


def _load() -> dict:
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text("utf-8"))
            merged = {**DEFAULTS, **data}
            merged["enhance"] = {**DEFAULTS["enhance"], **(data.get("enhance") or {})}
            return merged
        except Exception:
            pass
    return {**DEFAULTS, "enhance": dict(DEFAULTS["enhance"])}


def _save(payload: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(payload, indent=2), "utf-8")


def _boot_wpf() -> None:
    """Bootstrap pythonnet against WindowsDesktop (WPF) runtime."""
    from clr_loader import get_coreclr
    from pythonnet import set_runtime, load
    cfg = {
        "runtimeOptions": {
            "tfm": "net8.0",
            "framework": {"name": "Microsoft.WindowsDesktop.App", "version": "8.0.0"},
        }
    }
    path = tempfile.mktemp(suffix=".runtimeconfig.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    set_runtime(get_coreclr(runtime_config=path))
    load()


XAML = r"""<Window
    xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
    xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
    Title="OpenWritr Settings"
    Width="560" Height="720"
    WindowStartupLocation="CenterScreen"
    ResizeMode="CanResize"
    Background="#14171F"
    FontFamily="Segoe UI Variable Text, Segoe UI"
    FontSize="13"
    Foreground="#E8ECF3"
    UseLayoutRounding="True"
    TextOptions.TextFormattingMode="Display">

  <Window.Resources>
    <SolidColorBrush x:Key="Accent" Color="#4F8CFF"/>
    <SolidColorBrush x:Key="AccentHover" Color="#6AA0FF"/>
    <SolidColorBrush x:Key="Surface" Color="#1B1F28"/>
    <SolidColorBrush x:Key="SurfaceAlt" Color="#252A36"/>
    <SolidColorBrush x:Key="Border" Color="#363D4C"/>
    <SolidColorBrush x:Key="Dim" Color="#9AA3B2"/>

    <Style x:Key="Section" TargetType="Border">
      <Setter Property="Background" Value="{StaticResource Surface}"/>
      <Setter Property="BorderBrush" Value="{StaticResource Border}"/>
      <Setter Property="BorderThickness" Value="1"/>
      <Setter Property="CornerRadius" Value="10"/>
      <Setter Property="Padding" Value="18,16"/>
      <Setter Property="Margin" Value="0,0,0,12"/>
    </Style>

    <Style x:Key="SectionTitle" TargetType="TextBlock">
      <Setter Property="FontFamily" Value="Segoe UI Variable Small, Segoe UI"/>
      <Setter Property="FontWeight" Value="SemiBold"/>
      <Setter Property="FontSize" Value="11"/>
      <Setter Property="Foreground" Value="{StaticResource Dim}"/>
      <Setter Property="Margin" Value="0,0,0,12"/>
    </Style>

    <Style TargetType="ToggleButton" x:Key="ChipToggle">
      <Setter Property="Padding" Value="14,7"/>
      <Setter Property="Margin" Value="0,0,8,0"/>
      <Setter Property="Foreground" Value="#E8ECF3"/>
      <Setter Property="Background" Value="#2A2F3A"/>
      <Setter Property="BorderBrush" Value="{StaticResource Border}"/>
      <Setter Property="BorderThickness" Value="1"/>
      <Setter Property="Cursor" Value="Hand"/>
      <Setter Property="FontWeight" Value="SemiBold"/>
      <Setter Property="Template">
        <Setter.Value>
          <ControlTemplate TargetType="ToggleButton">
            <Border x:Name="Bd" CornerRadius="999"
                    Background="{TemplateBinding Background}"
                    BorderBrush="{TemplateBinding BorderBrush}"
                    BorderThickness="{TemplateBinding BorderThickness}"
                    Padding="{TemplateBinding Padding}">
              <ContentPresenter HorizontalAlignment="Center" VerticalAlignment="Center"/>
            </Border>
            <ControlTemplate.Triggers>
              <Trigger Property="IsMouseOver" Value="True">
                <Setter TargetName="Bd" Property="BorderBrush" Value="{StaticResource Accent}"/>
              </Trigger>
              <Trigger Property="IsChecked" Value="True">
                <Setter TargetName="Bd" Property="Background" Value="{StaticResource Accent}"/>
                <Setter TargetName="Bd" Property="BorderBrush" Value="{StaticResource Accent}"/>
                <Setter Property="Foreground" Value="White"/>
              </Trigger>
            </ControlTemplate.Triggers>
          </ControlTemplate>
        </Setter.Value>
      </Setter>
    </Style>

    <Style TargetType="CheckBox" x:Key="Switch">
      <Setter Property="Cursor" Value="Hand"/>
      <Setter Property="Template">
        <Setter.Value>
          <ControlTemplate TargetType="CheckBox">
            <Grid Width="42" Height="22">
              <Border x:Name="Track" CornerRadius="11"
                      Background="#2A2F3A" BorderBrush="{StaticResource Border}" BorderThickness="1"/>
              <Ellipse x:Name="Knob" Width="16" Height="16" Fill="#D5DBE6"
                       HorizontalAlignment="Left" Margin="3,0,0,0"/>
            </Grid>
            <ControlTemplate.Triggers>
              <Trigger Property="IsChecked" Value="True">
                <Setter TargetName="Track" Property="Background" Value="{StaticResource Accent}"/>
                <Setter TargetName="Track" Property="BorderBrush" Value="{StaticResource Accent}"/>
                <Setter TargetName="Knob" Property="HorizontalAlignment" Value="Right"/>
                <Setter TargetName="Knob" Property="Margin" Value="0,0,3,0"/>
                <Setter TargetName="Knob" Property="Fill" Value="White"/>
              </Trigger>
            </ControlTemplate.Triggers>
          </ControlTemplate>
        </Setter.Value>
      </Setter>
    </Style>

    <Style TargetType="TextBox">
      <Setter Property="Background" Value="{StaticResource SurfaceAlt}"/>
      <Setter Property="Foreground" Value="#E8ECF3"/>
      <Setter Property="BorderBrush" Value="{StaticResource Border}"/>
      <Setter Property="Padding" Value="10,8"/>
      <Setter Property="Margin" Value="0,4,0,8"/>
      <Setter Property="CaretBrush" Value="#E8ECF3"/>
    </Style>

    <Style TargetType="PasswordBox">
      <Setter Property="Background" Value="{StaticResource SurfaceAlt}"/>
      <Setter Property="Foreground" Value="#E8ECF3"/>
      <Setter Property="BorderBrush" Value="{StaticResource Border}"/>
      <Setter Property="Padding" Value="10,8"/>
      <Setter Property="Margin" Value="0,4,0,8"/>
    </Style>

    <Style TargetType="ComboBox">
      <Setter Property="Background" Value="{StaticResource SurfaceAlt}"/>
      <Setter Property="Foreground" Value="#E8ECF3"/>
      <Setter Property="BorderBrush" Value="{StaticResource Border}"/>
      <Setter Property="Padding" Value="10,6"/>
      <Setter Property="Margin" Value="0,4,0,8"/>
    </Style>

    <Style TargetType="Button">
      <Setter Property="Padding" Value="20,9"/>
      <Setter Property="Background" Value="#2A2F3A"/>
      <Setter Property="Foreground" Value="#E8ECF3"/>
      <Setter Property="BorderBrush" Value="{StaticResource Border}"/>
      <Setter Property="FontWeight" Value="SemiBold"/>
      <Setter Property="Cursor" Value="Hand"/>
      <Setter Property="Template">
        <Setter.Value>
          <ControlTemplate TargetType="Button">
            <Border x:Name="Bd" CornerRadius="6"
                    Background="{TemplateBinding Background}"
                    BorderBrush="{TemplateBinding BorderBrush}"
                    BorderThickness="1"
                    Padding="{TemplateBinding Padding}">
              <ContentPresenter HorizontalAlignment="Center" VerticalAlignment="Center"/>
            </Border>
            <ControlTemplate.Triggers>
              <Trigger Property="IsMouseOver" Value="True">
                <Setter TargetName="Bd" Property="Background" Value="#343B49"/>
              </Trigger>
            </ControlTemplate.Triggers>
          </ControlTemplate>
        </Setter.Value>
      </Setter>
    </Style>

    <Style TargetType="Button" x:Key="Primary" BasedOn="{StaticResource {x:Type Button}}">
      <Setter Property="Background" Value="{StaticResource Accent}"/>
      <Setter Property="BorderBrush" Value="{StaticResource Accent}"/>
      <Setter Property="Foreground" Value="White"/>
    </Style>

    <Style TargetType="TextBlock" x:Key="Hint">
      <Setter Property="Foreground" Value="{StaticResource Dim}"/>
      <Setter Property="FontSize" Value="12"/>
      <Setter Property="Margin" Value="0,4,0,0"/>
      <Setter Property="TextWrapping" Value="Wrap"/>
    </Style>

    <Style TargetType="TextBlock" x:Key="FieldLabel">
      <Setter Property="Foreground" Value="{StaticResource Dim}"/>
      <Setter Property="FontSize" Value="12"/>
      <Setter Property="Margin" Value="0,6,0,2"/>
    </Style>
  </Window.Resources>

  <ScrollViewer VerticalScrollBarVisibility="Auto" HorizontalScrollBarVisibility="Disabled">
    <StackPanel Margin="28,24,28,28">
      <TextBlock Text="OpenWritr" FontSize="26" FontWeight="SemiBold"
                 FontFamily="Segoe UI Variable Display, Segoe UI"/>
      <TextBlock Text="Voice-to-text for Windows on ARM" Foreground="{StaticResource Dim}" Margin="0,2,0,18"/>

      <Border Style="{StaticResource Section}">
        <StackPanel>
          <TextBlock Style="{StaticResource SectionTitle}" Text="HOTKEY (HOLD TO RECORD)"/>
          <StackPanel Orientation="Horizontal">
            <ToggleButton x:Name="ModCtrl"  Style="{StaticResource ChipToggle}" Content="Ctrl"/>
            <ToggleButton x:Name="ModShift" Style="{StaticResource ChipToggle}" Content="Shift"/>
            <ToggleButton x:Name="ModAlt"   Style="{StaticResource ChipToggle}" Content="Alt"/>
            <ToggleButton x:Name="ModWin"   Style="{StaticResource ChipToggle}" Content="Win"/>
          </StackPanel>
          <TextBlock Style="{StaticResource Hint}"
                     Text="… plus Space (always required). Hold the combo to dictate; release to transcribe."/>
        </StackPanel>
      </Border>

      <Border Style="{StaticResource Section}">
        <StackPanel>
          <TextBlock Style="{StaticResource SectionTitle}" Text="BEHAVIOUR"/>
          <Grid Margin="0,4">
            <Grid.ColumnDefinitions>
              <ColumnDefinition Width="*"/>
              <ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>
            <TextBlock Text="Auto-paste at cursor" VerticalAlignment="Center"/>
            <CheckBox x:Name="AutoPaste" Style="{StaticResource Switch}" Grid.Column="1"/>
          </Grid>
          <Grid Margin="0,8">
            <Grid.ColumnDefinitions>
              <ColumnDefinition Width="*"/>
              <ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>
            <TextBlock Text="Show overlay while recording" VerticalAlignment="Center"/>
            <CheckBox x:Name="OverlayOn" Style="{StaticResource Switch}" Grid.Column="1"/>
          </Grid>
          <Grid Margin="0,8,0,0">
            <Grid.ColumnDefinitions>
              <ColumnDefinition Width="*"/>
              <ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>
            <TextBlock Text="Play start/stop sounds" VerticalAlignment="Center"/>
            <CheckBox x:Name="SoundsOn" Style="{StaticResource Switch}" Grid.Column="1"/>
          </Grid>
        </StackPanel>
      </Border>

      <Border Style="{StaticResource Section}">
        <StackPanel>
          <TextBlock Style="{StaticResource SectionTitle}" Text="ENHANCE (PUNCTUATION + CLEANUP)"/>
          <TextBlock Style="{StaticResource Hint}" Margin="0,0,0,4"
                     Text="Hold the hotkey with Alt also pressed to trigger an LLM cleanup pass after transcription."/>
          <TextBlock Style="{StaticResource FieldLabel}" Text="Provider"/>
          <ComboBox x:Name="Provider" SelectedIndex="0">
            <ComboBoxItem Content="Off" Tag="off"/>
            <ComboBoxItem Content="GitHub Copilot (uses gh auth token)" Tag="github_copilot"/>
            <ComboBoxItem Content="OpenAI-compatible API" Tag="openai_compatible"/>
          </ComboBox>
          <TextBlock Style="{StaticResource FieldLabel}" Text="Base URL (OpenAI-compatible only)"/>
          <TextBox x:Name="BaseUrl"/>
          <TextBlock Style="{StaticResource FieldLabel}" Text="API key (OpenAI-compatible only)"/>
          <PasswordBox x:Name="ApiKey"/>
          <TextBlock Style="{StaticResource FieldLabel}" Text="Model"/>
          <TextBox x:Name="Model"/>
        </StackPanel>
      </Border>

      <StackPanel Orientation="Horizontal" HorizontalAlignment="Right" Margin="0,8,0,0">
        <Button x:Name="CancelBtn" Content="Cancel" Margin="0,0,10,0"/>
        <Button x:Name="SaveBtn"   Content="Save"   Style="{StaticResource Primary}"/>
      </StackPanel>
    </StackPanel>
  </ScrollViewer>
</Window>
"""


def main() -> int:
    _boot_wpf()
    import clr
    clr.AddReference("PresentationFramework")
    clr.AddReference("PresentationCore")
    clr.AddReference("WindowsBase")
    clr.AddReference("System.Xaml")

    from System.Threading import Thread, ApartmentState, ThreadStart

    result = [0]

    def _run():
        result[0] = _run_window()

    t = Thread(ThreadStart(_run))
    t.SetApartmentState(ApartmentState.STA)
    t.Start()
    t.Join()
    return result[0]


def _run_window() -> int:
    import clr
    from System.Windows import Application
    from System.Windows.Markup import XamlReader

    settings = _load()

    window = XamlReader.Parse(XAML)
    app = Application()

    def find(name):
        return window.FindName(name)

    # Apply Mica backdrop via DWM. Best-effort.
    try:
        from System.Windows.Interop import WindowInteropHelper
        import ctypes
        from ctypes import wintypes

        def _apply_mica(hwnd: int) -> None:
            dwm = ctypes.windll.dwmapi
            dark = ctypes.c_int(1)
            backdrop = ctypes.c_int(2)  # DWMSBT_MAINWINDOW = Mica
            dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), 20,
                                      ctypes.byref(dark), ctypes.sizeof(dark))
            dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), 38,
                                      ctypes.byref(backdrop), ctypes.sizeof(backdrop))

        def _on_loaded(sender, args):
            hwnd = WindowInteropHelper(window).Handle.ToInt64()
            if hwnd:
                _apply_mica(hwnd)

        window.Loaded += _on_loaded
    except Exception:
        pass

    # Initial state.
    have = set(settings.get("hotkey_modifiers") or [])
    find("ModCtrl").IsChecked = "ctrl" in have
    find("ModShift").IsChecked = "shift" in have
    find("ModAlt").IsChecked = "alt" in have
    find("ModWin").IsChecked = "win" in have
    find("AutoPaste").IsChecked = bool(settings.get("auto_paste", True))
    find("OverlayOn").IsChecked = bool(settings.get("overlay", True))
    find("SoundsOn").IsChecked = bool(settings.get("sounds", True))
    enh = settings.get("enhance") or {}
    provider_value = enh.get("provider", "off")
    cb = find("Provider")
    for i in range(cb.Items.Count):
        if str(cb.Items[i].Tag) == provider_value:
            cb.SelectedIndex = i
            break
    find("BaseUrl").Text = enh.get("base_url", "https://api.openai.com/v1")
    find("ApiKey").Password = enh.get("api_key", "")
    find("Model").Text = enh.get("model", "gpt-4o-mini") or "gpt-4o-mini"

    def on_save(sender, args):
        mods = []
        if find("ModCtrl").IsChecked:  mods.append("ctrl")
        if find("ModShift").IsChecked: mods.append("shift")
        if find("ModAlt").IsChecked:   mods.append("alt")
        if find("ModWin").IsChecked:   mods.append("win")
        if not mods:
            mods = ["ctrl", "shift"]
        item = find("Provider").SelectedItem
        provider_tag = str(item.Tag) if item is not None else "off"
        payload = {
            "hotkey_modifiers": mods,
            "auto_paste": bool(find("AutoPaste").IsChecked),
            "overlay": bool(find("OverlayOn").IsChecked),
            "sounds": bool(find("SoundsOn").IsChecked),
            "min_record_seconds": settings.get("min_record_seconds", 0.25),
            "max_record_seconds": settings.get("max_record_seconds", 60),
            "enhance": {
                "provider": provider_tag,
                "base_url": find("BaseUrl").Text.strip(),
                "api_key":  find("ApiKey").Password.strip(),
                "model":    find("Model").Text.strip() or "gpt-4o-mini",
            },
        }
        _save(payload)
        window.Close()

    def on_cancel(sender, args):
        window.Close()

    find("SaveBtn").Click += on_save
    find("CancelBtn").Click += on_cancel

    app.Run(window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
