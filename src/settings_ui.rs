//! egui settings dialog launched as a subprocess.

use crate::credentials::{store_verified, CredentialStore, WindowsCredentialStore};
use crate::enhance::{self, CopilotReadiness};
use crate::model_manager::{ModelCacheState, ModelInfo, ModelManager};
use crate::paths::settings_path;
use crate::settings::{EnhanceMode, Settings, SettingsRevision};
use crate::single_instance::SettingsTransaction;
use anyhow::{anyhow, Result};
use eframe::egui;
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, TryRecvError};

pub fn run_dialog() -> Result<()> {
    let (settings, initial_error, migration_blocked, load_failed, settings_revision) =
        match Settings::load_runtime() {
            Ok(loaded) => {
                let blocked = loaded.credential_health.requires_user_resolution;
                let load_failed = loaded.settings_error.is_some();
                let initial_error = [loaded.settings_error, loaded.credential_health.message]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join("\n");
                (
                    loaded.settings,
                    (!initial_error.is_empty()).then_some(initial_error),
                    blocked,
                    load_failed,
                    Some(loaded.revision),
                )
            }
            Err(error) => {
                let revision = Settings::revision().ok();
                (
                Settings::default(),
                Some(format!(
                    "Settings could not be loaded. Defaults are shown, but replacing the existing file requires explicit confirmation: {error}"
                )),
                false,
                true,
                revision,
            )
            }
        };
    let credential_status = read_credential_status(&WindowsCredentialStore);
    let engine_states = load_engine_states();
    let copilot_rx = start_copilot_probe();
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([640.0, 760.0])
            .with_min_inner_size([600.0, 640.0])
            .with_resizable(true)
            .with_title("OpenWritr Settings"),
        ..Default::default()
    };
    eframe::run_native(
        "OpenWritr Settings",
        opts,
        Box::new(move |cc| {
            apply_dark_theme(&cc.egui_ctx);
            Ok(Box::new(SettingsApp {
                settings,
                dirty: false,
                api_key_input: String::new(),
                remove_api_key: false,
                migration_blocked,
                load_failed,
                replace_load_failed: false,
                settings_revision,
                error: initial_error,
                credential_status,
                engine_states,
                copilot_readiness: None,
                copilot_rx: Some(copilot_rx),
                confirm_discard: false,
            }))
        }),
    )
    .map_err(|error| anyhow!("egui run failed: {error}"))
}

fn apply_dark_theme(ctx: &egui::Context) {
    use egui::{Color32, FontFamily, FontId, Stroke, Style, Visuals};
    let mut style = Style::default();
    style.visuals = Visuals::dark();
    style.visuals.window_fill = Color32::from_rgb(20, 23, 31);
    style.visuals.panel_fill = Color32::from_rgb(20, 23, 31);
    style.visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(27, 31, 40);
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(42, 47, 58);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(79, 140, 255);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(52, 58, 73);
    style.visuals.widgets.noninteractive.fg_stroke =
        Stroke::new(1.0, Color32::from_rgb(232, 236, 243));
    style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, Color32::from_rgb(232, 236, 243));
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    style.visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::from_rgb(232, 236, 243));
    style.visuals.window_stroke = Stroke::new(1.0, Color32::from_rgb(54, 61, 76));
    style.visuals.window_rounding = 8.0.into();
    style.text_styles.insert(
        egui::TextStyle::Heading,
        FontId::new(22.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        egui::TextStyle::Body,
        FontId::new(13.0, FontFamily::Proportional),
    );
    ctx.set_style(style);
}

struct SettingsApp {
    settings: Settings,
    dirty: bool,
    api_key_input: String,
    remove_api_key: bool,
    migration_blocked: bool,
    load_failed: bool,
    replace_load_failed: bool,
    settings_revision: Option<SettingsRevision>,
    error: Option<String>,
    credential_status: CredentialStatus,
    engine_states: HashMap<String, EngineUiState>,
    copilot_readiness: Option<CopilotReadiness>,
    copilot_rx: Option<Receiver<CopilotReadiness>>,
    confirm_discard: bool,
}

impl SettingsApp {
    fn persist(&mut self) -> Result<()> {
        let settings = self.settings.clone();
        settings.validate()?;
        if self.load_failed && !self.replace_load_failed {
            return Err(anyhow!(
                "Confirm replacement of the invalid or unreadable settings file before saving."
            ));
        }

        let replacement = self.api_key_input.trim();
        if self.migration_blocked && replacement.is_empty() && !self.remove_api_key {
            return Err(anyhow!(
                "Enter the API key again to secure it, or explicitly select Remove the legacy API key. The plaintext key has not been removed."
            ));
        }
        let change = requested_credential_change(
            &settings,
            self.migration_blocked,
            self.remove_api_key,
            replacement,
        );
        let path = settings_path();
        let expected_revision = self.settings_revision.as_ref().ok_or_else(|| {
            anyhow!(
                "The current settings file revision could not be captured; it was not replaced."
            )
        })?;
        let _transaction = SettingsTransaction::acquire(&path)?;
        ensure_revision_unchanged(&path, expected_revision)?;
        persist_with_credential_change(&WindowsCredentialStore, change, || {
            ensure_revision_unchanged(&path, expected_revision)?;
            settings.save_to(&path).map_err(Into::into)
        })?;
        self.api_key_input.clear();
        self.remove_api_key = false;
        self.migration_blocked = false;
        self.load_failed = false;
        self.replace_load_failed = false;
        self.error = None;
        self.dirty = false;
        Ok(())
    }

    fn poll_copilot_probe(&mut self) {
        let Some(receiver) = &self.copilot_rx else {
            return;
        };
        match receiver.try_recv() {
            Ok(status) => {
                self.copilot_readiness = Some(status);
                self.copilot_rx = None;
            }
            Err(TryRecvError::Disconnected) => {
                self.copilot_readiness = Some(CopilotReadiness::Failed(
                    "GitHub CLI readiness check stopped unexpectedly".into(),
                ));
                self.copilot_rx = None;
            }
            Err(TryRecvError::Empty) => {}
        }
    }

    fn restart_copilot_probe(&mut self) {
        self.copilot_readiness = None;
        self.copilot_rx = Some(start_copilot_probe());
    }

    fn refresh_engine_states(&mut self) {
        self.engine_states = load_engine_states();
    }

    fn selected_engine_error(&self) -> Option<String> {
        let state = self.engine_states.get(&self.settings.engine)?;
        match &state.support {
            Ok(support) if support.is_supported() => None,
            Ok(support) => Some(
                support
                    .reason()
                    .unwrap_or("The selected engine is unavailable.")
                    .to_string(),
            ),
            Err(error) => Some(format!("Could not determine engine support: {error}")),
        }
    }

    fn blocking_error(&self) -> Option<String> {
        if let Err(error) = self.settings.validate() {
            return Some(error.to_string());
        }
        if let Some(error) = self.selected_engine_error() {
            return Some(error);
        }
        if self.load_failed && !self.replace_load_failed {
            return Some("Confirm replacement of the invalid or unreadable settings file.".into());
        }
        if self.migration_blocked && self.api_key_input.trim().is_empty() && !self.remove_api_key {
            return Some(
                "Re-enter the legacy API key to secure it, or choose to remove it.".into(),
            );
        }
        let changes_secure_credential = !self.api_key_input.trim().is_empty()
            || (self.remove_api_key && !self.migration_blocked);
        if changes_secure_credential && matches!(self.credential_status, CredentialStatus::Error(_))
        {
            return Some(
                "Windows Credential Manager must be available before changing the API key.".into(),
            );
        }
        if self.settings.enhance.mode.is_enabled() {
            match self.settings.enhance.provider.as_str() {
                "github_copilot" => match self.copilot_readiness.as_ref() {
                    Some(CopilotReadiness::Ready) => {}
                    Some(CopilotReadiness::CliMissing) => {
                        return Some("GitHub CLI is not installed or is not on PATH.".into())
                    }
                    Some(CopilotReadiness::NotAuthenticated) => {
                        return Some(
                            "GitHub CLI is not authenticated. Run `gh auth login` first.".into(),
                        )
                    }
                    Some(CopilotReadiness::Failed(error)) => return Some(error.clone()),
                    None => return Some("Checking GitHub CLI authentication...".into()),
                },
                "openai_compatible" => {
                    if let CredentialStatus::Error(error) = &self.credential_status {
                        return Some(format!(
                            "Windows Credential Manager could not be read: {error}"
                        ));
                    }
                    if !has_openai_credential(
                        &self.credential_status,
                        self.migration_blocked,
                        self.remove_api_key,
                        &self.api_key_input,
                    ) {
                        return Some(
                            "Enter an API key or keep an existing saved credential.".into(),
                        );
                    }
                }
                _ => {}
            }
        }
        if self.settings_revision.is_none() {
            return Some("The settings file revision could not be captured.".into());
        }
        None
    }
}

#[derive(Clone, Debug)]
enum CredentialStatus {
    Saved,
    Missing,
    Error(String),
}

fn has_openai_credential(
    status: &CredentialStatus,
    migration_blocked: bool,
    remove_api_key: bool,
    replacement: &str,
) -> bool {
    let removes_secure_credential = remove_api_key && !migration_blocked;
    !removes_secure_credential
        && (!replacement.trim().is_empty() || matches!(status, CredentialStatus::Saved))
}

#[derive(Clone, Debug)]
struct EngineUiState {
    support: std::result::Result<crate::asr::EngineSupport, String>,
    model: std::result::Result<ModelInfo, String>,
}

fn read_credential_status(store: &dyn CredentialStore) -> CredentialStatus {
    match store.read() {
        Ok(Some(secret)) if !secret.trim().is_empty() => CredentialStatus::Saved,
        Ok(_) => CredentialStatus::Missing,
        Err(error) => CredentialStatus::Error(error.to_string()),
    }
}

fn load_engine_states() -> HashMap<String, EngineUiState> {
    ["parakeet_cpu", "parakeet_npu", "whisper_npu"]
        .into_iter()
        .map(|engine| {
            (
                engine.to_string(),
                EngineUiState {
                    support: crate::asr::engine_support(engine).map_err(|error| error.to_string()),
                    model: ModelManager::inspect(engine).map_err(|error| error.to_string()),
                },
            )
        })
        .collect()
}

fn start_copilot_probe() -> Receiver<CopilotReadiness> {
    let (sender, receiver) = mpsc::channel();
    let _ = std::thread::Builder::new()
        .name("settings-copilot-readiness".into())
        .spawn(move || {
            let _ = sender.send(enhance::github_copilot_readiness());
        });
    receiver
}

fn ensure_revision_unchanged(path: &std::path::Path, expected: &SettingsRevision) -> Result<()> {
    let current = Settings::revision_from(path)?;
    if &current != expected {
        return Err(anyhow!(
            "Settings changed after this window opened. Close and reopen Settings before saving."
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CredentialChange<'a> {
    Keep,
    Set(&'a str),
    Delete,
}

fn requested_credential_change<'a>(
    settings: &Settings,
    migration_blocked: bool,
    remove_api_key: bool,
    replacement: &'a str,
) -> CredentialChange<'a> {
    if remove_api_key {
        return if migration_blocked {
            CredentialChange::Keep
        } else {
            CredentialChange::Delete
        };
    }
    if !replacement.is_empty()
        && (migration_blocked
            || (settings.enhance.mode.is_enabled()
                && settings.enhance.provider == "openai_compatible"))
    {
        CredentialChange::Set(replacement)
    } else {
        CredentialChange::Keep
    }
}

fn persist_with_credential_change(
    store: &dyn CredentialStore,
    change: CredentialChange<'_>,
    commit_settings: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if matches!(change, CredentialChange::Keep) {
        return commit_settings();
    }

    let previous = store.read()?;
    let apply_result = match change {
        CredentialChange::Keep => unreachable!(),
        CredentialChange::Set(secret) => store_verified(store, secret),
        CredentialChange::Delete => store.delete(),
    };
    if let Err(error) = apply_result {
        return Err(error_with_rollback(
            store,
            previous.as_deref(),
            format!("Credential Manager update failed: {error}"),
        ));
    }

    if let Err(error) = commit_settings() {
        return Err(error_with_rollback(
            store,
            previous.as_deref(),
            format!("Settings were not saved: {error}"),
        ));
    }
    Ok(())
}

fn error_with_rollback(
    store: &dyn CredentialStore,
    previous: Option<&str>,
    primary: String,
) -> anyhow::Error {
    let rollback = match previous {
        Some(secret) => store_verified(store, secret),
        None => store.delete(),
    };
    match rollback {
        Ok(()) => anyhow!("{primary}. The previous secure credential was restored."),
        Err(error) => {
            anyhow!("{primary}. Restoring the previous secure credential also failed: {error}")
        }
    }
}

impl SettingsApp {
    fn render_shortcut(&mut self, ui: &mut egui::Ui) {
        ui.label("Modifier keys");
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            for (name, label) in [
                ("ctrl", "Ctrl"),
                ("shift", "Shift"),
                ("alt", "Alt"),
                ("win", "Win"),
            ] {
                let mut enabled = self
                    .settings
                    .hotkey_modifiers
                    .iter()
                    .any(|modifier| modifier == name);
                if ui.checkbox(&mut enabled, label).changed() {
                    self.settings
                        .hotkey_modifiers
                        .retain(|modifier| modifier != name);
                    if enabled {
                        self.settings.hotkey_modifiers.push(name.to_string());
                    }
                    changed = true;
                }
            }
        });
        self.settings
            .hotkey_modifiers
            .sort_by_key(|modifier| modifier_order(modifier));

        ui.add_space(6.0);
        ui.label("Optional trigger key");
        let triggers = [
            "none",
            "space",
            "tab",
            "caps_lock",
            "scroll_lock",
            "pause",
            "insert",
            "right_ctrl",
            "f13",
            "f14",
            "f15",
            "f16",
            "f17",
            "f18",
            "f19",
            "f20",
        ];
        egui::ComboBox::from_id_salt("trigger")
            .selected_text(label_for_trigger(&self.settings.hotkey_trigger))
            .show_ui(ui, |ui| {
                for trigger in triggers {
                    changed |= ui
                        .selectable_value(
                            &mut self.settings.hotkey_trigger,
                            trigger.to_string(),
                            label_for_trigger(trigger),
                        )
                        .changed();
                }
            });
        self.dirty |= changed;

        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(format!(
                "Hold {} to record; release any shortcut key to transcribe.",
                format_shortcut(&self.settings, false)
            ))
            .strong(),
        );
        match self.settings.enhance.mode {
            EnhanceMode::Never => {}
            EnhanceMode::WithShift => {
                ui.label(
                    egui::RichText::new(format!(
                        "Hold {} to also improve the transcript.",
                        format_shortcut(&self.settings, true)
                    ))
                    .small()
                    .color(secondary_text()),
                );
            }
            EnhanceMode::Always => {
                ui.label(
                    egui::RichText::new("Every transcript will be improved.")
                        .small()
                        .color(secondary_text()),
                );
            }
        }
        if let Err(error) = self.settings.validate_shortcut() {
            inline_error(ui, error.to_string());
        }
    }

    fn render_engines(&mut self, ui: &mut egui::Ui) {
        for descriptor in ENGINE_DESCRIPTORS {
            let state = self.engine_states.get(descriptor.id).cloned();
            let supported = state
                .as_ref()
                .and_then(|state| state.support.as_ref().ok())
                .map(|support| support.is_supported())
                .unwrap_or(false);
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.add_enabled_ui(supported, |ui| {
                    if ui
                        .radio(self.settings.engine == descriptor.id, descriptor.title)
                        .clicked()
                    {
                        self.settings.engine = descriptor.id.to_string();
                        self.dirty = true;
                    }
                });
                ui.label(
                    egui::RichText::new(descriptor.description)
                        .small()
                        .color(secondary_text()),
                );
                if let Some(state) = state {
                    match state.support {
                        Ok(support) if support.is_supported() => {
                            let detail = support
                                .detail()
                                .map(|detail| format!("Available: {detail}"))
                                .unwrap_or_else(|| "Available on this device".into());
                            status_label(ui, &detail, status_green());
                        }
                        Ok(support) => status_label(
                            ui,
                            &format!(
                                "Unavailable: {}",
                                support.reason().unwrap_or("unsupported hardware")
                            ),
                            warning_color(),
                        ),
                        Err(error) => inline_error(
                            ui,
                            format!("Could not determine hardware support: {error}"),
                        ),
                    }
                    match state.model {
                        Ok(info) => {
                            let cache = match info.cache_state {
                                ModelCacheState::Missing => "Not downloaded",
                                ModelCacheState::Installed => "Installed",
                                ModelCacheState::Incomplete => {
                                    "Incomplete; repaired on next engine load"
                                }
                            };
                            ui.label(
                                egui::RichText::new(format!(
                                    "{cache}. First-use download: {}.",
                                    format_bytes(info.download_bytes)
                                ))
                                .small()
                                .color(secondary_text()),
                            );
                        }
                        Err(error) => inline_error(
                            ui,
                            format!("Could not inspect the local model cache: {error}"),
                        ),
                    }
                }
            });
            ui.add_space(6.0);
        }
        if ui
            .small_button("Refresh availability and cache status")
            .clicked()
        {
            self.refresh_engine_states();
        }
        if let Some(error) = self.selected_engine_error() {
            inline_error(ui, error);
        }
    }

    fn render_output(&mut self, ui: &mut egui::Ui) {
        let before = self.settings.auto_paste;
        ui.radio_value(
            &mut self.settings.auto_paste,
            true,
            "Paste into the active app",
        );
        ui.label(
            egui::RichText::new(
                "Uses the clipboard temporarily, pastes at the cursor, then restores unchanged clipboard text.",
            )
            .small()
            .color(secondary_text()),
        );
        ui.add_space(4.0);
        ui.radio_value(
            &mut self.settings.auto_paste,
            false,
            "Copy to the clipboard",
        );
        ui.label(
            egui::RichText::new("Keeps the completed transcript on the clipboard for manual use.")
                .small()
                .color(secondary_text()),
        );
        self.dirty |= before != self.settings.auto_paste;

        ui.add_space(8.0);
        self.dirty |= ui
            .checkbox(
                &mut self.settings.overlay,
                "Show recording waveform and status messages",
            )
            .changed();
        self.dirty |= ui
            .checkbox(
                &mut self.settings.sounds,
                "Play recording start and stop sounds",
            )
            .changed();
    }

    fn render_enhancement(&mut self, ui: &mut egui::Ui) {
        ui.label("When should OpenWritr improve punctuation and wording?");
        let before_mode = self.settings.enhance.mode;
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.settings.enhance.mode, EnhanceMode::Never, "Never");
            ui.radio_value(
                &mut self.settings.enhance.mode,
                EnhanceMode::WithShift,
                "When Shift is additionally held",
            );
            ui.radio_value(
                &mut self.settings.enhance.mode,
                EnhanceMode::Always,
                "Always",
            );
        });
        if before_mode != self.settings.enhance.mode {
            if self.settings.enhance.mode == EnhanceMode::Never {
                self.api_key_input.clear();
            }
            self.dirty = true;
        }

        if self.migration_blocked {
            ui.add_space(8.0);
            self.render_legacy_credential_resolution(ui);
        }

        if self.settings.enhance.mode == EnhanceMode::Never {
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("No transcript text is sent to an external provider.")
                    .small()
                    .color(secondary_text()),
            );
            if !self.migration_blocked {
                self.render_disabled_credential_summary(ui);
            }
            return;
        }

        ui.add_space(8.0);
        ui.label("Provider");
        let before_provider = self.settings.enhance.provider.clone();
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(
                &mut self.settings.enhance.provider,
                "github_copilot".into(),
                "GitHub Copilot",
            );
            ui.radio_value(
                &mut self.settings.enhance.provider,
                "openai_compatible".into(),
                "OpenAI-compatible API",
            );
        });
        if before_provider != self.settings.enhance.provider {
            if self.settings.enhance.provider != "openai_compatible" {
                self.api_key_input.clear();
                self.remove_api_key = false;
            }
            self.dirty = true;
        }

        ui.label(
            egui::RichText::new(
                "Only the recognized transcript text is sent to this provider; recorded audio stays on this PC.",
            )
            .small()
            .color(secondary_text()),
        );
        ui.add_space(8.0);

        match self.settings.enhance.provider.as_str() {
            "github_copilot" => self.render_github_copilot(ui),
            "openai_compatible" => self.render_openai_compatible(ui),
            _ => inline_error(ui, "Select a supported enhancement provider."),
        }

        if let Err(error) = self.settings.validate_enhancement() {
            inline_error(ui, error.to_string());
        }
        if let Err(error) = self.settings.validate_shortcut() {
            if self.settings.enhance.mode == EnhanceMode::WithShift {
                inline_error(ui, error.to_string());
            }
        }
    }

    fn render_github_copilot(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            match self.copilot_readiness.as_ref() {
                None => status_label(
                    ui,
                    "Checking GitHub CLI authentication...",
                    secondary_text(),
                ),
                Some(CopilotReadiness::Ready) => {
                    status_label(ui, "GitHub CLI authentication is ready.", status_green())
                }
                Some(CopilotReadiness::CliMissing) => status_label(
                    ui,
                    "GitHub CLI is not installed or is not on PATH.",
                    error_color(),
                ),
                Some(CopilotReadiness::NotAuthenticated) => status_label(
                    ui,
                    "GitHub CLI is not authenticated. Run `gh auth login`.",
                    error_color(),
                ),
                Some(CopilotReadiness::Failed(error)) => status_label(ui, error, error_color()),
            }
            if !matches!(self.copilot_readiness, Some(CopilotReadiness::Ready))
                && ui.small_button("Retry").clicked()
            {
                self.restart_copilot_probe();
            }
        });
        ui.add_space(6.0);
        ui.label("Model");
        const MODELS: &[(&str, &str)] = &[
            ("gpt-5-mini", "GPT-5 Mini"),
            ("claude-haiku-4.5", "Claude Haiku 4.5"),
        ];
        let is_custom = !MODELS
            .iter()
            .any(|(model, _)| *model == self.settings.enhance.model);
        let mut selection = if is_custom {
            "custom".to_string()
        } else {
            self.settings.enhance.model.clone()
        };
        let selected_label = MODELS
            .iter()
            .find(|(model, _)| *model == selection)
            .map(|(_, label)| *label)
            .unwrap_or("Custom model ID");
        let before_selection = selection.clone();
        egui::ComboBox::from_id_salt("copilot_model")
            .selected_text(selected_label)
            .show_ui(ui, |ui| {
                for (model, label) in MODELS {
                    ui.selectable_value(&mut selection, (*model).into(), *label);
                }
                ui.selectable_value(&mut selection, "custom".into(), "Custom model ID");
            });
        if selection != before_selection {
            if selection == "custom" {
                self.settings.enhance.model.clear();
            } else {
                self.settings.enhance.model = selection.clone();
            }
            self.dirty = true;
        }
        if selection == "custom" {
            self.dirty |= ui
                .add(
                    egui::TextEdit::singleline(&mut self.settings.enhance.model)
                        .hint_text("Model ID"),
                )
                .changed();
        }
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(
                    "Model availability and premium usage depend on your current Copilot plan.",
                )
                .small()
                .color(secondary_text()),
            );
            ui.hyperlink_to(
                egui::RichText::new("View GitHub model documentation").small(),
                "https://docs.github.com/en/copilot/reference/ai-models/supported-models",
            );
        });
    }

    fn render_openai_compatible(&mut self, ui: &mut egui::Ui) {
        ui.label("Base URL");
        self.dirty |= ui
            .add(
                egui::TextEdit::singleline(&mut self.settings.enhance.base_url)
                    .hint_text("https://api.openai.com/v1"),
            )
            .changed();
        if insecure_remote_http(&self.settings.enhance.base_url) {
            status_label(
                ui,
                "This non-local endpoint uses unencrypted HTTP.",
                warning_color(),
            );
        }

        ui.add_space(4.0);
        ui.label("Model ID");
        self.dirty |= ui
            .add(
                egui::TextEdit::singleline(&mut self.settings.enhance.model)
                    .hint_text("Provider-specific model ID"),
            )
            .changed();

        if self.migration_blocked {
            return;
        }

        ui.add_space(6.0);
        match &self.credential_status {
            CredentialStatus::Saved if !self.remove_api_key => status_label(
                ui,
                "API key saved in Windows Credential Manager.",
                status_green(),
            ),
            CredentialStatus::Saved => status_label(
                ui,
                "The saved API key will be removed when Settings are saved.",
                warning_color(),
            ),
            CredentialStatus::Missing => status_label(ui, "No API key is saved.", secondary_text()),
            CredentialStatus::Error(error) => inline_error(
                ui,
                format!("Windows Credential Manager could not be read: {error}"),
            ),
        }
        let hint = if matches!(self.credential_status, CredentialStatus::Saved) {
            "Enter a new key to replace the saved credential"
        } else {
            "Enter API key"
        };
        if ui
            .add(
                egui::TextEdit::singleline(&mut self.api_key_input)
                    .password(true)
                    .hint_text(hint),
            )
            .changed()
        {
            self.remove_api_key = false;
            self.dirty = true;
        }
        if matches!(self.credential_status, CredentialStatus::Saved) {
            let label = if self.remove_api_key {
                "Keep saved API key"
            } else {
                "Remove saved API key"
            };
            if ui.small_button(label).clicked() {
                self.remove_api_key = !self.remove_api_key;
                if self.remove_api_key {
                    self.api_key_input.clear();
                }
                self.dirty = true;
            }
        }
    }

    fn render_legacy_credential_resolution(&mut self, ui: &mut egui::Ui) {
        message_frame(
            ui,
            "A legacy plaintext API key must be secured in Windows Credential Manager or removed before Settings can be saved.",
            warning_color(),
        );
        if ui
            .add(
                egui::TextEdit::singleline(&mut self.api_key_input)
                    .password(true)
                    .hint_text("Re-enter API key"),
            )
            .changed()
        {
            self.remove_api_key = false;
            self.dirty = true;
        }
        let label = if self.remove_api_key {
            "Keep the legacy API key"
        } else {
            "Remove the legacy API key"
        };
        if ui.small_button(label).clicked() {
            self.remove_api_key = !self.remove_api_key;
            if self.remove_api_key {
                self.api_key_input.clear();
            }
            self.dirty = true;
        }
    }

    fn render_disabled_credential_summary(&mut self, ui: &mut egui::Ui) {
        match &self.credential_status {
            CredentialStatus::Saved if self.remove_api_key => {
                status_label(
                    ui,
                    "The saved OpenAI-compatible API key will be removed when Settings are saved.",
                    warning_color(),
                );
                if ui.small_button("Keep saved API key").clicked() {
                    self.remove_api_key = false;
                    self.dirty = true;
                }
            }
            CredentialStatus::Saved => {
                status_label(
                    ui,
                    "An OpenAI-compatible API key is stored for later use.",
                    secondary_text(),
                );
                if ui.small_button("Remove saved API key").clicked() {
                    self.remove_api_key = true;
                    self.dirty = true;
                }
            }
            CredentialStatus::Missing => {}
            CredentialStatus::Error(error) => inline_error(
                ui,
                format!("Windows Credential Manager could not be read: {error}"),
            ),
        }
    }

    fn render_advanced(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Stop recording automatically after");
            self.dirty |= ui
                .add(
                    egui::DragValue::new(&mut self.settings.max_record_seconds)
                        .range(1.0..=3600.0)
                        .clamp_existing_to_range(false)
                        .speed(1.0)
                        .suffix(" s"),
                )
                .changed();
        });
        ui.label(
            egui::RichText::new(format!(
                "Current limit: {}. Reaching it automatically finishes the recording.",
                format_duration(self.settings.max_record_seconds)
            ))
            .small()
            .color(secondary_text()),
        );
    }
}

impl eframe::App for SettingsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_copilot_probe();

        if ctx.input(|input| input.viewport().close_requested()) && self.dirty {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.confirm_discard = true;
        }

        let mut save_requested = false;
        let mut cancel_requested = false;
        let blocking_error = self.blocking_error();

        egui::TopBottomPanel::top("settings_header").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.heading("OpenWritr");
                    ui.label(
                        egui::RichText::new("Local voice-to-text for Windows")
                            .color(secondary_text()),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    status_label(ui, architecture_label(), secondary_text());
                });
            });
            ui.add_space(8.0);
        });

        egui::TopBottomPanel::bottom("settings_footer").show(ctx, |ui| {
            ui.add_space(6.0);
            if let Some(error) = &blocking_error {
                ui.label(egui::RichText::new(error).small().color(error_color()));
                ui.add_space(4.0);
            }
            ui.horizontal(|ui| {
                ui.hyperlink_to(
                    egui::RichText::new("OpenWritr on GitHub").size(12.0),
                    "https://github.com/trsdn/openwritr-windows",
                );
                ui.label(
                    egui::RichText::new(format!(
                        "v{} - {}",
                        env!("CARGO_PKG_VERSION"),
                        architecture_label()
                    ))
                    .small()
                    .color(secondary_text()),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(
                            blocking_error.is_none(),
                            egui::Button::new(
                                egui::RichText::new("Save").color(egui::Color32::WHITE),
                            )
                            .fill(egui::Color32::from_rgb(79, 140, 255)),
                        )
                        .clicked()
                    {
                        save_requested = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel_requested = true;
                    }
                });
            });
            ui.add_space(6.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.set_min_width(ui.available_width());
                    ui.add_space(8.0);

                    if let Some(error) = &self.error {
                        message_frame(ui, error, error_color());
                        ui.add_space(8.0);
                    }
                    if self.load_failed {
                        let changed = ui
                            .checkbox(
                                &mut self.replace_load_failed,
                                "Replace the invalid or unreadable settings file when saving",
                            )
                            .changed();
                        self.dirty |= changed;
                        ui.add_space(8.0);
                    }

                    section(
                        ui,
                        "Recording shortcut",
                        "Choose the keys you hold while speaking.",
                        |ui| self.render_shortcut(ui),
                    );
                    section(
                        ui,
                        "Transcription",
                        "Choose where local speech recognition runs.",
                        |ui| self.render_engines(ui),
                    );
                    section(
                        ui,
                        "Output and feedback",
                        "Choose where completed text goes and what OpenWritr shows.",
                        |ui| self.render_output(ui),
                    );
                    section(
                        ui,
                        "Text enhancement",
                        "Optionally send transcript text for punctuation and cleanup.",
                        |ui| self.render_enhancement(ui),
                    );
                    section(ui, "Advanced", "Recording safety limits.", |ui| {
                        self.render_advanced(ui)
                    });
                    ui.add_space(12.0);
                });
        });

        if cancel_requested {
            if self.dirty {
                self.confirm_discard = true;
            } else {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }

        if save_requested {
            match self.persist() {
                Ok(()) => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
                Err(error) => self.error = Some(error.to_string()),
            }
        }

        if self.confirm_discard {
            let mut keep_editing = false;
            let mut discard = false;
            let response = egui::Modal::new(egui::Id::new("discard_settings")).show(ctx, |ui| {
                ui.heading("Discard unsaved changes?");
                ui.label("Your changes have not been saved.");
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Keep editing").clicked() {
                        keep_editing = true;
                    }
                    if ui.button("Discard changes").clicked() {
                        discard = true;
                    }
                });
            });
            if keep_editing || response.should_close() {
                self.confirm_discard = false;
            }
            if discard {
                self.dirty = false;
                self.confirm_discard = false;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }

        if self.copilot_rx.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
    }
}

struct EngineDescriptor {
    id: &'static str,
    title: &'static str,
    description: &'static str,
}

const ENGINE_DESCRIPTORS: &[EngineDescriptor] = &[
    EngineDescriptor {
        id: "parakeet_cpu",
        title: "Parakeet TDT v3 - CPU",
        description: "Runs locally on the CPU and works on Intel, AMD, and ARM64 PCs.",
    },
    EngineDescriptor {
        id: "parakeet_npu",
        title: "Parakeet TDT v3 - Snapdragon NPU",
        description: "Runs the encoder on the Snapdragon X Elite Hexagon NPU.",
    },
    EngineDescriptor {
        id: "whisper_npu",
        title: "Whisper Large v3 Turbo - Snapdragon NPU",
        description: "Runs the multilingual Whisper encoder and decoder on Snapdragon X Elite.",
    },
];

fn section(ui: &mut egui::Ui, title: &str, description: &str, body: impl FnOnce(&mut egui::Ui)) {
    ui.label(egui::RichText::new(title).size(16.0).strong());
    ui.label(
        egui::RichText::new(description)
            .small()
            .color(secondary_text()),
    );
    ui.add_space(4.0);
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        body(ui);
    });
    ui.add_space(14.0);
}

fn inline_error(ui: &mut egui::Ui, message: impl Into<String>) {
    ui.label(
        egui::RichText::new(message.into())
            .small()
            .color(error_color()),
    );
}

fn status_label(ui: &mut egui::Ui, message: &str, color: egui::Color32) {
    ui.label(egui::RichText::new(message).small().color(color));
}

fn message_frame(ui: &mut egui::Ui, message: &str, color: egui::Color32) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        ui.label(egui::RichText::new(message).color(color));
    });
}

fn secondary_text() -> egui::Color32 {
    egui::Color32::from_rgb(154, 163, 178)
}

fn status_green() -> egui::Color32 {
    egui::Color32::from_rgb(74, 222, 128)
}

fn warning_color() -> egui::Color32 {
    egui::Color32::from_rgb(250, 204, 21)
}

fn error_color() -> egui::Color32 {
    egui::Color32::from_rgb(248, 113, 113)
}

fn architecture_label() -> &'static str {
    match std::env::consts::ARCH {
        "aarch64" => "ARM64",
        "x86_64" => "x64",
        other => other,
    }
}

fn modifier_order(modifier: &str) -> usize {
    match modifier {
        "ctrl" => 0,
        "shift" => 1,
        "alt" => 2,
        "win" => 3,
        _ => usize::MAX,
    }
}

fn format_shortcut(settings: &Settings, add_shift: bool) -> String {
    let mut modifiers = settings.hotkey_modifiers.clone();
    if add_shift && !modifiers.iter().any(|modifier| modifier == "shift") {
        modifiers.push("shift".into());
    }
    modifiers.sort_by_key(|modifier| modifier_order(modifier));
    let mut parts = modifiers
        .iter()
        .map(|modifier| match modifier.as_str() {
            "ctrl" => "Ctrl".to_string(),
            "shift" => "Shift".to_string(),
            "alt" => "Alt".to_string(),
            "win" => "Win".to_string(),
            other => other.to_string(),
        })
        .collect::<Vec<_>>();
    if settings.hotkey_trigger != "none" {
        parts.push(label_for_trigger(&settings.hotkey_trigger));
    }
    if parts.is_empty() {
        "an incomplete shortcut".into()
    } else {
        parts.join(" + ")
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else {
        format!("{:.0} MB", bytes as f64 / 1_000_000.0)
    }
}

fn format_duration(seconds: f32) -> String {
    if seconds >= 60.0 && seconds % 60.0 == 0.0 {
        let minutes = seconds / 60.0;
        if minutes == 1.0 {
            "1 minute".into()
        } else {
            format!("{minutes:.0} minutes")
        }
    } else if seconds == 1.0 {
        "1 second".into()
    } else {
        format!("{seconds:.0} seconds")
    }
}

fn insecure_remote_http(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url.trim()) else {
        return false;
    };
    if url.scheme() != "http" {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    if host.eq_ignore_ascii_case("localhost") {
        return false;
    }
    host.parse::<std::net::IpAddr>()
        .map(|address| !address.is_loopback())
        .unwrap_or(true)
}

fn label_for_trigger(trigger: &str) -> String {
    match trigger {
        "none" => "None (modifiers only)".into(),
        "space" => "Space".into(),
        "tab" => "Tab".into(),
        "caps_lock" => "Caps Lock".into(),
        "scroll_lock" => "Scroll Lock".into(),
        "pause" => "Pause / Break".into(),
        "insert" => "Insert".into(),
        "right_ctrl" => "Right Ctrl".into(),
        other if other.starts_with('f') => other.to_uppercase(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::CredentialError;
    use parking_lot::Mutex;

    #[derive(Default)]
    struct FakeStore {
        secret: Mutex<Option<String>>,
    }

    impl CredentialStore for FakeStore {
        fn read(&self) -> std::result::Result<Option<String>, CredentialError> {
            Ok(self.secret.lock().clone())
        }

        fn write(&self, secret: &str) -> std::result::Result<(), CredentialError> {
            *self.secret.lock() = Some(secret.to_string());
            Ok(())
        }

        fn delete(&self) -> std::result::Result<(), CredentialError> {
            *self.secret.lock() = None;
            Ok(())
        }
    }

    #[test]
    fn legacy_plaintext_removal_keeps_a_distinct_secure_credential() {
        let mut settings = Settings::default();
        assert_eq!(
            requested_credential_change(&settings, true, true, ""),
            CredentialChange::Keep
        );
        assert!(has_openai_credential(
            &CredentialStatus::Saved,
            true,
            true,
            ""
        ));
        assert_eq!(
            requested_credential_change(&settings, false, true, ""),
            CredentialChange::Delete
        );
        assert!(!has_openai_credential(
            &CredentialStatus::Saved,
            false,
            true,
            ""
        ));

        settings.enhance.mode = EnhanceMode::Always;
        settings.enhance.provider = "openai_compatible".into();
        assert_eq!(
            requested_credential_change(&settings, false, false, "replacement"),
            CredentialChange::Set("replacement")
        );
    }

    #[test]
    fn failed_settings_commit_restores_the_previous_credential() {
        let store = FakeStore::default();
        *store.secret.lock() = Some("previous".into());

        let result =
            persist_with_credential_change(&store, CredentialChange::Set("replacement"), || {
                Err(anyhow!("injected settings failure"))
            });

        assert!(result.is_err());
        assert_eq!(store.secret.lock().as_deref(), Some("previous"));
    }

    #[test]
    fn failed_settings_commit_removes_a_new_credential_when_none_existed() {
        let store = FakeStore::default();

        let result =
            persist_with_credential_change(&store, CredentialChange::Set("replacement"), || {
                Err(anyhow!("injected settings failure"))
            });

        assert!(result.is_err());
        assert!(store.secret.lock().is_none());
    }

    #[test]
    fn failed_settings_commit_restores_a_deleted_credential() {
        let store = FakeStore::default();
        *store.secret.lock() = Some("previous".into());

        let result = persist_with_credential_change(&store, CredentialChange::Delete, || {
            Err(anyhow!("injected settings failure"))
        });

        assert!(result.is_err());
        assert_eq!(store.secret.lock().as_deref(), Some("previous"));
    }

    #[test]
    fn stale_settings_revision_is_rejected_before_persistence() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("settings.json");
        std::fs::write(&path, b"first").unwrap();
        let revision = Settings::revision_from(&path).unwrap();
        std::fs::write(&path, b"second").unwrap();

        let error = ensure_revision_unchanged(&path, &revision).unwrap_err();

        assert!(error
            .to_string()
            .contains("changed after this window opened"));
    }

    #[test]
    fn credential_status_never_exposes_the_secret() {
        let store = FakeStore::default();
        *store.secret.lock() = Some("super-secret".into());

        assert!(matches!(
            read_credential_status(&store),
            CredentialStatus::Saved
        ));
    }

    #[test]
    fn shortcut_preview_is_ordered_and_includes_optional_shift() {
        let mut settings = Settings::default();
        settings.hotkey_modifiers = vec!["win".into(), "ctrl".into()];
        assert_eq!(format_shortcut(&settings, false), "Ctrl + Win");
        assert_eq!(format_shortcut(&settings, true), "Ctrl + Shift + Win");
    }

    #[test]
    fn insecure_http_warning_excludes_loopback_endpoints() {
        assert!(!insecure_remote_http("http://localhost:11434/v1"));
        assert!(!insecure_remote_http("http://127.0.0.1:11434/v1"));
        assert!(insecure_remote_http("http://example.com/v1"));
        assert!(!insecure_remote_http("https://example.com/v1"));
    }
}
