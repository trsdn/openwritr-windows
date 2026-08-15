//! egui settings dialog launched as a subprocess.

use crate::about;
use crate::cleanup::{catalog, PromptSource, PromptTarget};
use crate::credentials::{store_verified, CredentialStore, WindowsCredentialStore};
use crate::diagnostics;
use crate::enhance::{self, CopilotReadiness};
use crate::model_manager::{ModelCacheState, ModelInfo, ModelManager};
use crate::paths::settings_path;
use crate::settings::{Enhance, EnhanceMode, Settings, SettingsRevision};
use crate::single_instance::SettingsTransaction;
use anyhow::{anyhow, Result};
use eframe::egui;
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, TryRecvError};

pub fn run_dialog(show_about: bool) -> Result<()> {
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
                prompt_editor: PromptEditorState::default(),
                show_about,
                about_error: None,
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
    style.interaction.tooltip_delay = 0.0;
    style.interaction.show_tooltips_only_when_still = false;
    style.spacing.tooltip_width = 360.0;
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
    prompt_editor: PromptEditorState,
    show_about: bool,
    about_error: Option<String>,
}

/// Ephemeral editor state. Draft text is never written until the outer
/// Settings Save button commits the whole settings document.
#[derive(Default)]
struct PromptEditorState {
    draft: Option<PromptDraft>,
    pending_target_switch: bool,
    reset_target: Option<PromptTarget>,
}

struct PromptDraft {
    target: PromptTarget,
    text: String,
    original_target_fields: PromptTargetFields,
}

struct PromptTargetFields {
    provider: String,
    base_url: String,
    model: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PromptTargetSwitchDecision {
    Save,
    Discard,
    Cancel,
}

impl PromptEditorState {
    fn begin_edit(&mut self, target: PromptTarget, prompt: String, enhance: &Enhance) {
        self.draft = Some(PromptDraft {
            target,
            text: prompt,
            original_target_fields: PromptTargetFields {
                provider: enhance.provider.clone(),
                base_url: enhance.base_url.clone(),
                model: enhance.model.clone(),
            },
        });
        self.pending_target_switch = false;
    }

    fn cancel_edit(&mut self) {
        self.draft = None;
        self.pending_target_switch = false;
    }

    fn request_target_switch(&mut self, target: &PromptTarget) -> bool {
        let needs_decision = self
            .draft
            .as_ref()
            .is_some_and(|draft| draft.target != *target);
        self.pending_target_switch = needs_decision;
        needs_decision
    }

    fn resolve_target_switch(
        &mut self,
        settings: &mut Settings,
        decision: PromptTargetSwitchDecision,
    ) -> bool {
        let Some(draft) = self.draft.take() else {
            self.pending_target_switch = false;
            return false;
        };
        let staged = match decision {
            PromptTargetSwitchDecision::Save => {
                settings.prompt_overrides.set(draft.target, draft.text);
                true
            }
            PromptTargetSwitchDecision::Discard => false,
            PromptTargetSwitchDecision::Cancel => {
                restore_prompt_target_fields(settings, &draft.original_target_fields);
                self.draft = Some(draft);
                false
            }
        };
        self.pending_target_switch = false;
        staged
    }

    fn has_open_draft(&self) -> bool {
        self.draft.is_some()
    }
}

fn restore_prompt_target_fields(settings: &mut Settings, fields: &PromptTargetFields) {
    settings.enhance.provider = fields.provider.clone();
    settings.enhance.base_url = fields.base_url.clone();
    settings.enhance.model = fields.model.clone();
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
        if self.prompt_editor.has_open_draft() {
            return Some("Save or cancel the prompt editor draft before saving Settings.".into());
        }
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
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(format!(
                "Hold {} to record; release any shortcut key to transcribe.",
                format_shortcut(&self.settings, false)
                ))
                .strong(),
            );
            match self.settings.enhance.mode {
                EnhanceMode::Never => {}
                EnhanceMode::WithShift => info_button(
                    ui,
                    "Press or release additional Shift at any time while recording to turn enhancement on or off.",
                ),
                EnhanceMode::Always => {
                    let shift_is_base_modifier = self
                        .settings
                        .hotkey_modifiers
                        .iter()
                        .any(|modifier| modifier == "shift");
                    let behavior = if shift_is_base_modifier {
                        "Every transcript is enhanced. Shift is part of the recording shortcut, so the additional-Shift bypass is unavailable."
                    } else {
                        "Press or release additional Shift at any time while recording to bypass or restore enhancement."
                    };
                    info_button(ui, behavior);
                }
            }
        });
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
                ui.horizontal_wrapped(|ui| {
                    ui.add_enabled_ui(supported, |ui| {
                        if ui
                            .radio(self.settings.engine == descriptor.id, descriptor.title)
                            .clicked()
                        {
                            self.settings.engine = descriptor.id.to_string();
                            self.dirty = true;
                        }
                    });
                    info_button(ui, descriptor.description);
                });
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
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(
                &mut self.settings.auto_paste,
                true,
                "Paste into the active app",
            );
            info_button(
                ui,
                "Uses the clipboard temporarily, pastes at the cursor, then restores unchanged clipboard text.",
            );
        });
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(
                &mut self.settings.auto_paste,
                false,
                "Copy to the clipboard",
            );
            info_button(
                ui,
                "Keeps the completed transcript on the clipboard for manual use.",
            );
        });
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
        ui.horizontal_wrapped(|ui| {
            ui.label("When should OpenWritr improve punctuation and wording?");
            info_button(
                ui,
                "Enhancement sends recognized transcript text, never recorded audio, to the selected provider.",
            );
        });
        let before_mode = self.settings.enhance.mode;
        let shift_is_base_modifier = self
            .settings
            .hotkey_modifiers
            .iter()
            .any(|modifier| modifier == "shift");
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.settings.enhance.mode, EnhanceMode::Never, "Never");
            info_button(ui, "Transcript text never leaves this PC for cleanup.");
            ui.radio_value(
                &mut self.settings.enhance.mode,
                EnhanceMode::WithShift,
                "When Shift is additionally held",
            );
            info_button(
                ui,
                "Press or release additional Shift during recording to turn enhancement on or off.",
            );
            ui.radio_value(
                &mut self.settings.enhance.mode,
                EnhanceMode::Always,
                "Always",
            );
            info_button(
                ui,
                if shift_is_base_modifier {
                    "Every transcript is enhanced. Bypass is unavailable because Shift is part of the recording shortcut."
                } else {
                    "Every transcript is enhanced unless additional Shift is held. Shift can be pressed or released during recording."
                },
            );
        });
        if before_mode != self.settings.enhance.mode {
            if self.settings.enhance.mode == EnhanceMode::Never {
                self.api_key_input.clear();
            }
            self.dirty = true;
        }
        if let Some(warning) = self.settings.prompt_override_warning() {
            ui.add_space(6.0);
            message_frame(ui, &warning, warning_color());
        }

        if self.migration_blocked {
            ui.add_space(8.0);
            self.render_legacy_credential_resolution(ui);
        }

        if self.settings.enhance.mode == EnhanceMode::Never {
            if !self.migration_blocked {
                self.render_disabled_credential_summary(ui);
            }
            self.render_prompt_editor(ui);
            return;
        }

        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            ui.label("Provider");
            info_button(
                ui,
                "Only recognized transcript text is sent to the provider. Recorded audio stays on this PC.",
            );
        });
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

        ui.add_space(8.0);

        match self.settings.enhance.provider.as_str() {
            "github_copilot" => self.render_github_copilot(ui),
            "openai_compatible" => self.render_openai_compatible(ui),
            _ => inline_error(ui, "Select a supported enhancement provider."),
        }
        self.render_prompt_editor(ui);

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
        let models = catalog::all();
        let mut selection = copilot_model_picker_value(&self.settings.enhance.model);
        let selected_label = models
            .iter()
            .find(|model| model.id == selection)
            .map(|model| model.display_name)
            .unwrap_or("Custom model ID");
        let before_selection = selection.clone();
        egui::ComboBox::from_id_salt("copilot_model")
            .selected_text(selected_label)
            .show_ui(ui, |ui| {
                for model in models {
                    ui.selectable_value(&mut selection, model.id.into(), model.display_name);
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
            info_button(
                ui,
                "Model availability and premium usage depend on your current Copilot plan.",
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

    fn render_prompt_editor(&mut self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Cleanup prompt").strong());
            info_button(
                ui,
                "Prompts are tuned per provider, endpoint, and exact model or deployment ID.",
            );
        });

        if self.settings.prompt_overrides.preserves_unsupported_raw() {
            status_label(
                ui,
                "This prompt document is preserved for a newer format and cannot be edited here.",
                warning_color(),
            );
            return;
        }

        let target = match self.settings.prompt_target() {
            Ok(target) => target,
            Err(error) => {
                inline_error(ui, format!("Prompt target is incomplete: {error}"));
                return;
            }
        };
        self.prompt_editor.request_target_switch(&target);

        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(format!("Target: {}", prompt_target_label(&target)))
                .small()
                .color(secondary_text()),
        );

        let resolved = self.settings.resolve_prompt(&target);
        let source = prompt_source_label(resolved.source);
        ui.label(
            egui::RichText::new(format!("Active prompt: {source}"))
                .small()
                .color(secondary_text()),
        );

        if self.prompt_editor.draft.is_some() {
            let pending_target_switch = self.prompt_editor.pending_target_switch;
            {
                let draft = self
                    .prompt_editor
                    .draft
                    .as_mut()
                    .expect("draft exists while editing");
                ui.add(
                    egui::TextEdit::multiline(&mut draft.text)
                        .desired_rows(8)
                        .hint_text("Write cleanup instructions"),
                );
            }
            let mut save_draft = false;
            let mut cancel_draft = false;
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(
                        !pending_target_switch,
                        egui::Button::new("Save prompt draft"),
                    )
                    .clicked()
                {
                    save_draft = true;
                }
                if ui
                    .add_enabled(
                        !pending_target_switch,
                        egui::Button::new("Cancel prompt draft"),
                    )
                    .clicked()
                {
                    cancel_draft = true;
                }
            });
            if save_draft {
                let draft = self
                    .prompt_editor
                    .draft
                    .take()
                    .expect("draft exists while saving");
                self.settings.prompt_overrides.set(draft.target, draft.text);
                self.prompt_editor.pending_target_switch = false;
                self.dirty = true;
            } else if cancel_draft {
                self.prompt_editor.cancel_edit();
            }
            if pending_target_switch {
                status_label(
                    ui,
                    "The target changed while this draft is open. Choose Save, Discard, or Cancel below.",
                    warning_color(),
                );
            }
        } else {
            let mut display = resolved.system;
            ui.add(
                egui::TextEdit::multiline(&mut display)
                    .desired_rows(8)
                    .interactive(false),
            );
            ui.horizontal(|ui| {
                if ui.button("Edit prompt").clicked() {
                    self.prompt_editor.begin_edit(
                        target.clone(),
                        display.clone(),
                        &self.settings.enhance,
                    );
                }
                if self.settings.prompt_overrides.get(&target).is_some()
                    && ui.small_button("Reset this target").clicked()
                {
                    self.prompt_editor.reset_target = Some(target.clone());
                }
            });
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
            info_button(
                ui,
                &format!(
                    "The current limit is {}. Reaching it finishes and transcribes the recording automatically.",
                    format_duration(self.settings.max_record_seconds)
                ),
            );
        });
    }

    fn render_about(&mut self, ui: &mut egui::Ui) {
        ui.heading("About OpenWritr");
        ui.label("Local push-to-talk voice-to-text for Windows.");
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(format!(
                "Version {} · {} · Publisher: {}",
                env!("CARGO_PKG_VERSION"),
                architecture_label(),
                about::PUBLISHER
            ))
            .small()
            .color(secondary_text()),
        );
        ui.label(
            egui::RichText::new(about::COPYRIGHT)
                .small()
                .color(secondary_text()),
        );

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Project and support").strong());
        ui.label(
            egui::RichText::new(
                "Official project links open in your default browser. To report a problem, export diagnostics from the tray menu and attach the resulting privacy-safe bundle.",
            )
            .small()
            .color(secondary_text()),
        );
        ui.add_space(4.0);

        let mut selected_link = None;
        ui.horizontal_wrapped(|ui| {
            for link in about::OFFICIAL_LINKS {
                if ui.link(link.label).clicked() {
                    selected_link = Some((link.label, link.url));
                }
            }
        });
        if let Some((label, url)) = selected_link {
            self.set_about_result(label, about::open_url(url));
        }

        ui.add_space(6.0);
        if ui.small_button("Open logs").clicked() {
            self.set_about_result("Open logs", diagnostics::open_logs_dir());
        }
        ui.label(
            egui::RichText::new(
                "Tray menu → Export diagnostics creates a bounded bundle without audio, transcript text, clipboard contents, or API keys.",
            )
            .small()
            .color(secondary_text()),
        );

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Legal").strong());
        ui.label(about::LICENSE_SUMMARY);
        ui.horizontal_wrapped(|ui| {
            if ui.small_button("Open MIT License").clicked() {
                self.set_about_result("Open MIT License", about::open_license());
            }
            if ui.small_button("Open privacy policy").clicked() {
                self.set_about_result("Open privacy policy", about::open_privacy_policy());
            }
            if ui.small_button("Open third-party licenses").clicked() {
                self.set_about_result(
                    "Open third-party licenses",
                    about::open_third_party_licenses(),
                );
            }
            if ui.link("MIT License online").clicked() {
                self.set_about_result("MIT License online", about::open_url(about::LICENSE_URL));
            }
        });

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Models, runtimes, and credits").strong());
        let mut selected_credit = None;
        for credit in about::CREDITS {
            ui.horizontal_wrapped(|ui| {
                if ui.link(credit.name).clicked() {
                    selected_credit = Some((credit.name, credit.url));
                }
                ui.label(
                    egui::RichText::new(credit.attribution)
                        .small()
                        .color(secondary_text()),
                );
            });
        }
        if let Some((label, url)) = selected_credit {
            self.set_about_result(label, about::open_url(url));
        }

        ui.add_space(10.0);
        ui.label(
            egui::RichText::new(about::DISCLAIMER)
                .small()
                .italics()
                .color(secondary_text()),
        );

        if let Some(error) = &self.about_error {
            ui.add_space(8.0);
            inline_error(ui, error);
        }
    }

    fn set_about_result(&mut self, action: &str, result: Result<()>) {
        self.about_error = result
            .err()
            .map(|error| format!("{action} failed: {error}"));
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
                if ui.small_button("About, credits & support").clicked() {
                    self.about_error = None;
                    self.show_about = true;
                }
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

        if self.show_about {
            let mut close_about = false;
            egui::Modal::new(egui::Id::new("about_openwritr")).show(ctx, |ui| {
                ui.set_min_width(520.0);
                egui::ScrollArea::vertical()
                    .max_height(560.0)
                    .show(ui, |ui| self.render_about(ui));
                ui.add_space(8.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        close_about = true;
                    }
                });
            });
            if close_about {
                self.show_about = false;
            }
        }

        if self.prompt_editor.pending_target_switch {
            let mut decision = None;
            let response =
                egui::Modal::new(egui::Id::new("prompt_target_switch")).show(ctx, |ui| {
                    ui.heading("Save prompt draft before changing target?");
                    ui.label(
                        "The provider, endpoint, or model changed. This draft belongs only to its original target.",
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Save draft").clicked() {
                            decision = Some(PromptTargetSwitchDecision::Save);
                        }
                        if ui.button("Discard draft").clicked() {
                            decision = Some(PromptTargetSwitchDecision::Discard);
                        }
                        if ui.button("Cancel target change").clicked() {
                            decision = Some(PromptTargetSwitchDecision::Cancel);
                        }
                    });
                });
            if let Some(decision) = decision {
                self.dirty |= self
                    .prompt_editor
                    .resolve_target_switch(&mut self.settings, decision);
            } else if response.should_close() {
                self.dirty |= self
                    .prompt_editor
                    .resolve_target_switch(&mut self.settings, PromptTargetSwitchDecision::Cancel);
            }
        }

        if let Some(target) = self.prompt_editor.reset_target.clone() {
            let mut reset = false;
            let mut keep = false;
            let response = egui::Modal::new(egui::Id::new("reset_prompt_override")).show(
                ctx,
                |ui| {
                    ui.heading("Reset custom prompt?");
                    ui.label(
                        "This removes the custom prompt only for the current provider, endpoint, and model target.",
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Reset this target").clicked() {
                            reset = true;
                        }
                        if ui.button("Keep custom prompt").clicked() {
                            keep = true;
                        }
                    });
                },
            );
            if reset {
                self.dirty |= self.settings.prompt_overrides.remove(&target);
                self.prompt_editor.reset_target = None;
            } else if keep || response.should_close() {
                self.prompt_editor.reset_target = None;
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
    ui.horizontal_wrapped(|ui| {
        ui.label(egui::RichText::new(title).size(16.0).strong());
        info_button(ui, description);
    });
    ui.add_space(4.0);
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_min_width(ui.available_width());
        body(ui);
    });
    ui.add_space(14.0);
}

fn info_button(ui: &mut egui::Ui, help: &str) {
    let _ = ui
        .small_button("i")
        .on_hover_cursor(egui::CursorIcon::Help)
        .on_hover_text_at_pointer(help);
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

fn prompt_target_label(target: &PromptTarget) -> String {
    match target.endpoint() {
        Some(endpoint) => format!(
            "OpenAI-compatible · {} · {}",
            endpoint.base_url(),
            target.model_id()
        ),
        None => format!("GitHub Copilot · {}", target.model_id()),
    }
}

fn prompt_source_label(source: PromptSource) -> &'static str {
    match source {
        PromptSource::CustomOverride => "custom override",
        PromptSource::ModelDefault => "bundled model default",
        PromptSource::ProviderDefault => "bundled provider default",
        PromptSource::GlobalDefault => "bundled global default",
    }
}

fn copilot_model_picker_value(model_id: &str) -> String {
    if catalog::lookup(model_id).is_some() {
        model_id.to_string()
    } else {
        "custom".to_string()
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
    fn prompt_editor_switch_save_stages_only_the_original_target() {
        let original = PromptTarget::github_copilot("custom-a").unwrap();
        let switched = PromptTarget::github_copilot("custom-b").unwrap();
        let mut editor = PromptEditorState::default();
        let mut settings = Settings::default();
        editor.begin_edit(original.clone(), "draft prompt".into(), &settings.enhance);

        assert!(editor.request_target_switch(&switched));
        assert!(editor.resolve_target_switch(&mut settings, PromptTargetSwitchDecision::Save));

        assert_eq!(
            settings.prompt_overrides.get(&original),
            Some("draft prompt")
        );
        assert!(settings.prompt_overrides.get(&switched).is_none());
        assert!(!editor.has_open_draft());
    }

    #[test]
    fn prompt_editor_switch_cancel_restores_target_and_keeps_draft() {
        let original = PromptTarget::openai_compatible(
            crate::cleanup::EndpointScope::parse("https://api.example.com/v1").unwrap(),
            "deployment-a",
        )
        .unwrap();
        let switched = PromptTarget::github_copilot("custom-b").unwrap();
        let mut editor = PromptEditorState::default();
        let mut settings = Settings::default();
        settings.enhance.provider = "openai_compatible".into();
        settings.enhance.base_url = "HTTPS://API.EXAMPLE.COM:443/v1/".into();
        settings.enhance.model = "deployment-a".into();
        editor.begin_edit(original.clone(), "draft prompt".into(), &settings.enhance);
        settings.enhance.provider = "github_copilot".into();
        settings.enhance.model = "custom-b".into();

        assert!(editor.request_target_switch(&switched));
        assert!(!editor.resolve_target_switch(&mut settings, PromptTargetSwitchDecision::Cancel));

        assert_eq!(settings.prompt_target().unwrap(), original);
        assert_eq!(settings.enhance.base_url, "HTTPS://API.EXAMPLE.COM:443/v1/");
        assert!(editor.has_open_draft());
        assert!(!editor.pending_target_switch);
    }

    #[test]
    fn advisory_picker_keeps_custom_model_ids_as_custom() {
        assert_eq!(copilot_model_picker_value("gpt-5.6-luna"), "gpt-5.6-luna");
        assert_eq!(
            copilot_model_picker_value("my-private/deployment-v42"),
            "custom"
        );
        let mut settings = Settings::default();
        settings.enhance.model = "my-private/deployment-v42".into();
        assert_eq!(
            settings.enhance.model, "my-private/deployment-v42",
            "custom picker selection must not replace the configured ID"
        );
    }

    #[test]
    fn insecure_http_warning_excludes_loopback_endpoints() {
        assert!(!insecure_remote_http("http://localhost:11434/v1"));
        assert!(!insecure_remote_http("http://127.0.0.1:11434/v1"));
        assert!(insecure_remote_http("http://example.com/v1"));
        assert!(!insecure_remote_http("https://example.com/v1"));
    }
}
