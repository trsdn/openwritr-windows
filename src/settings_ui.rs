//! egui settings dialog launched as a subprocess.

use crate::credentials::{store_verified, CredentialStore, WindowsCredentialStore};
use crate::paths::settings_path;
use crate::settings::{Settings, SettingsRevision};
use crate::single_instance::SettingsTransaction;
use anyhow::{anyhow, Result};
use eframe::egui;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

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
    let settings = Arc::new(Mutex::new(settings));
    let whisper_hardware_status = crate::asr::whisper_hardware_status()
        .unwrap_or_else(|error| format!("unavailable: {error}"));
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([560.0, 720.0])
            .with_min_inner_size([460.0, 600.0])
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
                saved_at: None,
                api_key_input: String::new(),
                remove_api_key: false,
                migration_blocked,
                load_failed,
                replace_load_failed: false,
                settings_revision,
                error: initial_error,
                whisper_hardware_status,
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
    settings: Arc<Mutex<Settings>>,
    dirty: bool,
    saved_at: Option<SystemTime>,
    api_key_input: String,
    remove_api_key: bool,
    migration_blocked: bool,
    load_failed: bool,
    replace_load_failed: bool,
    settings_revision: Option<SettingsRevision>,
    error: Option<String>,
    whisper_hardware_status: String,
}

impl SettingsApp {
    fn persist(&mut self) -> Result<()> {
        let settings = self
            .settings
            .lock()
            .map_err(|_| anyhow!("settings lock is unavailable"))?
            .clone();
        settings.validate()?;
        if self.load_failed && !self.replace_load_failed {
            return Err(anyhow!(
                "Confirm replacement of the invalid or unreadable settings file before saving."
            ));
        }

        let replacement = self.api_key_input.as_str();
        if self.migration_blocked && replacement.is_empty() && !self.remove_api_key {
            return Err(anyhow!(
                "Enter the API key again to secure it, or explicitly select Remove saved API key. The plaintext key has not been removed."
            ));
        }
        let change = if self.remove_api_key {
            CredentialChange::Delete
        } else if !replacement.is_empty() {
            CredentialChange::Set(replacement)
        } else {
            CredentialChange::Keep
        };
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
        self.saved_at = Some(SystemTime::now());
        Ok(())
    }
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

#[derive(Clone, Copy)]
enum CredentialChange<'a> {
    Keep,
    Set(&'a str),
    Delete,
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

impl eframe::App for SettingsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut save_requested = false;
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(10.0);
            ui.heading("OpenWritr");
            ui.label(
                egui::RichText::new("Voice-to-text for Windows on ARM")
                    .color(egui::Color32::from_rgb(154, 163, 178)),
            );
            ui.add_space(16.0);

            let mut settings = self.settings.lock().unwrap();

            section(ui, "HOTKEY (HOLD TO RECORD)", |ui| {
                ui.label("Modifier keys (any combination)");
                ui.horizontal(|ui| {
                    for (name, label) in [
                        ("ctrl", "Ctrl"),
                        ("shift", "Shift"),
                        ("alt", "Alt"),
                        ("win", "Win"),
                    ] {
                        let mut enabled =
                            settings.hotkey_modifiers.iter().any(|modifier| modifier == name);
                        if ui.selectable_label(enabled, label).clicked() {
                            enabled = !enabled;
                            settings
                                .hotkey_modifiers
                                .retain(|modifier| modifier != name);
                            if enabled {
                                settings.hotkey_modifiers.push(name.to_string());
                            }
                            self.dirty = true;
                        }
                    }
                });
                ui.add_space(6.0);
                ui.label("Trigger key (hold + release)");
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
                    .selected_text(label_for_trigger(&settings.hotkey_trigger))
                    .show_ui(ui, |ui| {
                        for trigger in triggers {
                            if ui
                                .selectable_value(
                                    &mut settings.hotkey_trigger,
                                    trigger.to_string(),
                                    label_for_trigger(trigger),
                                )
                                .changed()
                            {
                                self.dirty = true;
                            }
                        }
                    });
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(
                        "Hold modifiers (+ trigger) to record; release any to stop. \
                         Hold Shift additionally to also run LLM cleanup.",
                    )
                    .small()
                    .color(egui::Color32::from_rgb(154, 163, 178)),
                );
            });

            section(ui, "TRANSCRIPTION ENGINE", |ui| {
                #[cfg(target_arch = "aarch64")]
                let (engines, whisper_unavailable) = {
                    let status = self.whisper_hardware_status.as_str();
                    let mut engines = vec![
                        ("parakeet_cpu", "Parakeet TDT v3 — CPU INT8"),
                        ("parakeet_npu", "Parakeet TDT v3 — Hexagon NPU"),
                    ];
                    let unavailable = if status.starts_with("supported:") {
                        engines.push(("whisper_npu", "Whisper Large v3 Turbo — NPU"));
                        None
                    } else {
                        Some(status.to_string())
                    };
                    (engines, unavailable)
                };
                #[cfg(not(target_arch = "aarch64"))]
                let (engines, whisper_unavailable) = (
                    vec![("parakeet_cpu", "Parakeet TDT v3 — CPU INT8")],
                    Some(self.whisper_hardware_status.clone()),
                );
                egui::ComboBox::from_id_salt("engine")
                    .selected_text(
                        engines
                            .iter()
                            .find(|(key, _)| *key == settings.engine)
                            .map(|(_, label)| *label)
                            .unwrap_or(settings.engine.as_str()),
                    )
                    .show_ui(ui, |ui| {
                        for &(key, label) in &engines {
                            if ui
                                .selectable_value(&mut settings.engine, key.to_string(), label)
                                .changed()
                            {
                                self.dirty = true;
                            }
                        }
                    });
                if let Some(status) = whisper_unavailable {
                    ui.label(
                        egui::RichText::new(format!("Whisper NPU {status}"))
                            .small()
                            .color(egui::Color32::from_rgb(154, 163, 178)),
                    );
                }
            });

            section(ui, "BEHAVIOUR", |ui| {
                self.dirty |= ui
                    .checkbox(&mut settings.auto_paste, "Auto-paste at cursor")
                    .changed();
                self.dirty |= ui
                    .checkbox(&mut settings.overlay, "Show overlay while recording")
                    .changed();
                self.dirty |= ui
                    .checkbox(&mut settings.sounds, "Play start/stop sounds")
                    .changed();
            });

            section(ui, "ENHANCE (PUNCTUATION + CLEANUP)", |ui| {
                ui.label(
                    egui::RichText::new(
                        "Hold the hotkey with Shift also pressed to trigger an LLM cleanup pass after transcription.",
                    )
                    .small()
                    .color(egui::Color32::from_rgb(154, 163, 178)),
                );
                ui.add_space(6.0);
                let providers = [
                    ("off", "Off"),
                    ("github_copilot", "GitHub Copilot (uses gh auth token)"),
                    ("openai_compatible", "OpenAI-compatible API"),
                ];
                egui::ComboBox::from_label("Provider")
                    .selected_text(
                        providers
                            .iter()
                            .find(|(key, _)| *key == settings.enhance.provider)
                            .map(|(_, label)| *label)
                            .unwrap_or(""),
                    )
                    .show_ui(ui, |ui| {
                        for (key, label) in providers {
                            if ui
                                .selectable_value(
                                    &mut settings.enhance.provider,
                                    key.to_string(),
                                    label,
                                )
                                .changed()
                            {
                                self.dirty = true;
                            }
                        }
                    });

                if settings.enhance.provider == "github_copilot" {
                    ui.add_space(4.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        ui.label(
                            egui::RichText::new(
                                "Uses your existing GitHub Copilot subscription via the gh CLI token.",
                            )
                            .small()
                            .color(egui::Color32::from_rgb(154, 163, 178)),
                        );
                        ui.hyperlink_to(
                            egui::RichText::new("Get GitHub Copilot").small(),
                            "https://github.com/features/copilot/plans",
                        );
                    });
                }

                ui.add_space(4.0);
                ui.label("Base URL (OpenAI-compatible only)");
                if ui
                    .text_edit_singleline(&mut settings.enhance.base_url)
                    .changed()
                {
                    self.dirty = true;
                }
                ui.label("API key (stored in Windows Credential Manager)");
                let key_edit = egui::TextEdit::singleline(&mut self.api_key_input)
                    .password(true)
                    .hint_text("Leave blank to keep the saved credential");
                if ui.add(key_edit).changed() {
                    self.remove_api_key = false;
                    self.dirty = true;
                }
                if ui
                    .checkbox(
                        &mut self.remove_api_key,
                        "Remove saved API key when settings are saved",
                    )
                    .changed()
                {
                    if self.remove_api_key {
                        self.api_key_input.clear();
                    }
                    self.dirty = true;
                }

                ui.label("Model");
                let models = [
                    ("gpt-4.1", "GPT-4.1  ✓ included", true),
                    ("gpt-5-mini", "GPT-5 Mini  ✓ included", true),
                    (
                        "claude-haiku-4.5",
                        "Claude Haiku 4.5  (premium request)",
                        false,
                    ),
                ];
                let current_label = models
                    .iter()
                    .find(|(key, _, _)| *key == settings.enhance.model)
                    .map(|(_, label, _)| *label)
                    .unwrap_or(settings.enhance.model.as_str());
                egui::ComboBox::from_id_salt("enhance_model")
                    .selected_text(current_label)
                    .show_ui(ui, |ui| {
                        for (key, label, included) in models {
                            let text = if included {
                                egui::RichText::new(label)
                                    .color(egui::Color32::from_rgb(34, 197, 94))
                            } else {
                                egui::RichText::new(label)
                            };
                            if ui
                                .selectable_value(
                                    &mut settings.enhance.model,
                                    key.to_string(),
                                    text,
                                )
                                .changed()
                            {
                                self.dirty = true;
                            }
                        }
                    });
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new(
                        "✓ included = no premium-request cost on GitHub Copilot. \
                         Claude Haiku is higher quality but consumes premium requests.",
                    )
                    .small()
                    .color(egui::Color32::from_rgb(154, 163, 178)),
                );
                ui.add_space(2.0);
                ui.label(
                    egui::RichText::new("Or type a custom model name:")
                        .color(egui::Color32::from_rgb(154, 163, 178))
                        .size(11.0),
                );
                if ui
                    .text_edit_singleline(&mut settings.enhance.model)
                    .changed()
                {
                    self.dirty = true;
                }
            });

            ui.add_space(12.0);
            if let Some(error) = &self.error {
                ui.colored_label(egui::Color32::from_rgb(248, 113, 113), error);
                ui.add_space(6.0);
            }
            if self.load_failed
                && ui
                    .checkbox(
                        &mut self.replace_load_failed,
                        "Replace the invalid/unreadable settings file when saving",
                    )
                    .changed()
            {
                self.dirty = true;
            }
            ui.horizontal(|ui| {
                if let Some(saved_at) = self.saved_at {
                    if saved_at
                        .elapsed()
                        .map(|duration| duration.as_secs() < 2)
                        .unwrap_or(false)
                    {
                        ui.label(
                            egui::RichText::new("Saved")
                                .color(egui::Color32::from_rgb(34, 197, 94)),
                        );
                    }
                }
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        if ui.button("Save & Close").clicked() {
                            save_requested = true;
                        }
                        if ui.button("Close").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    },
                );
            });

            ui.add_space(10.0);
            ui.separator();
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.hyperlink_to(
                    egui::RichText::new("⭐ OpenWritr on GitHub").size(12.0),
                    "https://github.com/trsdn/openwritr-windows",
                );
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        ui.label(
                            egui::RichText::new(concat!("v", env!("CARGO_PKG_VERSION")))
                                .small()
                                .color(egui::Color32::from_rgb(120, 128, 143)),
                        );
                    },
                );
            });
        });

        if save_requested {
            match self.persist() {
                Ok(()) => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
                Err(error) => self.error = Some(error.to_string()),
            }
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(500));
    }
}

fn section(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(8.0);
    ui.label(
        egui::RichText::new(title)
            .small()
            .color(egui::Color32::from_rgb(154, 163, 178))
            .strong(),
    );
    egui::Frame::group(ui.style()).show(ui, |ui| body(ui));
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
}
