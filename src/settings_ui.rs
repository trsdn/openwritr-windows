//! egui settings dialog launched as a subprocess.
//!
//! Spawning a separate process keeps the main tray app's winit loop
//! single-window (winit gets cranky about multiple windows on Windows when
//! one is invisible / borderless / focus-stealing). The subprocess reads
//! and writes the same settings.json the tray app uses; we tail its mtime
//! to know when to hot-swap engines.

use crate::paths::settings_path;
use crate::settings::{Enhance, Settings};
use anyhow::Result;
use eframe::egui;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

pub fn run_dialog() -> Result<()> {
    let settings = Arc::new(Mutex::new(Settings::load()));
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([560.0, 720.0])
            .with_min_inner_size([460.0, 600.0])
            .with_resizable(true)
            .with_title("OpenWritr Settings"),
        ..Default::default()
    };
    let s = settings.clone();
    eframe::run_native(
        "OpenWritr Settings",
        opts,
        Box::new(|cc| {
            apply_dark_theme(&cc.egui_ctx);
            Ok(Box::new(SettingsApp { settings: s, dirty: false, saved_at: None }))
        }),
    )
    .map_err(|e| anyhow::anyhow!("egui run failed: {e}"))
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
    style.visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, Color32::from_rgb(232, 236, 243));
    style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, Color32::from_rgb(232, 236, 243));
    style.visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    style.visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::from_rgb(232, 236, 243));
    style.visuals.window_stroke = Stroke::new(1.0, Color32::from_rgb(54, 61, 76));
    style.visuals.window_rounding = 8.0.into();
    style.text_styles.insert(egui::TextStyle::Heading, FontId::new(22.0, FontFamily::Proportional));
    style.text_styles.insert(egui::TextStyle::Body, FontId::new(13.0, FontFamily::Proportional));
    ctx.set_style(style);
}

struct SettingsApp {
    settings: Arc<Mutex<Settings>>,
    dirty: bool,
    saved_at: Option<SystemTime>,
}

impl eframe::App for SettingsApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(10.0);
            ui.heading("OpenWritr");
            ui.label(egui::RichText::new("Voice-to-text for Windows on ARM")
                .color(egui::Color32::from_rgb(154, 163, 178)));
            ui.add_space(16.0);

            let mut s = self.settings.lock().unwrap();

            // Hotkey
            section(ui, "HOTKEY (HOLD TO RECORD)", |ui| {
                ui.label("Modifier keys (any combination)");
                ui.horizontal(|ui| {
                    for (name, lbl) in [("ctrl","Ctrl"),("shift","Shift"),("alt","Alt"),("win","Win")] {
                        let mut on = s.hotkey_modifiers.iter().any(|m| m == name);
                        if ui.selectable_label(on, lbl).clicked() {
                            on = !on;
                            s.hotkey_modifiers.retain(|m| m != name);
                            if on { s.hotkey_modifiers.push(name.to_string()); }
                            self.dirty = true;
                        }
                    }
                });
                ui.add_space(6.0);
                ui.label("Trigger key (hold + release)");
                let triggers = [
                    "none",
                    "space","tab","caps_lock","scroll_lock","pause","insert","right_ctrl",
                    "f13","f14","f15","f16","f17","f18","f19","f20",
                ];
                egui::ComboBox::from_id_source("trigger")
                    .selected_text(label_for_trigger(&s.hotkey_trigger))
                    .show_ui(ui, |ui| {
                        for t in triggers {
                            if ui.selectable_value(&mut s.hotkey_trigger, t.to_string(), label_for_trigger(t)).changed() {
                                self.dirty = true;
                            }
                        }
                    });
                ui.add_space(4.0);
                ui.label(egui::RichText::new(
                    "Hold modifiers (+ trigger) to record; release any to stop. \
                     Hold Shift additionally to also run LLM cleanup."
                ).small().color(egui::Color32::from_rgb(154, 163, 178)));
            });

            // Engine
            section(ui, "TRANSCRIPTION ENGINE", |ui| {
                let engines = [
                    ("parakeet_cpu", "Parakeet TDT v3 — CPU INT8 (default)"),
                    ("parakeet_npu", "Parakeet TDT v3 — NPU INT8 (CPU fallback in native)"),
                    ("whisper_npu",  "Whisper Large v3 Turbo — NPU (CPU fallback in native)"),
                ];
                egui::ComboBox::from_id_source("engine")
                    .selected_text(engines.iter().find(|(k, _)| *k == s.engine).map(|(_, v)| *v).unwrap_or(s.engine.as_str()))
                    .show_ui(ui, |ui| {
                        for (k, lbl) in engines {
                            if ui.selectable_value(&mut s.engine, k.to_string(), lbl).changed() {
                                self.dirty = true;
                            }
                        }
                    });
            });

            // Behaviour
            section(ui, "BEHAVIOUR", |ui| {
                self.dirty |= ui.checkbox(&mut s.auto_paste, "Auto-paste at cursor").changed();
                self.dirty |= ui.checkbox(&mut s.overlay, "Show overlay while recording").changed();
                self.dirty |= ui.checkbox(&mut s.sounds, "Play start/stop sounds").changed();
            });

            // Enhance
            section(ui, "ENHANCE (PUNCTUATION + CLEANUP)", |ui| {
                ui.label(egui::RichText::new(
                    "Hold the hotkey with Shift also pressed to trigger an LLM cleanup pass after transcription."
                ).small().color(egui::Color32::from_rgb(154, 163, 178)));
                ui.add_space(6.0);
                let providers = [
                    ("off", "Off"),
                    ("github_copilot", "GitHub Copilot (uses gh auth token)"),
                    ("openai_compatible", "OpenAI-compatible API"),
                ];
                egui::ComboBox::from_label("Provider")
                    .selected_text(providers.iter().find(|(k, _)| *k == s.enhance.provider).map(|(_, v)| *v).unwrap_or(""))
                    .show_ui(ui, |ui| {
                        for (k, lbl) in providers {
                            if ui.selectable_value(&mut s.enhance.provider, k.to_string(), lbl).changed() {
                                self.dirty = true;
                            }
                        }
                    });

                if s.enhance.provider == "github_copilot" {
                    ui.add_space(4.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        ui.label(egui::RichText::new(
                            "Uses your existing GitHub Copilot subscription via the gh CLI token."
                        ).small().color(egui::Color32::from_rgb(154, 163, 178)));
                        ui.hyperlink_to(
                            egui::RichText::new("Get GitHub Copilot").small(),
                            "https://github.com/features/copilot/plans",
                        );
                    });
                }

                ui.add_space(4.0);
                ui.label("Base URL (OpenAI-compatible only)");
                if ui.text_edit_singleline(&mut s.enhance.base_url).changed() { self.dirty = true; }
                ui.label("API key (OpenAI-compatible only)");
                if ui.add(egui::TextEdit::singleline(&mut s.enhance.api_key).password(true)).changed() { self.dirty = true; }
                ui.label("Model");
                // "included" = no premium-request cost on GitHub Copilot plans.
                // GPT-4.1 and GPT-5 mini are GitHub's included base models;
                // Claude Haiku consumes premium requests. Included ones first
                // so the zero-cost default is the obvious pick.
                let models = [
                    ("gpt-4.1", "GPT-4.1  ✓ included", true),
                    ("gpt-5-mini", "GPT-5 Mini  ✓ included", true),
                    ("claude-haiku-4.5", "Claude Haiku 4.5  (premium request)", false),
                ];
                let current_label = models
                    .iter()
                    .find(|(k, _, _)| *k == s.enhance.model)
                    .map(|(_, v, _)| *v)
                    .unwrap_or(s.enhance.model.as_str());
                egui::ComboBox::from_id_source("enhance_model")
                    .selected_text(current_label)
                    .show_ui(ui, |ui| {
                        for (k, lbl, included) in models {
                            let text = if included {
                                egui::RichText::new(lbl).color(egui::Color32::from_rgb(34, 197, 94))
                            } else {
                                egui::RichText::new(lbl)
                            };
                            if ui.selectable_value(&mut s.enhance.model, k.to_string(), text).changed() {
                                self.dirty = true;
                            }
                        }
                    });
                ui.add_space(2.0);
                ui.label(egui::RichText::new(
                    "\u{2713} included = no premium-request cost on GitHub Copilot. \
                     Claude Haiku is higher quality but consumes premium requests."
                ).small().color(egui::Color32::from_rgb(154, 163, 178)));
                ui.add_space(2.0);
                ui.label(egui::RichText::new("Or type a custom model name:")
                    .color(egui::Color32::from_rgb(154, 163, 178))
                    .size(11.0));
                if ui.text_edit_singleline(&mut s.enhance.model).changed() { self.dirty = true; }
            });

            ui.add_space(14.0);
            ui.horizontal(|ui| {
                if let Some(t) = self.saved_at {
                    if t.elapsed().map(|d| d.as_secs() < 2).unwrap_or(false) {
                        ui.label(egui::RichText::new("Saved").color(egui::Color32::from_rgb(34, 197, 94)));
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Save & Close").clicked() {
                        let _ = s.save_to(&settings_path());
                        self.saved_at = Some(SystemTime::now());
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    if ui.button("Close").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });

            // Footer: project link + version.
            ui.add_space(10.0);
            ui.separator();
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.hyperlink_to(
                    egui::RichText::new("\u{2B50} OpenWritr on GitHub").size(12.0),
                    "https://github.com/trsdn/openwritr-windows",
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new(concat!("v", env!("CARGO_PKG_VERSION")))
                        .small()
                        .color(egui::Color32::from_rgb(120, 128, 143)));
                });
            });
        });

        // Light refresh so the timer-driven "Saved" badge fades.
        ctx.request_repaint_after(std::time::Duration::from_millis(500));
    }
}

fn section(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui)) {
    ui.add_space(8.0);
    ui.label(egui::RichText::new(title)
        .small()
        .color(egui::Color32::from_rgb(154, 163, 178))
        .strong());
    egui::Frame::group(ui.style()).show(ui, |ui| body(ui));
}

fn label_for_trigger(t: &str) -> String {
    match t {
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

// Settings::save_to lives in settings.rs — added in this commit.
#[allow(dead_code)]
fn _x() -> Enhance { Enhance::default() }
