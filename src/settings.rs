use crate::paths::settings_path;
use serde::{Deserialize, Serialize};
use std::fs;
use tracing::warn;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Enhance {
    pub provider: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl Default for Enhance {
    fn default() -> Self {
        Self {
            provider: "off".into(),
            base_url: "https://api.openai.com/v1".into(),
            api_key: String::new(),
            model: "gpt-4o-mini".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub hotkey_modifiers: Vec<String>,
    pub hotkey_trigger: String,
    pub engine: String,
    pub auto_paste: bool,
    pub overlay: bool,
    pub sounds: bool,
    pub min_record_seconds: f32,
    pub max_record_seconds: f32,
    #[serde(default)]
    pub enhance: Enhance,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey_modifiers: vec!["ctrl".into(), "win".into()],
            hotkey_trigger: "space".into(),
            engine: "parakeet_cpu".into(),
            auto_paste: true,
            overlay: true,
            sounds: true,
            min_record_seconds: 0.25,
            max_record_seconds: 60.0,
            enhance: Enhance::default(),
        }
    }
}

impl Settings {
    pub fn load() -> Self {
        match fs::read_to_string(settings_path()) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
                warn!(error = %e, "settings parse failed, using defaults");
                Settings::default()
            }),
            Err(_) => Settings::default(),
        }
    }

    pub fn save_to(&self, path: &std::path::Path) {
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).ok();
        }
        if let Ok(s) = serde_json::to_string_pretty(self) {
            fs::write(path, s).ok();
        }
    }

    #[allow(dead_code)]
    pub fn save(&self) {
        self.save_to(&settings_path());
    }
}
