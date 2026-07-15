use crate::settings::Settings;
use anyhow::Result;
use image::{ImageBuffer, Rgba};
use tray_icon::menu::{Menu, MenuId, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

pub struct Tray {
    pub _icon: TrayIcon,
    pub menu_quit_id: MenuId,
    pub menu_settings_id: MenuId,
    pub menu_open_logs_id: MenuId,
    pub menu_export_diagnostics_id: MenuId,
    pub menu_cancel_model_id: MenuId,
    pub menu_retry_engine_id: MenuId,
}

#[derive(Clone, Copy, Debug)]
pub enum IconColor {
    Idle,
    Recording,
    Transcribing,
    Error,
}

impl Tray {
    pub fn new(_s: &Settings) -> Result<Self> {
        let menu = Menu::new();
        let info = MenuItem::new("OpenWritr — hold hotkey to dictate", false, None);
        let retry_engine = MenuItem::new("Retry model / engine", true, None);
        let cancel_model = MenuItem::new("Cancel model download", true, None);
        let model_sep = PredefinedMenuItem::separator();
        let open_logs = MenuItem::new("Open logs", true, None);
        let export_diagnostics = MenuItem::new("Export diagnostics…", true, None);
        let settings = MenuItem::new("Settings…", true, None);
        let sep = PredefinedMenuItem::separator();
        let sep2 = PredefinedMenuItem::separator();
        let quit = MenuItem::new("Quit", true, None);
        menu.append(&info)?;
        menu.append(&sep)?;
        menu.append(&retry_engine)?;
        menu.append(&cancel_model)?;
        menu.append(&model_sep)?;
        menu.append(&open_logs)?;
        menu.append(&export_diagnostics)?;
        menu.append(&settings)?;
        menu.append(&sep2)?;
        menu.append(&quit)?;

        let icon = make_icon(IconColor::Idle)?;
        let tray = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip("OpenWritr")
            .with_icon(icon)
            .build()?;

        Ok(Self {
            _icon: tray,
            menu_quit_id: quit.id().clone(),
            menu_settings_id: settings.id().clone(),
            menu_open_logs_id: open_logs.id().clone(),
            menu_export_diagnostics_id: export_diagnostics.id().clone(),
            menu_cancel_model_id: cancel_model.id().clone(),
            menu_retry_engine_id: retry_engine.id().clone(),
        })
    }

    pub fn set_color(&self, color: IconColor) {
        if let Ok(icon) = make_icon(color) {
            let _ = self._icon.set_icon(Some(icon));
        }
    }

    pub fn set_status(&self, color: IconColor, tooltip: &str) {
        self.set_color(color);
        self.set_tooltip(tooltip);
    }

    pub fn set_tooltip(&self, tooltip: &str) {
        let _ = self._icon.set_tooltip(Some(tooltip));
    }
}

fn make_icon(color: IconColor) -> Result<Icon> {
    let (r, g, b) = match color {
        IconColor::Idle => (74, 144, 226),
        IconColor::Recording => (220, 38, 38),
        IconColor::Transcribing => (245, 158, 11),
        IconColor::Error => (107, 114, 128),
    };
    let mut img = ImageBuffer::from_pixel(64u32, 64u32, Rgba([0u8, 0, 0, 0]));
    let fg = Rgba([r, g, b, 255]);
    // Mic body
    for y in 12..38 {
        for x in 24..40 {
            img.put_pixel(x, y, fg);
        }
    }
    // Stem + base
    for y in 38..50 {
        img.put_pixel(31, y, fg);
        img.put_pixel(32, y, fg);
    }
    for x in 20..45 {
        img.put_pixel(x, 50, fg);
        img.put_pixel(x, 51, fg);
    }

    Ok(Icon::from_rgba(img.into_raw(), 64, 64)?)
}
