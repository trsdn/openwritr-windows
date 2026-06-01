// Clipboard save -> set transcript -> SendInput Ctrl+V -> restore clipboard.
//
// Mirrors the macOS PasteManager.swift behaviour. enigo gives us a portable
// SendInput wrapper; arboard handles the clipboard.

use anyhow::Result;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use std::{thread, time::Duration};

#[derive(Clone)]
pub struct Paster;

impl Paster {
    pub fn new() -> Self { Self }

    pub fn paste(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let mut clip = arboard::Clipboard::new()?;
        let saved = clip.get_text().ok();
        clip.set_text(text.to_string())?;

        let mut enigo = Enigo::new(&Settings::default())?;
        enigo.key(Key::Control, Direction::Press)?;
        enigo.key(Key::Unicode('v'), Direction::Click)?;
        enigo.key(Key::Control, Direction::Release)?;

        // Give the target app a beat to consume the paste before we restore.
        thread::sleep(Duration::from_millis(80));
        if let Some(prev) = saved {
            let _ = clip.set_text(prev);
        }
        Ok(())
    }
}
