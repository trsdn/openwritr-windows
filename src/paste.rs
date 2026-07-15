use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings as EnigoSettings};
use std::thread;
use std::time::Duration;
use tracing::warn;
use windows_sys::Win32::System::DataExchange::GetClipboardSequenceNumber;

pub fn paste(text: &str) {
    let mut clipboard = match Clipboard::new() {
        Ok(clipboard) => clipboard,
        Err(error) => {
            warn!(error = %error, "clipboard open failed");
            return;
        }
    };
    let saved = clipboard.get_text().ok();
    if clipboard.set_text(text.to_string()).is_err() {
        warn!("clipboard write failed");
        return;
    }
    let transcript_sequence = unsafe { GetClipboardSequenceNumber() };

    if let Ok(mut enigo) = Enigo::new(&EnigoSettings::default()) {
        let _ = enigo.key(Key::Control, Direction::Press);
        let _ = enigo.key(Key::Unicode('v'), Direction::Click);
        let _ = enigo.key(Key::Control, Direction::Release);
    } else {
        warn!("enigo init failed");
    }

    if let Some(previous) = saved {
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(400));
            let current_sequence = unsafe { GetClipboardSequenceNumber() };
            if should_restore(transcript_sequence, current_sequence) {
                if let Ok(mut clipboard) = Clipboard::new() {
                    let _ = clipboard.set_text(previous);
                }
            } else {
                tracing::info!("clipboard changed externally; preserving newer contents");
            }
        });
    }
}

fn should_restore(transcript_sequence: u32, current_sequence: u32) -> bool {
    transcript_sequence != 0 && transcript_sequence == current_sequence
}

#[cfg(test)]
mod tests {
    use super::should_restore;

    #[test]
    fn restores_only_when_the_clipboard_is_unchanged() {
        assert!(should_restore(42, 42));
        assert!(!should_restore(42, 43));
        assert!(!should_restore(0, 0));
    }
}
