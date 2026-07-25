use anyhow::{anyhow, Context, Result};
use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings as EnigoSettings};
use std::thread;
use std::time::Duration;
use windows_sys::Win32::System::DataExchange::GetClipboardSequenceNumber;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeliveryMode {
    Paste,
    Clipboard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeliveryOutcome {
    Pasted,
    Copied,
}

pub fn deliver(text: &str, mode: DeliveryMode) -> Result<DeliveryOutcome> {
    match mode {
        DeliveryMode::Paste => paste(text),
        DeliveryMode::Clipboard => copy(text),
    }
}

fn copy(text: &str) -> Result<DeliveryOutcome> {
    let mut clipboard = Clipboard::new().context("open the Windows clipboard")?;
    clipboard
        .set_text(text.to_string())
        .context("write the transcript to the Windows clipboard")?;
    Ok(DeliveryOutcome::Copied)
}

fn paste(text: &str) -> Result<DeliveryOutcome> {
    let mut clipboard = Clipboard::new().context("open the Windows clipboard")?;
    let saved = clipboard.get_text().ok();
    clipboard
        .set_text(text.to_string())
        .context("write the transcript to the Windows clipboard")?;
    let transcript_sequence = unsafe { GetClipboardSequenceNumber() };

    let mut enigo = Enigo::new(&EnigoSettings::default())
        .map_err(|error| anyhow!("could not initialize keyboard paste: {error}"))?;
    let paste_result = (|| {
        enigo
            .key(Key::Control, Direction::Press)
            .context("press Ctrl for paste")?;
        enigo
            .key(Key::Unicode('v'), Direction::Click)
            .context("press V for paste")?;
        enigo
            .key(Key::Control, Direction::Release)
            .context("release Ctrl after paste")
    })();
    if let Err(error) = paste_result {
        let _ = enigo.key(Key::Control, Direction::Release);
        return Err(anyhow!(
            "{error}; the transcript remains on the Windows clipboard"
        ));
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
    Ok(DeliveryOutcome::Pasted)
}

fn should_restore(transcript_sequence: u32, current_sequence: u32) -> bool {
    transcript_sequence != 0 && transcript_sequence == current_sequence
}

#[cfg(test)]
mod tests {
    use super::{should_restore, DeliveryMode};

    #[test]
    fn restores_only_when_the_clipboard_is_unchanged() {
        assert!(should_restore(42, 42));
        assert!(!should_restore(42, 43));
        assert!(!should_restore(0, 0));
    }

    #[test]
    fn delivery_modes_are_explicit() {
        assert_ne!(DeliveryMode::Paste, DeliveryMode::Clipboard);
    }
}
