//! Soft start/stop UI cues — generated on first use as PCM WAVs in %TEMP%,
//! played via winsound through windows API.

use parking_lot::Mutex;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use windows::core::PCSTR;
use windows::Win32::Media::Audio::{PlaySoundA, SND_ASYNC, SND_FILENAME, SND_NODEFAULT, SND_NOSTOP};

const SR: u32 = 44_100;

static START_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);
static STOP_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

fn temp_dir() -> PathBuf { std::env::temp_dir().join("openwritr-cues") }

fn render_tone(freq: f32, duration_s: f32) -> Vec<i16> {
    let n = (SR as f32 * duration_s) as usize;
    let attack_n = (SR as f32 * 0.025) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / SR as f32;
        let base = (2.0 * std::f32::consts::PI * freq * t).sin();
        let octave = 0.25 * (2.0 * std::f32::consts::PI * freq * 2.0 * t).sin();
        let mut body = base + octave;
        // 60 ms sub-thump.
        if i < (SR as f32 * 0.06) as usize {
            let env_t = 1.0 - (i as f32 / (SR as f32 * 0.06));
            body += 0.18 * env_t * (2.0 * std::f32::consts::PI * 80.0 * t).sin();
        }
        let attack = if i < attack_n {
            0.5 - 0.5 * (std::f32::consts::PI * i as f32 / attack_n as f32).cos()
        } else { 1.0 };
        let decay = (-t / (duration_s * 0.32)).exp();
        let s = body * attack * decay * 0.18;
        out.push((s.clamp(-1.0, 1.0) * 32767.0) as i16);
    }
    out
}

fn write_wav(path: &PathBuf, samples: &[i16]) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    let n = samples.len();
    let data_size = (n * 2) as u32;
    let total = 36 + data_size;
    f.write_all(b"RIFF")?;
    f.write_all(&total.to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;     // chunk size
    f.write_all(&1u16.to_le_bytes())?;      // PCM
    f.write_all(&1u16.to_le_bytes())?;      // mono
    f.write_all(&SR.to_le_bytes())?;
    f.write_all(&(SR * 2).to_le_bytes())?;  // byte rate
    f.write_all(&2u16.to_le_bytes())?;      // block align
    f.write_all(&16u16.to_le_bytes())?;     // bits per sample
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for s in samples {
        f.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

fn ensure(path_slot: &Mutex<Option<PathBuf>>, name: &str, freq: f32) -> Option<PathBuf> {
    let mut slot = path_slot.lock();
    if let Some(p) = slot.as_ref() {
        if p.exists() { return Some(p.clone()); }
    }
    let dir = temp_dir();
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(name);
    let samples = render_tone(freq, 0.32);
    if write_wav(&path, &samples).is_ok() {
        *slot = Some(path.clone());
        Some(path)
    } else {
        None
    }
}

pub fn play_start() {
    if let Some(p) = ensure(&START_PATH, "start.wav", 196.0) {
        play(&p);
    }
}

pub fn play_stop() {
    if let Some(p) = ensure(&STOP_PATH, "stop.wav", 164.81) {
        play(&p);
    }
}

fn play(path: &PathBuf) {
    let path_str = path.to_string_lossy();
    let mut c = path_str.as_bytes().to_vec();
    c.push(0); // null-terminate
    unsafe {
        let _ = PlaySoundA(
            PCSTR(c.as_ptr()),
            None,
            SND_FILENAME | SND_ASYNC | SND_NODEFAULT | SND_NOSTOP,
        );
    }
}
