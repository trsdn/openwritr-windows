#[cfg(all(target_arch = "aarch64", windows))]
use anyhow::Context;
use anyhow::{anyhow, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineSupport {
    Supported { detail: Option<String> },
    Unsupported { reason: String },
}

impl EngineSupport {
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported { .. })
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Supported { .. } => None,
            Self::Unsupported { reason } => Some(reason),
        }
    }

    pub fn detail(&self) -> Option<&str> {
        match self {
            Self::Supported { detail } => detail.as_deref(),
            Self::Unsupported { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpuTarget {
    V73, // Snapdragon X Elite (X1E)
    V81, // Snapdragon X2 Elite (X2E)
}

impl NpuTarget {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::V73 => "v73",
            Self::V81 => "v81",
        }
    }

    #[allow(dead_code)]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::V73 => "Snapdragon X Elite",
            Self::V81 => "Snapdragon X2 Elite",
        }
    }
}

pub fn detect_npu_target(processor: &str) -> Option<NpuTarget> {
    let p = processor.to_ascii_lowercase();
    if !p.contains("snapdragon") {
        return None;
    }
    if p.contains("x2 elite") || p.contains("x2e") {
        Some(NpuTarget::V81)
    } else if p.contains("x elite") || p.contains("x1e") {
        Some(NpuTarget::V73)
    } else {
        None
    }
}

pub fn current_npu_target() -> Option<NpuTarget> {
    #[cfg(all(target_arch = "aarch64", windows))]
    {
        processor_name().ok().as_deref().and_then(detect_npu_target)
    }
    #[cfg(not(all(target_arch = "aarch64", windows)))]
    {
        None
    }
}

pub fn engine_support(engine: &str) -> Result<EngineSupport> {
    match engine {
        "parakeet_cpu" => Ok(EngineSupport::Supported { detail: None }),
        "whisper_npu" => current_whisper_npu_support(),
        "parakeet_npu" => current_parakeet_npu_support(),
        other => Err(anyhow!("unknown transcription engine {other}")),
    }
}

pub fn ensure_engine_supported(engine: &str) -> Result<()> {
    match engine_support(engine)? {
        EngineSupport::Supported { .. } => Ok(()),
        EngineSupport::Unsupported { reason } => Err(anyhow!("{engine} is unavailable: {reason}")),
    }
}

fn current_whisper_npu_support() -> Result<EngineSupport> {
    #[cfg(not(target_arch = "aarch64"))]
    {
        return Ok(whisper_npu_support_for("x86_64", true, None));
    }
    #[cfg(all(target_arch = "aarch64", not(windows)))]
    {
        return Ok(whisper_npu_support_for("aarch64", false, None));
    }
    #[cfg(all(target_arch = "aarch64", windows))]
    {
        let processor = processor_name()?;
        Ok(whisper_npu_support_for("aarch64", true, Some(&processor)))
    }
}

pub fn whisper_npu_support_for(
    architecture: &str,
    windows: bool,
    processor: Option<&str>,
) -> EngineSupport {
    if architecture != "aarch64" {
        return EngineSupport::Unsupported {
            reason: "Requires the ARM64 build on Snapdragon X Elite or X2 Elite.".into(),
        };
    }
    if !windows {
        return EngineSupport::Unsupported {
            reason: "Requires Windows on Snapdragon X Elite or X2 Elite.".into(),
        };
    }
    let Some(processor) = processor else {
        return EngineSupport::Unsupported {
            reason: "Could not determine the processor model.".into(),
        };
    };
    if let Some(_target) = detect_npu_target(processor) {
        EngineSupport::Supported {
            detail: Some(processor.to_string()),
        }
    } else {
        EngineSupport::Unsupported {
            reason: format!(
                "Detected {processor}; this model requires Snapdragon X Elite or X2 Elite."
            ),
        }
    }
}

fn current_parakeet_npu_support() -> Result<EngineSupport> {
    #[cfg(not(target_arch = "aarch64"))]
    {
        return Ok(parakeet_npu_support_for("x86_64", true, None));
    }
    #[cfg(all(target_arch = "aarch64", not(windows)))]
    {
        return Ok(parakeet_npu_support_for("aarch64", false, None));
    }
    #[cfg(all(target_arch = "aarch64", windows))]
    {
        let processor = processor_name()?;
        Ok(parakeet_npu_support_for("aarch64", true, Some(&processor)))
    }
}

pub fn parakeet_npu_support_for(
    architecture: &str,
    windows: bool,
    processor: Option<&str>,
) -> EngineSupport {
    if architecture != "aarch64" {
        return EngineSupport::Unsupported {
            reason: "Requires the ARM64 build on Snapdragon X Elite or X2 Elite.".into(),
        };
    }
    if !windows {
        return EngineSupport::Unsupported {
            reason: "Requires Windows on Snapdragon X Elite or X2 Elite.".into(),
        };
    }
    let Some(processor) = processor else {
        return EngineSupport::Unsupported {
            reason: "Could not determine the processor model.".into(),
        };
    };
    if let Some(_target) = detect_npu_target(processor) {
        EngineSupport::Supported {
            detail: Some(processor.to_string()),
        }
    } else {
        EngineSupport::Unsupported {
            reason: format!(
                "Detected {processor}; this model is compiled for Snapdragon X Elite or X2 Elite."
            ),
        }
    }
}

#[allow(dead_code)]
pub fn is_snapdragon_x_elite(processor: &str) -> bool {
    detect_npu_target(processor) == Some(NpuTarget::V73)
}

#[cfg(all(target_arch = "aarch64", windows))]
fn processor_name() -> Result<String> {
    use windows::core::w;
    use windows::Win32::System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ};

    let mut bytes = 0_u32;
    unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            w!("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
            w!("ProcessorNameString"),
            RRF_RT_REG_SZ,
            None,
            None,
            Some(&mut bytes),
        )
        .ok()
        .context("read processor name size")?;
    }
    if bytes < 2 {
        return Err(anyhow!("Windows returned an empty processor name"));
    }

    let mut buffer = vec![0_u16; (bytes as usize).div_ceil(2)];
    unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            w!("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
            w!("ProcessorNameString"),
            RRF_RT_REG_SZ,
            None,
            Some(buffer.as_mut_ptr().cast()),
            Some(&mut bytes),
        )
        .ok()
        .context("read processor name")?;
    }
    let length = buffer
        .iter()
        .position(|value| *value == 0)
        .unwrap_or(buffer.len());
    String::from_utf16(&buffer[..length]).context("decode processor name")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_supported_everywhere() {
        assert!(engine_support("parakeet_cpu").unwrap().is_supported());
    }

    #[test]
    fn whisper_npu_support_for_x_elite_and_x2_elite() {
        assert!(!whisper_npu_support_for("x86_64", true, None).is_supported());
        assert!(!whisper_npu_support_for("aarch64", false, None).is_supported());
        assert!(
            !whisper_npu_support_for("aarch64", true, Some("Snapdragon X Plus X1P64100"))
                .is_supported()
        );
        assert!(whisper_npu_support_for(
            "aarch64",
            true,
            Some("Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz")
        )
        .is_supported());
        assert!(whisper_npu_support_for(
            "aarch64",
            true,
            Some("Snapdragon(R) X2 Elite Extreme - X2E94100 - Qualcomm Oryon(TM) CPU")
        )
        .is_supported());
    }

    #[test]
    fn parakeet_npu_support_for_x_elite_and_x2_elite() {
        assert!(!parakeet_npu_support_for("x86_64", true, None).is_supported());
        assert!(!parakeet_npu_support_for("aarch64", false, None).is_supported());
        assert!(
            !parakeet_npu_support_for("aarch64", true, Some("Snapdragon X Plus X1P64100"))
                .is_supported()
        );
        assert!(parakeet_npu_support_for(
            "aarch64",
            true,
            Some("Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz")
        )
        .is_supported());
        assert!(parakeet_npu_support_for(
            "aarch64",
            true,
            Some("Snapdragon(R) X2 Elite Extreme - X2E94100 - Qualcomm Oryon(TM) CPU")
        )
        .is_supported());
    }
}
