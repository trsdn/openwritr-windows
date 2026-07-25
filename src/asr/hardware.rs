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

pub fn engine_support(engine: &str) -> Result<EngineSupport> {
    match engine {
        "parakeet_cpu" => Ok(EngineSupport::Supported { detail: None }),
        "parakeet_npu" | "whisper_npu" => current_npu_support(),
        other => Err(anyhow!("unknown transcription engine {other}")),
    }
}

pub fn ensure_engine_supported(engine: &str) -> Result<()> {
    match engine_support(engine)? {
        EngineSupport::Supported { .. } => Ok(()),
        EngineSupport::Unsupported { reason } => Err(anyhow!("{engine} is unavailable: {reason}")),
    }
}

fn current_npu_support() -> Result<EngineSupport> {
    #[cfg(not(target_arch = "aarch64"))]
    {
        return Ok(npu_support_for("x86_64", true, None));
    }
    #[cfg(all(target_arch = "aarch64", not(windows)))]
    {
        return Ok(npu_support_for("aarch64", false, None));
    }
    #[cfg(all(target_arch = "aarch64", windows))]
    {
        let processor = processor_name()?;
        Ok(npu_support_for("aarch64", true, Some(&processor)))
    }
}

fn npu_support_for(architecture: &str, windows: bool, processor: Option<&str>) -> EngineSupport {
    if architecture != "aarch64" {
        return EngineSupport::Unsupported {
            reason: "Requires the ARM64 build on Snapdragon X Elite.".into(),
        };
    }
    if !windows {
        return EngineSupport::Unsupported {
            reason: "Requires Windows on Snapdragon X Elite.".into(),
        };
    }
    let Some(processor) = processor else {
        return EngineSupport::Unsupported {
            reason: "Could not determine the processor model.".into(),
        };
    };
    if is_snapdragon_x_elite(processor) {
        EngineSupport::Supported {
            detail: Some(processor.to_string()),
        }
    } else {
        EngineSupport::Unsupported {
            reason: format!("Detected {processor}; this model is compiled for Snapdragon X Elite."),
        }
    }
}

fn is_snapdragon_x_elite(processor: &str) -> bool {
    let processor = processor.to_ascii_lowercase();
    processor.contains("snapdragon") && (processor.contains("x elite") || processor.contains("x1e"))
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
    fn npu_support_requires_windows_arm64_snapdragon_x_elite() {
        assert!(!npu_support_for("x86_64", true, None).is_supported());
        assert!(!npu_support_for("aarch64", false, None).is_supported());
        assert!(
            !npu_support_for("aarch64", true, Some("Snapdragon X Plus X1P64100")).is_supported()
        );
        assert!(npu_support_for(
            "aarch64",
            true,
            Some("Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz")
        )
        .is_supported());
    }
}
