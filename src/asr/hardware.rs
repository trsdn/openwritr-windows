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

/// Hexagon HTP architecture the shipped precompiled NPU models (Parakeet
/// encoder, Whisper) are compiled for. QNN context binaries only load on this
/// exact architecture; bump it together with the models.
const REQUIRED_HTP_ARCH: u32 = 73;

/// What probing the NPU through the pinned QNN runtime found.
#[derive(Debug)]
enum NpuProbe {
    /// Not a Windows ARM64 build, so there is no QNN runtime to ask.
    UnsupportedPlatform,
    Failed(String),
    NoHtp,
    Htp(u32),
}

fn current_npu_support() -> Result<EngineSupport> {
    #[cfg(not(all(target_arch = "aarch64", windows)))]
    {
        return Ok(npu_support_for(NpuProbe::UnsupportedPlatform, None));
    }
    #[cfg(all(target_arch = "aarch64", windows))]
    {
        let probe = match super::qnn_ffi::htp_architecture() {
            Ok(Some(arch)) => NpuProbe::Htp(arch),
            Ok(None) => NpuProbe::NoHtp,
            Err(error) => {
                tracing::warn!(error = %error, "Hexagon NPU probe failed");
                NpuProbe::Failed(error.to_string())
            }
        };
        let processor = processor_name()
            .map_err(|error| tracing::warn!(error = %error, "processor name unavailable"))
            .ok();
        Ok(npu_support_for(probe, processor.as_deref()))
    }
}

fn npu_support_for(probe: NpuProbe, processor: Option<&str>) -> EngineSupport {
    match probe {
        NpuProbe::Htp(arch) if arch == REQUIRED_HTP_ARCH => EngineSupport::Supported {
            detail: Some(match processor {
                Some(processor) => format!("{processor} (Hexagon V{arch})"),
                None => format!("Hexagon V{arch} NPU"),
            }),
        },
        NpuProbe::Htp(arch) => EngineSupport::Unsupported {
            reason: format!(
                "Detected a Hexagon V{arch} NPU, but the NPU models are compiled for Hexagon V{REQUIRED_HTP_ARCH}. Use the CPU engine instead."
            ),
        },
        NpuProbe::NoHtp => EngineSupport::Unsupported {
            reason: "No Qualcomm Hexagon NPU was found. Use the CPU engine instead.".into(),
        },
        NpuProbe::Failed(error) => EngineSupport::Unsupported {
            reason: format!("Could not query the Hexagon NPU ({error}). Use the CPU engine instead."),
        },
        NpuProbe::UnsupportedPlatform => EngineSupport::Unsupported {
            reason: format!(
                "Requires the Windows ARM64 build on a Hexagon V{REQUIRED_HTP_ARCH} NPU (Snapdragon X Elite / X Plus)."
            ),
        },
    }
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
    fn npu_support_requires_the_models_htp_architecture() {
        let processor = Some("Snapdragon(R) X 10-core X1P64100 @ 3.40 GHz");
        let supported = npu_support_for(NpuProbe::Htp(REQUIRED_HTP_ARCH), processor);
        assert_eq!(
            supported.detail(),
            Some("Snapdragon(R) X 10-core X1P64100 @ 3.40 GHz (Hexagon V73)")
        );
        assert!(npu_support_for(NpuProbe::Htp(REQUIRED_HTP_ARCH), None).is_supported());

        // Older and newer Hexagon generations cannot load the V73 context binaries.
        for arch in [68, 69, 75, 79, 81] {
            let support = npu_support_for(NpuProbe::Htp(arch), processor);
            assert!(!support.is_supported(), "V{arch} must be rejected");
            assert!(support.reason().unwrap().contains(&format!("V{arch}")));
        }
        assert!(!npu_support_for(NpuProbe::NoHtp, processor).is_supported());
        assert!(!npu_support_for(NpuProbe::Failed("boom".into()), processor).is_supported());
        assert!(!npu_support_for(NpuProbe::UnsupportedPlatform, None).is_supported());
    }

    /// Probes the real NPU through the pinned runtime staged in target/<profile>.
    /// Run on Snapdragon hardware: `cargo test --release -- --ignored real_npu`
    #[test]
    #[ignore = "needs Windows ARM64 hardware with the QNN runtime staged"]
    #[cfg(all(target_arch = "aarch64", windows))]
    fn real_npu_reports_required_htp_architecture() {
        let arch = super::super::qnn_ffi::htp_architecture().expect("probe the Hexagon NPU");
        assert_eq!(arch, Some(REQUIRED_HTP_ARCH));
        assert!(engine_support("parakeet_npu").unwrap().is_supported());
    }
}
