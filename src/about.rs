use anyhow::{bail, Context, Result};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
#[cfg(windows)]
const DETACHED_PROCESS: u32 = 0x0000_0008;

pub const PUBLISHER: &str = "Torsten Mahr";
pub const COPYRIGHT: &str = "Copyright © 2026 Torsten Mahr and contributors.";
pub const LICENSE_SUMMARY: &str = "OpenWritr is licensed under the MIT License.";
pub const DISCLAIMER: &str = "OpenWritr is an independent open-source project and is not affiliated with NVIDIA, Qualcomm, or Microsoft.";

pub const WEBSITE_URL: &str = "https://trsdn.github.io/openwritr-windows/";
pub const STORE_URL: &str = "https://apps.microsoft.com/detail/9MSQWR701P2Q";
pub const REPOSITORY_URL: &str = "https://github.com/trsdn/openwritr-windows";
pub const REPORT_ISSUE_URL: &str = "https://github.com/trsdn/openwritr-windows/issues/new/choose";
pub const ISSUES_URL: &str = "https://github.com/trsdn/openwritr-windows/issues";
pub const RELEASES_URL: &str = "https://github.com/trsdn/openwritr-windows/releases/latest";
pub const PRIVACY_URL: &str = "https://github.com/trsdn/openwritr-windows/blob/main/PRIVACY.md";
pub const LICENSE_URL: &str = "https://github.com/trsdn/openwritr-windows/blob/main/LICENSE";

pub const PARAKEET_URL: &str = "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3";
pub const PARAKEET_ONNX_URL: &str = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx";
pub const PARAKEET_NPU_URL: &str = "https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s";
pub const WHISPER_QNN_URL: &str = "https://huggingface.co/qualcomm/Whisper-Large-V3-Turbo";
pub const ONNX_RUNTIME_URL: &str = "https://github.com/microsoft/onnxruntime";
pub const QUALCOMM_AI_HUB_URL: &str = "https://aihub.qualcomm.com/";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OfficialLink {
    pub label: &'static str,
    pub url: &'static str,
}

pub const OFFICIAL_LINKS: &[OfficialLink] = &[
    OfficialLink {
        label: "Project website",
        url: WEBSITE_URL,
    },
    OfficialLink {
        label: "Microsoft Store",
        url: STORE_URL,
    },
    OfficialLink {
        label: "Source repository",
        url: REPOSITORY_URL,
    },
    OfficialLink {
        label: "Report an issue",
        url: REPORT_ISSUE_URL,
    },
    OfficialLink {
        label: "Existing issues",
        url: ISSUES_URL,
    },
    OfficialLink {
        label: "Latest releases",
        url: RELEASES_URL,
    },
    OfficialLink {
        label: "Privacy policy",
        url: PRIVACY_URL,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Credit {
    pub name: &'static str,
    pub attribution: &'static str,
    pub url: &'static str,
}

pub const CREDITS: &[Credit] = &[
    Credit {
        name: "NVIDIA Parakeet TDT 0.6B v3",
        attribution: "NVIDIA · CC-BY-4.0",
        url: PARAKEET_URL,
    },
    Credit {
        name: "Parakeet ONNX assets",
        attribution: "istupakov/parakeet-tdt-0.6b-v3-onnx · CC-BY-4.0",
        url: PARAKEET_ONNX_URL,
    },
    Credit {
        name: "OpenWritr Parakeet NPU build",
        attribution: "Compiled through Qualcomm AI Hub · CC-BY-4.0",
        url: PARAKEET_NPU_URL,
    },
    Credit {
        name: "Whisper Large v3 Turbo QNN assets",
        attribution: "Qualcomm · Apache-2.0 / BSD-3-Clause",
        url: WHISPER_QNN_URL,
    },
    Credit {
        name: "ONNX Runtime",
        attribution: "Microsoft · MIT",
        url: ONNX_RUNTIME_URL,
    },
    Credit {
        name: "Qualcomm QNN runtime",
        attribution: "Qualcomm AI Engine Direct redistributable license",
        url: QUALCOMM_AI_HUB_URL,
    },
];

pub fn open_url(url: &str) -> Result<()> {
    if !url.starts_with("https://") {
        bail!("refusing to open a non-HTTPS URL");
    }
    shell_open(OsStr::new(url)).with_context(|| format!("open {url}"))
}

pub fn open_license() -> Result<()> {
    open_legal_file("LICENSE")
}

pub fn open_privacy_policy() -> Result<()> {
    open_legal_file("PRIVACY.md")
}

pub fn open_third_party_licenses() -> Result<()> {
    let directory = resolve_installed_path("third-party-licenses")?;
    if !directory.is_dir() {
        bail!(
            "third-party license directory is unavailable at {}",
            directory.display()
        );
    }
    shell_open(directory.as_os_str()).with_context(|| format!("open {}", directory.display()))
}

fn open_legal_file(relative: &str) -> Result<()> {
    let path = resolve_installed_path(relative)?;
    if !path.is_file() {
        bail!("legal file is unavailable at {}", path.display());
    }
    shell_open(path.as_os_str()).with_context(|| format!("open {}", path.display()))
}

fn resolve_installed_path(relative: &str) -> Result<PathBuf> {
    let runtime = std::env::current_exe()
        .context("resolve current executable")?
        .parent()
        .context("current executable has no parent directory")?
        .to_path_buf();
    Ok(resolve_legal_path(
        &runtime,
        Some(Path::new(env!("CARGO_MANIFEST_DIR"))),
        relative,
    )
    .unwrap_or_else(|| runtime.join(relative)))
}

fn resolve_legal_path(
    runtime_dir: &Path,
    repository_dir: Option<&Path>,
    relative: &str,
) -> Option<PathBuf> {
    let installed = runtime_dir.join(relative);
    if installed.exists() {
        return Some(installed);
    }
    repository_dir
        .map(|directory| directory.join(relative))
        .filter(|path| path.exists())
}

fn shell_open(target: &OsStr) -> Result<()> {
    #[cfg(windows)]
    let mut command = {
        let mut command = Command::new("explorer.exe");
        command.creation_flags(DETACHED_PROCESS | CREATE_NO_WINDOW);
        command
    };
    #[cfg(not(windows))]
    let mut command = Command::new("xdg-open");

    command
        .arg(target)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("start the system shell opener")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn official_links_are_unique_https_urls() {
        let mut urls = HashSet::new();
        for link in OFFICIAL_LINKS {
            assert!(link.url.starts_with("https://"));
            assert!(urls.insert(link.url));
        }
    }

    #[test]
    fn legal_path_prefers_the_installed_copy() {
        let runtime = tempfile::tempdir().unwrap();
        let repository = tempfile::tempdir().unwrap();
        std::fs::write(runtime.path().join("LICENSE"), "installed").unwrap();
        std::fs::write(repository.path().join("LICENSE"), "repository").unwrap();

        assert_eq!(
            resolve_legal_path(runtime.path(), Some(repository.path()), "LICENSE"),
            Some(runtime.path().join("LICENSE"))
        );
    }

    #[test]
    fn legal_path_uses_the_repository_during_development() {
        let runtime = tempfile::tempdir().unwrap();
        let repository = tempfile::tempdir().unwrap();
        std::fs::write(repository.path().join("PRIVACY.md"), "privacy").unwrap();

        assert_eq!(
            resolve_legal_path(runtime.path(), Some(repository.path()), "PRIVACY.md"),
            Some(repository.path().join("PRIVACY.md"))
        );
    }

    #[test]
    fn legal_path_reports_missing_files() {
        let runtime = tempfile::tempdir().unwrap();
        let repository = tempfile::tempdir().unwrap();

        assert_eq!(
            resolve_legal_path(runtime.path(), Some(repository.path()), "LICENSE"),
            None
        );
    }
}
