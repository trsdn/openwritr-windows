use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use zip::write::SimpleFileOptions;

fn main() -> Result<()> {
    let architecture = parse_architecture(std::env::args().skip(1))?;
    run_python(&["scripts/fetch_runtime.py", "--arch", architecture.as_str()])?;
    run_python(&[
        "scripts/prepare_release.py",
        "--arch",
        architecture.as_str(),
    ])?;

    let stage = PathBuf::from("target")
        .join("stage")
        .join(architecture.as_str());
    let files = collect_files(&stage)?;
    let dist = PathBuf::from("target").join("dist");
    std::fs::create_dir_all(&dist).context("create target/dist")?;
    let output = dist.join(format!(
        "openwritr-windows-{}-v{}.zip",
        architecture.as_str(),
        env!("CARGO_PKG_VERSION")
    ));
    write_zip(&output, &stage, &files)?;

    run_python(&[
        "scripts/verify_artifact.py",
        "--arch",
        architecture.as_str(),
        "--artifact",
        output
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("ZIP path is not valid UTF-8"))?,
        "--format",
        "zip",
    ])?;
    println!(
        "packaged {} ({:.2} MB)",
        output.display(),
        output.metadata()?.len() as f64 / 1_000_000.0
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum Architecture {
    Arm64,
    X64,
}

impl Architecture {
    fn as_str(self) -> &'static str {
        match self {
            Self::Arm64 => "arm64",
            Self::X64 => "x64",
        }
    }
}

fn parse_architecture(args: impl Iterator<Item = String>) -> Result<Architecture> {
    let args = args.collect::<Vec<_>>();
    let mut architecture = Architecture::Arm64;
    let mut index = 0;
    while index < args.len() {
        if args[index] != "--arch" {
            bail!("unknown package argument {}", args[index]);
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| anyhow::anyhow!("--arch requires arm64 or x64"))?;
        architecture = match value.as_str() {
            "arm64" => Architecture::Arm64,
            "x64" => Architecture::X64,
            other => bail!("unsupported package architecture {other}"),
        };
        index += 2;
    }
    Ok(architecture)
}

fn run_python(args: &[&str]) -> Result<()> {
    let python = std::env::var_os("PYTHON").unwrap_or_else(|| "python".into());
    let status = Command::new(&python)
        .args(args)
        .status()
        .with_context(|| format!("run {} {}", Path::new(&python).display(), args.join(" ")))?;
    if !status.success() {
        bail!(
            "{} {} failed with exit code {}",
            Path::new(&python).display(),
            args.join(" "),
            status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "terminated".to_string())
        );
    }
    Ok(())
}

fn collect_files(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.is_dir() {
        bail!("release stage does not exist: {}", root.display());
    }
    let mut files = Vec::new();
    collect_files_recursive(root, root, &mut files)?;
    files.sort();
    if files.is_empty() {
        bail!("release stage is empty: {}", root.display());
    }
    Ok(files)
}

fn collect_files_recursive(root: &Path, directory: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(directory)
        .with_context(|| format!("read release stage {}", directory.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_files_recursive(root, &path, files)?;
        } else if entry.file_type()?.is_file() {
            files.push(
                path.strip_prefix(root)
                    .expect("release file is below stage root")
                    .to_path_buf(),
            );
        }
    }
    Ok(())
}

fn write_zip(output: &Path, stage: &Path, files: &[PathBuf]) -> Result<()> {
    let temporary = output.with_extension("zip.tmp");
    let _ = std::fs::remove_file(&temporary);
    let result = (|| {
        let mut archive = zip::ZipWriter::new(
            File::create(&temporary)
                .with_context(|| format!("create temporary ZIP {}", temporary.display()))?,
        );
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for relative in files {
            let source = stage.join(relative);
            if source.metadata()?.len() == 0 {
                bail!("release stage contains empty file {}", source.display());
            }
            let zip_name = relative.to_string_lossy().replace('\\', "/");
            archive.start_file(&zip_name, options)?;
            let mut input =
                File::open(&source).with_context(|| format!("open {}", source.display()))?;
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                let read = input.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                archive.write_all(&buffer[..read])?;
            }
            println!("  + {zip_name}");
        }
        archive.finish()?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    if output.exists() {
        std::fs::remove_file(output)
            .with_context(|| format!("remove previous ZIP {}", output.display()))?;
    }
    std::fs::rename(&temporary, output)
        .with_context(|| format!("publish ZIP {}", output.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_architecture_is_strict() {
        assert!(matches!(
            parse_architecture(std::iter::empty()).unwrap(),
            Architecture::Arm64
        ));
        assert!(matches!(
            parse_architecture(["--arch".into(), "x64".into()].into_iter()).unwrap(),
            Architecture::X64
        ));
        assert!(parse_architecture(["--arch".into(), "mips".into()].into_iter()).is_err());
        assert!(parse_architecture(["--unknown".into()].into_iter()).is_err());
    }
}
