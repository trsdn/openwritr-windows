use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use zip::write::SimpleFileOptions;

fn main() -> anyhow::Result<()> {
    let target = PathBuf::from("target/release");
    let dist = PathBuf::from("target/dist");
    let venv_qnn = PathBuf::from(".venv/Lib/site-packages/onnxruntime_qnn");
    std::fs::create_dir_all(&dist)?;

    let version = env!("CARGO_PKG_VERSION");
    let out_zip = dist.join(format!("openwritr-windows-arm64-v{version}.zip"));
    println!("packaging -> {}", out_zip.display());

    let mut z = zip::ZipWriter::new(File::create(&out_zip)?);
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    // Files we expect to exist in target/release/ — built by cargo, no fallback.
    let must_have = ["openwritr.exe", "onnxruntime.dll"];
    // QNN runtime: copied from `.venv\Lib\site-packages\onnxruntime_qnn\` by
    // pip install onnxruntime-qnn. We resolve each file from target/release/
    // (where the user — or this very script — staged it) OR fall back to the
    // venv on a fresh checkout. Anything pulled from venv is also copied into
    // target/release/ so subsequent local `openwritr.exe` runs find their DLLs.
    let qnn_runtime = [
        "onnxruntime_providers_qnn.dll",
        // EP plugin + HTP backend chain.
        "QnnHtp.dll", "QnnHtpPrepare.dll",
        // Per-Hexagon-arch stub DLLs. V73 = Snapdragon X Elite / 8 Gen 3,
        // V81 = next-gen. Without their sibling Skel.so + .cat files the
        // stub fails LoadLibrary with err=126 (ERROR_MOD_NOT_FOUND), and
        // QnnHtp's CreateSession then aborts with STATUS_STACK_BUFFER_OVERRUN
        // — not an obvious error, hence the misery hunting it down.
        "QnnHtpV73Stub.dll", "QnnHtpV81Stub.dll",
        "libQnnHtpV73Skel.so", "libQnnHtpV81Skel.so",
        "libqnnhtpv73.cat", "libqnnhtpv81.cat",
        // Sibling backends — not used by openwritr today but cheap to ship.
        "QnnSystem.dll", "QnnCpu.dll", "QnnGpu.dll", "QnnIr.dll", "Genie.dll",
    ];

    // Required artifacts: hard-fail if missing.
    for name in &must_have {
        add_file(&mut z, &target.join(name), name, opts)?;
    }

    // QNN runtime: try staged target/release, then venv. Mirror into
    // target/release so `cargo run` (sans this packager) also works.
    for name in &qnn_runtime {
        let staged = target.join(name);
        let from_venv = venv_qnn.join(name);
        let src = if staged.exists() {
            staged.clone()
        } else if from_venv.exists() {
            std::fs::copy(&from_venv, &staged)
                .map_err(|e| anyhow::anyhow!("stage {name} from venv to target/release: {e}"))?;
            println!("  (staged {name} from venv → target/release)");
            staged
        } else {
            eprintln!("  WARN: {name} not found in target/release or {} — skipping",
                      venv_qnn.display());
            continue;
        };
        add_file(&mut z, &src, name, opts)?;
    }

    add_file(&mut z, Path::new("README.md"), "README.md", opts)?;
    if Path::new("LICENSE").exists() {
        add_file(&mut z, Path::new("LICENSE"), "LICENSE", opts)?;
    }

    // Third-party licence files for the bundled Qualcomm QNN runtime DLLs.
    for (src_name, zip_name) in [
        ("Qualcomm_LICENSE.pdf", "third-party-licenses/Qualcomm_LICENSE.pdf"),
        ("ThirdPartyNotices.txt", "third-party-licenses/ThirdPartyNotices.txt"),
        ("LICENSE", "third-party-licenses/onnxruntime-qnn-LICENSE.txt"),
        ("Privacy.md", "third-party-licenses/onnxruntime-qnn-Privacy.md"),
    ] {
        let candidates = [target.join(src_name), venv_qnn.join(src_name)];
        if let Some(p) = candidates.iter().find(|p| p.exists()) {
            add_file(&mut z, p, zip_name, opts)?;
        }
    }

    z.finish()?;
    let size = out_zip.metadata()?.len();
    println!("done -> {} ({:.2} MB)", out_zip.display(), size as f32 / 1_000_000.0);
    Ok(())
}

fn add_file<W: Write + std::io::Seek>(z: &mut zip::ZipWriter<W>, src: &Path, name: &str, opts: SimpleFileOptions) -> anyhow::Result<()> {
    z.start_file(name, opts)?;
    let mut f = File::open(src).map_err(|e| anyhow::anyhow!("open {}: {e}", src.display()))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    z.write_all(&buf)?;
    println!("  + {} ({:.1} KB)", name, buf.len() as f32 / 1000.0);
    Ok(())
}
