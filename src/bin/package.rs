use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use zip::write::SimpleFileOptions;

fn main() -> anyhow::Result<()> {
    let target = PathBuf::from("target/release");
    let dist = PathBuf::from("target/dist");
    std::fs::create_dir_all(&dist)?;

    let version = env!("CARGO_PKG_VERSION");
    let out_zip = dist.join(format!("openwritr-windows-arm64-v{version}.zip"));
    println!("packaging -> {}", out_zip.display());

    let mut z = zip::ZipWriter::new(File::create(&out_zip)?);
    let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    let must_have = ["openwritr.exe", "onnxruntime.dll"];
    let should_have = [
        "onnxruntime_providers_shared.dll", "onnxruntime_providers_qnn.dll",
        "QnnHtp.dll", "QnnHtpPrepare.dll", "QnnHtpV73Stub.dll", "QnnHtpV81Stub.dll",
        "QnnSystem.dll", "QnnCpu.dll", "QnnGpu.dll", "QnnIr.dll", "Genie.dll",
        "libQnnHtpV73Skel.so", "libQnnHtpV81Skel.so",
        "libqnnhtpv73.cat", "libqnnhtpv81.cat",
    ];

    for name in &must_have { add_file(&mut z, &target.join(name), name, opts)?; }
    for name in &should_have {
        let p = target.join(name);
        if p.exists() { add_file(&mut z, &p, name, opts)?; }
    }
    add_file(&mut z, Path::new("README.md"), "README.md", opts)?;
    if Path::new("LICENSE").exists() {
        add_file(&mut z, Path::new("LICENSE"), "LICENSE", opts)?;
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
