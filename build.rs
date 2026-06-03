//! Build script: embed the app icon + version metadata into openwritr.exe
//! on Windows. The icon lives at installer/openwritr.ico (regenerate with
//! `python installer/make_icon.py`). On non-Windows targets this is a no-op
//! so `cargo check` still works on a dev box.

fn main() {
    #[cfg(target_os = "windows")]
    {
        // Only the GUI binary needs the icon; the `package` helper bin doesn't,
        // but winresource attaches per-crate so both get it — harmless.
        let mut res = winresource::WindowsResource::new();
        res.set_icon("installer/openwritr.ico");
        res.set("ProductName", "OpenWritr");
        res.set("FileDescription", "OpenWritr — push-to-talk voice-to-text");
        res.set("CompanyName", "Torsten Mahr");
        res.set("LegalCopyright", "MIT License");
        if let Err(e) = res.compile() {
            // Don't hard-fail the build if the resource compiler isn't on PATH
            // (e.g. a shell without vcvars). Just warn — the exe builds, sans icon.
            println!("cargo:warning=icon embed skipped: {e}");
        }
        println!("cargo:rerun-if-changed=installer/openwritr.ico");
    }
}
