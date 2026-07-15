//! Standalone path-based QNN session and inference probe.
//!
//! Run from the project root with the pinned runtime staged in target/release:
//!
//!     cargo run --release --example test_npu_ffi
//!     cargo run --release --example test_npu_ffi -- --probe-only

#[path = "../src/asr/qnn_ffi.rs"]
mod qnn_ffi;

use anyhow::{anyhow, bail, Context, Result};
use ort::AsPointer;
use qnn_ffi::{
    acquire_qnn_provider, enumerate_qnn_npu_devices, initialize_ort_runtime, QnnSession,
    SessionContract, TensorElementType, TensorInput, TensorSpec,
};
use std::path::PathBuf;

fn main() -> Result<()> {
    let executable = std::env::current_exe().context("resolve current executable")?;
    let executable_dir = executable
        .parent()
        .ok_or_else(|| anyhow!("current executable has no parent directory"))?;
    let runtime_dir = if executable_dir.ends_with("examples") {
        executable_dir
            .parent()
            .ok_or_else(|| anyhow!("example executable has no target directory"))?
            .to_path_buf()
    } else {
        executable_dir.to_path_buf()
    };

    let initialized_runtime = initialize_ort_runtime()?;
    if initialized_runtime != runtime_dir {
        bail!(
            "runtime directory mismatch: expected {}, initialized {}",
            runtime_dir.display(),
            initialized_runtime.display()
        );
    }

    let qnn_dll = runtime_dir.join("onnxruntime_providers_qnn.dll");
    let provider_lease = acquire_qnn_provider(&qnn_dll)?;
    let environment = ort::environment::Environment::current()
        .map_err(|error| anyhow!("Environment::current: {error}"))?;
    let devices = enumerate_qnn_npu_devices(environment.ptr())?;
    if devices.is_empty() {
        bail!("no QNN NPU device available");
    }
    println!(
        "verified runtime {} with {} QNN NPU device(s)",
        runtime_dir.display(),
        devices.len()
    );

    if std::env::args().any(|argument| argument == "--probe-only") {
        return Ok(());
    }

    let model = std::env::var_os("OPENWRITR_NPU_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let mut path =
                PathBuf::from(std::env::var_os("LOCALAPPDATA").expect("LOCALAPPDATA is set"));
            path.push("OpenWritr/models/parakeet-tdt-0.6b-v3-htp-int8-8s/encoder-model.onnx");
            path
        });
    if !model.is_file() {
        bail!(
            "missing model wrapper {}; set OPENWRITR_NPU_MODEL to override",
            model.display()
        );
    }

    let contract = SessionContract::new(
        vec![
            TensorSpec::new("audio_signal", TensorElementType::F32, vec![1, 128, 801]),
            TensorSpec::new("length", TensorElementType::I32, vec![1]),
        ],
        vec![
            TensorSpec::new("output_0", TensorElementType::F32, vec![1, 1024, -1]),
            TensorSpec::new("output_1", TensorElementType::I32, vec![1]),
        ],
    )?;
    let mut session = QnnSession::load(
        environment.ptr(),
        &devices,
        &model,
        contract,
        provider_lease,
    )?;

    let audio = vec![0.0_f32; 128 * 801];
    let audio_shape = [1_i64, 128, 801];
    let length = [801_i32];
    let length_shape = [1_i64];
    let started = std::time::Instant::now();
    let outputs = session.run(&[
        TensorInput::f32("audio_signal", &audio_shape, &audio),
        TensorInput::i32("length", &length_shape, &length),
    ])?;

    println!(
        "path-based NPU inference completed in {:?}: {} {:?}, {} {:?}",
        started.elapsed(),
        outputs[0].name,
        outputs[0].dimensions,
        outputs[1].name,
        outputs[1].dimensions
    );
    Ok(())
}
