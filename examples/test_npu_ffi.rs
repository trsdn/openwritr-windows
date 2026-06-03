//! Standalone reproducer for the QnnHtp crash that hits openwritr.exe but
//! not the Python validator at scripts/test_npu_encoder.py.
//!
//! Run from the project root with the ORT/QNN DLLs already staged in
//! target/release (i.e. after `cargo run --release --bin package` once):
//!
//!     cargo run --release --example test_npu_ffi
//!
//! Expected behaviour:
//!   - Steps 1..4 log cleanly
//!   - CreateSessionFromArray either succeeds (then we know the crash is
//!     caused by some other subsystem in the main app) or crashes with
//!     STATUS_STACK_BUFFER_OVERRUN (then we have a minimal Rust repro).

use std::ffi::CString;
use std::path::PathBuf;
use std::ptr;

use anyhow::{anyhow, bail, Context, Result};
use ort::sys;

fn check(api: &sys::OrtApi, status: sys::OrtStatusPtr, ctx: &'static str) -> Result<()> {
    if status.0.is_null() { return Ok(()); }
    unsafe {
        let msg_ptr = (api.GetErrorMessage)(status.0);
        let msg = std::ffi::CStr::from_ptr(msg_ptr).to_string_lossy().into_owned();
        (api.ReleaseStatus)(status.0);
        Err(anyhow!("{ctx}: {msg}"))
    }
}

fn main() -> Result<()> {
    // 0) Resolve ORT_DYLIB_PATH like the main app does.
    let exe = std::env::current_exe().context("current_exe")?;
    let exe_dir = exe.parent().ok_or_else(|| anyhow!("no exe parent"))?;
    // examples/ binaries land in target/release/examples/ — the DLLs are one
    // level up in target/release/.
    let dll_dir = if exe_dir.ends_with("examples") {
        exe_dir.parent().unwrap()
    } else {
        exe_dir
    }.to_path_buf();
    let ort_dll = dll_dir.join("onnxruntime.dll");
    if !ort_dll.exists() {
        bail!("onnxruntime.dll not found at {} — run `cargo run --release --bin package` first", ort_dll.display());
    }
    std::env::set_var("ORT_DYLIB_PATH", &ort_dll);
    println!("ORT_DYLIB_PATH = {}", ort_dll.display());

    // 0a) Force the DLL directory + prepend PATH so EVERY LoadLibrary call
    //     QnnHtp issues finds its siblings, regardless of which loader-flag
    //     it uses. AddDllDirectory only affects LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
    //     calls; SetDllDirectory is the legacy-PATH override that hooks all
    //     LoadLibrary calls. Both, plus PATH prepend, belt-and-suspenders.
    #[cfg(windows)]
    unsafe {
        use std::os::windows::ffi::OsStrExt;
        use windows::core::PCWSTR;
        use windows::Win32::System::LibraryLoader::{
            AddDllDirectory, LoadLibraryW, SetDllDirectoryW,
        };
        let dir_w: Vec<u16> = dll_dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let _ = AddDllDirectory(PCWSTR(dir_w.as_ptr()));
        let _ = SetDllDirectoryW(PCWSTR(dir_w.as_ptr()));
        let path = std::env::var("PATH").unwrap_or_default();
        std::env::set_var("PATH", format!("{};{}", dll_dir.display(), path));
        println!("PATH prepended with {}", dll_dir.display());

        // Preload every Qnn DLL we ship — make sure they're all in-memory
        // before QnnHtp's internal init tries to resolve them by name.
        use windows::Win32::Foundation::GetLastError;
        for dll in [
            "QnnSystem.dll",
            "QnnHtpPrepare.dll",
            "QnnHtp.dll",
            "QnnHtpV68Stub.dll",
            "QnnHtpV73Stub.dll",
            "QnnHtpV81Stub.dll",
            "QnnHtpNetRunExtensions.dll",
            "QnnCpu.dll",
            "QnnGpu.dll",
        ] {
            let p = dll_dir.join(dll);
            if !p.exists() {
                println!("preload {dll}: (file missing)");
                continue;
            }
            let p_w: Vec<u16> = p.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
            match LoadLibraryW(PCWSTR(p_w.as_ptr())) {
                Ok(_) => println!("preload {dll}: ok"),
                Err(_) => {
                    let err = GetLastError().0;
                    println!("preload {dll}: FAIL err={err} (0x{err:x})");
                }
            }
        }
    }

    // 1) Bring up the ort Environment so we have an OrtEnv and the QNN plugin EP registered.
    let _ = ort::init().with_name("test_npu_ffi").commit();
    let env = ort::environment::Environment::current()
        .map_err(|e| anyhow!("Environment::current: {e:?}"))?;

    let qnn_plugin = dll_dir.join("onnxruntime_providers_qnn.dll");
    if !qnn_plugin.exists() {
        bail!("missing {}", qnn_plugin.display());
    }
    env.register_ep_library("QNNExecutionProvider", &qnn_plugin)
        .map_err(|e| anyhow!("register_ep_library: {e:?}"))?;
    println!("QNN EP library registered");

    use ort::AsPointer;
    let env_ptr = env.ptr();
    let api = ort::api();

    // 2) Enumerate NPU devices that the QNN EP claims.
    let mut devs_ptr: *const *const sys::OrtEpDevice = ptr::null();
    let mut n: usize = 0;
    unsafe {
        check(api, (api.GetEpDevices)(env_ptr, &mut devs_ptr, &mut n), "GetEpDevices")?;
    }
    let devs = unsafe { std::slice::from_raw_parts(devs_ptr, n) };
    let mut qnn_npu_devs: Vec<*const sys::OrtEpDevice> = Vec::new();
    for &dev in devs {
        unsafe {
            let ep_name = std::ffi::CStr::from_ptr((api.EpDevice_EpName)(dev))
                .to_string_lossy();
            let hw = (api.EpDevice_Device)(dev);
            let ty = (api.HardwareDevice_Type)(hw);
            println!("device: ep={ep_name} hw_type={:?}", ty);
            if ty == sys::OrtHardwareDeviceType::OrtHardwareDeviceType_NPU
                && ep_name == "QNNExecutionProvider"
            {
                qnn_npu_devs.push(dev);
            }
        }
    }
    if qnn_npu_devs.is_empty() {
        bail!("no QNN NPU device available");
    }
    println!("QNN NPU device count: {}", qnn_npu_devs.len());

    // 3) Build SessionOptions + append QNN EP via V2 API.
    let mut so: *mut sys::OrtSessionOptions = ptr::null_mut();
    unsafe {
        check(api, (api.CreateSessionOptions)(&mut so), "CreateSessionOptions")?;
        check(api, (api.SetSessionGraphOptimizationLevel)(
            so, sys::GraphOptimizationLevel::ORT_ENABLE_ALL,
        ), "SetGraphOpt")?;

        let perf_key = CString::new("htp_performance_mode")?;
        let perf_val = CString::new("burst")?;
        let key_ptrs = [perf_key.as_ptr()];
        let val_ptrs = [perf_val.as_ptr()];
        check(api, (api.SessionOptionsAppendExecutionProvider_V2)(
            so,
            env_ptr as *mut sys::OrtEnv,
            qnn_npu_devs.as_ptr(),
            qnn_npu_devs.len(),
            key_ptrs.as_ptr(),
            val_ptrs.as_ptr(),
            1,
        ), "AppendExecutionProvider_V2")?;
    }
    println!("session options ready");

    // 4) Load the wrapper bytes + chdir to the model dir.
    let model: PathBuf = std::env::var_os("OPENWRITR_NPU_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let mut p = PathBuf::from(std::env::var_os("LOCALAPPDATA").expect("LOCALAPPDATA"));
            p.push("OpenWritr/models/parakeet-tdt-0.6b-v3-htp-int8-8s/encoder-model.onnx");
            p
        });
    if !model.exists() {
        bail!("missing model wrapper: {} — set OPENWRITR_NPU_MODEL to override", model.display());
    }
    let bytes = std::fs::read(&model)
        .with_context(|| format!("read {}", model.display()))?;
    let model_dir = model.parent().unwrap();
    let saved = std::env::current_dir().context("getcwd")?;
    std::env::set_current_dir(model_dir).with_context(|| format!("chdir {}", model_dir.display()))?;
    println!("model: {} ({} bytes), chdir={}", model.display(), bytes.len(), model_dir.display());

    // 5) THE call that crashes openwritr.exe.
    println!("--- about to call CreateSessionFromArray ---");
    let mut session: *mut sys::OrtSession = ptr::null_mut();
    let result = unsafe {
        check(api, (api.CreateSessionFromArray)(
            env_ptr,
            bytes.as_ptr().cast(),
            bytes.len(),
            so,
            &mut session,
        ), "CreateSessionFromArray")
    };
    let _ = std::env::set_current_dir(&saved);
    result?;
    println!("CreateSessionFromArray OK, session = {:?}", session);

    // 6) Tiny run to prove inference works.
    let mut mem_info: *mut sys::OrtMemoryInfo = ptr::null_mut();
    unsafe {
        check(api, (api.CreateCpuMemoryInfo)(
            sys::OrtAllocatorType::OrtArenaAllocator,
            sys::OrtMemType::OrtMemTypeDefault,
            &mut mem_info,
        ), "CreateCpuMemoryInfo")?;
    }

    let mel_bins = 128;
    let t_fixed = 801;
    let audio: Vec<f32> = vec![0.0; mel_bins * t_fixed];
    let audio_shape: [i64; 3] = [1, mel_bins as i64, t_fixed as i64];
    let length_buf = [t_fixed as i32];
    let length_shape: [i64; 1] = [1];

    let in_name1 = CString::new("audio_signal")?;
    let in_name2 = CString::new("length")?;
    let out_name1 = CString::new("output_0")?;
    let out_name2 = CString::new("output_1")?;

    unsafe {
        let mut audio_val: *mut sys::OrtValue = ptr::null_mut();
        check(api, (api.CreateTensorWithDataAsOrtValue)(
            mem_info, audio.as_ptr() as *mut _,
            audio.len() * std::mem::size_of::<f32>(),
            audio_shape.as_ptr(), 3,
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut audio_val,
        ), "CreateTensor audio")?;
        let mut length_val: *mut sys::OrtValue = ptr::null_mut();
        check(api, (api.CreateTensorWithDataAsOrtValue)(
            mem_info, length_buf.as_ptr() as *mut _,
            std::mem::size_of::<i32>(),
            length_shape.as_ptr(), 1,
            sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            &mut length_val,
        ), "CreateTensor length")?;

        let input_names = [in_name1.as_ptr(), in_name2.as_ptr()];
        let input_values = [audio_val as *const sys::OrtValue, length_val as *const sys::OrtValue];
        let output_names = [out_name1.as_ptr(), out_name2.as_ptr()];
        let mut output_values: [*mut sys::OrtValue; 2] = [ptr::null_mut(), ptr::null_mut()];

        let t0 = std::time::Instant::now();
        check(api, (api.Run)(
            session, ptr::null(),
            input_names.as_ptr(), input_values.as_ptr(), 2,
            output_names.as_ptr(), 2, output_values.as_mut_ptr(),
        ), "Run")?;
        let dt = t0.elapsed();
        println!("Run OK in {:?}", dt);

        (api.ReleaseValue)(audio_val);
        (api.ReleaseValue)(length_val);
        (api.ReleaseValue)(output_values[0]);
        (api.ReleaseValue)(output_values[1]);
    }

    unsafe {
        (api.ReleaseMemoryInfo)(mem_info);
        (api.ReleaseSession)(session);
        (api.ReleaseSessionOptions)(so);
    }
    println!("DONE — NPU FFI works in standalone reproducer");
    Ok(())
}
