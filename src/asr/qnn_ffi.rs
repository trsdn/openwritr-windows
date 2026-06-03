//! Direct C-API FFI for the QNN HTP encoder session.
//!
//! `ort` 2.0-rc.12's session builders crash with `STATUS_STACK_BUFFER_OVERRUN`
//! (`0xC0000409`, /GS-cookie corruption inside QnnHtp) when consuming an
//! EPContext-wrapper ONNX, regardless of API path or how the ONNX is fed in.
//! The exact same C-API call sequence works fine from Python at ~67 ms steady
//! state. So we bypass `ort`'s wrappers entirely for the NPU encoder and call
//! the ONNX Runtime C API directly through `ort_sys` (re-exported as
//! `ort::sys`).
//!
//! Scope is narrow: session creation + a single 2-in/2-out `Run`. The
//! preprocessor and TDT decoder still go through `ort::Session`.

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array3, ArrayView3};
use ort::sys;
use std::ffi::CString;
use std::path::Path;
use std::ptr;
use tracing::info;

pub struct NpuEncoderFfi {
    api: &'static sys::OrtApi,
    session: *mut sys::OrtSession,
    session_options: *mut sys::OrtSessionOptions,
    mem_info: *mut sys::OrtMemoryInfo,
    in_audio_name: CString,
    in_length_name: CString,
    out_features_name: CString,
    out_lens_name: CString,
}

unsafe impl Send for NpuEncoderFfi {}
unsafe impl Sync for NpuEncoderFfi {}

fn check(api: &sys::OrtApi, status: sys::OrtStatusPtr, ctx: &'static str) -> Result<()> {
    if status.0.is_null() {
        return Ok(());
    }
    unsafe {
        let msg_ptr = (api.GetErrorMessage)(status.0);
        let msg = std::ffi::CStr::from_ptr(msg_ptr).to_string_lossy().into_owned();
        (api.ReleaseStatus)(status.0);
        Err(anyhow!("{ctx}: {msg}"))
    }
}

impl NpuEncoderFfi {
    pub fn load(
        env_ptr: *const sys::OrtEnv,
        qnn_npu_devices: &[*const sys::OrtEpDevice],
        model_path: &Path,
    ) -> Result<Self> {
        if qnn_npu_devices.is_empty() {
            bail!("FFI: no QNN NPU device passed in");
        }
        // COM apartment hygiene: the engine-loader runs on a fresh worker
        // thread which has no COM apartment by default. QnnHtp's loader does
        // some COM-via-RPC calls internally; if the thread is uninitialized
        // when it tries to marshal, it asserts. Initialize MTA explicitly.
        #[cfg(windows)]
        unsafe {
            use windows::Win32::System::Com::{
                CoInitializeEx, COINIT_MULTITHREADED, COINIT_DISABLE_OLE1DDE,
            };
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED | COINIT_DISABLE_OLE1DDE);
        }
        let api = ort::api();
        unsafe {
            info!("FFI step 1: CreateSessionOptions");
            // 1) SessionOptions
            let mut so: *mut sys::OrtSessionOptions = ptr::null_mut();
            check(api, (api.CreateSessionOptions)(&mut so), "CreateSessionOptions")?;

            info!("FFI step 2: SetSessionGraphOptimizationLevel");
            check(api, (api.SetSessionGraphOptimizationLevel)(
                so, sys::GraphOptimizationLevel::ORT_ENABLE_ALL,
            ), "SetSessionGraphOptimizationLevel")?;

            // 2) Provider options as parallel C-string arrays.
            //    Note: with V2 API, prefixes are stripped; just pass plain keys.
            let perf_key = CString::new("htp_performance_mode")?;
            let perf_val = CString::new("burst")?;
            let key_ptrs = [perf_key.as_ptr()];
            let val_ptrs = [perf_val.as_ptr()];

            info!("FFI step 3: AppendExecutionProvider_V2 with {} devices", qnn_npu_devices.len());
            // 3) Append QNN EP for the chosen NPU device(s).
            check(api, (api.SessionOptionsAppendExecutionProvider_V2)(
                so,
                env_ptr as *mut sys::OrtEnv,
                qnn_npu_devices.as_ptr(),
                qnn_npu_devices.len(),
                key_ptrs.as_ptr(),
                val_ptrs.as_ptr(),
                1,
            ), "SessionOptionsAppendExecutionProvider_V2")?;

            // 4) Read the wrapper ONNX bytes + CreateSessionFromArray.
            //    ORT QNN EP resolves `ep_cache_context` relative to CWD, not
            //    relative to the model file. Temporarily chdir to the model
            //    directory so the sibling `.bin` is found, then restore.
            let bytes = std::fs::read(model_path)
                .with_context(|| format!("read {}", model_path.display()))?;
            let model_dir = model_path.parent()
                .ok_or_else(|| anyhow!("model path has no parent: {}", model_path.display()))?;
            let saved_cwd = std::env::current_dir().context("getcwd")?;
            std::env::set_current_dir(model_dir)
                .with_context(|| format!("chdir to {}", model_dir.display()))?;
            info!("FFI step 4: bytes_len={} chdir={}", bytes.len(), model_dir.display());
            info!("FFI step 5: CreateSessionFromArray (this is where it usually crashes)");
            let mut session: *mut sys::OrtSession = ptr::null_mut();
            let create_result = check(api, (api.CreateSessionFromArray)(
                env_ptr,
                bytes.as_ptr().cast(),
                bytes.len(),
                so,
                &mut session,
            ), "CreateSessionFromArray");
            // Always restore CWD, regardless of outcome.
            let _ = std::env::set_current_dir(&saved_cwd);
            create_result?;

            // 5) Reusable MemoryInfo for input tensors (CPU side).
            let mut mem_info: *mut sys::OrtMemoryInfo = ptr::null_mut();
            check(api, (api.CreateCpuMemoryInfo)(
                sys::OrtAllocatorType::OrtArenaAllocator,
                sys::OrtMemType::OrtMemTypeDefault,
                &mut mem_info,
            ), "CreateCpuMemoryInfo")?;

            info!("FFI NPU session created");
            Ok(Self {
                api,
                session,
                session_options: so,
                mem_info,
                in_audio_name: CString::new("audio_signal")?,
                in_length_name: CString::new("length")?,
                out_features_name: CString::new("output_0")?,
                out_lens_name: CString::new("output_1")?,
            })
        }
    }

    pub fn run(
        &self,
        audio_signal: ArrayView3<f32>,
        length: i32,
    ) -> Result<(Array3<f32>, i32)> {
        let api = self.api;
        let (b, m, t) = audio_signal.dim();
        if b != 1 {
            bail!("FFI: expected batch 1, got {b}");
        }
        let audio_shape: [i64; 3] = [b as i64, m as i64, t as i64];
        let audio_contig = audio_signal.as_standard_layout();
        let audio_bytes = audio_contig.len() * std::mem::size_of::<f32>();

        let length_buf = [length];
        let length_shape: [i64; 1] = [1];

        unsafe {
            let mut audio_value: *mut sys::OrtValue = ptr::null_mut();
            check(api, (api.CreateTensorWithDataAsOrtValue)(
                self.mem_info,
                audio_contig.as_ptr() as *mut _,
                audio_bytes,
                audio_shape.as_ptr(),
                3,
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                &mut audio_value,
            ), "CreateTensor audio_signal")?;

            let mut length_value: *mut sys::OrtValue = ptr::null_mut();
            check(api, (api.CreateTensorWithDataAsOrtValue)(
                self.mem_info,
                length_buf.as_ptr() as *mut _,
                std::mem::size_of::<i32>(),
                length_shape.as_ptr(),
                1,
                sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                &mut length_value,
            ), "CreateTensor length")?;

            let input_names = [self.in_audio_name.as_ptr(), self.in_length_name.as_ptr()];
            let input_values = [
                audio_value as *const sys::OrtValue,
                length_value as *const sys::OrtValue,
            ];
            let output_names = [self.out_features_name.as_ptr(), self.out_lens_name.as_ptr()];
            let mut output_values: [*mut sys::OrtValue; 2] = [ptr::null_mut(), ptr::null_mut()];

            check(api, (api.Run)(
                self.session,
                ptr::null(),
                input_names.as_ptr(),
                input_values.as_ptr(),
                2,
                output_names.as_ptr(),
                2,
                output_values.as_mut_ptr(),
            ), "Run")?;

            let out0 = output_values[0];
            let out0_arr = extract_f32_3d(api, out0)?;
            let out1 = output_values[1];
            let mut out1_data: *mut std::ffi::c_void = ptr::null_mut();
            check(api, (api.GetTensorMutableData)(out1, &mut out1_data),
                  "GetTensorMutableData out1")?;
            let encoded_len: i32 = *(out1_data as *const i32);

            (api.ReleaseValue)(audio_value);
            (api.ReleaseValue)(length_value);
            (api.ReleaseValue)(out0);
            (api.ReleaseValue)(out1);

            Ok((out0_arr, encoded_len))
        }
    }
}

unsafe fn extract_f32_3d(api: &sys::OrtApi, val: *mut sys::OrtValue) -> Result<Array3<f32>> {
    let mut shape_info: *mut sys::OrtTensorTypeAndShapeInfo = ptr::null_mut();
    check(api, (api.GetTensorTypeAndShape)(val, &mut shape_info),
          "GetTensorTypeAndShape")?;
    let mut dim_count: usize = 0;
    check(api, (api.GetDimensionsCount)(shape_info, &mut dim_count),
          "GetDimensionsCount")?;
    if dim_count != 3 {
        (api.ReleaseTensorTypeAndShapeInfo)(shape_info);
        bail!("expected rank 3 output, got {dim_count}");
    }
    let mut dims = vec![0i64; dim_count];
    check(api, (api.GetDimensions)(shape_info, dims.as_mut_ptr(), dim_count),
          "GetDimensions")?;
    (api.ReleaseTensorTypeAndShapeInfo)(shape_info);

    let (b, c, t) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
    let n = b.checked_mul(c).and_then(|x| x.checked_mul(t))
        .ok_or_else(|| anyhow!("output shape overflow"))?;

    let mut data: *mut std::ffi::c_void = ptr::null_mut();
    check(api, (api.GetTensorMutableData)(val, &mut data),
          "GetTensorMutableData out0")?;

    let slice = std::slice::from_raw_parts(data as *const f32, n);
    Array3::from_shape_vec((b, c, t), slice.to_vec())
        .map_err(|e| anyhow!("shape mismatch: {e}"))
}

impl Drop for NpuEncoderFfi {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                (self.api.ReleaseSession)(self.session);
            }
            if !self.session_options.is_null() {
                (self.api.ReleaseSessionOptions)(self.session_options);
            }
            if !self.mem_info.is_null() {
                (self.api.ReleaseMemoryInfo)(self.mem_info);
            }
        }
    }
}

pub fn enumerate_qnn_npu_devices(env_ptr: *const sys::OrtEnv) -> Result<Vec<*const sys::OrtEpDevice>> {
    let api = ort::api();
    unsafe {
        let mut devs_ptr: *const *const sys::OrtEpDevice = ptr::null();
        let mut n: usize = 0;
        check(api, (api.GetEpDevices)(env_ptr, &mut devs_ptr, &mut n),
              "GetEpDevices")?;
        let slice = std::slice::from_raw_parts(devs_ptr, n);
        let mut out = Vec::new();
        for &dev in slice {
            let ep_name_ptr = (api.EpDevice_EpName)(dev);
            let ep_name = std::ffi::CStr::from_ptr(ep_name_ptr).to_string_lossy();
            let hw = (api.EpDevice_Device)(dev);
            let ty = (api.HardwareDevice_Type)(hw);
            if ty == sys::OrtHardwareDeviceType::OrtHardwareDeviceType_NPU
                && ep_name == "QNNExecutionProvider"
            {
                out.push(dev);
            }
        }
        Ok(out)
    }
}
