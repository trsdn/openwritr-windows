//! Reusable ONNX Runtime C-API session layer for QNN execution-provider models.

use anyhow::{anyhow, bail, Context, Result};
use once_cell::sync::Lazy;
use ort::sys;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::ptr::{self, NonNull};
use tracing::{info, warn};

const EXPECTED_ORT_CRATE: &str = "2.0.0-rc.12";
const EXPECTED_ORT_NATIVE: &str = "1.24.4";
const EXPECTED_QNN_PACKAGE: &str = "2.1.1";
const EXPECTED_QAIRT: &str = "2.45.41";
const COMMON_RUNTIME_FILES: &[&str] = &["onnxruntime.dll"];
const ARM64_QNN_RUNTIME_FILES: &[&str] = &[
    "onnxruntime_providers_qnn.dll",
    "QnnHtp.dll",
    "QnnHtpPrepare.dll",
    "QnnHtpNetRunExtensions.dll",
    "QnnHtpV73Stub.dll",
    "libQnnHtpV73Skel.so",
    "libqnnhtpv73.cat",
    "QnnSystem.dll",
];

type ReleaseFn<T> = unsafe extern "system" fn(*mut T);

struct OrtHandle<T> {
    ptr: NonNull<T>,
    release: ReleaseFn<T>,
}

impl<T> OrtHandle<T> {
    unsafe fn from_raw(ptr: *mut T, release: ReleaseFn<T>, context: &str) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(ptr)
                .ok_or_else(|| anyhow!("{context} returned a null handle without an error"))?,
            release,
        })
    }

    fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for OrtHandle<T> {
    fn drop(&mut self) {
        unsafe {
            (self.release)(self.ptr.as_ptr());
        }
    }
}

fn create_handle<T>(
    api: &'static sys::OrtApi,
    release: ReleaseFn<T>,
    context: &'static str,
    create: impl FnOnce(*mut *mut T) -> sys::OrtStatusPtr,
) -> Result<OrtHandle<T>> {
    let mut ptr = ptr::null_mut();
    let result = check(api, create(&mut ptr), context);
    finish_created_handle(ptr, release, context, result)
}

fn finish_created_handle<T>(
    ptr: *mut T,
    release: ReleaseFn<T>,
    context: &str,
    result: Result<()>,
) -> Result<OrtHandle<T>> {
    if let Err(error) = result {
        if let Some(ptr) = NonNull::new(ptr) {
            unsafe {
                release(ptr.as_ptr());
            }
        }
        return Err(error);
    }
    unsafe { OrtHandle::from_raw(ptr, release, context) }
}

fn check(api: &sys::OrtApi, status: sys::OrtStatusPtr, context: &str) -> Result<()> {
    if status.0.is_null() {
        return Ok(());
    }
    unsafe {
        let message = CStr::from_ptr((api.GetErrorMessage)(status.0))
            .to_string_lossy()
            .into_owned();
        (api.ReleaseStatus)(status.0);
        Err(anyhow!("{context}: {message}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorElementType {
    F32,
    F16,
    I32,
    I64,
    U8,
}

impl TensorElementType {
    fn to_onnx(self) -> sys::ONNXTensorElementDataType {
        use sys::ONNXTensorElementDataType as OrtType;
        match self {
            Self::F32 => OrtType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            Self::F16 => OrtType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            Self::I32 => OrtType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            Self::I64 => OrtType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            Self::U8 => OrtType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
        }
    }

    fn from_onnx(value: sys::ONNXTensorElementDataType) -> Result<Self> {
        use sys::ONNXTensorElementDataType as OrtType;
        match value {
            OrtType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => Ok(Self::F32),
            OrtType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => Ok(Self::F16),
            OrtType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => Ok(Self::I32),
            OrtType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => Ok(Self::I64),
            OrtType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => Ok(Self::U8),
            other => bail!("unsupported tensor element type {other:?}"),
        }
    }

    fn byte_width(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 => 2,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub element_type: TensorElementType,
    pub dimensions: Vec<i64>,
}

impl TensorSpec {
    pub fn new(
        name: impl Into<String>,
        element_type: TensorElementType,
        dimensions: impl Into<Vec<i64>>,
    ) -> Self {
        Self {
            name: name.into(),
            element_type,
            dimensions: dimensions.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SessionContract {
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
}

impl SessionContract {
    pub fn new(inputs: Vec<TensorSpec>, outputs: Vec<TensorSpec>) -> Result<Self> {
        validate_contract_definition(&inputs, "input")?;
        validate_contract_definition(&outputs, "output")?;
        Ok(Self { inputs, outputs })
    }
}

#[allow(dead_code)]
pub enum TensorData<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    U8(&'a [u8]),
}

impl TensorData<'_> {
    fn element_type(&self) -> TensorElementType {
        match self {
            Self::F32(_) => TensorElementType::F32,
            Self::F16(_) => TensorElementType::F16,
            Self::I32(_) => TensorElementType::I32,
            Self::I64(_) => TensorElementType::I64,
            Self::U8(_) => TensorElementType::U8,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F16(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U8(values) => values.len(),
        }
    }

    fn as_mut_ptr(&self) -> *mut std::ffi::c_void {
        match self {
            Self::F32(values) => values.as_ptr().cast_mut().cast(),
            Self::F16(values) => values.as_ptr().cast_mut().cast(),
            Self::I32(values) => values.as_ptr().cast_mut().cast(),
            Self::I64(values) => values.as_ptr().cast_mut().cast(),
            Self::U8(values) => values.as_ptr().cast_mut().cast(),
        }
    }
}

pub struct TensorInput<'a> {
    pub name: &'a str,
    pub dimensions: &'a [i64],
    pub data: TensorData<'a>,
}

#[allow(dead_code)]
impl<'a> TensorInput<'a> {
    pub fn f32(name: &'a str, dimensions: &'a [i64], data: &'a [f32]) -> Self {
        Self {
            name,
            dimensions,
            data: TensorData::F32(data),
        }
    }

    pub fn i32(name: &'a str, dimensions: &'a [i64], data: &'a [i32]) -> Self {
        Self {
            name,
            dimensions,
            data: TensorData::I32(data),
        }
    }

    pub fn f16(name: &'a str, dimensions: &'a [i64], data: &'a [u16]) -> Self {
        Self {
            name,
            dimensions,
            data: TensorData::F16(data),
        }
    }

    pub fn i64(name: &'a str, dimensions: &'a [i64], data: &'a [i64]) -> Self {
        Self {
            name,
            dimensions,
            data: TensorData::I64(data),
        }
    }

    pub fn u8(name: &'a str, dimensions: &'a [i64], data: &'a [u8]) -> Self {
        Self {
            name,
            dimensions,
            data: TensorData::U8(data),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum TensorDataOwned {
    F32(Vec<f32>),
    F16(Vec<u16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
}

#[derive(Debug)]
pub struct TensorOutput {
    pub name: String,
    pub dimensions: Vec<i64>,
    pub data: TensorDataOwned,
}

#[allow(dead_code)]
impl TensorOutput {
    pub fn into_f32(self) -> Result<(Vec<i64>, Vec<f32>)> {
        match self.data {
            TensorDataOwned::F32(values) => Ok((self.dimensions, values)),
            other => bail!("output {} is not f32: {other:?}", self.name),
        }
    }

    pub fn into_i32(self) -> Result<(Vec<i64>, Vec<i32>)> {
        match self.data {
            TensorDataOwned::I32(values) => Ok((self.dimensions, values)),
            other => bail!("output {} is not i32: {other:?}", self.name),
        }
    }

    pub fn into_f16(self) -> Result<(Vec<i64>, Vec<u16>)> {
        match self.data {
            TensorDataOwned::F16(values) => Ok((self.dimensions, values)),
            other => bail!("output {} is not f16: {other:?}", self.name),
        }
    }

    pub fn into_i64(self) -> Result<(Vec<i64>, Vec<i64>)> {
        match self.data {
            TensorDataOwned::I64(values) => Ok((self.dimensions, values)),
            other => bail!("output {} is not i64: {other:?}", self.name),
        }
    }

    pub fn into_u8(self) -> Result<(Vec<i64>, Vec<u8>)> {
        match self.data {
            TensorDataOwned::U8(values) => Ok((self.dimensions, values)),
            other => bail!("output {} is not u8: {other:?}", self.name),
        }
    }
}

pub struct QnnSession {
    api: &'static sys::OrtApi,
    session: OrtHandle<sys::OrtSession>,
    memory_info: OrtHandle<sys::OrtMemoryInfo>,
    contract: SessionContract,
    input_names: Vec<CString>,
    output_names: Vec<CString>,
    // Field order is intentional: the ORT session must be released before
    // the final provider lease unregisters and unloads the QNN EP library.
    _provider_lease: QnnProviderLease,
}

// The inference worker moves a session onto its dedicated thread and calls it
// serially. QnnSession deliberately does not implement Sync.
unsafe impl Send for QnnSession {}

impl QnnSession {
    pub fn load(
        env_ptr: *const sys::OrtEnv,
        qnn_npu_devices: &[*const sys::OrtEpDevice],
        model_path: &Path,
        contract: SessionContract,
        provider_lease: QnnProviderLease,
    ) -> Result<Self> {
        if qnn_npu_devices.is_empty() {
            bail!("no QNN NPU device was provided");
        }
        ensure_com_apartment()?;
        let api = ort::api();
        let session_options = create_handle(
            api,
            api.ReleaseSessionOptions,
            "CreateSessionOptions",
            |out| unsafe { (api.CreateSessionOptions)(out) },
        )?;
        check(
            api,
            unsafe {
                (api.SetSessionGraphOptimizationLevel)(
                    session_options.as_ptr(),
                    sys::GraphOptimizationLevel::ORT_ENABLE_ALL,
                )
            },
            "SetSessionGraphOptimizationLevel",
        )?;

        let performance_key = CString::new("htp_performance_mode")?;
        let performance_value = CString::new("burst")?;
        let keys = [performance_key.as_ptr()];
        let values = [performance_value.as_ptr()];
        check(
            api,
            unsafe {
                (api.SessionOptionsAppendExecutionProvider_V2)(
                    session_options.as_ptr(),
                    env_ptr.cast_mut(),
                    qnn_npu_devices.as_ptr(),
                    qnn_npu_devices.len(),
                    keys.as_ptr(),
                    values.as_ptr(),
                    1,
                )
            },
            "SessionOptionsAppendExecutionProvider_V2",
        )?;

        let model_path = path_to_ort_chars(model_path)?;
        info!(model = %model_path_display(model_path.as_slice()), "creating path-based QNN session");
        let session = create_handle(api, api.ReleaseSession, "CreateSession", |out| unsafe {
            (api.CreateSession)(env_ptr, model_path.as_ptr(), session_options.as_ptr(), out)
        })?;
        validate_session_contract(api, session.as_ptr(), &contract)?;
        let memory_info = create_handle(
            api,
            api.ReleaseMemoryInfo,
            "CreateCpuMemoryInfo",
            |out| unsafe {
                (api.CreateCpuMemoryInfo)(
                    sys::OrtAllocatorType::OrtArenaAllocator,
                    sys::OrtMemType::OrtMemTypeDefault,
                    out,
                )
            },
        )?;
        let input_names = contract
            .inputs
            .iter()
            .map(|spec| CString::new(spec.name.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let output_names = contract
            .outputs
            .iter()
            .map(|spec| CString::new(spec.name.as_str()))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(Self {
            api,
            session,
            memory_info,
            contract,
            input_names,
            output_names,
            _provider_lease: provider_lease,
        })
    }

    pub fn run(&mut self, inputs: &[TensorInput<'_>]) -> Result<Vec<TensorOutput>> {
        ensure_com_apartment()?;
        validate_inputs(&self.contract.inputs, inputs)?;
        let mut input_values = Vec::with_capacity(self.contract.inputs.len());
        for spec in &self.contract.inputs {
            let input = inputs
                .iter()
                .find(|input| input.name == spec.name)
                .expect("validated input is missing");
            input_values.push(self.create_input_value(input)?);
        }

        let input_name_ptrs = self
            .input_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let input_value_ptrs = input_values
            .iter()
            .map(|value| value.as_ptr().cast_const())
            .collect::<Vec<_>>();
        let output_name_ptrs = self
            .output_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let mut output_slots = OutputSlots::new(self.api, self.contract.outputs.len());
        let status = unsafe {
            (self.api.Run)(
                self.session.as_ptr(),
                ptr::null(),
                input_name_ptrs.as_ptr(),
                input_value_ptrs.as_ptr(),
                input_value_ptrs.len(),
                output_name_ptrs.as_ptr(),
                output_name_ptrs.len(),
                output_slots.as_mut_ptr(),
            )
        };
        check(self.api, status, "Run")?;

        let mut outputs = Vec::with_capacity(self.contract.outputs.len());
        for (index, spec) in self.contract.outputs.iter().enumerate() {
            let value = output_slots.take(index)?;
            outputs.push(extract_output(self.api, &value, spec)?);
        }
        Ok(outputs)
    }

    fn create_input_value(&self, input: &TensorInput<'_>) -> Result<OrtHandle<sys::OrtValue>> {
        let elements = element_count(input.dimensions)?;
        if elements != input.data.len() {
            bail!(
                "input {} shape {:?} requires {elements} elements, got {}",
                input.name,
                input.dimensions,
                input.data.len()
            );
        }
        let bytes = elements
            .checked_mul(input.data.element_type().byte_width())
            .ok_or_else(|| anyhow!("input {} byte size overflow", input.name))?;
        create_handle(
            self.api,
            self.api.ReleaseValue,
            "CreateTensorWithDataAsOrtValue",
            |out| unsafe {
                (self.api.CreateTensorWithDataAsOrtValue)(
                    self.memory_info.as_ptr(),
                    input.data.as_mut_ptr(),
                    bytes,
                    input.dimensions.as_ptr(),
                    input.dimensions.len(),
                    input.data.element_type().to_onnx(),
                    out,
                )
            },
        )
        .with_context(|| format!("create input tensor {}", input.name))
    }
}

struct OutputSlots {
    api: &'static sys::OrtApi,
    values: Vec<*mut sys::OrtValue>,
}

impl OutputSlots {
    fn new(api: &'static sys::OrtApi, count: usize) -> Self {
        Self {
            api,
            values: vec![ptr::null_mut(); count],
        }
    }

    fn as_mut_ptr(&mut self) -> *mut *mut sys::OrtValue {
        self.values.as_mut_ptr()
    }

    fn take(&mut self, index: usize) -> Result<OrtHandle<sys::OrtValue>> {
        let ptr = std::mem::replace(&mut self.values[index], ptr::null_mut());
        unsafe { OrtHandle::from_raw(ptr, self.api.ReleaseValue, "Run output") }
    }
}

impl Drop for OutputSlots {
    fn drop(&mut self) {
        for value in self.values.drain(..).filter(|value| !value.is_null()) {
            unsafe {
                (self.api.ReleaseValue)(value);
            }
        }
    }
}

fn extract_output(
    api: &'static sys::OrtApi,
    value: &OrtHandle<sys::OrtValue>,
    spec: &TensorSpec,
) -> Result<TensorOutput> {
    let shape_info = create_handle(
        api,
        api.ReleaseTensorTypeAndShapeInfo,
        "GetTensorTypeAndShape",
        |out| unsafe { (api.GetTensorTypeAndShape)(value.as_ptr(), out) },
    )?;
    let (element_type, dimensions) = tensor_metadata(api, shape_info.as_ptr())?;
    if element_type != spec.element_type {
        bail!(
            "output {} type mismatch: expected {:?}, got {:?}",
            spec.name,
            spec.element_type,
            element_type
        );
    }
    if !shape_matches(&spec.dimensions, &dimensions) {
        bail!(
            "output {} shape mismatch: expected {:?}, got {:?}",
            spec.name,
            spec.dimensions,
            dimensions
        );
    }
    let elements = element_count(&dimensions)?;
    let data = if elements == 0 {
        copy_tensor_data(element_type, ptr::null_mut(), 0)?
    } else {
        let mut data = ptr::null_mut();
        check(
            api,
            unsafe { (api.GetTensorMutableData)(value.as_ptr(), &mut data) },
            "GetTensorMutableData",
        )?;
        copy_tensor_data(element_type, data, elements)
            .with_context(|| format!("read output {}", spec.name))?
    };
    Ok(TensorOutput {
        name: spec.name.clone(),
        dimensions,
        data,
    })
}

fn copy_tensor_data(
    element_type: TensorElementType,
    data: *mut std::ffi::c_void,
    elements: usize,
) -> Result<TensorDataOwned> {
    if elements == 0 {
        return Ok(match element_type {
            TensorElementType::F32 => TensorDataOwned::F32(Vec::new()),
            TensorElementType::F16 => TensorDataOwned::F16(Vec::new()),
            TensorElementType::I32 => TensorDataOwned::I32(Vec::new()),
            TensorElementType::I64 => TensorDataOwned::I64(Vec::new()),
            TensorElementType::U8 => TensorDataOwned::U8(Vec::new()),
        });
    }
    if data.is_null() {
        bail!("tensor returned a null data pointer");
    }
    unsafe {
        Ok(match element_type {
            TensorElementType::F32 => {
                TensorDataOwned::F32(std::slice::from_raw_parts(data.cast(), elements).to_vec())
            }
            TensorElementType::F16 => {
                TensorDataOwned::F16(std::slice::from_raw_parts(data.cast(), elements).to_vec())
            }
            TensorElementType::I32 => {
                TensorDataOwned::I32(std::slice::from_raw_parts(data.cast(), elements).to_vec())
            }
            TensorElementType::I64 => {
                TensorDataOwned::I64(std::slice::from_raw_parts(data.cast(), elements).to_vec())
            }
            TensorElementType::U8 => {
                TensorDataOwned::U8(std::slice::from_raw_parts(data.cast(), elements).to_vec())
            }
        })
    }
}

fn validate_inputs(expected: &[TensorSpec], inputs: &[TensorInput<'_>]) -> Result<()> {
    if expected.len() != inputs.len() {
        bail!(
            "input count mismatch: expected {}, got {}",
            expected.len(),
            inputs.len()
        );
    }
    let mut names = HashSet::new();
    for input in inputs {
        if !names.insert(input.name) {
            bail!("duplicate input {}", input.name);
        }
    }
    for spec in expected {
        let input = inputs
            .iter()
            .find(|input| input.name == spec.name)
            .ok_or_else(|| anyhow!("missing input {}", spec.name))?;
        if input.data.element_type() != spec.element_type {
            bail!(
                "input {} type mismatch: expected {:?}, got {:?}",
                spec.name,
                spec.element_type,
                input.data.element_type()
            );
        }
        if !shape_matches(&spec.dimensions, input.dimensions) {
            bail!(
                "input {} shape mismatch: expected {:?}, got {:?}",
                spec.name,
                spec.dimensions,
                input.dimensions
            );
        }
    }
    Ok(())
}

fn validate_contract_definition(specs: &[TensorSpec], kind: &str) -> Result<()> {
    let mut names = HashSet::new();
    for spec in specs {
        if spec.name.is_empty() {
            bail!("{kind} name cannot be empty");
        }
        CString::new(spec.name.as_str())
            .with_context(|| format!("{kind} name contains NUL: {}", spec.name))?;
        if !names.insert(spec.name.as_str()) {
            bail!("duplicate {kind} name {}", spec.name);
        }
        if spec.dimensions.iter().any(|dimension| *dimension == 0) {
            bail!("{kind} {} contains a zero dimension", spec.name);
        }
    }
    Ok(())
}

fn validate_session_contract(
    api: &'static sys::OrtApi,
    session: *mut sys::OrtSession,
    expected: &SessionContract,
) -> Result<()> {
    let actual_inputs = inspect_session_tensors(api, session, true)?;
    let actual_outputs = inspect_session_tensors(api, session, false)?;
    validate_specs(&expected.inputs, &actual_inputs, "input")?;
    validate_specs(&expected.outputs, &actual_outputs, "output")
}

fn validate_specs(expected: &[TensorSpec], actual: &[TensorSpec], kind: &str) -> Result<()> {
    if expected.len() != actual.len() {
        bail!(
            "{kind} count mismatch: expected {}, got {} ({actual:?})",
            expected.len(),
            actual.len()
        );
    }
    for spec in expected {
        let actual = actual
            .iter()
            .find(|actual| actual.name == spec.name)
            .ok_or_else(|| anyhow!("missing model {kind} {}", spec.name))?;
        if actual.element_type != spec.element_type {
            bail!(
                "model {kind} {} type mismatch: expected {:?}, got {:?}",
                spec.name,
                spec.element_type,
                actual.element_type
            );
        }
        if !shape_matches(&spec.dimensions, &actual.dimensions) {
            bail!(
                "model {kind} {} shape mismatch: expected {:?}, got {:?}",
                spec.name,
                spec.dimensions,
                actual.dimensions
            );
        }
    }
    Ok(())
}

fn inspect_session_tensors(
    api: &'static sys::OrtApi,
    session: *mut sys::OrtSession,
    inputs: bool,
) -> Result<Vec<TensorSpec>> {
    let mut count = 0;
    check(
        api,
        unsafe {
            if inputs {
                (api.SessionGetInputCount)(session, &mut count)
            } else {
                (api.SessionGetOutputCount)(session, &mut count)
            }
        },
        if inputs {
            "SessionGetInputCount"
        } else {
            "SessionGetOutputCount"
        },
    )?;
    let mut allocator = ptr::null_mut();
    check(
        api,
        unsafe { (api.GetAllocatorWithDefaultOptions)(&mut allocator) },
        "GetAllocatorWithDefaultOptions",
    )?;
    let allocator = NonNull::new(allocator).ok_or_else(|| anyhow!("default allocator is null"))?;
    let mut specs = Vec::with_capacity(count);
    for index in 0..count {
        let name = session_tensor_name(api, session, allocator, index, inputs)?;
        let type_info = create_handle(
            api,
            api.ReleaseTypeInfo,
            if inputs {
                "SessionGetInputTypeInfo"
            } else {
                "SessionGetOutputTypeInfo"
            },
            |out| unsafe {
                if inputs {
                    (api.SessionGetInputTypeInfo)(session, index, out)
                } else {
                    (api.SessionGetOutputTypeInfo)(session, index, out)
                }
            },
        )?;
        let mut onnx_type = sys::ONNXType::ONNX_TYPE_UNKNOWN;
        check(
            api,
            unsafe { (api.GetOnnxTypeFromTypeInfo)(type_info.as_ptr(), &mut onnx_type) },
            "GetOnnxTypeFromTypeInfo",
        )?;
        if onnx_type != sys::ONNXType::ONNX_TYPE_TENSOR {
            bail!("{name} is not a tensor: {onnx_type:?}");
        }
        let mut tensor_info = ptr::null();
        check(
            api,
            unsafe { (api.CastTypeInfoToTensorInfo)(type_info.as_ptr(), &mut tensor_info) },
            "CastTypeInfoToTensorInfo",
        )?;
        let tensor_info = NonNull::new(tensor_info.cast_mut())
            .ok_or_else(|| anyhow!("{name} tensor info is null"))?;
        let (element_type, dimensions) = tensor_metadata(api, tensor_info.as_ptr())?;
        specs.push(TensorSpec::new(name, element_type, dimensions));
    }
    Ok(specs)
}

fn session_tensor_name(
    api: &'static sys::OrtApi,
    session: *mut sys::OrtSession,
    allocator: NonNull<sys::OrtAllocator>,
    index: usize,
    input: bool,
) -> Result<String> {
    let mut name = ptr::null_mut();
    check(
        api,
        unsafe {
            if input {
                (api.SessionGetInputName)(session, index, allocator.as_ptr(), &mut name)
            } else {
                (api.SessionGetOutputName)(session, index, allocator.as_ptr(), &mut name)
            }
        },
        if input {
            "SessionGetInputName"
        } else {
            "SessionGetOutputName"
        },
    )?;
    let name = NonNull::new(name).ok_or_else(|| anyhow!("session tensor name is null"))?;
    let text = unsafe { CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    check(
        api,
        unsafe { (api.AllocatorFree)(allocator.as_ptr(), name.as_ptr().cast()) },
        "AllocatorFree tensor name",
    )?;
    Ok(text)
}

fn tensor_metadata(
    api: &'static sys::OrtApi,
    shape_info: *mut sys::OrtTensorTypeAndShapeInfo,
) -> Result<(TensorElementType, Vec<i64>)> {
    let mut element_type = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    check(
        api,
        unsafe { (api.GetTensorElementType)(shape_info, &mut element_type) },
        "GetTensorElementType",
    )?;
    let mut rank = 0;
    check(
        api,
        unsafe { (api.GetDimensionsCount)(shape_info, &mut rank) },
        "GetDimensionsCount",
    )?;
    let mut dimensions = vec![0; rank];
    check(
        api,
        unsafe { (api.GetDimensions)(shape_info, dimensions.as_mut_ptr(), rank) },
        "GetDimensions",
    )?;
    Ok((TensorElementType::from_onnx(element_type)?, dimensions))
}

fn shape_matches(expected: &[i64], actual: &[i64]) -> bool {
    expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(expected, actual)| *expected < 0 || *actual < 0 || expected == actual)
}

fn element_count(dimensions: &[i64]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |count, dimension| {
        let dimension = usize::try_from(*dimension)
            .map_err(|_| anyhow!("runtime tensor has negative dimension {dimension}"))?;
        count
            .checked_mul(dimension)
            .ok_or_else(|| anyhow!("tensor element count overflow"))
    })
}

#[cfg(windows)]
fn path_to_ort_chars(path: &Path) -> Result<Vec<sys::os_char>> {
    use std::os::windows::ffi::OsStrExt;
    Ok(path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect())
}

#[cfg(not(windows))]
fn path_to_ort_chars(path: &Path) -> Result<Vec<sys::os_char>> {
    use std::os::unix::ffi::OsStrExt;
    let path = CString::new(path.as_os_str().as_bytes())?;
    Ok(path
        .as_bytes_with_nul()
        .iter()
        .map(|byte| *byte as _)
        .collect())
}

#[cfg(windows)]
fn model_path_display(path: &[sys::os_char]) -> String {
    String::from_utf16_lossy(path.strip_suffix(&[0]).unwrap_or(path))
}

#[cfg(not(windows))]
fn model_path_display(path: &[sys::os_char]) -> String {
    String::from_utf8_lossy(
        &path
            .iter()
            .copied()
            .take_while(|byte| *byte != 0)
            .map(|byte| byte as u8)
            .collect::<Vec<_>>(),
    )
    .into_owned()
}

thread_local! {
    static COM_APARTMENT: std::result::Result<ComApartment, String> =
        ComApartment::initialize().map_err(|error| error.to_string());
}

struct ComApartment;

impl ComApartment {
    fn initialize() -> Result<Self> {
        #[cfg(windows)]
        unsafe {
            use windows::Win32::System::Com::{
                CoInitializeEx, COINIT_DISABLE_OLE1DDE, COINIT_MULTITHREADED,
            };
            CoInitializeEx(None, COINIT_MULTITHREADED | COINIT_DISABLE_OLE1DDE)
                .ok()
                .context("CoInitializeEx for QNN worker")?;
        }
        Ok(Self)
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            windows::Win32::System::Com::CoUninitialize();
        }
    }
}

fn ensure_com_apartment() -> Result<()> {
    COM_APARTMENT.with(|apartment| match apartment {
        Ok(_) => Ok(()),
        Err(error) => Err(anyhow!(error.clone())),
    })
}

static VERIFIED_RUNTIME: Lazy<Mutex<Option<PathBuf>>> = Lazy::new(|| Mutex::new(None));
static ORT_INITIALIZED: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

#[derive(Default)]
struct QnnProviderState {
    registered: bool,
    leases: usize,
}

static QNN_PROVIDER: Lazy<Mutex<QnnProviderState>> =
    Lazy::new(|| Mutex::new(QnnProviderState::default()));

pub struct QnnProviderLease {
    active: bool,
}

impl Drop for QnnProviderLease {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut state = QNN_PROVIDER.lock();
        if let Err(error) = release_provider_state(&mut state, unregister_qnn_provider) {
            warn!(%error, "failed to unregister QNN execution provider");
        }
        self.active = false;
    }
}

#[cfg(windows)]
struct LoadedModule(windows::Win32::Foundation::HMODULE);

// Windows module handles are process-wide and FreeLibrary may be called from
// any non-DllMain thread. The mutex serializes access to the retained handles.
#[cfg(windows)]
unsafe impl Send for LoadedModule {}

#[cfg(windows)]
impl Drop for LoadedModule {
    fn drop(&mut self) {
        unsafe {
            let _ = windows::Win32::Foundation::FreeLibrary(self.0);
        }
    }
}

#[cfg(windows)]
struct QnnPreload {
    _modules: Vec<LoadedModule>,
}

#[cfg(windows)]
static QNN_PRELOAD: Lazy<Mutex<Option<QnnPreload>>> = Lazy::new(|| Mutex::new(None));

pub fn acquire_qnn_provider(qnn_dll: &Path) -> Result<QnnProviderLease> {
    let runtime_dir = qnn_dll
        .parent()
        .ok_or_else(|| anyhow!("QNN provider path has no parent: {}", qnn_dll.display()))?;
    verify_runtime_contract(runtime_dir)?;
    preload_qnn_runtime(runtime_dir)?;
    let mut state = QNN_PROVIDER.lock();
    let was_registered = state.registered;
    acquire_provider_state(&mut state, || {
        if !qnn_dll.is_file() {
            bail!("{} not found", qnn_dll.display());
        }
        let environment = ort::environment::Environment::current()
            .map_err(|error| anyhow!("environment::current: {error}"))?;
        environment
            .register_ep_library("QNNExecutionProvider", qnn_dll)
            .map_err(|error| anyhow!("register QNN execution provider: {error}"))?;
        Ok(())
    })?;
    if !was_registered {
        info!(
            provider = %qnn_dll.display(),
            ort_crate = EXPECTED_ORT_CRATE,
            ort_native = EXPECTED_ORT_NATIVE,
            qnn_package = EXPECTED_QNN_PACKAGE,
            qairt = EXPECTED_QAIRT,
            ort_build = ort::info(),
            "QNN execution provider registered with pinned runtime"
        );
    }
    Ok(QnnProviderLease { active: true })
}

fn acquire_provider_state(
    state: &mut QnnProviderState,
    register: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if !state.registered {
        register()?;
        state.registered = true;
    }
    state.leases = state
        .leases
        .checked_add(1)
        .ok_or_else(|| anyhow!("QNN provider lease count overflow"))?;
    Ok(())
}

fn release_provider_state(
    state: &mut QnnProviderState,
    unregister: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if state.leases == 0 {
        bail!("QNN provider lease count underflow");
    }
    state.leases -= 1;
    if state.leases == 0 && state.registered {
        unregister()?;
        state.registered = false;
    }
    Ok(())
}

fn unregister_qnn_provider() -> Result<()> {
    use ort::AsPointer;
    let environment = ort::environment::Environment::current()
        .map_err(|error| anyhow!("environment::current: {error}"))?;
    let registration_name = CString::new("QNNExecutionProvider")?;
    check(
        ort::api(),
        unsafe {
            (ort::api().UnregisterExecutionProviderLibrary)(
                environment.ptr().cast_mut(),
                registration_name.as_ptr(),
            )
        },
        "UnregisterExecutionProviderLibrary",
    )?;
    info!("QNN execution provider unregistered after final session");
    Ok(())
}

pub(crate) fn verify_runtime_contract(runtime_dir: &Path) -> Result<()> {
    let runtime_dir = runtime_dir
        .canonicalize()
        .with_context(|| format!("resolve runtime directory {}", runtime_dir.display()))?;
    let mut verified = VERIFIED_RUNTIME.lock();
    if verified.as_ref() == Some(&runtime_dir) {
        return Ok(());
    }
    verify_runtime_contract_uncached(&runtime_dir)?;
    *verified = Some(runtime_dir);
    Ok(())
}

pub fn initialize_ort_runtime() -> Result<PathBuf> {
    let runtime_dir = runtime_directory()?;
    let mut initialized = ORT_INITIALIZED.lock();
    if !*initialized {
        verify_runtime_contract(&runtime_dir)?;
        std::env::set_var("ORT_DYLIB_PATH", runtime_dir.join("onnxruntime.dll"));
        let _ = ort::init().with_name("openwritr").commit();
        *initialized = true;
    }
    Ok(runtime_dir)
}

fn runtime_directory() -> Result<PathBuf> {
    let executable = std::env::current_exe().context("resolve current executable")?;
    let directory = executable
        .parent()
        .ok_or_else(|| anyhow!("current executable has no runtime directory"))?;
    if directory.ends_with("deps") || directory.ends_with("examples") {
        directory
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| anyhow!("runtime executable directory has no parent"))
    } else {
        Ok(directory.to_path_buf())
    }
}

fn verify_runtime_contract_uncached(runtime_dir: &Path) -> Result<()> {
    let receipt_path = runtime_dir.join("runtime-versions.json");
    let receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&receipt_path)
            .with_context(|| format!("read {}", receipt_path.display()))?,
    )
    .with_context(|| format!("parse {}", receipt_path.display()))?;
    require_json_string(&receipt, &["rust_ort", "crate_version"], EXPECTED_ORT_CRATE)?;
    require_json_u64(
        &receipt,
        &["rust_ort", "api_version"],
        u64::from(sys::ORT_API_VERSION),
    )?;
    require_package_version(&receipt, "onnxruntime", EXPECTED_ORT_NATIVE)?;
    let expected_arch = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else if cfg!(target_arch = "x86_64") {
        "x64"
    } else {
        std::env::consts::ARCH
    };
    require_json_string(&receipt, &["architecture"], expected_arch)?;
    if cfg!(target_arch = "aarch64") {
        require_package_version(&receipt, "onnxruntime-qnn", EXPECTED_QNN_PACKAGE)?;
        require_json_string(&receipt, &["qnn", "qairt_version"], EXPECTED_QAIRT)?;
    }
    verify_runtime_files(runtime_dir, &receipt)?;
    let native_version = native_ort_version(&runtime_dir.join("onnxruntime.dll"))?;
    if native_version != EXPECTED_ORT_NATIVE {
        bail!(
            "loaded ONNX Runtime version mismatch: expected {EXPECTED_ORT_NATIVE}, got {native_version}"
        );
    }
    info!(
        runtime = %runtime_dir.display(),
        ort_native = %native_version,
        architecture = expected_arch,
        "verified pinned native runtime files"
    );
    Ok(())
}

fn verify_runtime_files(runtime_dir: &Path, receipt: &serde_json::Value) -> Result<()> {
    let files = receipt
        .get("files")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow!("runtime receipt is missing files"))?;
    for expected_path in COMMON_RUNTIME_FILES.iter().copied().chain(
        cfg!(target_arch = "aarch64")
            .then_some(ARM64_QNN_RUNTIME_FILES)
            .into_iter()
            .flatten()
            .copied(),
    ) {
        let entry = files
            .iter()
            .find(|entry| {
                entry.get("path").and_then(serde_json::Value::as_str) == Some(expected_path)
            })
            .ok_or_else(|| anyhow!("runtime receipt is missing file {expected_path}"))?;
        verify_runtime_file(runtime_dir, entry)?;
    }
    Ok(())
}

fn verify_runtime_file(runtime_dir: &Path, entry: &serde_json::Value) -> Result<()> {
    let relative = entry
        .get("path")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("runtime receipt contains a file without a path"))?;
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("runtime receipt contains unsafe path {relative}");
    }
    let expected_bytes = entry
        .get("bytes")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow!("runtime receipt file {relative} is missing bytes"))?;
    let expected_sha256 = entry
        .get("sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("runtime receipt file {relative} is missing sha256"))?;
    if expected_sha256.len() != 64 || !expected_sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("runtime receipt file {relative} has an invalid sha256");
    }
    let path = runtime_dir.join(relative_path);
    let actual_bytes = path
        .metadata()
        .with_context(|| format!("inspect runtime file {}", path.display()))?
        .len();
    if actual_bytes != expected_bytes {
        bail!(
            "runtime file {} size mismatch: expected {expected_bytes}, got {actual_bytes}",
            path.display()
        );
    }
    let actual_sha256 = sha256_file(&path)?;
    if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
        bail!(
            "runtime file {} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}",
            path.display()
        );
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut source = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = source
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(windows)]
fn native_ort_version(path: &Path) -> Result<String> {
    use std::os::windows::ffi::OsStrExt;
    use windows::core::{s, PCWSTR};
    use windows::Win32::Foundation::FreeLibrary;
    use windows::Win32::System::LibraryLoader::{
        GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
    };

    let path_w: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let module = unsafe {
        LoadLibraryExW(
            PCWSTR(path_w.as_ptr()),
            None,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
    }
    .with_context(|| format!("load {}", path.display()))?;
    let result = (|| unsafe {
        type OrtGetApiBase = unsafe extern "system" fn() -> *const sys::OrtApiBase;
        let symbol = GetProcAddress(module, s!("OrtGetApiBase"))
            .ok_or_else(|| anyhow!("{} does not export OrtGetApiBase", path.display()))?;
        let get_api_base: OrtGetApiBase = std::mem::transmute(symbol);
        let base = get_api_base();
        if base.is_null() {
            bail!("OrtGetApiBase returned null for {}", path.display());
        }
        let version = ((*base).GetVersionString)();
        if version.is_null() {
            bail!("GetVersionString returned null for {}", path.display());
        }
        CStr::from_ptr(version)
            .to_str()
            .map(str::to_owned)
            .context("ONNX Runtime returned a non-UTF-8 version")
    })();
    unsafe {
        let _ = FreeLibrary(module);
    }
    result
}

#[cfg(not(windows))]
fn native_ort_version(_path: &Path) -> Result<String> {
    bail!("native ONNX Runtime validation is supported only on Windows")
}

#[cfg(windows)]
fn preload_qnn_runtime(runtime_dir: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows::core::PCWSTR;
    use windows::Win32::System::LibraryLoader::{
        LoadLibraryExW, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
    };

    let mut preload = QNN_PRELOAD.lock();
    if preload.is_some() {
        return Ok(());
    }
    let mut modules = Vec::new();
    unsafe {
        for name in [
            "QnnSystem.dll",
            "QnnHtpPrepare.dll",
            "QnnHtpNetRunExtensions.dll",
            "QnnHtp.dll",
        ] {
            let path = runtime_dir.join(name);
            let path_w: Vec<u16> = path
                .as_os_str()
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let module = LoadLibraryExW(
                PCWSTR(path_w.as_ptr()),
                None,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
            .with_context(|| format!("preload {}", path.display()))?;
            modules.push(LoadedModule(module));
        }
    }
    *preload = Some(QnnPreload { _modules: modules });
    Ok(())
}

#[cfg(not(windows))]
fn preload_qnn_runtime(_runtime_dir: &Path) -> Result<()> {
    bail!("QNN runtime preloading is supported only on Windows")
}

fn require_json_string(value: &serde_json::Value, path: &[&str], expected: &str) -> Result<()> {
    let actual = json_path(value, path)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("runtime receipt is missing {}", path.join(".")))?;
    if actual != expected {
        bail!(
            "runtime receipt {} mismatch: expected {expected}, got {actual}",
            path.join(".")
        );
    }
    Ok(())
}

fn require_json_u64(value: &serde_json::Value, path: &[&str], expected: u64) -> Result<()> {
    let actual = json_path(value, path)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow!("runtime receipt is missing {}", path.join(".")))?;
    if actual != expected {
        bail!(
            "runtime receipt {} mismatch: expected {expected}, got {actual}",
            path.join(".")
        );
    }
    Ok(())
}

fn require_package_version(
    receipt: &serde_json::Value,
    package: &str,
    expected: &str,
) -> Result<()> {
    let actual = receipt
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .and_then(|packages| {
            packages.iter().find(|entry| {
                entry.get("name").and_then(serde_json::Value::as_str) == Some(package)
            })
        })
        .and_then(|entry| entry.get("version"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow!("runtime receipt is missing package {package}"))?;
    if actual != expected {
        bail!("runtime package {package} mismatch: expected {expected}, got {actual}");
    }
    Ok(())
}

fn json_path<'a>(mut value: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    for component in path {
        value = value.get(*component)?;
    }
    Some(value)
}

pub fn enumerate_qnn_npu_devices(
    env_ptr: *const sys::OrtEnv,
) -> Result<Vec<*const sys::OrtEpDevice>> {
    let api = ort::api();
    unsafe {
        let mut devices = ptr::null();
        let mut count = 0;
        check(
            api,
            (api.GetEpDevices)(env_ptr, &mut devices, &mut count),
            "GetEpDevices",
        )?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let devices = NonNull::new(devices.cast_mut())
            .ok_or_else(|| anyhow!("GetEpDevices returned null"))?;
        let mut qnn_devices = Vec::new();
        for &device in std::slice::from_raw_parts(devices.as_ptr(), count) {
            if device.is_null() {
                continue;
            }
            let ep_name = (api.EpDevice_EpName)(device);
            let hardware = (api.EpDevice_Device)(device);
            if ep_name.is_null() || hardware.is_null() {
                continue;
            }
            let ep_name = CStr::from_ptr(ep_name).to_string_lossy();
            let hardware_type = (api.HardwareDevice_Type)(hardware);
            if hardware_type == sys::OrtHardwareDeviceType::OrtHardwareDeviceType_NPU
                && ep_name == "QNNExecutionProvider"
            {
                qnn_devices.push(device);
            }
        }
        Ok(qnn_devices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[repr(C)]
    struct FakeHandle;

    static RELEASES: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "system" fn release_fake(handle: *mut FakeHandle) {
        RELEASES.fetch_add(1, Ordering::SeqCst);
        drop(Box::from_raw(handle));
    }

    #[test]
    fn owned_handles_release_during_error_unwind() {
        RELEASES.store(0, Ordering::SeqCst);
        let result = (|| -> Result<()> {
            let raw = Box::into_raw(Box::new(FakeHandle));
            let _handle = unsafe { OrtHandle::from_raw(raw, release_fake, "fake")? };
            bail!("injected failure")
        })();

        assert!(result.is_err());
        assert_eq!(RELEASES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_creation_releases_a_partial_handle() {
        RELEASES.store(0, Ordering::SeqCst);
        let raw = Box::into_raw(Box::new(FakeHandle));
        let result: Result<OrtHandle<FakeHandle>> = finish_created_handle(
            raw,
            release_fake,
            "fake",
            Err(anyhow!("injected creation failure")),
        );

        assert!(result.is_err());
        assert_eq!(RELEASES.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn zero_element_tensor_does_not_dereference_a_null_pointer() {
        let output = copy_tensor_data(TensorElementType::F32, ptr::null_mut(), 0).unwrap();
        assert!(matches!(output, TensorDataOwned::F32(values) if values.is_empty()));
        assert!(copy_tensor_data(TensorElementType::F32, ptr::null_mut(), 1).is_err());
    }

    #[test]
    fn runtime_file_validation_detects_content_changes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("onnxruntime.dll");
        std::fs::write(&path, b"pinned").unwrap();
        let sha256 = sha256_file(&path).unwrap();
        let entry = serde_json::json!({
            "path": "onnxruntime.dll",
            "bytes": 6,
            "sha256": sha256,
        });

        verify_runtime_file(directory.path(), &entry).unwrap();
        std::fs::write(&path, b"staled").unwrap();
        let error = verify_runtime_file(directory.path(), &entry).unwrap_err();
        assert!(error.to_string().contains("SHA-256 mismatch"));
    }

    #[test]
    fn provider_registration_retries_and_unregisters_after_last_lease() {
        let mut state = QnnProviderState::default();
        let attempts = AtomicUsize::new(0);
        let unregisters = AtomicUsize::new(0);

        assert!(acquire_provider_state(&mut state, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            bail!("injected")
        })
        .is_err());
        acquire_provider_state(&mut state, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        acquire_provider_state(&mut state, || {
            attempts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();

        assert!(state.registered);
        assert_eq!(state.leases, 2);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        release_provider_state(&mut state, || {
            unregisters.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        assert!(state.registered);
        assert_eq!(unregisters.load(Ordering::SeqCst), 0);
        release_provider_state(&mut state, || {
            unregisters.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        assert!(!state.registered);
        assert_eq!(unregisters.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn tensor_contract_rejects_mismatched_shapes_and_duplicates() {
        assert!(shape_matches(&[1, 128, -1], &[1, 128, 801]));
        assert!(!shape_matches(&[1, 128, 801], &[1, 80, 801]));
        assert!(SessionContract::new(
            vec![
                TensorSpec::new("input", TensorElementType::F32, vec![1]),
                TensorSpec::new("input", TensorElementType::F32, vec![1]),
            ],
            vec![],
        )
        .is_err());
    }

    #[test]
    fn input_validation_rejects_wrong_type_and_element_count() {
        let specs = vec![TensorSpec::new("audio", TensorElementType::F32, vec![1, 2])];
        let values = [1_i32, 2];
        let dimensions = [1, 2];
        assert!(
            validate_inputs(&specs, &[TensorInput::i32("audio", &dimensions, &values)]).is_err()
        );
        assert_eq!(element_count(&[1, 2, 3]).unwrap(), 6);
        assert!(element_count(&[1, -1]).is_err());
    }
}
