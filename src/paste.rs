use anyhow::{anyhow, Context, Result};
use arboard::Clipboard;
use enigo::{Direction, Enigo, Key, Keyboard, Settings as EnigoSettings};
use once_cell::sync::{Lazy, OnceCell};
use parking_lot::{Condvar, Mutex};
use std::{
    mem::size_of,
    ptr,
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::{sync_channel, Receiver, SyncSender},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};
use windows::Win32::{
    Foundation::CLIPBRD_E_CANT_OPEN,
    System::{
        Com::IDataObject,
        Ole::{
            OleFlushClipboard, OleGetClipboard, OleInitialize, OleSetClipboard, OleUninitialize,
            CF_UNICODETEXT,
        },
    },
};
use windows_sys::Win32::{
    Foundation::{GetLastError, GlobalFree, SetLastError},
    System::{
        DataExchange::{
            CloseClipboard, EmptyClipboard, EnumClipboardFormats, GetClipboardData,
            GetClipboardOwner, GetClipboardSequenceNumber, OpenClipboard, RegisterClipboardFormatW,
            SetClipboardData,
        },
        Memory::{GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock, GMEM_MOVEABLE},
    },
};

const CLIPBOARD_OPEN_ATTEMPTS: usize = 5;
const CLIPBOARD_OPEN_RETRY_DELAY: Duration = Duration::from_millis(5);
const CLIPBOARD_RESTORE_DELAY: Duration = Duration::from_millis(400);
const OPENWRITR_MARKER_FORMAT_NAME: &str = "OpenWritr.AutoPasteMarker";

static NEXT_MARKER_ID: AtomicU64 = AtomicU64::new(1);
static OPENWRITR_MARKER_FORMAT: OnceCell<u32> = OnceCell::new();
static RESTORATION_BARRIER: Lazy<Arc<RestorationBarrier>> =
    Lazy::new(|| Arc::new(RestorationBarrier::default()));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeliveryMode {
    Paste,
    Clipboard,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeliveryOutcome {
    Pasted,
    Copied,
    CopiedWithWarning {
        warning: DeliveryWarning,
        detail: String,
    },
    CancelledClipboardChanged,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeliveryWarning {
    ClipboardPreparationFailed,
    KeyInjectionFailed,
}

#[derive(Default)]
pub struct DeliveryInterlock {
    pending_epoch: AtomicU64,
    injection_gate: Mutex<()>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ClipboardIdentity {
    sequence: u32,
    owner: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ClipboardObservation {
    identity: ClipboardIdentity,
    text: Option<String>,
    marker: Option<String>,
    formats: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InjectedClipboardPayload {
    text: String,
    marker: String,
    marker_format: u32,
    required_formats: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RestoreDecision {
    Restore,
    SkipCancelled,
    SkipSequenceChanged,
    SkipOwnerChanged,
    SkipTextChanged,
    SkipMarkerChanged,
    SkipFormatsChanged,
    SkipHandoffChanged,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuardRejectionAction {
    CopyTranscript,
    PreserveChangedClipboard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InjectionFailureAction {
    PreserveTranscript,
    RestoreAfterDelay,
}

enum ClipboardSnapshot {
    Empty,
    DataObject(IDataObject),
}

enum RestoreCommand {
    RestoreAfterDelay,
    RestoreNow(SyncSender<Result<RestoreDecision>>),
    KeepTranscript,
}

struct PendingClipboardRestore {
    restore_tx: Option<SyncSender<RestoreCommand>>,
}

struct ClipboardOpenGuard {
    closed: bool,
}

struct OleClipboardApartment;

struct PasteSimulationError {
    error: anyhow::Error,
    paste_attempted: bool,
}

struct ClipboardPreparationError {
    error: anyhow::Error,
    staging_started: bool,
}

#[derive(Default)]
struct RestorationState {
    pending: usize,
    cancel_requested: bool,
}

#[derive(Default)]
struct RestorationBarrier {
    state: Mutex<RestorationState>,
    changed: Condvar,
}

struct RestorationPermit {
    barrier: Arc<RestorationBarrier>,
}

pub fn cancel_pending_restorations() {
    RESTORATION_BARRIER.cancel();
}

pub fn wait_for_pending_restorations(timeout: Duration) -> bool {
    RESTORATION_BARRIER.wait_for_completion(timeout)
}

pub fn deliver(text: &str, mode: DeliveryMode) -> Result<DeliveryOutcome> {
    deliver_guarded(text, mode, None, || true)
}

/// Deliver text while revalidating the captured target immediately before
/// keyboard injection. The guard is evaluated before clipboard preparation
/// and once more after preparation. The optional interlock serializes the
/// final pending-press check with key injection.
pub fn deliver_guarded(
    text: &str,
    mode: DeliveryMode,
    interlock: Option<&DeliveryInterlock>,
    mut paste_allowed: impl FnMut() -> bool,
) -> Result<DeliveryOutcome> {
    match mode {
        DeliveryMode::Paste => paste(text, interlock, &mut paste_allowed),
        DeliveryMode::Clipboard => copy(text),
    }
}

impl DeliveryInterlock {
    pub fn mark_press_pending(&self, next_epoch: &mut u64) -> u64 {
        let _gate = self.injection_gate.lock();
        *next_epoch = next_epoch.wrapping_add(1);
        if *next_epoch == 0 {
            *next_epoch = 1;
        }
        self.pending_epoch.store(*next_epoch, Ordering::Release);
        *next_epoch
    }

    pub fn consume_press(&self, epoch: u64) {
        let _ = self
            .pending_epoch
            .compare_exchange(epoch, 0, Ordering::AcqRel, Ordering::Acquire);
    }

    pub fn press_pending(&self) -> bool {
        self.pending_epoch.load(Ordering::Acquire) != 0
    }
}

fn copy(text: &str) -> Result<DeliveryOutcome> {
    let mut clipboard = Clipboard::new().context("open the Windows clipboard")?;
    clipboard
        .set_text(text.to_string())
        .context("write the transcript to the Windows clipboard")?;
    Ok(DeliveryOutcome::Copied)
}

fn copy_with_warning(
    text: &str,
    warning: DeliveryWarning,
    detail: impl Into<String>,
) -> Result<DeliveryOutcome> {
    let detail = detail.into();
    copy(text).map(|_| DeliveryOutcome::CopiedWithWarning { warning, detail })
}

fn paste(
    text: &str,
    interlock: Option<&DeliveryInterlock>,
    paste_allowed: &mut dyn FnMut() -> bool,
) -> Result<DeliveryOutcome> {
    if interlock.is_some_and(DeliveryInterlock::press_pending) || !paste_allowed() {
        return copy(text);
    }
    let mut enigo = match Enigo::new(&EnigoSettings::default()) {
        Ok(enigo) => enigo,
        Err(error) => {
            let error = anyhow!("could not initialize keyboard paste: {error}");
            return copy_with_warning(text, DeliveryWarning::KeyInjectionFailed, error.to_string())
                .with_context(|| {
                    format!(
                        "{error}; additionally failed to preserve the transcript on the clipboard"
                    )
                });
        }
    };
    let pending_restore = match PendingClipboardRestore::prepare(text) {
        Ok(pending_restore) => pending_restore,
        Err(error) => {
            return copy_with_warning(
                text,
                DeliveryWarning::ClipboardPreparationFailed,
                error.to_string(),
            )
            .with_context(|| {
                format!("{error}; additionally failed to preserve the transcript on the clipboard")
            });
        }
    };
    let injection_guard = interlock.map(|interlock| interlock.injection_gate.lock());
    if interlock.is_some_and(DeliveryInterlock::press_pending) || !paste_allowed() {
        drop(injection_guard);
        match pending_restore.restore_now() {
            Ok(decision) => match guard_rejection_action(decision) {
                GuardRejectionAction::CopyTranscript => return copy(text),
                GuardRejectionAction::PreserveChangedClipboard => {
                    return Ok(DeliveryOutcome::CancelledClipboardChanged);
                }
            },
            Err(error) => {
                return Err(error).context(
                    "target changed after paste preparation and clipboard state could not be finalized safely",
                );
            }
        }
    }

    let paste_result = send_paste_keys(&mut enigo);
    drop(injection_guard);
    match paste_result {
        Ok(()) => {
            pending_restore.restore_after_delay().map_err(|error| {
                anyhow!("keyboard paste completed, but clipboard restoration could not be scheduled: {error}")
            })?;
            Ok(DeliveryOutcome::Pasted)
        }
        Err(PasteSimulationError {
            error,
            paste_attempted,
        }) => match injection_failure_action(paste_attempted) {
            InjectionFailureAction::RestoreAfterDelay => {
                pending_restore.restore_after_delay().map_err(|restore_error| {
                    anyhow!("{error}; additionally failed to schedule clipboard restoration: {restore_error}")
                })?;
                Err(anyhow!(
                    "{error}; OpenWritr will restore the previous clipboard only if the injected payload is still current"
                ))
            }
            InjectionFailureAction::PreserveTranscript => {
                let keep_error = pending_restore.keep_transcript().err();
                let detail = match keep_error {
                    Some(keep_error) => format!(
                        "{error}; additionally failed to finalize clipboard preservation: {keep_error}"
                    ),
                    None => error.to_string(),
                };
                copy_with_warning(text, DeliveryWarning::KeyInjectionFailed, detail).with_context(
                    || {
                        format!(
                            "{error}; additionally failed to keep the transcript on the clipboard"
                        )
                    },
                )
            }
        },
    }
}

fn injection_failure_action(paste_attempted: bool) -> InjectionFailureAction {
    if paste_attempted {
        InjectionFailureAction::RestoreAfterDelay
    } else {
        InjectionFailureAction::PreserveTranscript
    }
}

fn guard_rejection_action(decision: RestoreDecision) -> GuardRejectionAction {
    match decision {
        RestoreDecision::Restore => GuardRejectionAction::CopyTranscript,
        RestoreDecision::SkipCancelled
        | RestoreDecision::SkipSequenceChanged
        | RestoreDecision::SkipOwnerChanged
        | RestoreDecision::SkipTextChanged
        | RestoreDecision::SkipMarkerChanged
        | RestoreDecision::SkipFormatsChanged
        | RestoreDecision::SkipHandoffChanged => GuardRejectionAction::PreserveChangedClipboard,
    }
}

impl PendingClipboardRestore {
    fn prepare(text: &str) -> Result<Self> {
        let (ready_tx, ready_rx) = sync_channel(1);
        let (restore_tx, restore_rx) = sync_channel(1);
        let text = text.to_string();
        let barrier = RESTORATION_BARRIER.clone();
        let permit = barrier.begin();

        thread::Builder::new()
            .name("clipboard-restore".into())
            .spawn(move || run_clipboard_worker(text, ready_tx, restore_rx, barrier, permit))
            .context("start the automatic paste clipboard worker")?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                restore_tx: Some(restore_tx),
            }),
            Ok(Err(error)) => Err(error),
            Err(_) => Err(anyhow!(
                "automatic paste aborted because the clipboard worker stopped before the transcript payload was ready"
            )),
        }
    }

    fn restore_after_delay(mut self) -> Result<()> {
        self.send_command(RestoreCommand::RestoreAfterDelay)
    }

    fn restore_now(mut self) -> Result<RestoreDecision> {
        let (reply_tx, reply_rx) = sync_channel(1);
        self.send_command(RestoreCommand::RestoreNow(reply_tx))?;
        reply_rx.recv().map_err(|_| {
            anyhow!(
                "automatic paste aborted because the clipboard worker stopped before reporting the restore result"
            )
        })?
    }

    fn keep_transcript(mut self) -> Result<()> {
        self.send_command(RestoreCommand::KeepTranscript)
    }

    fn send_command(&mut self, command: RestoreCommand) -> Result<()> {
        let restore_tx = self.restore_tx.take().ok_or_else(|| {
            anyhow!("automatic paste clipboard restoration had already been finalized")
        })?;
        restore_tx.send(command).map_err(|_| {
            anyhow!("automatic paste aborted because the clipboard worker exited unexpectedly")
        })
    }
}

impl Drop for PendingClipboardRestore {
    fn drop(&mut self) {
        if let Some(restore_tx) = self.restore_tx.take() {
            let _ = restore_tx.send(RestoreCommand::RestoreAfterDelay);
        }
    }
}

impl RestorationBarrier {
    fn begin(self: &Arc<Self>) -> RestorationPermit {
        self.state.lock().pending += 1;
        RestorationPermit {
            barrier: self.clone(),
        }
    }

    fn cancel(&self) {
        self.state.lock().cancel_requested = true;
        self.changed.notify_all();
    }

    fn wait_for_completion(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut state = self.state.lock();
        while state.pending != 0 {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            self.changed.wait_for(&mut state, deadline - now);
        }
        true
    }

    fn wait_for_delay_or_cancel(&self, delay: Duration) -> bool {
        let deadline = Instant::now() + delay;
        let mut state = self.state.lock();
        loop {
            if state.cancel_requested {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            self.changed.wait_for(&mut state, deadline - now);
        }
    }

    fn cancel_requested(&self) -> bool {
        self.state.lock().cancel_requested
    }
}

impl Drop for RestorationPermit {
    fn drop(&mut self) {
        let mut state = self.barrier.state.lock();
        state.pending = state.pending.saturating_sub(1);
        drop(state);
        self.barrier.changed.notify_all();
    }
}

impl ClipboardSnapshot {
    fn capture() -> Result<(Self, ClipboardIdentity)> {
        let initial_identity = current_clipboard_identity()
            .context("read the Windows clipboard sequence before automatic paste")?;
        let snapshot = if clipboard_has_formats()
            .context("inspect the current clipboard before automatic paste")?
        {
            Self::DataObject(ole_clipboard_retry(
                "snapshot the clipboard contents",
                || unsafe { OleGetClipboard() },
            )?)
        } else {
            Self::Empty
        };
        let final_identity = current_clipboard_identity()
            .context("re-read the Windows clipboard sequence after snapshotting it")?;

        if initial_identity.sequence != final_identity.sequence {
            return Err(anyhow!(
                "automatic paste aborted because the clipboard changed while OpenWritr was snapshotting it"
            ));
        }

        Ok((snapshot, final_identity))
    }

    fn restore_unconditionally(&self) -> Result<()> {
        match self {
            ClipboardSnapshot::Empty => clear_clipboard(),
            ClipboardSnapshot::DataObject(data_object) => {
                ole_clipboard_retry("restore the clipboard snapshot", || unsafe {
                    OleSetClipboard(data_object)
                })?;
                ole_clipboard_retry("persist the restored clipboard snapshot", || unsafe {
                    OleFlushClipboard()
                })?;
                Ok(())
            }
        }
    }

    fn restore_if_current(&self, expected_identity: ClipboardIdentity) -> Result<RestoreDecision> {
        match self {
            ClipboardSnapshot::Empty => {
                with_open_clipboard("restore an empty clipboard snapshot", || {
                    if current_clipboard_identity()? != expected_identity {
                        return Ok(RestoreDecision::SkipHandoffChanged);
                    }
                    unsafe {
                        if EmptyClipboard() == 0 {
                            return Err(std::io::Error::last_os_error())
                                .context("empty the Windows clipboard");
                        }
                    }
                    Ok(RestoreDecision::Restore)
                })
            }
            ClipboardSnapshot::DataObject(data_object) => {
                if current_clipboard_identity()? != expected_identity {
                    return Ok(RestoreDecision::SkipHandoffChanged);
                }
                ole_clipboard_retry("restore the clipboard snapshot", || unsafe {
                    OleSetClipboard(data_object)
                })?;
                ole_clipboard_retry("persist the restored clipboard snapshot", || unsafe {
                    OleFlushClipboard()
                })?;
                Ok(RestoreDecision::Restore)
            }
        }
    }
}

impl InjectedClipboardPayload {
    fn new(text: &str) -> Result<Self> {
        let marker_format = openwritr_marker_format()?;
        let normalized_text = normalize_clipboard_text(text);
        let marker = format!(
            "openwritr:{}:{}",
            std::process::id(),
            NEXT_MARKER_ID.fetch_add(1, Ordering::Relaxed)
        );

        Ok(Self {
            text: normalized_text,
            marker,
            marker_format,
            required_formats: vec![u32::from(CF_UNICODETEXT.0), marker_format],
        })
    }
}

impl ClipboardOpenGuard {
    fn open(action: &str) -> Result<Self> {
        let mut attempts = CLIPBOARD_OPEN_ATTEMPTS;
        loop {
            if unsafe { OpenClipboard(ptr::null_mut()) } != 0 {
                return Ok(Self { closed: false });
            }

            if attempts == 0 {
                return Err(std::io::Error::last_os_error())
                    .context(format!("open the Windows clipboard to {action}"));
            }

            attempts -= 1;
            thread::sleep(CLIPBOARD_OPEN_RETRY_DELAY);
        }
    }

    fn close(mut self) -> Result<()> {
        self.closed = true;
        if unsafe { CloseClipboard() } != 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error()).context("close the Windows clipboard")
        }
    }
}

impl Drop for ClipboardOpenGuard {
    fn drop(&mut self) {
        if !self.closed {
            unsafe {
                CloseClipboard();
            }
        }
    }
}

impl OleClipboardApartment {
    fn initialize() -> Result<Self> {
        unsafe { OleInitialize(None) }.context("initialize the OLE clipboard apartment")?;
        Ok(Self)
    }
}

impl Drop for OleClipboardApartment {
    fn drop(&mut self) {
        unsafe {
            OleUninitialize();
        }
    }
}

impl PasteSimulationError {
    fn new(error: anyhow::Error, paste_attempted: bool) -> Self {
        Self {
            error,
            paste_attempted,
        }
    }
}

fn run_clipboard_worker(
    text: String,
    ready_tx: SyncSender<Result<()>>,
    restore_rx: Receiver<RestoreCommand>,
    barrier: Arc<RestorationBarrier>,
    _permit: RestorationPermit,
) {
    let mut ready_sent = false;
    let result: Result<()> = (|| {
        let _ole = OleClipboardApartment::initialize()?;
        let (snapshot, captured_identity) = ClipboardSnapshot::capture()?;
        let payload = InjectedClipboardPayload::new(&text)?;
        let injected_identity = match write_injected_payload(captured_identity, &payload) {
            Ok(identity) => identity,
            Err(error) => {
                if !error.staging_started {
                    return Err(error.error);
                }
                match snapshot.restore_unconditionally() {
                    Ok(()) => return Err(error.error),
                    Err(restore_error) => {
                        return Err(anyhow!(
                            "{}; additionally failed to restore the original clipboard after paste preparation failed: {restore_error}",
                            error.error
                        ))
                    }
                }
            }
        };

        ready_tx.send(Ok(())).map_err(|_| {
            anyhow!("automatic paste caller dropped before the clipboard payload was ready")
        })?;
        ready_sent = true;

        match restore_rx.recv() {
            Ok(RestoreCommand::RestoreAfterDelay) => log_restore_result(restore_snapshot_if_safe(
                &snapshot,
                injected_identity,
                &payload,
                CLIPBOARD_RESTORE_DELAY,
                &barrier,
            )),
            Ok(RestoreCommand::RestoreNow(reply_tx)) => {
                let _ = reply_tx.send(restore_snapshot_if_safe(
                    &snapshot,
                    injected_identity,
                    &payload,
                    Duration::ZERO,
                    &barrier,
                ));
            }
            Ok(RestoreCommand::KeepTranscript) => {}
            Err(_) => log_restore_result(restore_snapshot_if_safe(
                &snapshot,
                injected_identity,
                &payload,
                CLIPBOARD_RESTORE_DELAY,
                &barrier,
            )),
        }

        Ok(())
    })();

    if let Err(error) = result {
        if !ready_sent {
            let _ = ready_tx.send(Err(error));
        } else {
            tracing::warn!(error = %error, "automatic paste clipboard worker failed");
        }
    }
}

fn write_injected_payload(
    captured_identity: ClipboardIdentity,
    payload: &InjectedClipboardPayload,
) -> std::result::Result<ClipboardIdentity, ClipboardPreparationError> {
    let mut staging_started = false;
    let result = with_open_clipboard("inject the transcript for automatic paste", || {
        let current_identity = current_clipboard_identity()?;
        if current_identity.sequence != captured_identity.sequence {
            return Err(anyhow!(
                "automatic paste aborted because the clipboard changed while OpenWritr was preparing the transcript payload"
            ));
        }

        unsafe {
            if EmptyClipboard() == 0 {
                return Err(std::io::Error::last_os_error())
                    .context("empty the Windows clipboard for automatic paste");
            }
        }
        staging_started = true;

        write_clipboard_text(&payload.text)?;
        write_clipboard_marker(payload.marker_format, &payload.marker)?;
        Ok(())
    });
    if let Err(error) = result {
        return Err(ClipboardPreparationError {
            error,
            staging_started,
        });
    }

    current_clipboard_identity()
        .context("read the Windows clipboard sequence after injecting the transcript")
        .map_err(|error| ClipboardPreparationError {
            error,
            staging_started: true,
        })
}

fn restore_snapshot_if_safe(
    snapshot: &ClipboardSnapshot,
    injected_identity: ClipboardIdentity,
    payload: &InjectedClipboardPayload,
    delay: Duration,
    barrier: &RestorationBarrier,
) -> Result<RestoreDecision> {
    if barrier.wait_for_delay_or_cancel(delay) {
        return Ok(RestoreDecision::SkipCancelled);
    }

    let observation = observe_current_clipboard(payload)
        .context("inspect the current clipboard before restore")?;
    if barrier.cancel_requested() {
        return Ok(RestoreDecision::SkipCancelled);
    }
    let decision = should_restore(injected_identity, payload, &observation);
    if decision != RestoreDecision::Restore {
        return Ok(decision);
    }
    if barrier.cancel_requested() {
        return Ok(RestoreDecision::SkipCancelled);
    }

    snapshot.restore_if_current(observation.identity)
}

fn log_restore_result(result: Result<RestoreDecision>) {
    match result {
        Ok(RestoreDecision::Restore) => {}
        Ok(RestoreDecision::SkipCancelled) => {
            tracing::info!("clipboard restoration cancelled during discard shutdown")
        }
        Ok(decision) => {
            tracing::info!(
                ?decision,
                "clipboard changed externally; preserving newer contents"
            )
        }
        Err(error) => {
            tracing::warn!(error = %error, "failed to restore the clipboard after automatic paste")
        }
    }
}

fn observe_current_clipboard(payload: &InjectedClipboardPayload) -> Result<ClipboardObservation> {
    with_open_clipboard("inspect the current clipboard before restore", || {
        let identity = current_clipboard_identity()?;
        let text = read_clipboard_unicode_text()?;
        let marker = read_clipboard_marker(payload.marker_format)?;
        let mut formats = Vec::with_capacity(2);
        if text.is_some() {
            formats.push(u32::from(CF_UNICODETEXT.0));
        }
        if marker.is_some() {
            formats.push(payload.marker_format);
        }

        Ok(ClipboardObservation {
            identity,
            text,
            marker,
            formats,
        })
    })
}

fn should_restore(
    injected_identity: ClipboardIdentity,
    payload: &InjectedClipboardPayload,
    current: &ClipboardObservation,
) -> RestoreDecision {
    if injected_identity.sequence == 0 || current.identity.sequence != injected_identity.sequence {
        return RestoreDecision::SkipSequenceChanged;
    }
    if current.identity.owner != injected_identity.owner {
        return RestoreDecision::SkipOwnerChanged;
    }
    if current.text.as_deref() != Some(payload.text.as_str()) {
        return RestoreDecision::SkipTextChanged;
    }
    if current.marker.as_deref() != Some(payload.marker.as_str()) {
        return RestoreDecision::SkipMarkerChanged;
    }
    if !payload
        .required_formats
        .iter()
        .all(|format| current.formats.contains(format))
    {
        return RestoreDecision::SkipFormatsChanged;
    }
    RestoreDecision::Restore
}

fn send_paste_keys(enigo: &mut Enigo) -> std::result::Result<(), PasteSimulationError> {
    if let Err(error) = enigo
        .key(Key::Control, Direction::Press)
        .context("press Ctrl for paste")
    {
        return Err(PasteSimulationError::new(error, false));
    }

    if let Err(error) = enigo
        .key(Key::Unicode('v'), Direction::Click)
        .context("press V for paste")
    {
        let _ = enigo.key(Key::Control, Direction::Release);
        return Err(PasteSimulationError::new(error, false));
    }

    if let Err(error) = enigo
        .key(Key::Control, Direction::Release)
        .context("release Ctrl after paste")
    {
        let _ = enigo.key(Key::Control, Direction::Release);
        return Err(PasteSimulationError::new(error, true));
    }

    Ok(())
}

fn clipboard_has_formats() -> Result<bool> {
    with_open_clipboard("inspect the current clipboard contents", || unsafe {
        SetLastError(0);
        let first_format = EnumClipboardFormats(0);
        if first_format == 0 {
            let error = GetLastError();
            if error == 0 {
                Ok(false)
            } else {
                Err(std::io::Error::last_os_error())
                    .context("enumerate the current Windows clipboard formats")
            }
        } else {
            Ok(true)
        }
    })
}

fn current_clipboard_identity() -> Result<ClipboardIdentity> {
    let sequence = unsafe { GetClipboardSequenceNumber() };
    if sequence == 0 {
        return Err(anyhow!(
            "read the Windows clipboard sequence number for automatic paste safety"
        ));
    }

    let owner = unsafe { GetClipboardOwner() };
    Ok(ClipboardIdentity {
        sequence,
        owner: (!owner.is_null()).then_some(owner as usize),
    })
}

fn clear_clipboard() -> Result<()> {
    with_open_clipboard("clear the Windows clipboard", || unsafe {
        if EmptyClipboard() == 0 {
            Err(std::io::Error::last_os_error()).context("empty the Windows clipboard")
        } else {
            Ok(())
        }
    })
}

fn write_clipboard_text(text: &str) -> Result<()> {
    let encoded = encode_utf16_with_nul(text);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            encoded.as_ptr().cast::<u8>(),
            encoded.len() * size_of::<u16>(),
        )
    };
    set_clipboard_memory(u32::from(CF_UNICODETEXT.0), bytes)
        .context("write the transcript to the Windows clipboard")
}

fn write_clipboard_marker(marker_format: u32, marker: &str) -> Result<()> {
    let mut bytes = marker.as_bytes().to_vec();
    bytes.push(0);
    set_clipboard_memory(marker_format, &bytes).context("write the OpenWritr clipboard marker")
}

fn set_clipboard_memory(format: u32, bytes: &[u8]) -> Result<()> {
    let handle = allocate_moveable_memory(bytes)?;
    if unsafe { SetClipboardData(format, handle) }.is_null() {
        unsafe {
            GlobalFree(handle);
        }
        return Err(std::io::Error::last_os_error())
            .context(format!("publish clipboard format {format}"));
    }
    Ok(())
}

fn allocate_moveable_memory(bytes: &[u8]) -> Result<*mut core::ffi::c_void> {
    let handle = unsafe { GlobalAlloc(GMEM_MOVEABLE, bytes.len()) };
    if handle.is_null() {
        return Err(std::io::Error::last_os_error())
            .context("allocate Windows global memory for the clipboard payload");
    }

    if bytes.is_empty() {
        return Ok(handle);
    }

    let locked = unsafe { GlobalLock(handle) };
    if locked.is_null() {
        unsafe {
            GlobalFree(handle);
        }
        return Err(std::io::Error::last_os_error())
            .context("lock Windows global memory for the clipboard payload");
    }

    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), locked.cast::<u8>(), bytes.len());
        GlobalUnlock(handle);
    }

    Ok(handle)
}

fn read_clipboard_unicode_text() -> Result<Option<String>> {
    let handle = unsafe { GetClipboardData(u32::from(CF_UNICODETEXT.0)) };
    if handle.is_null() {
        return Ok(None);
    }

    Ok(Some(read_global_utf16_string(handle).context(
        "read CF_UNICODETEXT from the Windows clipboard",
    )?))
}

fn read_clipboard_marker(marker_format: u32) -> Result<Option<String>> {
    let handle = unsafe { GetClipboardData(marker_format) };
    if handle.is_null() {
        return Ok(None);
    }

    Ok(Some(
        read_global_utf8_string(handle).context("read the OpenWritr clipboard marker")?,
    ))
}

fn read_global_utf16_string(handle: *mut core::ffi::c_void) -> Result<String> {
    let byte_len = unsafe { GlobalSize(handle) };
    if byte_len == 0 {
        return Ok(String::new());
    }
    if byte_len % size_of::<u16>() != 0 {
        return Err(anyhow!(
            "clipboard text payload length {byte_len} is not valid UTF-16"
        ));
    }

    let locked = unsafe { GlobalLock(handle) };
    if locked.is_null() {
        return Err(std::io::Error::last_os_error()).context("lock CF_UNICODETEXT clipboard data");
    }

    let text = unsafe {
        let units = std::slice::from_raw_parts(locked.cast::<u16>(), byte_len / size_of::<u16>());
        let text_len = units
            .iter()
            .position(|unit| *unit == 0)
            .unwrap_or(units.len());
        String::from_utf16(&units[..text_len])
    }
    .context("decode CF_UNICODETEXT from the Windows clipboard");

    unsafe {
        GlobalUnlock(handle);
    }

    text
}

fn read_global_utf8_string(handle: *mut core::ffi::c_void) -> Result<String> {
    let bytes = read_global_bytes(handle)?;
    let text_len = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8(bytes[..text_len].to_vec())
        .context("decode the OpenWritr clipboard marker payload")
}

fn read_global_bytes(handle: *mut core::ffi::c_void) -> Result<Vec<u8>> {
    let byte_len = unsafe { GlobalSize(handle) };
    if byte_len == 0 {
        return Ok(Vec::new());
    }

    let locked = unsafe { GlobalLock(handle) };
    if locked.is_null() {
        return Err(std::io::Error::last_os_error()).context("lock custom clipboard data");
    }

    let bytes = unsafe { std::slice::from_raw_parts(locked.cast::<u8>(), byte_len).to_vec() };
    unsafe {
        GlobalUnlock(handle);
    }

    Ok(bytes)
}

fn ole_clipboard_retry<T>(
    action: &str,
    mut operation: impl FnMut() -> windows::core::Result<T>,
) -> Result<T> {
    let mut attempts = CLIPBOARD_OPEN_ATTEMPTS;
    loop {
        match operation() {
            Ok(value) => return Ok(value),
            Err(error) if error.code() == CLIPBRD_E_CANT_OPEN && attempts > 0 => {
                attempts -= 1;
                thread::sleep(CLIPBOARD_OPEN_RETRY_DELAY);
            }
            Err(error) => {
                return Err(error).with_context(|| format!("{action} through the OLE clipboard"));
            }
        }
    }
}

fn with_open_clipboard<T>(action: &str, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    let guard = ClipboardOpenGuard::open(action)?;
    let result = operation();
    let close_result = guard.close();

    match (result, close_result) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(close_error)) => Err(close_error),
        (Err(error), Err(close_error)) => Err(anyhow!(
            "{error}; additionally failed to close the Windows clipboard: {close_error}"
        )),
    }
}

fn openwritr_marker_format() -> Result<u32> {
    OPENWRITR_MARKER_FORMAT
        .get_or_try_init(|| {
            let name = wide_null(OPENWRITR_MARKER_FORMAT_NAME);
            let format = unsafe { RegisterClipboardFormatW(name.as_ptr()) };
            if format == 0 {
                Err(std::io::Error::last_os_error())
                    .context("register the OpenWritr clipboard marker format")
            } else {
                Ok(format)
            }
        })
        .copied()
}

fn normalize_clipboard_text(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\n', "\r\n")
}

fn encode_utf16_with_nul(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

fn wide_null(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(test)]
mod tests {
    use super::{
        guard_rejection_action, injection_failure_action, normalize_clipboard_text, should_restore,
        ClipboardIdentity, ClipboardObservation, DeliveryMode, GuardRejectionAction,
        InjectedClipboardPayload, InjectionFailureAction, RestorationBarrier, RestoreDecision,
    };
    use std::sync::Arc;
    use std::time::Duration;
    use windows::Win32::System::Ole::CF_UNICODETEXT;

    const MARKER_FORMAT: u32 = 0xC123;

    fn payload(text: &str) -> InjectedClipboardPayload {
        InjectedClipboardPayload {
            text: normalize_clipboard_text(text),
            marker: "openwritr:test:1".to_string(),
            marker_format: MARKER_FORMAT,
            required_formats: vec![u32::from(CF_UNICODETEXT.0), MARKER_FORMAT],
        }
    }

    fn observation(
        sequence: u32,
        owner: Option<usize>,
        text: Option<&str>,
        marker: Option<&str>,
        formats: Vec<u32>,
    ) -> ClipboardObservation {
        ClipboardObservation {
            identity: ClipboardIdentity { sequence, owner },
            text: text.map(str::to_string),
            marker: marker.map(str::to_string),
            formats,
        }
    }

    #[test]
    fn restores_only_when_the_clipboard_is_unchanged_and_still_contains_openwritr_payload() {
        let payload = payload("hello");
        let decision = should_restore(
            ClipboardIdentity {
                sequence: 42,
                owner: Some(7),
            },
            &payload,
            &observation(
                42,
                Some(7),
                Some("hello"),
                Some("openwritr:test:1"),
                vec![u32::from(CF_UNICODETEXT.0), MARKER_FORMAT],
            ),
        );

        assert_eq!(decision, RestoreDecision::Restore);
    }

    #[test]
    fn skips_restore_when_the_clipboard_sequence_changes() {
        let payload = payload("hello");
        let decision = should_restore(
            ClipboardIdentity {
                sequence: 42,
                owner: Some(7),
            },
            &payload,
            &observation(
                43,
                Some(7),
                Some("hello"),
                Some("openwritr:test:1"),
                vec![u32::from(CF_UNICODETEXT.0), MARKER_FORMAT],
            ),
        );

        assert_eq!(decision, RestoreDecision::SkipSequenceChanged);
    }

    #[test]
    fn skips_restore_when_the_clipboard_owner_changes() {
        let payload = payload("hello");
        let decision = should_restore(
            ClipboardIdentity {
                sequence: 42,
                owner: Some(7),
            },
            &payload,
            &observation(
                42,
                Some(8),
                Some("hello"),
                Some("openwritr:test:1"),
                vec![u32::from(CF_UNICODETEXT.0), MARKER_FORMAT],
            ),
        );

        assert_eq!(decision, RestoreDecision::SkipOwnerChanged);
    }

    #[test]
    fn skips_restore_when_the_transcript_text_changes() {
        let payload = payload("hello");
        let decision = should_restore(
            ClipboardIdentity {
                sequence: 42,
                owner: Some(7),
            },
            &payload,
            &observation(
                42,
                Some(7),
                Some("goodbye"),
                Some("openwritr:test:1"),
                vec![u32::from(CF_UNICODETEXT.0), MARKER_FORMAT],
            ),
        );

        assert_eq!(decision, RestoreDecision::SkipTextChanged);
    }

    #[test]
    fn skips_restore_when_the_openwritr_marker_changes() {
        let payload = payload("hello");
        let decision = should_restore(
            ClipboardIdentity {
                sequence: 42,
                owner: Some(7),
            },
            &payload,
            &observation(
                42,
                Some(7),
                Some("hello"),
                Some("someone-else"),
                vec![u32::from(CF_UNICODETEXT.0), MARKER_FORMAT],
            ),
        );

        assert_eq!(decision, RestoreDecision::SkipMarkerChanged);
    }

    #[test]
    fn skips_restore_when_required_formats_are_missing() {
        let payload = payload("hello");
        let decision = should_restore(
            ClipboardIdentity {
                sequence: 42,
                owner: Some(7),
            },
            &payload,
            &observation(
                42,
                Some(7),
                Some("hello"),
                Some("openwritr:test:1"),
                vec![u32::from(CF_UNICODETEXT.0)],
            ),
        );

        assert_eq!(decision, RestoreDecision::SkipFormatsChanged);
    }

    #[test]
    fn accepts_synthetic_multi_format_representations_when_required_formats_are_present() {
        let payload = payload("hello");
        let decision = should_restore(
            ClipboardIdentity {
                sequence: 42,
                owner: Some(7),
            },
            &payload,
            &observation(
                42,
                Some(7),
                Some("hello"),
                Some("openwritr:test:1"),
                vec![u32::from(CF_UNICODETEXT.0), MARKER_FORMAT, 0xC124, 0xC125],
            ),
        );

        assert_eq!(decision, RestoreDecision::Restore);
    }

    #[test]
    fn normalizes_line_endings_for_windows_clipboard_text() {
        assert_eq!(
            normalize_clipboard_text("one\ntwo\r\nthree\rfour"),
            "one\r\ntwo\r\nthree\r\nfour"
        );
    }

    #[test]
    fn delivery_modes_are_explicit() {
        assert_ne!(DeliveryMode::Paste, DeliveryMode::Clipboard);
    }

    #[test]
    fn post_prepare_guard_rejection_preserves_newer_external_clipboard() {
        assert_eq!(
            guard_rejection_action(RestoreDecision::Restore),
            GuardRejectionAction::CopyTranscript
        );
        assert_eq!(
            guard_rejection_action(RestoreDecision::SkipSequenceChanged),
            GuardRejectionAction::PreserveChangedClipboard
        );
        assert_eq!(
            guard_rejection_action(RestoreDecision::SkipHandoffChanged),
            GuardRejectionAction::PreserveChangedClipboard
        );
    }

    #[test]
    fn key_failure_before_paste_preserves_transcript_instead_of_restoring_snapshot() {
        assert_eq!(
            injection_failure_action(false),
            InjectionFailureAction::PreserveTranscript
        );
        assert_eq!(
            injection_failure_action(true),
            InjectionFailureAction::RestoreAfterDelay
        );
    }

    #[test]
    fn shutdown_wait_does_not_complete_with_a_pending_restore() {
        let barrier = Arc::new(RestorationBarrier::default());
        let permit = barrier.begin();

        assert!(!barrier.wait_for_completion(Duration::ZERO));
        drop(permit);
        assert!(barrier.wait_for_completion(Duration::ZERO));
    }

    #[test]
    fn discard_shutdown_cancels_delayed_restore_and_settles_barrier() {
        let barrier = Arc::new(RestorationBarrier::default());
        let permit = barrier.begin();

        barrier.cancel();
        assert!(barrier.wait_for_delay_or_cancel(Duration::from_secs(60)));
        drop(permit);
        assert!(barrier.wait_for_completion(Duration::ZERO));
    }
}
