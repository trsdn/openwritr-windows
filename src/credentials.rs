use thiserror::Error;

const SERVICE: &str = "OpenWritr";
const USER: &str = "openai-compatible";
const TARGET: &str = "OpenWritr/OpenAICompatibleApiKey";

#[derive(Debug, Error)]
pub enum CredentialError {
    #[error("credential is empty")]
    Empty,
    #[error("Windows Credential Manager {operation} failed: {message}")]
    Backend {
        operation: &'static str,
        message: String,
    },
    #[error("credential verification failed")]
    VerificationFailed,
    #[error("stored credential conflicts with the legacy settings credential")]
    Conflict,
    #[error("credential transaction lock failed: {0}")]
    Lock(String),
}

pub trait CredentialStore {
    fn read(&self) -> Result<Option<String>, CredentialError>;
    fn write(&self, secret: &str) -> Result<(), CredentialError>;
    fn delete(&self) -> Result<(), CredentialError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WindowsCredentialStore;

impl WindowsCredentialStore {
    fn entry() -> Result<keyring::Entry, CredentialError> {
        keyring::Entry::new_with_target(TARGET, SERVICE, USER)
            .map_err(|error| backend_error("entry creation", error))
    }
}

impl CredentialStore for WindowsCredentialStore {
    fn read(&self) -> Result<Option<String>, CredentialError> {
        match Self::entry()?.get_password() {
            Ok(secret) => Ok(Some(secret)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(error) => Err(backend_error("read", error)),
        }
    }

    fn write(&self, secret: &str) -> Result<(), CredentialError> {
        if secret.is_empty() {
            return Err(CredentialError::Empty);
        }
        Self::entry()?
            .set_password(secret)
            .map_err(|error| backend_error("write", error))
    }

    fn delete(&self) -> Result<(), CredentialError> {
        match Self::entry()?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(error) => Err(backend_error("delete", error)),
        }
    }
}

pub fn read_openai_api_key() -> Result<Option<String>, CredentialError> {
    let _transaction =
        crate::single_instance::SettingsTransaction::acquire(&crate::paths::settings_path())
            .map_err(|error| CredentialError::Lock(error.to_string()))?;
    WindowsCredentialStore.read()
}

pub fn store_verified(store: &dyn CredentialStore, secret: &str) -> Result<(), CredentialError> {
    if secret.is_empty() {
        return Err(CredentialError::Empty);
    }
    store.write(secret)?;
    match store.read()? {
        Some(saved) if saved == secret => Ok(()),
        _ => Err(CredentialError::VerificationFailed),
    }
}

fn backend_error(operation: &'static str, error: keyring::Error) -> CredentialError {
    CredentialError::Backend {
        operation,
        message: error.to_string(),
    }
}
