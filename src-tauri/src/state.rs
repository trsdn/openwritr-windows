use parking_lot::RwLock;

pub struct App {
    pub active_ep: RwLock<String>,
}

impl App {
    pub fn new() -> Self {
        Self { active_ep: RwLock::new("probing…".into()) }
    }
}

// pulled in via tauri's transitive dep tree — declared here to surface the use
extern crate parking_lot;
