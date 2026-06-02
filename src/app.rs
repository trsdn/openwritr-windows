//! App orchestration: tray + winit event loop + hotkey polling.

use crate::{audio::Recorder, hotkey, settings::Settings, tray};
use anyhow::Result;
use std::time::Instant;
use tracing::info;
use tray_icon::menu::MenuEvent;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

struct State {
    settings: Settings,
    recorder: Recorder,
    hotkey_mgr: hotkey::HotkeyManager,
    tray: tray::Tray,
    pressed: bool,
    record_started: Option<Instant>,
}

pub fn run() -> Result<()> {
    let settings = Settings::load();
    let recorder = Recorder::new()?;
    let hotkey_mgr = hotkey::HotkeyManager::register(&settings)?;
    let tray = tray::Tray::new(&settings)?;
    let state = State {
        settings,
        recorder,
        hotkey_mgr,
        tray,
        pressed: false,
        record_started: None,
    };

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = AppHandler { state };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct AppHandler { state: State }

impl ApplicationHandler for AppHandler {
    fn resumed(&mut self, _el: &ActiveEventLoop) {
        info!("event loop ready");
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, _ev: WindowEvent) {}

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        // Tray menu events.
        if let Ok(ev) = MenuEvent::receiver().try_recv() {
            if ev.id == self.state.tray.menu_quit_id {
                info!("quit via tray");
                el.exit();
                return;
            }
            if ev.id == self.state.tray.menu_settings_id {
                info!("settings clicked (UI not implemented in v0.2 skeleton)");
            }
        }

        // Hotkey edge polling.
        if let Some(event) = hotkey::poll_state(&self.state.hotkey_mgr, &mut self.state.pressed) {
            match event {
                hotkey::Event::Press => {
                    if self.state.record_started.is_none() {
                        self.state.recorder.start();
                        self.state.tray.set_color(tray::IconColor::Recording);
                        self.state.record_started = Some(Instant::now());
                        info!("recording start");
                    }
                }
                hotkey::Event::Release => {
                    if let Some(started) = self.state.record_started.take() {
                        let samples = self.state.recorder.stop();
                        self.state.tray.set_color(tray::IconColor::Idle);
                        let dur = started.elapsed();
                        if dur.as_secs_f32() < self.state.settings.min_record_seconds {
                            info!(secs = dur.as_secs_f32(), "below min — discarded");
                        } else {
                            info!(
                                secs = dur.as_secs_f32(),
                                samples = samples.len(),
                                sr = self.state.recorder.sample_rate,
                                "captured (ASR pending)"
                            );
                        }
                    }
                }
            }
        }

        hotkey::poll_sleep();
        el.set_control_flow(ControlFlow::Poll);
    }
}
