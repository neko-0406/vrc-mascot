//! Single-`EventLoop` 2-window manager (`winit` + `wgpu` + `egui` + Win32 + `tray-icon`).

pub mod perch;
pub mod tray;
pub mod win32;

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use glam::{Mat4, Vec2, Vec3};
use tray_icon::menu::MenuEvent;
use tray_icon::{MouseButton, MouseButtonState, TrayIconEvent};
use vrc_physics::SimInput;
use vrc_renderer::MascotRenderer;
use vrc_schema::{AudioLinkData, EmotionTag, MascotState};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{
    ElementState, Modifiers, MouseButton as WinitMouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

use crate::app_state::{current_time_of_day_label, AppState, PendingPerchAction};
use crate::status_ui::StatusUiState;
use perch::{
    PerchController, PerchStepInput, BASE_LOGICAL_HEIGHT, BASE_LOGICAL_WIDTH, MAX_SCALE, MIN_SCALE,
};
use tray::{TrayAction, TrayMenuSpec, TrayServiceHandle};

#[derive(Debug)]
enum UserEvent {
    TrayIcon(TrayIconEvent),
    Menu(MenuEvent),
    AvatarTick,
}

struct GpuContext {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct AvatarWindowBundle {
    window: Arc<Window>,
    hwnd: isize,
    surface: wgpu::Surface<'static>,
    surface_cfg: wgpu::SurfaceConfiguration,
    renderer: MascotRenderer,
}

struct StatusWindowBundle {
    window: Arc<Window>,
    hwnd: isize,
    surface: wgpu::Surface<'static>,
    surface_cfg: wgpu::SurfaceConfiguration,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    ui_state: StatusUiState,
}

/// Launches the single `winit` 2-window event loop and blocks until exit.
pub fn run_event_loop(app_state: AppState) -> anyhow::Result<()> {
    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let proxy_tray = event_loop.create_proxy();
    TrayIconEvent::set_event_handler(Some(move |ev| {
        let _ = proxy_tray.send_event(UserEvent::TrayIcon(ev));
    }));
    let proxy_menu = event_loop.create_proxy();
    MenuEvent::set_event_handler(Some(move |ev| {
        let _ = proxy_menu.send_event(UserEvent::Menu(ev));
    }));
    let proxy_tick = event_loop.create_proxy();

    let mut runner = MascotAppRunner::new(app_state, proxy_tick);
    event_loop.run_app(&mut runner)?;
    runner.shutdown_background_threads();
    Ok(())
}

struct MascotAppRunner {
    app_state: AppState,
    gpu: Option<GpuContext>,
    avatar: Option<AvatarWindowBundle>,
    status: Option<StatusWindowBundle>,
    tray: Option<TrayServiceHandle>,
    perch: PerchController,
    modifiers: Modifiers,
    grabbing_physbone: bool,
    drag_window_accel: Vec3,
    last_world_matrices: Vec<Mat4>,
    last_frame_time: Instant,
    next_frame_deadline: Instant,
    last_decision_check: Instant,
    last_env_poll: Instant,
    idle_seconds: f32,
    paused_by_fullscreen: bool,
    hidden_by_fullscreen: bool,
    last_fg_process: Option<String>,
    last_fg_title: Option<String>,
    visible_windows_cache: Vec<win32::ExternalWindowInfo>,
    audiolink_was_active: bool,
    fps_accum_frames: u32,
    fps_accum_time: f32,
    status_repaint_requested: bool,
    status_repaint_deadline: Option<Instant>,
    last_status_telemetry_redraw: Instant,
    status_visible_flag: Arc<AtomicBool>,
    live_tick_stop: Arc<AtomicBool>,
    live_tick_max_fps: Arc<AtomicU32>,
    live_tick_thread: Option<JoinHandle<()>>,
    live_tick_proxy: EventLoopProxy<UserEvent>,
}

impl Drop for MascotAppRunner {
    fn drop(&mut self) {
        self.shutdown_background_threads();
    }
}

impl MascotAppRunner {
    fn new(app_state: AppState, live_tick_proxy: EventLoopProxy<UserEvent>) -> Self {
        let work_area = win32::get_primary_work_area();
        let perch = PerchController::new(work_area, app_state.config.display.scale);
        let now = Instant::now();
        let status_visible_flag = Arc::new(AtomicBool::new(app_state.status_window_visible));
        let live_tick_stop = Arc::new(AtomicBool::new(false));
        let live_tick_max_fps = Arc::new(AtomicU32::new(
            app_state.config.display.max_fps.clamp(15, 240),
        ));

        Self {
            app_state,
            gpu: None,
            avatar: None,
            status: None,
            tray: None,
            perch,
            modifiers: Modifiers::default(),
            grabbing_physbone: false,
            drag_window_accel: Vec3::ZERO,
            last_world_matrices: Vec::new(),
            last_frame_time: now,
            next_frame_deadline: now,
            last_decision_check: now,
            last_env_poll: now - Duration::from_secs(1),
            idle_seconds: 0.0,
            paused_by_fullscreen: false,
            hidden_by_fullscreen: false,
            last_fg_process: None,
            last_fg_title: None,
            visible_windows_cache: Vec::new(),
            audiolink_was_active: false,
            fps_accum_frames: 0,
            fps_accum_time: 0.0,
            status_repaint_requested: true,
            status_repaint_deadline: None,
            last_status_telemetry_redraw: now - Duration::from_secs(1),
            status_visible_flag,
            live_tick_stop,
            live_tick_max_fps,
            live_tick_thread: None,
            live_tick_proxy,
        }
    }

    fn ensure_live_tick_thread(&mut self) {
        if self.live_tick_thread.is_some() {
            return;
        }
        let visible_flag = Arc::clone(&self.status_visible_flag);
        let stop_flag = Arc::clone(&self.live_tick_stop);
        let max_fps_atomic = Arc::clone(&self.live_tick_max_fps);
        let proxy = self.live_tick_proxy.clone();
        if let Ok(handle) = thread::Builder::new()
            .name("vrc-avatar-live-tick".to_owned())
            .spawn(move || {
                while !stop_flag.load(Ordering::Acquire) {
                    if !visible_flag.load(Ordering::Acquire) {
                        thread::park();
                        continue;
                    }
                    let fps = max_fps_atomic.load(Ordering::Relaxed).clamp(15, 240);
                    let interval = Duration::from_secs_f64(1.0 / f64::from(fps));
                    thread::sleep(interval);
                    if !stop_flag.load(Ordering::Acquire) && visible_flag.load(Ordering::Acquire) {
                        let _ = proxy.send_event(UserEvent::AvatarTick);
                    }
                }
            })
        {
            self.live_tick_thread = Some(handle);
        }
    }

    fn shutdown_background_threads(&mut self) {
        self.live_tick_stop.store(true, Ordering::Release);
        self.status_visible_flag.store(false, Ordering::Release);
        if let Some(handle) = self.live_tick_thread.take() {
            handle.thread().unpark();
            let _ = handle.join();
        }
        if let Some(tray) = &mut self.tray {
            tray.shutdown();
        }
        self.tray = None;
    }

    fn init_windows_and_gpu(&mut self, event_loop: &ActiveEventLoop) -> anyhow::Result<()> {
        let work_area = win32::get_primary_work_area();
        self.perch = PerchController::new(work_area, self.app_state.config.display.scale);
        let (init_w, init_h) = self.perch.size();
        let (init_x, init_y) = self.perch.position();

        #[allow(unused_mut)]
        let mut avatar_attrs = Window::default_attributes()
            .with_title("VRC Desktop Mascot")
            .with_decorations(false)
            .with_transparent(true)
            .with_inner_size(LogicalSize::new(init_w as f64, init_h as f64))
            .with_position(PhysicalPosition::new(init_x, init_y));
        #[cfg(windows)]
        {
            use winit::platform::windows::WindowAttributesExtWindows;
            avatar_attrs = avatar_attrs.with_no_redirection_bitmap(true);
        }
        let avatar_window = Arc::new(event_loop.create_window(avatar_attrs)?);
        let avatar_hwnd = win32::hwnd_from_winit(&avatar_window).unwrap_or(0);
        let dpi = avatar_window.scale_factor();
        self.perch.set_dpi_scale(dpi, work_area);
        let (adj_x, adj_y) = self.perch.position();
        avatar_window.set_outer_position(PhysicalPosition::new(adj_x, adj_y));

        win32::configure_avatar_window(
            avatar_hwnd,
            self.app_state.config.display.always_on_top,
            self.app_state.config.display.click_through_locked,
        );

        let status_icon =
            winit::window::Icon::from_rgba(tray::create_mascot_icon_rgba(), 32, 32).ok();
        let mut status_attrs = Window::default_attributes()
            .with_title("VRC Desktop Mascot - ステータス・設定")
            .with_inner_size(LogicalSize::new(920.0, 700.0))
            .with_visible(self.app_state.status_window_visible);
        if let Some(icon) = status_icon {
            status_attrs = status_attrs.with_window_icon(Some(icon));
        }
        let status_window = Arc::new(event_loop.create_window(status_attrs)?);
        let status_hwnd = win32::hwnd_from_winit(&status_window).unwrap_or(0);

        // Shared single wgpu Instance / Adapter / Device / Queue across both windows.
        // Prefer DX12 DirectComposition (`DxgiFromVisual`) on Windows for native transparent
        // swapchain compositing; fall back to default backends if DX12 adapter creation fails.
        let (instance, avatar_surface, status_surface, adapter) =
            create_instance_surfaces_and_adapter(&avatar_window, &status_window)?;
        let adapter_info = adapter.get_info();

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;

        // Configure transparent avatar surface
        let avatar_caps = avatar_surface.get_capabilities(&adapter);
        let avatar_format = avatar_caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Bgra8UnormSrgb)
            .or_else(|| avatar_caps.formats.iter().copied().find(|f| f.is_srgb()))
            .unwrap_or(avatar_caps.formats[0]);
        let avatar_alpha = avatar_caps
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == wgpu::CompositeAlphaMode::PreMultiplied)
            .or_else(|| {
                avatar_caps
                    .alpha_modes
                    .iter()
                    .copied()
                    .find(|m| *m == wgpu::CompositeAlphaMode::PostMultiplied)
            })
            .unwrap_or(avatar_caps.alpha_modes[0]);

        let av_phys = avatar_window.inner_size();
        let avatar_surface_cfg = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: avatar_format,
            width: av_phys.width.max(1),
            height: av_phys.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: avatar_alpha,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: Default::default(),
        };
        avatar_surface.configure(&device, &avatar_surface_cfg);

        let mut renderer = MascotRenderer::new(
            &device,
            &queue,
            avatar_format,
            avatar_surface_cfg.width,
            avatar_surface_cfg.height,
        );
        if let Some((model, manifest)) = self.app_state.pending_model_reload.take() {
            self.last_world_matrices = model.compute_rest_world_matrices();
            renderer.set_model(&device, &queue, model, &manifest);
        }

        // Configure Status & Settings egui surface
        let status_caps = status_surface.get_capabilities(&adapter);
        let status_format = status_caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                *f == wgpu::TextureFormat::Bgra8Unorm || *f == wgpu::TextureFormat::Rgba8Unorm
            })
            .or_else(|| status_caps.formats.iter().copied().find(|f| !f.is_srgb()))
            .unwrap_or(status_caps.formats[0]);
        let status_alpha = status_caps
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == wgpu::CompositeAlphaMode::Opaque)
            .unwrap_or(status_caps.alpha_modes[0]);
        let status_present_mode = [
            wgpu::PresentMode::AutoNoVsync,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::AutoVsync,
        ]
        .into_iter()
        .find(|m| status_caps.present_modes.contains(m))
        .unwrap_or(wgpu::PresentMode::AutoVsync);
        let st_phys = status_window.inner_size();
        let status_surface_cfg = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: status_format,
            width: st_phys.width.max(1),
            height: st_phys.height.max(1),
            present_mode: status_present_mode,
            alpha_mode: status_alpha,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: Default::default(),
        };
        status_surface.configure(&device, &status_surface_cfg);

        let egui_ctx = egui::Context::default();
        StatusUiState::configure_fonts(&egui_ctx);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &status_window,
            Some(status_window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            status_format,
            egui_wgpu::RendererOptions::default(),
        );

        let tray = TrayServiceHandle::spawn(&self.build_tray_spec())?;

        if self.app_state.status_window_visible {
            status_window.set_visible(true);
            status_window.focus_window();
            status_window.request_redraw();
        }
        avatar_window.request_redraw();

        tracing::info!(
            adapter = %adapter_info.name,
            backend = ?adapter_info.backend,
            avatar_format = ?avatar_format,
            avatar_alpha = ?avatar_alpha,
            avatar_pos = ?(adj_x, adj_y),
            avatar_size = ?(avatar_surface_cfg.width, avatar_surface_cfg.height),
            status_format = ?status_format,
            status_alpha = ?status_alpha,
            status_visible = self.app_state.status_window_visible,
            "Initialized GPU, windows, and system tray"
        );

        self.gpu = Some(GpuContext {
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
        });
        self.avatar = Some(AvatarWindowBundle {
            window: avatar_window,
            hwnd: avatar_hwnd,
            surface: avatar_surface,
            surface_cfg: avatar_surface_cfg,
            renderer,
        });
        self.status = Some(StatusWindowBundle {
            window: status_window,
            hwnd: status_hwnd,
            surface: status_surface,
            surface_cfg: status_surface_cfg,
            egui_ctx,
            egui_state,
            egui_renderer,
            ui_state: StatusUiState::default(),
        });
        self.tray = Some(tray);
        self.ensure_live_tick_thread();
        if self.app_state.status_window_visible {
            if let Some(handle) = &self.live_tick_thread {
                handle.thread().unpark();
            }
        }

        Ok(())
    }

    fn build_tray_spec(&self) -> TrayMenuSpec<'_> {
        TrayMenuSpec {
            avatars: self.app_state.registry.entries(),
            active_avatar_id: self.app_state.active_avatar_id.as_deref(),
            expressions: &self.app_state.manifest.expressions.menu,
            autonomous_enabled: self.app_state.config.motion.autonomous_enabled,
            click_through_locked: self.app_state.config.display.click_through_locked,
            status_window_visible: self.app_state.status_window_visible,
        }
    }

    fn rebuild_tray_if_dirty(&mut self) {
        if !self.app_state.menu_dirty {
            return;
        }
        self.app_state.menu_dirty = false;
        let spec = TrayMenuSpec {
            avatars: self.app_state.registry.entries(),
            active_avatar_id: self.app_state.active_avatar_id.as_deref(),
            expressions: &self.app_state.manifest.expressions.menu,
            autonomous_enabled: self.app_state.config.motion.autonomous_enabled,
            click_through_locked: self.app_state.config.display.click_through_locked,
            status_window_visible: self.app_state.status_window_visible,
        };
        if let Some(tray) = &mut self.tray {
            if let Err(err) = tray.rebuild(&spec) {
                tracing::warn!(error = %err, "Failed to rebuild tray menu");
            }
        }
    }

    fn set_status_window_visible(&mut self, visible: bool) {
        self.app_state.status_window_visible = visible;
        self.app_state.menu_dirty = true;
        self.status_visible_flag.store(visible, Ordering::Release);
        if visible {
            self.status_repaint_requested = true;
            self.ensure_live_tick_thread();
            if let Some(handle) = &self.live_tick_thread {
                handle.thread().unpark();
            }
        }
        if let Some(status) = &self.status {
            status.window.set_visible(visible);
            if visible {
                status.window.focus_window();
                status.window.request_redraw();
            }
        }
    }

    fn handle_tray_action(&mut self, action: TrayAction, event_loop: &ActiveEventLoop) {
        match action {
            TrayAction::ToggleStatusWindow => {
                let next = !self.app_state.status_window_visible;
                self.set_status_window_visible(next);
            }
            TrayAction::SwitchAvatar(avatar_id) => {
                self.app_state.switch_avatar(avatar_id.as_deref());
            }
            TrayAction::TriggerExpression(item_name) => {
                self.app_state.simulation.trigger_menu_item(&item_name);
            }
            TrayAction::ResetPosition => {
                let work_area = win32::get_primary_work_area();
                self.perch.reset_position(work_area);
                let (px, py) = self.perch.position();
                if let Some(av) = &self.avatar {
                    av.window.set_outer_position(PhysicalPosition::new(px, py));
                }
            }
            TrayAction::ToggleAutonomous => {
                let next = !self.app_state.config.motion.autonomous_enabled;
                self.app_state.config.motion.autonomous_enabled = next;
                self.app_state.draft_config.motion.autonomous_enabled = next;
                self.app_state.menu_dirty = true;
            }
            TrayAction::ToggleClickThroughLock => {
                let next = !self.app_state.config.display.click_through_locked;
                self.app_state.config.display.click_through_locked = next;
                self.app_state.draft_config.display.click_through_locked = next;
                if let Some(av) = &self.avatar {
                    if next {
                        win32::set_click_through(av.hwnd, true);
                    }
                }
                self.app_state.menu_dirty = true;
            }
            TrayAction::Exit => {
                self.shutdown_background_threads();
                event_loop.exit();
            }
        }
    }

    fn apply_scale_change(&mut self, new_scale: f32) {
        let clamped = new_scale.clamp(MIN_SCALE, MAX_SCALE);
        self.app_state.config.display.scale = clamped;
        self.app_state.draft_config.display.scale = clamped;
        let work_area = win32::get_primary_work_area();
        self.perch.set_scale(clamped, work_area);
        let (lw, lh) = self.perch.size();
        let (px, py) = self.perch.position();
        if let Some(av) = &self.avatar {
            let _ = av
                .window
                .request_inner_size(LogicalSize::new(lw as f64, lh as f64));
            av.window.set_outer_position(PhysicalPosition::new(px, py));
        }
    }

    fn apply_config_effects_to_windows(&mut self) {
        self.live_tick_max_fps.store(
            self.app_state.config.display.max_fps.clamp(15, 240),
            Ordering::Relaxed,
        );
        let work_area = win32::get_primary_work_area();
        if (self.app_state.config.display.scale - self.perch.scale()).abs() > 1e-3 {
            self.apply_scale_change(self.app_state.config.display.scale);
        }
        if let Some(av) = &self.avatar {
            win32::set_always_on_top(av.hwnd, self.app_state.config.display.always_on_top);
            if self.app_state.config.display.click_through_locked {
                win32::set_click_through(av.hwnd, true);
            }
        }
        let _ = work_area;
    }

    fn trigger_autonomous_decision_now(&mut self) {
        let ctx = vrc_ai::sanitize_decision_context(
            self.idle_seconds,
            self.app_state.simulation.pet_count_recent(),
            self.last_fg_process.as_deref(),
            self.last_fg_title.as_deref(),
            current_time_of_day_label(),
            self.app_state.current_mood.clone(),
            &self.app_state.config.ai,
            &self.app_state.config.motion.perch_exclude_apps,
        );
        self.app_state.trigger_decision(ctx);
    }

    fn step_simulation_and_render_avatar(&mut self, dt: f32) {
        // Upload any newly loaded/imported avatar model before stepping.
        if let Some((model, manifest)) = self.app_state.pending_model_reload.take() {
            if let (Some(gpu), Some(av)) = (&self.gpu, &mut self.avatar) {
                self.last_world_matrices = model.compute_rest_world_matrices();
                av.renderer
                    .set_model(&gpu.device, &gpu.queue, model, &manifest);
            }
        }

        let status_hwnd = self.status.as_ref().map(|s| s.hwnd).unwrap_or(0);
        let Some(gpu) = &self.gpu else {
            return;
        };
        let Some(av) = &mut self.avatar else {
            return;
        };
        let exclude_hwnds = [av.hwnd, status_hwnd];

        // Non-blocking hit-test staging readback
        av.renderer.poll_hit_test_readback(&gpu.device);

        // Execute pending perch actions from Decisions AI
        let work_area = win32::get_primary_work_area();
        let taskbar_rect = win32::get_taskbar_rect();
        if let Some(action) = self.app_state.pending_perch_action.take() {
            match action {
                PendingPerchAction::MoveToTaskbar => {
                    self.perch.move_to_taskbar(work_area, taskbar_rect);
                }
                PendingPerchAction::SitOnWindow => {
                    self.visible_windows_cache = win32::enumerate_visible_windows(&exclude_hwnds);
                    let _ = self.perch.sit_on_top_window(
                        &self.visible_windows_cache,
                        &self.app_state.config.motion.perch_exclude_apps,
                        work_area,
                    );
                }
            }
        }

        // Poll desktop environment (~4 Hz) for fullscreen state & visible windows
        if self.last_env_poll.elapsed() >= Duration::from_millis(250) {
            self.last_env_poll = Instant::now();
            self.visible_windows_cache = win32::enumerate_visible_windows(&exclude_hwnds);
            if let Some(fg) = win32::get_foreground_window_info(&exclude_hwnds) {
                self.last_fg_process = Some(fg.process_name);
                self.last_fg_title = Some(fg.title);
                let hide = fg.is_fullscreen && self.app_state.config.motion.hide_on_fullscreen;
                let pause = fg.is_fullscreen && self.app_state.config.motion.pause_on_fullscreen;
                if hide != self.hidden_by_fullscreen {
                    self.hidden_by_fullscreen = hide;
                    av.window.set_visible(!hide);
                }
                self.paused_by_fullscreen = pause;
            } else {
                if self.hidden_by_fullscreen {
                    self.hidden_by_fullscreen = false;
                    av.window.set_visible(true);
                }
                self.paused_by_fullscreen = false;
            }
        }

        if self.paused_by_fullscreen || self.hidden_by_fullscreen {
            return;
        }

        // Handle active window dragging or PhysBone grab via global cursor polling
        let global_cursor = win32::get_cursor_pos();
        let local_cursor =
            global_cursor.and_then(|(gx, gy)| win32::screen_to_client(av.hwnd, gx, gy));

        if self.perch.state() == MascotState::Dragged {
            if let Some(gc) = global_cursor {
                self.drag_window_accel = self.perch.update_drag(gc, dt, work_area);
                self.idle_seconds = 0.0;
            }
        } else {
            self.drag_window_accel = Vec3::ZERO;
        }

        if self.grabbing_physbone {
            if let Some((cx, cy)) = local_cursor {
                let plane_z = av.renderer.camera().target.z;
                let world_pt = av
                    .renderer
                    .screen_ray_to_world_plane(cx as f32, cy as f32, plane_z);
                self.app_state.simulation.update_grab(world_pt);
                self.idle_seconds = 0.0;
            }
        }

        // Step perch & gravity state machine
        let target_win_rect = self.perch.seated_hwnd().and_then(win32::query_window_rect);
        let perch_out = self.perch.step(&PerchStepInput {
            dt,
            work_area,
            taskbar_rect,
            visible_windows: &self.visible_windows_cache,
            target_window_rect: target_win_rect,
            motion_cfg: &self.app_state.config.motion,
            scale: self.app_state.config.display.scale,
        });
        self.app_state.mascot_state = perch_out.state;
        if perch_out.position_changed {
            av.window.set_outer_position(PhysicalPosition::new(
                perch_out.position.0,
                perch_out.position.1,
            ));
        }

        // Dynamic click-through toggle from offscreen HitTestPass
        let surf_w = av.surface_cfg.width.max(1);
        let surf_h = av.surface_cfg.height.max(1);
        if self.app_state.config.display.click_through_locked {
            win32::set_click_through(av.hwnd, true);
        } else if self.perch.state() == MascotState::Dragged || self.grabbing_physbone {
            win32::set_click_through(av.hwnd, false);
        } else if let Some((cx, cy)) = local_cursor {
            if cx >= 0 && cy >= 0 && (cx as u32) < surf_w && (cy as u32) < surf_h {
                let hit = av.renderer.hit_test(cx as u32, cy as u32);
                win32::set_click_through(av.hwnd, !hit.is_opaque);
            } else {
                win32::set_click_through(av.hwnd, true);
            }
        }

        // Project Head node for petting detection & compute cursor_ndc for gaze tracking
        let head_path = self
            .app_state
            .manifest
            .humanoid_bones
            .get("Head")
            .map(String::as_str)
            .unwrap_or("Head");
        let head_screen_pos = av
            .renderer
            .project_node_screen(head_path, &self.last_world_matrices)
            .map(Vec2::from);

        let (cursor_ndc, cursor_screen_pos) = if let Some((cx, cy)) = local_cursor {
            let nx = ((cx as f32 / surf_w as f32) * 2.0 - 1.0).clamp(-1.5, 1.5);
            let ny = (1.0 - (cy as f32 / surf_h as f32) * 2.0).clamp(-1.5, 1.5);
            let in_bounds = cx >= 0 && cy >= 0 && (cx as u32) < surf_w && (cy as u32) < surf_h;
            (
                Some(Vec2::new(nx, ny)),
                if in_bounds {
                    Some(Vec2::new(cx as f32, cy as f32))
                } else {
                    None
                },
            )
        } else {
            (None, None)
        };

        let viseme_weights = self.app_state.sample_current_visemes();
        let combined_accel = self.drag_window_accel + perch_out.window_accel;

        let sim_input = SimInput {
            window_accel: combined_accel,
            mascot_state: perch_out.state,
            cursor_ndc,
            head_screen_pos,
            cursor_screen_pos,
            viseme_weights,
        };
        let sim_out = self.app_state.simulation.step(
            dt,
            &sim_input,
            &self.app_state.config.motion,
            &self.app_state.config.physics_shader,
        );
        self.last_world_matrices = sim_out.world_matrices;

        if sim_out.petting_triggered {
            self.idle_seconds = 0.0;
            self.app_state.current_mood = EmotionTag::Happy;
            self.app_state.simulation.set_emotion(EmotionTag::Happy);
        }

        av.renderer.update_pose(
            &gpu.queue,
            &self.last_world_matrices,
            &sim_out.blendshape_weights,
            &sim_out.node_visibility,
        );

        if self.app_state.config.physics_shader.audiolink_enabled {
            let al = self.app_state.audiolink_capture.latest_data();
            av.renderer.update_audiolink(&gpu.queue, &al);
            self.audiolink_was_active = true;
        } else if self.audiolink_was_active {
            av.renderer
                .update_audiolink(&gpu.queue, &AudioLinkData::default());
            self.audiolink_was_active = false;
        }

        // Render avatar frame
        match av.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("vrc_avatar_encoder"),
                        });
                av.renderer.render(
                    &gpu.device,
                    &gpu.queue,
                    &mut encoder,
                    &view,
                    &self.app_state.config.physics_shader,
                );
                drop(view);
                gpu.queue.submit(Some(encoder.finish()));
                gpu.queue.present(frame);
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                av.surface.configure(&gpu.device, &av.surface_cfg);
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {}
            wgpu::CurrentSurfaceTexture::Validation => {
                tracing::warn!("Avatar surface validation error");
            }
        }
    }

    fn render_status_window(&mut self) {
        if !self.app_state.status_window_visible {
            return;
        }
        let Some(gpu) = &self.gpu else {
            return;
        };
        let Some(status) = &mut self.status else {
            return;
        };

        let raw_input = status.egui_state.take_egui_input(&status.window);
        let mut effects = Default::default();
        let full_output = status.egui_ctx.run_ui(raw_input, |ui| {
            effects = status.ui_state.draw(ui, &mut self.app_state);
        });

        status
            .egui_state
            .handle_platform_output(&status.window, full_output.platform_output);

        let clipped_primitives = status
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [
                status.surface_cfg.width.max(1),
                status.surface_cfg.height.max(1),
            ],
            pixels_per_point: status.window.scale_factor() as f32,
        };

        for (id, image_deltas) in &full_output.textures_delta.set {
            for image_delta in image_deltas {
                status
                    .egui_renderer
                    .update_texture(&gpu.device, &gpu.queue, *id, image_delta);
            }
        }

        match status.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("vrc_status_egui_encoder"),
                        });
                let user_cmds = status.egui_renderer.update_buffers(
                    &gpu.device,
                    &gpu.queue,
                    &mut encoder,
                    &clipped_primitives,
                    &screen_descriptor,
                );
                {
                    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("vrc_status_egui_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.10,
                                    g: 0.11,
                                    b: 0.13,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    let mut pass = pass.forget_lifetime();
                    status
                        .egui_renderer
                        .render(&mut pass, &clipped_primitives, &screen_descriptor);
                }

                let mut submit_cmds = user_cmds;
                submit_cmds.push(encoder.finish());
                drop(view);
                gpu.queue.submit(submit_cmds);
                gpu.queue.present(frame);
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                status.surface.configure(&gpu.device, &status.surface_cfg);
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {}
            wgpu::CurrentSurfaceTexture::Validation => {
                tracing::warn!("Status surface validation error");
            }
        }

        for id in &full_output.textures_delta.free {
            status.egui_renderer.free_texture(id);
        }

        if let Some(vp) = full_output.viewport_output.get(&egui::ViewportId::ROOT) {
            if vp.repaint_delay.is_zero() {
                self.status_repaint_requested = true;
            } else if vp.repaint_delay < Duration::from_secs(10) {
                self.status_repaint_deadline = Some(Instant::now() + vp.repaint_delay);
            }
        }

        // Apply UI-requested immediate effects outside the `status` borrow
        if effects.reset_position {
            let work_area = win32::get_primary_work_area();
            self.perch.reset_position(work_area);
            let (px, py) = self.perch.position();
            if let Some(av) = &self.avatar {
                av.window.set_outer_position(PhysicalPosition::new(px, py));
            }
        }
        if effects.config_applied {
            self.apply_config_effects_to_windows();
        }
        if effects.trigger_decision {
            self.trigger_autonomous_decision_now();
        }
    }
}

impl ApplicationHandler<UserEvent> for MascotAppRunner {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            if let Err(err) = self.init_windows_and_gpu(event_loop) {
                tracing::error!(error = %err, "Failed to initialize windows & GPU");
                event_loop.exit();
            }
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::TrayIcon(TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            })
            | UserEvent::TrayIcon(TrayIconEvent::DoubleClick {
                button: MouseButton::Left,
                ..
            }) => {
                let next = !self.app_state.status_window_visible;
                self.set_status_window_visible(next);
            }
            UserEvent::TrayIcon(_) => {}
            UserEvent::Menu(menu_ev) => {
                if let Some(action) = self
                    .tray
                    .as_ref()
                    .and_then(|t| t.resolve_action(&menu_ev.id))
                {
                    self.handle_tray_action(action, event_loop);
                }
            }
            UserEvent::AvatarTick => {
                let now = Instant::now();
                if now >= self.next_frame_deadline {
                    let max_fps = self.app_state.config.display.max_fps.clamp(15, 240);
                    let frame_interval = Duration::from_secs_f64(1.0 / f64::from(max_fps));
                    let elapsed_sec = (now - self.last_frame_time).as_secs_f32().clamp(0.0, 0.25);
                    if self.perch.state() != MascotState::Dragged && !self.grabbing_physbone {
                        self.idle_seconds += elapsed_sec;
                    }
                    self.next_frame_deadline = now + frame_interval;
                    if !self.hidden_by_fullscreen {
                        if let Some(av) = &self.avatar {
                            av.window.request_redraw();
                        }
                    }
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let avatar_id = self.avatar.as_ref().map(|a| a.window.id());
        let status_id = self.status.as_ref().map(|s| s.window.id());

        if Some(status_id) == Some(Some(window_id)) {
            if let Some(status) = &mut self.status {
                let resp = status.egui_state.on_window_event(&status.window, &event);
                if resp.repaint && self.app_state.status_window_visible {
                    self.status_repaint_requested = true;
                    status.window.request_redraw();
                }
            }
            match event {
                WindowEvent::CloseRequested => {
                    // Never exit on Status Window close — hide and remain resident in tray!
                    self.set_status_window_visible(false);
                }
                WindowEvent::Resized(new_size) => {
                    if new_size.width > 0 && new_size.height > 0 {
                        if let (Some(gpu), Some(status)) = (&self.gpu, &mut self.status) {
                            status.surface_cfg.width = new_size.width;
                            status.surface_cfg.height = new_size.height;
                            status.surface.configure(&gpu.device, &status.surface_cfg);
                            self.status_repaint_requested = true;
                        }
                    }
                }
                WindowEvent::RedrawRequested if self.app_state.status_window_visible => {
                    self.render_status_window();
                }
                _ => {}
            }
            return;
        }

        if Some(avatar_id) == Some(Some(window_id)) {
            match event {
                WindowEvent::CloseRequested => {
                    self.shutdown_background_threads();
                    event_loop.exit();
                }
                WindowEvent::ModifiersChanged(mods) => {
                    self.modifiers = mods;
                }
                WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                    let wa = win32::get_primary_work_area();
                    self.perch.set_dpi_scale(scale_factor, wa);
                }
                WindowEvent::Resized(new_size) => {
                    if new_size.width > 0 && new_size.height > 0 {
                        if let (Some(gpu), Some(av)) = (&self.gpu, &mut self.avatar) {
                            av.surface_cfg.width = new_size.width;
                            av.surface_cfg.height = new_size.height;
                            av.surface.configure(&gpu.device, &av.surface_cfg);
                            av.renderer
                                .resize(&gpu.device, new_size.width, new_size.height);
                        }
                    }
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: WinitMouseButton::Left,
                    ..
                } => {
                    if self.app_state.config.display.click_through_locked {
                        return;
                    }
                    self.idle_seconds = 0.0;
                    if self.modifiers.state().shift_key() {
                        if let Some(av) = &self.avatar {
                            if let Some((gx, gy)) = win32::get_cursor_pos() {
                                if let Some((cx, cy)) = win32::screen_to_client(av.hwnd, gx, gy) {
                                    let plane_z = av.renderer.camera().target.z;
                                    let world_pt = av
                                        .renderer
                                        .screen_ray_to_world_plane(cx as f32, cy as f32, plane_z);
                                    if self
                                        .app_state
                                        .simulation
                                        .grab_physbone_near(world_pt, 0.35)
                                        .is_some()
                                    {
                                        self.grabbing_physbone = true;
                                    }
                                }
                            }
                        }
                    } else if let Some(gc) = win32::get_cursor_pos() {
                        self.perch.begin_drag(gc);
                    }
                }
                WindowEvent::MouseInput {
                    state: ElementState::Released,
                    button: WinitMouseButton::Left,
                    ..
                } => {
                    if self.grabbing_physbone {
                        self.app_state.simulation.release_grab();
                        self.grabbing_physbone = false;
                    }
                    if self.perch.state() == MascotState::Dragged {
                        self.perch
                            .end_drag(self.app_state.config.motion.gravity_enabled);
                    }
                }
                WindowEvent::MouseInput {
                    state: ElementState::Released,
                    button: WinitMouseButton::Right,
                    ..
                } => {
                    if let Some(tray) = &self.tray {
                        tray.show_context_menu();
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    if self.app_state.config.display.click_through_locked {
                        return;
                    }
                    let dy = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) / 120.0,
                    };
                    if dy.abs() > 1e-4 {
                        let next_scale = (self.app_state.config.display.scale + dy * 0.1)
                            .clamp(MIN_SCALE, MAX_SCALE);
                        self.apply_scale_change(next_scale);
                        self.idle_seconds = 0.0;
                    }
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let dt = (now - self.last_frame_time)
                        .as_secs_f32()
                        .clamp(1.0 / 240.0, 0.1);
                    self.last_frame_time = now;

                    self.fps_accum_frames += 1;
                    self.fps_accum_time += dt;
                    if self.fps_accum_time >= 0.5 {
                        self.app_state.fps_estimate =
                            self.fps_accum_frames as f32 / self.fps_accum_time;
                        self.fps_accum_frames = 0;
                        self.fps_accum_time = 0.0;
                    }

                    self.step_simulation_and_render_avatar(dt);
                }
                _ => {}
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.app_state.poll_events();
        self.rebuild_tray_if_dirty();

        let now = Instant::now();
        let max_fps = self.app_state.config.display.max_fps.clamp(15, 240);
        let frame_interval = Duration::from_secs_f64(1.0 / f64::from(max_fps));

        // Periodic autonomous behavior timer (uses remote AI when enabled/available, or local fallback)
        if self.app_state.config.motion.autonomous_enabled && !self.paused_by_fullscreen {
            let interval =
                Duration::from_secs(self.app_state.config.ai.decision_interval_sec.max(5));
            if now.duration_since(self.last_decision_check) >= interval {
                self.last_decision_check = now;
                self.trigger_autonomous_decision_now();
            }
        } else {
            self.last_decision_check = now;
        }

        if now >= self.next_frame_deadline {
            let elapsed_sec = (now - self.last_frame_time).as_secs_f32().clamp(0.0, 0.25);
            if self.perch.state() != MascotState::Dragged && !self.grabbing_physbone {
                self.idle_seconds += elapsed_sec;
            }
            self.next_frame_deadline = now + frame_interval;

            if !self.hidden_by_fullscreen {
                if let Some(av) = &self.avatar {
                    av.window.request_redraw();
                }
            }
        }

        // Redraw Status & Settings window on demand (egui repaint or ~4 Hz telemetry refresh)
        // so it never VSync-stalls the 3D avatar window.
        if self.app_state.status_window_visible {
            let due_for_egui = self.status_repaint_requested
                || self.status_repaint_deadline.is_some_and(|d| now >= d);
            let due_for_telemetry =
                now.duration_since(self.last_status_telemetry_redraw) >= Duration::from_millis(250);
            if due_for_egui || due_for_telemetry {
                self.status_repaint_requested = false;
                self.status_repaint_deadline = None;
                self.last_status_telemetry_redraw = now;
                if let Some(status) = &self.status {
                    status.window.request_redraw();
                }
            }
        }

        let _ = (BASE_LOGICAL_WIDTH, BASE_LOGICAL_HEIGHT);
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame_deadline));
    }
}

fn create_instance_surfaces_and_adapter(
    avatar_window: &Arc<Window>,
    status_window: &Arc<Window>,
) -> anyhow::Result<(
    wgpu::Instance,
    wgpu::Surface<'static>,
    wgpu::Surface<'static>,
    wgpu::Adapter,
)> {
    let mut dx12_desc = wgpu::InstanceDescriptor::new_without_display_handle();
    dx12_desc.backends = wgpu::Backends::DX12;
    dx12_desc.backend_options.dx12.presentation_system = wgpu::Dx12SwapchainKind::DxgiFromVisual;
    let dx12_instance = wgpu::Instance::new(dx12_desc);
    match try_create_surfaces_and_adapter(dx12_instance, avatar_window, status_window) {
        Ok(res) => Ok(res),
        Err(err) => {
            tracing::warn!(
                error = %err,
                "DX12 DirectComposition setup failed; falling back to default wgpu backends"
            );
            try_create_surfaces_and_adapter(wgpu::Instance::default(), avatar_window, status_window)
        }
    }
}

fn try_create_surfaces_and_adapter(
    instance: wgpu::Instance,
    avatar_window: &Arc<Window>,
    status_window: &Arc<Window>,
) -> anyhow::Result<(
    wgpu::Instance,
    wgpu::Surface<'static>,
    wgpu::Surface<'static>,
    wgpu::Adapter,
)> {
    let avatar_surface = instance.create_surface(Arc::clone(avatar_window))?;
    let status_surface = instance.create_surface(Arc::clone(status_window))?;
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: Some(&avatar_surface),
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .or_else(|_| {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&avatar_surface),
            force_fallback_adapter: true,
            apply_limit_buckets: false,
        }))
    })?;
    Ok((instance, avatar_surface, status_surface, adapter))
}
