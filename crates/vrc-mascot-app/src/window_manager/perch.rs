//! Pure unit-testable window positioning, scaling, drag inertia, gravity fall,
//! and desktop window-perch state machine (`PerchController`).

use glam::Vec3;
use vrc_ai::is_excluded_app;
use vrc_schema::{MascotState, MotionConfig};

use super::win32::{ExternalWindowInfo, ScreenRect};

/// Base logical width of the transparent mascot window (`1.0x` scale).
pub const BASE_LOGICAL_WIDTH: f32 = 400.0;
/// Base logical height of the transparent mascot window (`1.0x` scale).
pub const BASE_LOGICAL_HEIGHT: f32 = 600.0;
/// Minimum allowed display scale.
pub const MIN_SCALE: f32 = 0.5;
/// Maximum allowed display scale.
pub const MAX_SCALE: f32 = 2.0;

const GRAVITY_PX_PER_SEC2: f32 = 1800.0;
const MAX_FALL_SPEED_PX_PER_SEC: f32 = 2400.0;
const PIXEL_TO_WORLD_ACCEL: f32 = 0.0025;
const MAX_WORLD_ACCEL: f32 = 28.0;
const BASE_GLIDE_SPEED_PX_PER_SEC: f32 = 520.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlideGoal {
    Taskbar {
        target_x: i32,
        target_y: i32,
    },
    Window {
        hwnd: isize,
        rel_x: i32,
        target_y: i32,
    },
}

/// Per-frame external environment snapshot passed into [`PerchController::step`].
#[derive(Debug, Clone)]
pub struct PerchStepInput<'a> {
    pub dt: f32,
    pub work_area: ScreenRect,
    pub taskbar_rect: Option<ScreenRect>,
    pub visible_windows: &'a [ExternalWindowInfo],
    pub target_window_rect: Option<ScreenRect>,
    pub motion_cfg: &'a MotionConfig,
    pub scale: f32,
}

/// Output of one [`PerchController::step`] evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerchStepOutput {
    pub position: (i32, i32),
    pub size: (u32, u32),
    pub state: MascotState,
    pub seated_hwnd: Option<isize>,
    pub window_accel: Vec3,
    pub position_changed: bool,
    pub size_changed: bool,
}

/// Stateful controller managing the mascot window's screen position, scale,
/// drag velocity inertia, gravity fall, and window-top perching.
#[derive(Debug, Clone)]
pub struct PerchController {
    x: f32,
    y: f32,
    scale: f32,
    dpi_scale: f64,
    state: MascotState,
    seated_hwnd: Option<isize>,
    seated_rel_x: i32,
    velocity_x: f32,
    velocity_y: f32,
    prev_drag_vx: f32,
    prev_drag_vy: f32,
    drag_offset_x: i32,
    drag_offset_y: i32,
    last_reported_pos: (i32, i32),
    glide_goal: Option<GlideGoal>,
    pending_impulse: Vec3,
}

impl PerchController {
    /// Creates a new `PerchController` positioned at the bottom-right of `work_area`
    /// (`SeatedOnTaskbar`).
    pub fn new(work_area: ScreenRect, scale: f32) -> Self {
        let clamped_scale = sanitize_scale(scale);
        let (w, h) = physical_size_i32(clamped_scale, 1.0);
        let (init_x, init_y) =
            clamp_pos(work_area.right - w, work_area.bottom - h, w, h, work_area);

        Self {
            x: init_x as f32,
            y: init_y as f32,
            scale: clamped_scale,
            dpi_scale: 1.0,
            state: MascotState::SeatedOnTaskbar,
            seated_hwnd: None,
            seated_rel_x: 0,
            velocity_x: 0.0,
            velocity_y: 0.0,
            prev_drag_vx: 0.0,
            prev_drag_vy: 0.0,
            drag_offset_x: 0,
            drag_offset_y: 0,
            last_reported_pos: (init_x, init_y),
            glide_goal: None,
            pending_impulse: Vec3::ZERO,
        }
    }

    /// Updates monitor DPI scale factor (`>= 1.0`) used when converting logical window size
    /// to physical Win32 screen coordinates.
    pub fn set_dpi_scale(&mut self, dpi_scale: f64, work_area: ScreenRect) {
        let clamped_dpi = if dpi_scale.is_finite() {
            dpi_scale.clamp(0.5, 4.0)
        } else {
            1.0
        };
        if (clamped_dpi - self.dpi_scale).abs() < 1e-4 {
            return;
        }
        let (old_w, old_h) = physical_size_i32(self.scale, self.dpi_scale);
        let (new_w, new_h) = physical_size_i32(self.scale, clamped_dpi);
        let bottom_center_x = self.x + (old_w as f32) * 0.5;
        let bottom_y = self.y + (old_h as f32);
        let raw_x = (bottom_center_x - (new_w as f32) * 0.5).round() as i32;
        let raw_y = (bottom_y - (new_h as f32)).round() as i32;
        let (cx, cy) = clamp_pos(raw_x, raw_y, new_w, new_h, work_area);
        self.dpi_scale = clamped_dpi;
        self.x = cx as f32;
        self.y = cy as f32;
    }

    /// Current top-left position in screen coordinates.
    pub fn position(&self) -> (i32, i32) {
        (self.x.round() as i32, self.y.round() as i32)
    }

    /// Current logical window size `(width, height)`.
    pub fn size(&self) -> (u32, u32) {
        let (w, h) = logical_size_i32(self.scale);
        (w as u32, h as u32)
    }

    /// Current scale factor in `0.5..=2.0`.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Current mascot movement/perch state.
    pub fn state(&self) -> MascotState {
        self.state
    }

    /// HWND of the window the mascot is currently perched on, if `SeatedOnWindow`.
    pub fn seated_hwnd(&self) -> Option<isize> {
        self.seated_hwnd
    }

    /// Resets position back to bottom-right of `work_area` (`SeatedOnTaskbar`).
    pub fn reset_position(&mut self, work_area: ScreenRect) {
        let (w, h) = physical_size_i32(self.scale, self.dpi_scale);
        let (nx, ny) = clamp_pos(work_area.right - w, work_area.bottom - h, w, h, work_area);
        self.x = nx as f32;
        self.y = ny as f32;
        self.velocity_x = 0.0;
        self.velocity_y = 0.0;
        self.seated_hwnd = None;
        self.glide_goal = None;
        self.state = MascotState::SeatedOnTaskbar;
    }

    /// Updates scale (`0.5..=2.0`), anchoring bottom-center and clamping inside `work_area`.
    pub fn set_scale(&mut self, new_scale: f32, work_area: ScreenRect) {
        let clamped = sanitize_scale(new_scale);
        if (clamped - self.scale).abs() < 1e-4 {
            return;
        }
        let (old_w, old_h) = physical_size_i32(self.scale, self.dpi_scale);
        let (new_w, new_h) = physical_size_i32(clamped, self.dpi_scale);

        let bottom_center_x = self.x + (old_w as f32) * 0.5;
        let bottom_y = self.y + (old_h as f32);

        let raw_x = (bottom_center_x - (new_w as f32) * 0.5).round() as i32;
        let raw_y = (bottom_y - (new_h as f32)).round() as i32;
        let (cx, cy) = clamp_pos(raw_x, raw_y, new_w, new_h, work_area);

        self.scale = clamped;
        self.x = cx as f32;
        self.y = cy as f32;
    }

    /// Starts dragging the mascot window from cursor position `cursor_screen`.
    pub fn begin_drag(&mut self, cursor_screen: (i32, i32)) {
        let (px, py) = self.position();
        self.drag_offset_x = cursor_screen.0 - px;
        self.drag_offset_y = cursor_screen.1 - py;
        self.state = MascotState::Dragged;
        self.seated_hwnd = None;
        self.glide_goal = None;
        self.velocity_x = 0.0;
        self.velocity_y = 0.0;
        self.prev_drag_vx = 0.0;
        self.prev_drag_vy = 0.0;
    }

    /// Updates dragged window position from cursor coordinates, returning inertial `window_accel`.
    pub fn update_drag(
        &mut self,
        cursor_screen: (i32, i32),
        dt: f32,
        work_area: ScreenRect,
    ) -> Vec3 {
        if self.state != MascotState::Dragged {
            return Vec3::ZERO;
        }
        let safe_dt = dt.max(1.0 / 240.0);
        let (w, h) = physical_size_i32(self.scale, self.dpi_scale);
        let prev_x = self.x;
        let prev_y = self.y;

        let target_x = cursor_screen.0 - self.drag_offset_x;
        let target_y = cursor_screen.1 - self.drag_offset_y;
        let (clamped_x, clamped_y) = clamp_pos(target_x, target_y, w, h, work_area);

        self.x = clamped_x as f32;
        self.y = clamped_y as f32;

        let vx = (self.x - prev_x) / safe_dt;
        let vy = (self.y - prev_y) / safe_dt;
        let ax = (vx - self.prev_drag_vx) / safe_dt;
        let ay = (vy - self.prev_drag_vy) / safe_dt;

        self.prev_drag_vx = vx;
        self.prev_drag_vy = vy;
        self.velocity_x = vx;
        self.velocity_y = vy;

        // Inertial force on PhysBones acts opposite to window acceleration.
        // Screen +Y is down, while 3D world +Y is up.
        let world_ax = (-ax * PIXEL_TO_WORLD_ACCEL).clamp(-MAX_WORLD_ACCEL, MAX_WORLD_ACCEL);
        let world_ay = (ay * PIXEL_TO_WORLD_ACCEL).clamp(-MAX_WORLD_ACCEL, MAX_WORLD_ACCEL);
        Vec3::new(world_ax, world_ay, 0.0)
    }

    /// Ends dragging, transitioning to `Falling` when `gravity_enabled` or `Idle` otherwise.
    pub fn end_drag(&mut self, gravity_enabled: bool) {
        if self.state != MascotState::Dragged {
            return;
        }
        self.prev_drag_vx = 0.0;
        self.prev_drag_vy = 0.0;
        if gravity_enabled {
            self.state = MascotState::Falling;
            self.velocity_x = 0.0;
            self.velocity_y = self.velocity_y.clamp(0.0, 400.0);
        } else {
            self.state = MascotState::Idle;
            self.velocity_x = 0.0;
            self.velocity_y = 0.0;
        }
    }

    /// Initiates an autonomous glide to the taskbar bottom-right area (Decisions `MOVE_TO_TASKBAR`).
    pub fn move_to_taskbar(&mut self, work_area: ScreenRect, taskbar_rect: Option<ScreenRect>) {
        if self.state == MascotState::Dragged {
            return;
        }
        let (w, h) = physical_size_i32(self.scale, self.dpi_scale);
        let floor_y = effective_floor_y(work_area, taskbar_rect, h);
        let (tx, ty) = clamp_pos(work_area.right - w, floor_y - h, w, h, work_area);
        self.seated_hwnd = None;
        self.velocity_x = 0.0;
        self.velocity_y = 0.0;
        self.glide_goal = Some(GlideGoal::Taskbar {
            target_x: tx,
            target_y: ty,
        });
        self.state = MascotState::Idle;
    }

    /// Initiates an autonomous glide onto the topmost eligible visible window
    /// (Decisions `SIT_ON_WINDOW`). Returns `false` if no eligible window exists.
    pub fn sit_on_top_window(
        &mut self,
        visible_windows: &[ExternalWindowInfo],
        exclude_apps: &[String],
        work_area: ScreenRect,
    ) -> bool {
        if self.state == MascotState::Dragged {
            return false;
        }
        let (w, h) = physical_size_i32(self.scale, self.dpi_scale);
        let Some(target) = visible_windows.iter().find(|win| {
            !is_excluded_app(Some(&win.process_name), exclude_apps)
                && is_perchable_window_rect(win.rect, work_area, w)
        }) else {
            return false;
        };

        let rel_x =
            ((target.rect.width() - w).max(0) / 2).min(target.rect.width().saturating_sub(w));
        let (_, target_y) = clamp_pos(
            target.rect.left + rel_x,
            target.rect.top - h,
            w,
            h,
            work_area,
        );
        self.seated_hwnd = None;
        self.velocity_x = 0.0;
        self.velocity_y = 0.0;
        self.glide_goal = Some(GlideGoal::Window {
            hwnd: target.hwnd,
            rel_x,
            target_y,
        });
        self.state = MascotState::Idle;
        true
    }

    /// Advances the perch state machine by `dt` seconds and returns updated geometry & impulse.
    pub fn step(&mut self, input: &PerchStepInput<'_>) -> PerchStepOutput {
        let prev_size = self.size();

        if (sanitize_scale(input.scale) - self.scale).abs() > 1e-4 {
            self.set_scale(input.scale, input.work_area);
        }

        let (w, h) = physical_size_i32(self.scale, self.dpi_scale);
        let dt = input.dt.clamp(0.0, 0.1);
        let mut frame_accel = std::mem::take(&mut self.pending_impulse);

        // Handle smooth autonomous glide if active
        if let Some(goal) = self.glide_goal {
            let speed =
                BASE_GLIDE_SPEED_PX_PER_SEC * input.motion_cfg.move_speed_scale.clamp(0.2, 4.0);
            match goal {
                GlideGoal::Taskbar { target_x, target_y } => {
                    let (arrived, accel) =
                        self.step_glide_towards(target_x as f32, target_y as f32, speed, dt);
                    frame_accel += accel;
                    if arrived {
                        self.glide_goal = None;
                        self.state = MascotState::SeatedOnTaskbar;
                        frame_accel += Vec3::new(0.0, 3.5, 0.0);
                    }
                }
                GlideGoal::Window {
                    hwnd,
                    rel_x,
                    target_y,
                } => {
                    if let Some(win) = input
                        .visible_windows
                        .iter()
                        .find(|w_info| w_info.hwnd == hwnd)
                    {
                        let (tx, ty) = clamp_pos(
                            win.rect.left + rel_x,
                            win.rect.top - h,
                            w,
                            h,
                            input.work_area,
                        );
                        let (arrived, accel) =
                            self.step_glide_towards(tx as f32, ty as f32, speed, dt);
                        frame_accel += accel;
                        if arrived {
                            self.glide_goal = None;
                            self.seated_hwnd = Some(hwnd);
                            self.seated_rel_x = tx - win.rect.left;
                            self.state = MascotState::SeatedOnWindow;
                            frame_accel += Vec3::new(0.0, 3.5, 0.0);
                        }
                    } else {
                        let _ = target_y;
                        self.glide_goal = None;
                        self.state = if input.motion_cfg.gravity_enabled {
                            MascotState::Falling
                        } else {
                            MascotState::Idle
                        };
                    }
                }
            }
        } else {
            match self.state {
                MascotState::Dragged => {}
                MascotState::Falling => {
                    let prev_y = self.y;
                    self.velocity_y =
                        (self.velocity_y + GRAVITY_PX_PER_SEC2 * dt).min(MAX_FALL_SPEED_PX_PER_SEC);
                    let next_y = prev_y + self.velocity_y * dt;

                    let prev_bottom = prev_y + h as f32;
                    let next_bottom = next_y + h as f32;
                    let floor_y = effective_floor_y(input.work_area, input.taskbar_rect, h);

                    let cur_x = self.x.round() as i32;
                    let (foot_left, foot_right) = foot_span(cur_x, w);

                    // Find highest crossed eligible window top in [prev_bottom - 4, next_bottom]
                    let mut best_window: Option<(isize, ScreenRect)> = None;
                    if input.motion_cfg.window_perch_enabled {
                        for win in input.visible_windows {
                            if is_excluded_app(
                                Some(&win.process_name),
                                &input.motion_cfg.perch_exclude_apps,
                            ) {
                                continue;
                            }
                            if !is_perchable_window_rect(win.rect, input.work_area, w) {
                                continue;
                            }
                            if foot_right <= win.rect.left || foot_left >= win.rect.right {
                                continue;
                            }
                            let top = win.rect.top as f32;
                            if top >= prev_bottom - 4.0
                                && next_bottom >= top
                                && top < floor_y as f32
                            {
                                let replace = match best_window {
                                    None => true,
                                    Some((_, best_rect)) => win.rect.top < best_rect.top,
                                };
                                if replace {
                                    best_window = Some((win.hwnd, win.rect));
                                }
                            }
                        }
                    }

                    if let Some((hwnd, rect)) = best_window {
                        let (_, landed_y) = clamp_pos(cur_x, rect.top - h, w, h, input.work_area);
                        self.y = landed_y as f32;
                        self.seated_hwnd = Some(hwnd);
                        self.seated_rel_x = cur_x - rect.left;
                        self.state = MascotState::SeatedOnWindow;
                        let impulse_y = (self.velocity_y * 0.012).clamp(2.0, 14.0);
                        self.velocity_y = 0.0;
                        frame_accel += Vec3::new(0.0, impulse_y, 0.0);
                    } else if next_bottom >= floor_y as f32 {
                        let (_, landed_y) = clamp_pos(cur_x, floor_y - h, w, h, input.work_area);
                        self.y = landed_y as f32;
                        self.seated_hwnd = None;
                        self.state = MascotState::SeatedOnTaskbar;
                        let impulse_y = (self.velocity_y * 0.012).clamp(2.0, 14.0);
                        self.velocity_y = 0.0;
                        frame_accel += Vec3::new(0.0, impulse_y, 0.0);
                    } else {
                        self.y = next_y.max(input.work_area.top as f32);
                    }
                }
                MascotState::SeatedOnWindow => {
                    let Some(target_rect) = input.target_window_rect else {
                        self.seated_hwnd = None;
                        self.velocity_y = 0.0;
                        self.state = if input.motion_cfg.gravity_enabled {
                            MascotState::Falling
                        } else {
                            MascotState::Idle
                        };
                        return self.finish_step(prev_size, frame_accel, input.work_area);
                    };

                    let desired_x = target_rect.left + self.seated_rel_x;
                    let desired_y = target_rect.top - h;
                    let (cx, cy) = clamp_pos(desired_x, desired_y, w, h, input.work_area);

                    let (foot_left, foot_right) = foot_span(cx, w);
                    let still_overlapping =
                        foot_right > target_rect.left && foot_left < target_rect.right;

                    if !still_overlapping {
                        self.seated_hwnd = None;
                        self.velocity_y = 0.0;
                        self.state = if input.motion_cfg.gravity_enabled {
                            MascotState::Falling
                        } else {
                            MascotState::Idle
                        };
                    } else {
                        let dx = cx as f32 - self.x;
                        let dy = cy as f32 - self.y;
                        self.x = cx as f32;
                        self.y = cy as f32;
                        if dt > 1e-4 && (dx.abs() > 0.5 || dy.abs() > 0.5) {
                            frame_accel += Vec3::new(
                                (-dx / dt * 0.02).clamp(-10.0, 10.0),
                                (dy / dt * 0.02).clamp(-10.0, 10.0),
                                0.0,
                            );
                        }
                    }
                }
                MascotState::SeatedOnTaskbar => {
                    let floor_y = effective_floor_y(input.work_area, input.taskbar_rect, h);
                    let (cx, cy) =
                        clamp_pos(self.x.round() as i32, floor_y - h, w, h, input.work_area);
                    self.x = cx as f32;
                    self.y = cy as f32;
                }
                MascotState::Idle => {
                    let (cx, cy) = clamp_pos(
                        self.x.round() as i32,
                        self.y.round() as i32,
                        w,
                        h,
                        input.work_area,
                    );
                    self.x = cx as f32;
                    self.y = cy as f32;
                }
            }
        }

        self.finish_step(prev_size, frame_accel, input.work_area)
    }

    fn finish_step(
        &mut self,
        prev_size: (u32, u32),
        window_accel: Vec3,
        work_area: ScreenRect,
    ) -> PerchStepOutput {
        let (w, h) = physical_size_i32(self.scale, self.dpi_scale);
        let (cx, cy) = clamp_pos(
            self.x.round() as i32,
            self.y.round() as i32,
            w,
            h,
            work_area,
        );
        self.x = cx as f32;
        self.y = cy as f32;

        let position = self.position();
        let size = self.size();
        let position_changed = position != self.last_reported_pos;
        self.last_reported_pos = position;
        PerchStepOutput {
            position,
            size,
            state: self.state,
            seated_hwnd: self.seated_hwnd,
            window_accel,
            position_changed,
            size_changed: size != prev_size,
        }
    }

    fn step_glide_towards(&mut self, tx: f32, ty: f32, speed: f32, dt: f32) -> (bool, Vec3) {
        let dx = tx - self.x;
        let dy = ty - self.y;
        let dist = (dx * dx + dy * dy).sqrt();
        let max_step = speed * dt;
        if dist <= max_step.max(2.0) {
            self.x = tx;
            self.y = ty;
            return (true, Vec3::ZERO);
        }
        let vx = (dx / dist) * speed;
        let vy = (dy / dist) * speed;
        self.x += vx * dt;
        self.y += vy * dt;
        let accel = Vec3::new(
            (-vx * 0.008).clamp(-8.0, 8.0),
            (vy * 0.008).clamp(-8.0, 8.0),
            0.0,
        );
        (false, accel)
    }
}

fn sanitize_scale(scale: f32) -> f32 {
    if scale.is_finite() {
        scale.clamp(MIN_SCALE, MAX_SCALE)
    } else {
        1.0
    }
}

fn logical_size_i32(scale: f32) -> (i32, i32) {
    let s = sanitize_scale(scale);
    (
        (BASE_LOGICAL_WIDTH * s).round() as i32,
        (BASE_LOGICAL_HEIGHT * s).round() as i32,
    )
}

fn physical_size_i32(scale: f32, dpi_scale: f64) -> (i32, i32) {
    let s = f64::from(sanitize_scale(scale)) * dpi_scale.max(0.5);
    (
        (f64::from(BASE_LOGICAL_WIDTH) * s).round() as i32,
        (f64::from(BASE_LOGICAL_HEIGHT) * s).round() as i32,
    )
}

fn clamp_pos(x: i32, y: i32, w: i32, h: i32, work_area: ScreenRect) -> (i32, i32) {
    let max_x = (work_area.right - w).max(work_area.left);
    let max_y = (work_area.bottom - h).max(work_area.top);
    (
        x.clamp(work_area.left, max_x),
        y.clamp(work_area.top, max_y),
    )
}

fn foot_span(x: i32, w: i32) -> (i32, i32) {
    let margin = (w / 4).max(10);
    (x + margin, x + w - margin)
}

fn effective_floor_y(
    work_area: ScreenRect,
    taskbar_rect: Option<ScreenRect>,
    mascot_h: i32,
) -> i32 {
    if let Some(tb) = taskbar_rect {
        // Bottom taskbar: top edge lies in the lower part of the screen
        if tb.top >= work_area.top + (mascot_h / 3) && tb.top <= work_area.bottom + 8 {
            return tb.top.min(work_area.bottom);
        }
    }
    work_area.bottom
}

fn is_perchable_window_rect(rect: ScreenRect, work_area: ScreenRect, mascot_w: i32) -> bool {
    // Skip minimized/tiny windows and maximized windows touching the top of the work area.
    rect.width() >= (mascot_w / 2).max(120)
        && rect.height() >= 60
        && rect.top > work_area.top + 20
        && rect.top < work_area.bottom - 16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work_area_1080p() -> ScreenRect {
        ScreenRect::new(0, 0, 1920, 1040)
    }

    fn taskbar_1080p() -> ScreenRect {
        ScreenRect::new(0, 1040, 1920, 1080)
    }

    #[test]
    fn initial_placement_and_scale_clamping() {
        let wa = work_area_1080p();
        let mut perch = PerchController::new(wa, 1.0);
        assert_eq!(perch.size(), (400, 600));
        assert_eq!(perch.position(), (1520, 440));
        assert_eq!(perch.state(), MascotState::SeatedOnTaskbar);

        // Scaling clamps to [0.5, 2.0] and stays within work_area
        perch.set_scale(3.5, wa);
        assert_eq!(perch.scale(), 2.0);
        assert_eq!(perch.size(), (800, 1200));
        assert_eq!(perch.position(), (1120, 0));

        perch.set_scale(0.1, wa);
        assert_eq!(perch.scale(), 0.5);
        assert_eq!(perch.size(), (200, 300));
        let (x, y) = perch.position();
        assert!(x >= wa.left && x + 200 <= wa.right);
        assert!(y >= wa.top && y + 300 <= wa.bottom);
    }

    #[test]
    fn drag_clamps_to_work_area_and_emits_inertia() {
        let wa = work_area_1080p();
        let mut perch = PerchController::new(wa, 1.0);
        perch.begin_drag((1700, 700));
        assert_eq!(perch.state(), MascotState::Dragged);

        let accel1 = perch.update_drag((1500, 500), 1.0 / 60.0, wa);
        // Dragged left & up -> inertial accel points right (+X) and down (-Y in 3D world)
        assert!(
            accel1.x > 0.0,
            "expected positive inertial X, got {accel1:?}"
        );
        assert!(
            accel1.y < 0.0,
            "expected negative inertial Y, got {accel1:?}"
        );

        // Trying to drag offscreen right/bottom clamps cleanly
        let _ = perch.update_drag((5000, 5000), 1.0 / 60.0, wa);
        assert_eq!(perch.position(), (1520, 440));
    }

    #[test]
    fn gravity_fall_lands_on_taskbar_and_emits_impulse() {
        let wa = work_area_1080p();
        let tb = taskbar_1080p();
        let cfg = MotionConfig::default();
        let mut perch = PerchController::new(wa, 1.0);

        perch.begin_drag((1700, 700));
        let _ = perch.update_drag((1000, 200), 1.0 / 60.0, wa);
        perch.end_drag(true);
        assert_eq!(perch.state(), MascotState::Falling);

        let mut landed_impulse = Vec3::ZERO;
        for _ in 0..180 {
            let out = perch.step(&PerchStepInput {
                dt: 1.0 / 60.0,
                work_area: wa,
                taskbar_rect: Some(tb),
                visible_windows: &[],
                target_window_rect: None,
                motion_cfg: &cfg,
                scale: 1.0,
            });
            if out.state == MascotState::SeatedOnTaskbar {
                landed_impulse = out.window_accel;
                break;
            }
        }

        assert_eq!(perch.state(), MascotState::SeatedOnTaskbar);
        assert_eq!(perch.position().1, 440);
        assert!(landed_impulse.y > 0.0, "expected upward bounce impulse");
    }

    #[test]
    fn gravity_fall_lands_on_non_excluded_window_and_skips_excluded() {
        let wa = work_area_1080p();
        let tb = taskbar_1080p();
        let cfg = MotionConfig::default();
        let mut perch = PerchController::new(wa, 1.0);

        let windows = vec![
            ExternalWindowInfo {
                hwnd: 101,
                title: "OBS".to_owned(),
                process_name: "obs64.exe".to_owned(), // excluded by default
                rect: ScreenRect::new(200, 700, 1200, 1000),
            },
            ExternalWindowInfo {
                hwnd: 202,
                title: "Editor".to_owned(),
                process_name: "Code.exe".to_owned(),
                rect: ScreenRect::new(300, 820, 1300, 1020),
            },
        ];

        // Position mascot near top-center (x=500, y=50 -> bottom=650, above both windows)
        perch.begin_drag((1700, 700));
        let _ = perch.update_drag((680, 310), 1.0 / 60.0, wa);
        perch.end_drag(true);
        assert!(perch.position().1 + 600 < 700);

        for _ in 0..180 {
            let out = perch.step(&PerchStepInput {
                dt: 1.0 / 60.0,
                work_area: wa,
                taskbar_rect: Some(tb),
                visible_windows: &windows,
                target_window_rect: None,
                motion_cfg: &cfg,
                scale: 1.0,
            });
            if out.state == MascotState::SeatedOnWindow {
                break;
            }
        }

        // Skips obs64.exe (top=700) and lands on Code.exe (top=820 -> y=220)
        assert_eq!(perch.state(), MascotState::SeatedOnWindow);
        assert_eq!(perch.seated_hwnd(), Some(202));
        assert_eq!(perch.position().1, 220);
    }

    #[test]
    fn seated_window_tracks_move_and_refalls_on_minimize_or_horizontal_slip() {
        let wa = work_area_1080p();
        let tb = taskbar_1080p();
        let cfg = MotionConfig::default();
        let mut perch = PerchController::new(wa, 1.0);

        let target = ExternalWindowInfo {
            hwnd: 555,
            title: "Browser".to_owned(),
            process_name: "chrome.exe".to_owned(),
            rect: ScreenRect::new(400, 800, 1200, 1020),
        };

        perch.begin_drag((1700, 700));
        let _ = perch.update_drag((700, 200), 1.0 / 60.0, wa);
        perch.end_drag(true);

        for _ in 0..180 {
            let out = perch.step(&PerchStepInput {
                dt: 1.0 / 60.0,
                work_area: wa,
                taskbar_rect: Some(tb),
                visible_windows: std::slice::from_ref(&target),
                target_window_rect: None,
                motion_cfg: &cfg,
                scale: 1.0,
            });
            if out.state == MascotState::SeatedOnWindow {
                break;
            }
        }
        assert_eq!(perch.seated_hwnd(), Some(555));
        let initial_x = perch.position().0;

        // Target moves right +50, down +40 -> mascot follows
        let moved_rect = ScreenRect::new(450, 840, 1250, 1020);
        let out_follow = perch.step(&PerchStepInput {
            dt: 1.0 / 60.0,
            work_area: wa,
            taskbar_rect: Some(tb),
            visible_windows: &[],
            target_window_rect: Some(moved_rect),
            motion_cfg: &cfg,
            scale: 1.0,
        });
        assert_eq!(out_follow.state, MascotState::SeatedOnWindow);
        assert_eq!(out_follow.position, (initial_x + 50, 240));

        // Target minimizes/closes (target_window_rect = None) -> falls again onto taskbar
        let out_lost = perch.step(&PerchStepInput {
            dt: 1.0 / 60.0,
            work_area: wa,
            taskbar_rect: Some(tb),
            visible_windows: &[],
            target_window_rect: None,
            motion_cfg: &cfg,
            scale: 1.0,
        });
        assert_eq!(out_lost.state, MascotState::Falling);
        assert_eq!(out_lost.seated_hwnd, None);
    }

    #[test]
    fn autonomous_sit_on_top_window_and_horizontal_slip_trigger_fall() {
        let wa = work_area_1080p();
        let tb = taskbar_1080p();
        let cfg = MotionConfig::default();
        let mut perch = PerchController::new(wa, 1.0);

        let target = ExternalWindowInfo {
            hwnd: 777,
            title: "Docs".to_owned(),
            process_name: "chrome.exe".to_owned(),
            rect: ScreenRect::new(200, 760, 1000, 1000),
        };

        assert!(perch.sit_on_top_window(
            std::slice::from_ref(&target),
            &cfg.perch_exclude_apps,
            wa
        ));
        for _ in 0..240 {
            let out = perch.step(&PerchStepInput {
                dt: 1.0 / 60.0,
                work_area: wa,
                taskbar_rect: Some(tb),
                visible_windows: std::slice::from_ref(&target),
                target_window_rect: None,
                motion_cfg: &cfg,
                scale: 1.0,
            });
            if out.state == MascotState::SeatedOnWindow {
                break;
            }
        }
        assert_eq!(perch.state(), MascotState::SeatedOnWindow);
        assert_eq!(perch.seated_hwnd(), Some(777));

        // Window shrinks so its right edge no longer overlaps mascot's feet -> falls!
        let shrunk = ScreenRect::new(200, 760, 260, 1000);
        let out_slip = perch.step(&PerchStepInput {
            dt: 1.0 / 60.0,
            work_area: wa,
            taskbar_rect: Some(tb),
            visible_windows: &[],
            target_window_rect: Some(shrunk),
            motion_cfg: &cfg,
            scale: 1.0,
        });
        assert_eq!(out_slip.state, MascotState::Falling);

        // Move back to taskbar autonomously
        perch.move_to_taskbar(wa, Some(tb));
        for _ in 0..240 {
            let out = perch.step(&PerchStepInput {
                dt: 1.0 / 60.0,
                work_area: wa,
                taskbar_rect: Some(tb),
                visible_windows: &[],
                target_window_rect: None,
                motion_cfg: &cfg,
                scale: 1.0,
            });
            if out.state == MascotState::SeatedOnTaskbar {
                break;
            }
        }
        assert_eq!(perch.state(), MascotState::SeatedOnTaskbar);
    }

    #[test]
    fn drag_emits_position_changed_every_frame_and_drops_without_warp() {
        let wa = work_area_1080p();
        let tb = taskbar_1080p();
        let cfg = MotionConfig {
            gravity_enabled: false,
            ..Default::default()
        };
        let mut perch = PerchController::new(wa, 1.0);
        let start_pos = perch.position();

        perch.begin_drag((1700, 700));
        for step_idx in 1..=5 {
            let cursor = (1700 - step_idx * 40, 700 - step_idx * 30);
            let _ = perch.update_drag(cursor, 1.0 / 60.0, wa);
            let out = perch.step(&PerchStepInput {
                dt: 1.0 / 60.0,
                work_area: wa,
                taskbar_rect: Some(tb),
                visible_windows: &[],
                target_window_rect: None,
                motion_cfg: &cfg,
                scale: 1.0,
            });
            assert_eq!(out.state, MascotState::Dragged);
            assert!(
                out.position_changed,
                "expected position_changed=true on drag frame {step_idx}"
            );
            assert_eq!(
                out.position,
                (start_pos.0 - step_idx * 40, start_pos.1 - step_idx * 30)
            );
        }

        let dropped_pos = perch.position();
        perch.end_drag(false);
        let after_drop = perch.step(&PerchStepInput {
            dt: 1.0 / 60.0,
            work_area: wa,
            taskbar_rect: Some(tb),
            visible_windows: &[],
            target_window_rect: None,
            motion_cfg: &cfg,
            scale: 1.0,
        });
        assert_eq!(after_drop.state, MascotState::Idle);
        assert_eq!(after_drop.position, dropped_pos);
        assert!(!after_drop.position_changed);
    }
}
