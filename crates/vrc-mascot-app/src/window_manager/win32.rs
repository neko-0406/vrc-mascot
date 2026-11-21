//! Win32 API helpers for layered/click-through window styling, cursor/work-area queries,
//! visible window enumeration, fullscreen detection, and auto-start registry integration.

use std::ffi::c_void;
use std::path::Path;

use windows::core::{BOOL, PCWSTR, PWSTR};
use windows::Win32::Foundation::{
    CloseHandle, COLORREF, ERROR_FILE_NOT_FOUND, HWND, LPARAM, POINT, RECT,
};
use windows::Win32::Graphics::Dwm::{
    DwmGetWindowAttribute, DWMWA_CLOAKED, DWMWA_EXTENDED_FRAME_BOUNDS,
};
use windows::Win32::Graphics::Gdi::{
    GetMonitorInfoW, MonitorFromWindow, ScreenToClient, MONITORINFO, MONITOR_DEFAULTTONEAREST,
};
use windows::Win32::System::Registry::{
    RegCloseKey, RegCreateKeyExW, RegDeleteValueW, RegOpenKeyExW, RegSetValueExW, HKEY,
    HKEY_CURRENT_USER, KEY_SET_VALUE, REG_OPTION_NON_VOLATILE, REG_SZ,
};
use windows::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::Shell::{SHAppBarMessage, ABM_GETTASKBARPOS, APPBARDATA};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetClassNameW, GetCursorPos, GetForegroundWindow, GetSystemMetrics,
    GetWindowLongPtrW, GetWindowRect, GetWindowTextLengthW, GetWindowTextW,
    GetWindowThreadProcessId, IsIconic, IsWindow, IsWindowVisible, SetLayeredWindowAttributes,
    SetWindowLongPtrW, SetWindowPos, SystemParametersInfoW, GWL_EXSTYLE, HWND_NOTOPMOST,
    HWND_TOPMOST, LWA_ALPHA, SM_CXSCREEN, SM_CYSCREEN, SPI_GETWORKAREA, SWP_FRAMECHANGED,
    SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS, WS_EX_APPWINDOW,
    WS_EX_LAYERED, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_EX_TRANSPARENT,
};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

const AUTO_START_SUBKEY: &str = "Software\\Microsoft\\Windows\\CurrentVersion\\Run";
const AUTO_START_VALUE_NAME: &str = "VrcDesktopMascot";

const SHELL_CLASSES: &[&str] = &[
    "Progman",
    "WorkerW",
    "Shell_TrayWnd",
    "Shell_SecondaryTrayWnd",
    "Windows.UI.Core.CoreWindow",
];

/// Screen-space rectangle in physical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScreenRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl ScreenRect {
    pub const fn new(left: i32, top: i32, right: i32, bottom: i32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    pub const fn width(self) -> i32 {
        if self.right > self.left {
            self.right - self.left
        } else {
            0
        }
    }

    pub const fn height(self) -> i32 {
        if self.bottom > self.top {
            self.bottom - self.top
        } else {
            0
        }
    }

    pub const fn is_empty(self) -> bool {
        self.right <= self.left || self.bottom <= self.top
    }
}

impl From<RECT> for ScreenRect {
    fn from(r: RECT) -> Self {
        Self {
            left: r.left,
            top: r.top,
            right: r.right,
            bottom: r.bottom,
        }
    }
}

/// Metadata for an external visible window in Z-order (front-to-back).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalWindowInfo {
    pub hwnd: isize,
    pub title: String,
    pub process_name: String,
    pub rect: ScreenRect,
}

/// Metadata for the current foreground window including fullscreen heuristic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForegroundWindowInfo {
    pub hwnd: isize,
    pub title: String,
    pub process_name: String,
    pub rect: ScreenRect,
    pub is_fullscreen: bool,
}

/// Extracts the Win32 `HWND` handle value (`isize`) from a `winit` window.
pub fn hwnd_from_winit(window: &winit::window::Window) -> Option<isize> {
    let handle = window.window_handle().ok()?;
    match handle.as_raw() {
        RawWindowHandle::Win32(win32) => Some(win32.hwnd.get()),
        _ => None,
    }
}

#[inline]
fn to_hwnd(raw: isize) -> HWND {
    HWND(raw as *mut c_void)
}

/// Applies initial desktop mascot Win32 extended styles (`WS_EX_LAYERED | WS_EX_TOOLWINDOW`,
/// optional `WS_EX_TOPMOST` and `WS_EX_TRANSPARENT`) and initializes layered opacity to 255
/// (`LWA_ALPHA`) so DWM does not treat the window as 0-alpha invisible.
pub fn configure_avatar_window(hwnd_raw: isize, always_on_top: bool, click_through: bool) {
    if hwnd_raw == 0 {
        return;
    }
    let hwnd = to_hwnd(hwnd_raw);
    unsafe {
        let current = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        let mut next = (current | (WS_EX_LAYERED.0 as isize) | (WS_EX_TOOLWINDOW.0 as isize))
            & !(WS_EX_APPWINDOW.0 as isize);
        if click_through {
            next |= WS_EX_TRANSPARENT.0 as isize;
        } else {
            next &= !(WS_EX_TRANSPARENT.0 as isize);
        }
        if next != current {
            let _ = SetWindowLongPtrW(hwnd, GWL_EXSTYLE, next);
        }
        // Without SetLayeredWindowAttributes (or UpdateLayeredWindow), WS_EX_LAYERED leaves
        // bAlpha=0 / dwFlags=0, making the window 100% invisible and halting WM_PAINT delivery.
        let _ = SetLayeredWindowAttributes(hwnd, COLORREF(0), 255, LWA_ALPHA);
        let insert_after = if always_on_top {
            HWND_TOPMOST
        } else {
            HWND_NOTOPMOST
        };
        let _ = SetWindowPos(
            hwnd,
            Some(insert_after),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        );
    }
}

/// Toggles `WS_EX_TRANSPARENT` only when the current style differs, avoiding Win32 flicker.
pub fn set_click_through(hwnd_raw: isize, transparent: bool) {
    if hwnd_raw == 0 {
        return;
    }
    let hwnd = to_hwnd(hwnd_raw);
    unsafe {
        let current = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        let had_layered = (current & (WS_EX_LAYERED.0 as isize)) != 0;
        let mut next = current | (WS_EX_LAYERED.0 as isize);
        if transparent {
            next |= WS_EX_TRANSPARENT.0 as isize;
        } else {
            next &= !(WS_EX_TRANSPARENT.0 as isize);
        }
        if next != current {
            let _ = SetWindowLongPtrW(hwnd, GWL_EXSTYLE, next);
            if !had_layered {
                let _ = SetLayeredWindowAttributes(hwnd, COLORREF(0), 255, LWA_ALPHA);
            }
        }
    }
}

/// Toggles `HWND_TOPMOST` / `HWND_NOTOPMOST` only when the current `WS_EX_TOPMOST` state differs.
pub fn set_always_on_top(hwnd_raw: isize, topmost: bool) {
    if hwnd_raw == 0 {
        return;
    }
    let hwnd = to_hwnd(hwnd_raw);
    unsafe {
        let current = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        let is_currently_topmost = (current & (WS_EX_TOPMOST.0 as isize)) != 0;
        if is_currently_topmost == topmost {
            return;
        }
        let insert_after = if topmost {
            HWND_TOPMOST
        } else {
            HWND_NOTOPMOST
        };
        let _ = SetWindowPos(
            hwnd,
            Some(insert_after),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        );
    }
}

/// Returns the current cursor position in screen coordinates.
pub fn get_cursor_pos() -> Option<(i32, i32)> {
    let mut pt = POINT::default();
    unsafe { GetCursorPos(&mut pt).ok()? };
    Some((pt.x, pt.y))
}

/// Converts screen coordinates `(x, y)` to client coordinates of `hwnd_raw`.
pub fn screen_to_client(hwnd_raw: isize, x: i32, y: i32) -> Option<(i32, i32)> {
    if hwnd_raw == 0 {
        return None;
    }
    let mut pt = POINT { x, y };
    let ok = unsafe { ScreenToClient(to_hwnd(hwnd_raw), &mut pt).as_bool() };
    if ok {
        Some((pt.x, pt.y))
    } else {
        None
    }
}

/// Returns the primary monitor work area (`SPI_GETWORKAREA`), falling back to full screen metrics.
pub fn get_primary_work_area() -> ScreenRect {
    let mut rect = RECT::default();
    let ok = unsafe {
        SystemParametersInfoW(
            SPI_GETWORKAREA,
            0,
            Some(&mut rect as *mut RECT as *mut c_void),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        )
        .is_ok()
    };
    if ok {
        let sr = ScreenRect::from(rect);
        if !sr.is_empty() {
            return sr;
        }
    }
    let w = unsafe { GetSystemMetrics(SM_CXSCREEN) }.max(800);
    let h = unsafe { GetSystemMetrics(SM_CYSCREEN) }.max(600);
    ScreenRect::new(0, 0, w, h)
}

/// Returns the Windows taskbar rectangle via `SHAppBarMessage(ABM_GETTASKBARPOS)`.
pub fn get_taskbar_rect() -> Option<ScreenRect> {
    let mut abd = APPBARDATA {
        cbSize: std::mem::size_of::<APPBARDATA>() as u32,
        ..Default::default()
    };
    let res = unsafe { SHAppBarMessage(ABM_GETTASKBARPOS, &mut abd) };
    if res == 0 {
        return None;
    }
    let rect = ScreenRect::from(abd.rc);
    if rect.is_empty() {
        None
    } else {
        Some(rect)
    }
}

/// Queries the extended DWM frame bounds of a visible, non-minimized, non-cloaked window.
/// Returns `None` if the window no longer exists, is hidden, or is minimized.
pub fn query_window_rect(hwnd_raw: isize) -> Option<ScreenRect> {
    if hwnd_raw == 0 {
        return None;
    }
    let hwnd = to_hwnd(hwnd_raw);
    unsafe {
        if !IsWindow(Some(hwnd)).as_bool()
            || !IsWindowVisible(hwnd).as_bool()
            || IsIconic(hwnd).as_bool()
        {
            return None;
        }
        if is_window_cloaked(hwnd) {
            return None;
        }
        get_dwm_or_window_rect(hwnd)
    }
}

/// Enumerates visible, non-minimized top-level windows in Z-order (front-to-back),
/// excluding `exclude_hwnd`, tool windows, cloaked windows, and desktop shell windows.
pub fn enumerate_visible_windows(exclude_hwnd: isize) -> Vec<ExternalWindowInfo> {
    struct EnumCtx {
        exclude_hwnd: isize,
        windows: Vec<ExternalWindowInfo>,
    }

    unsafe extern "system" fn enum_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let ctx = &mut *(lparam.0 as *mut EnumCtx);
        let hwnd_raw = hwnd.0 as isize;
        if hwnd_raw == 0 || hwnd_raw == ctx.exclude_hwnd {
            return BOOL(1);
        }
        if !IsWindowVisible(hwnd).as_bool() || IsIconic(hwnd).as_bool() {
            return BOOL(1);
        }
        if is_window_cloaked(hwnd) {
            return BOOL(1);
        }
        let ex_style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        if (ex_style & (WS_EX_TOOLWINDOW.0 as isize)) != 0 {
            return BOOL(1);
        }
        let class_name = read_window_class(hwnd);
        if is_shell_class(&class_name) {
            return BOOL(1);
        }
        let title = read_window_title(hwnd);
        if title.is_empty() {
            return BOOL(1);
        }
        let Some(rect) = get_dwm_or_window_rect(hwnd) else {
            return BOOL(1);
        };
        if rect.width() < 80 || rect.height() < 40 {
            return BOOL(1);
        }
        let process_name = read_window_process_name(hwnd).unwrap_or_default();
        ctx.windows.push(ExternalWindowInfo {
            hwnd: hwnd_raw,
            title,
            process_name,
            rect,
        });
        BOOL(1)
    }

    let mut ctx = EnumCtx {
        exclude_hwnd,
        windows: Vec::new(),
    };
    unsafe {
        let _ = EnumWindows(Some(enum_proc), LPARAM(&mut ctx as *mut EnumCtx as isize));
    }
    ctx.windows
}

/// Returns information about the current foreground window and whether it covers its monitor fullscreen.
pub fn get_foreground_window_info(exclude_hwnd: isize) -> Option<ForegroundWindowInfo> {
    let hwnd = unsafe { GetForegroundWindow() };
    let hwnd_raw = hwnd.0 as isize;
    if hwnd_raw == 0 || hwnd_raw == exclude_hwnd {
        return None;
    }
    unsafe {
        if !IsWindowVisible(hwnd).as_bool() || IsIconic(hwnd).as_bool() {
            return None;
        }
    }
    let class_name = unsafe { read_window_class(hwnd) };
    if is_shell_class(&class_name) {
        return None;
    }

    let rect = unsafe { get_dwm_or_window_rect(hwnd) }?;
    let title = unsafe { read_window_title(hwnd) };
    let process_name = unsafe { read_window_process_name(hwnd) }.unwrap_or_default();

    let is_fullscreen = unsafe {
        let hmon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        let mut mi = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if GetMonitorInfoW(hmon, &mut mi).as_bool() {
            let mon = ScreenRect::from(mi.rcMonitor);
            rect.left <= mon.left + 1
                && rect.top <= mon.top + 1
                && rect.right >= mon.right - 1
                && rect.bottom >= mon.bottom - 1
        } else {
            false
        }
    };

    Some(ForegroundWindowInfo {
        hwnd: hwnd_raw,
        title,
        process_name,
        rect,
        is_fullscreen,
    })
}

/// Enables or disables Windows user startup auto-launch in `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`.
pub fn set_auto_start(enabled: bool) -> anyhow::Result<()> {
    let subkey = to_wide_null(AUTO_START_SUBKEY);
    let value_name = to_wide_null(AUTO_START_VALUE_NAME);

    if enabled {
        let exe_path = std::env::current_exe()?;
        let command_value = format!("\"{}\"", exe_path.display());
        let wide_cmd = to_wide_null(&command_value);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(wide_cmd.as_ptr().cast::<u8>(), wide_cmd.len() * 2)
        };

        let mut hkey = HKEY::default();
        let status = unsafe {
            RegCreateKeyExW(
                HKEY_CURRENT_USER,
                PCWSTR(subkey.as_ptr()),
                None,
                None,
                REG_OPTION_NON_VOLATILE,
                KEY_SET_VALUE,
                None,
                &mut hkey,
                None,
            )
        };
        if status.is_err() {
            return Err(anyhow::anyhow!(
                "RegCreateKeyExW failed for auto-start: {:?}",
                status
            ));
        }

        let set_res =
            unsafe { RegSetValueExW(hkey, PCWSTR(value_name.as_ptr()), None, REG_SZ, Some(bytes)) };
        unsafe {
            let _ = RegCloseKey(hkey);
        }
        if set_res.is_err() {
            return Err(anyhow::anyhow!(
                "RegSetValueExW failed for auto-start: {:?}",
                set_res
            ));
        }
    } else {
        let mut hkey = HKEY::default();
        let open_res = unsafe {
            RegOpenKeyExW(
                HKEY_CURRENT_USER,
                PCWSTR(subkey.as_ptr()),
                None,
                KEY_SET_VALUE,
                &mut hkey,
            )
        };
        if open_res == ERROR_FILE_NOT_FOUND {
            return Ok(());
        }
        if open_res.is_err() {
            return Err(anyhow::anyhow!(
                "RegOpenKeyExW failed for auto-start removal: {:?}",
                open_res
            ));
        }
        let del_res = unsafe { RegDeleteValueW(hkey, PCWSTR(value_name.as_ptr())) };
        unsafe {
            let _ = RegCloseKey(hkey);
        }
        if del_res.is_err() && del_res != ERROR_FILE_NOT_FOUND {
            return Err(anyhow::anyhow!(
                "RegDeleteValueW failed for auto-start: {:?}",
                del_res
            ));
        }
    }
    Ok(())
}

unsafe fn is_window_cloaked(hwnd: HWND) -> bool {
    let mut cloaked: u32 = 0;
    let hr = DwmGetWindowAttribute(
        hwnd,
        DWMWA_CLOAKED,
        &mut cloaked as *mut u32 as *mut c_void,
        std::mem::size_of::<u32>() as u32,
    );
    hr.is_ok() && cloaked != 0
}

unsafe fn get_dwm_or_window_rect(hwnd: HWND) -> Option<ScreenRect> {
    let mut rect = RECT::default();
    let dwm_ok = DwmGetWindowAttribute(
        hwnd,
        DWMWA_EXTENDED_FRAME_BOUNDS,
        &mut rect as *mut RECT as *mut c_void,
        std::mem::size_of::<RECT>() as u32,
    )
    .is_ok();
    if !dwm_ok && GetWindowRect(hwnd, &mut rect).is_err() {
        return None;
    }
    let sr = ScreenRect::from(rect);
    if sr.is_empty() {
        None
    } else {
        Some(sr)
    }
}

unsafe fn read_window_title(hwnd: HWND) -> String {
    let len = GetWindowTextLengthW(hwnd);
    if len <= 0 {
        return String::new();
    }
    let mut buf = vec![0u16; (len as usize) + 1];
    let copied = GetWindowTextW(hwnd, &mut buf);
    if copied <= 0 {
        String::new()
    } else {
        String::from_utf16_lossy(&buf[..copied as usize])
    }
}

unsafe fn read_window_class(hwnd: HWND) -> String {
    let mut buf = [0u16; 256];
    let copied = GetClassNameW(hwnd, &mut buf);
    if copied <= 0 {
        String::new()
    } else {
        String::from_utf16_lossy(&buf[..copied as usize])
    }
}

unsafe fn read_window_process_name(hwnd: HWND) -> Option<String> {
    let mut pid: u32 = 0;
    let _ = GetWindowThreadProcessId(hwnd, Some(&mut pid));
    if pid == 0 {
        return None;
    }
    let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
    let mut buf = vec![0u16; 512];
    let mut size = buf.len() as u32;
    let ok = QueryFullProcessImageNameW(
        handle,
        PROCESS_NAME_WIN32,
        PWSTR(buf.as_mut_ptr()),
        &mut size,
    )
    .is_ok();
    let _ = CloseHandle(handle);
    if !ok || size == 0 {
        return None;
    }
    let full = String::from_utf16_lossy(&buf[..size as usize]);
    Path::new(&full)
        .file_name()
        .and_then(|n| n.to_str())
        .map(ToOwned::to_owned)
}

fn is_shell_class(class_name: &str) -> bool {
    SHELL_CLASSES.contains(&class_name)
}

fn to_wide_null(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DestroyWindow, GetLayeredWindowAttributes, WINDOW_EX_STYLE, WS_OVERLAPPED,
    };

    struct TestHwnd(HWND);

    impl TestHwnd {
        fn new_hidden() -> Self {
            let class = to_wide_null("STATIC");
            let title = to_wide_null("vrc_mascot_win32_test");
            let hwnd = unsafe {
                CreateWindowExW(
                    WINDOW_EX_STYLE(0),
                    PCWSTR(class.as_ptr()),
                    PCWSTR(title.as_ptr()),
                    WS_OVERLAPPED,
                    0,
                    0,
                    16,
                    16,
                    None,
                    None,
                    None,
                    None,
                )
            }
            .expect("create hidden test HWND");
            Self(hwnd)
        }
    }

    impl Drop for TestHwnd {
        fn drop(&mut self) {
            unsafe {
                let _ = DestroyWindow(self.0);
            }
        }
    }

    #[test]
    fn configure_avatar_window_initializes_layered_alpha_and_toolwindow() {
        let win = TestHwnd::new_hidden();
        let raw = win.0 .0 as isize;
        configure_avatar_window(raw, true, false);

        let ex = unsafe { GetWindowLongPtrW(win.0, GWL_EXSTYLE) };
        assert_ne!(ex & (WS_EX_LAYERED.0 as isize), 0);
        assert_ne!(ex & (WS_EX_TOOLWINDOW.0 as isize), 0);
        assert_ne!(ex & (WS_EX_TOPMOST.0 as isize), 0);
        assert_eq!(ex & (WS_EX_APPWINDOW.0 as isize), 0);
        assert_eq!(ex & (WS_EX_TRANSPARENT.0 as isize), 0);

        let mut cr_key = COLORREF(0);
        let mut alpha = 0u8;
        let mut flags = Default::default();
        unsafe {
            GetLayeredWindowAttributes(
                win.0,
                Some(&mut cr_key),
                Some(&mut alpha),
                Some(&mut flags),
            )
            .expect("GetLayeredWindowAttributes must succeed");
        }
        assert_eq!(alpha, 255);
        assert_eq!(flags, LWA_ALPHA);
    }

    #[test]
    fn set_click_through_toggles_transparent_and_initializes_layered_alpha() {
        let win = TestHwnd::new_hidden();
        let raw = win.0 .0 as isize;

        set_click_through(raw, true);
        let ex_on = unsafe { GetWindowLongPtrW(win.0, GWL_EXSTYLE) };
        assert_ne!(ex_on & (WS_EX_LAYERED.0 as isize), 0);
        assert_ne!(ex_on & (WS_EX_TRANSPARENT.0 as isize), 0);

        let mut alpha = 0u8;
        let mut flags = Default::default();
        unsafe {
            GetLayeredWindowAttributes(win.0, None, Some(&mut alpha), Some(&mut flags))
                .expect("GetLayeredWindowAttributes must succeed");
        }
        assert_eq!(alpha, 255);
        assert_eq!(flags, LWA_ALPHA);

        set_click_through(raw, false);
        let ex_off = unsafe { GetWindowLongPtrW(win.0, GWL_EXSTYLE) };
        assert_eq!(ex_off & (WS_EX_TRANSPARENT.0 as isize), 0);
        assert_ne!(ex_off & (WS_EX_LAYERED.0 as isize), 0);
    }
}
