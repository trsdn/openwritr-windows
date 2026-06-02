"""Floating overlay with DPI awareness, supersampled PIL rendering, Windows
11 Mica backdrop, and a baseline-pulsing audio-level meter."""
from __future__ import annotations
import ctypes
import math
import time
import tkinter as tk
from dataclasses import dataclass
from ctypes import wintypes
from PIL import Image, ImageDraw, ImageFont, ImageTk


def enable_dpi_awareness() -> float:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return max(1.0, dpi / 96.0)
    except Exception:
        return 1.0


# Windows 11 Mica / Acrylic via DwmSetWindowAttribute.
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMSBT_MAINWINDOW = 2  # Mica


def apply_mica(hwnd: int) -> None:
    try:
        dwm = ctypes.windll.dwmapi
        dark = ctypes.c_int(1)
        dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), DWMWA_USE_IMMERSIVE_DARK_MODE,
                                  ctypes.byref(dark), ctypes.sizeof(dark))
        backdrop = ctypes.c_int(DWMSBT_MAINWINDOW)
        dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), DWMWA_SYSTEMBACKDROP_TYPE,
                                  ctypes.byref(backdrop), ctypes.sizeof(backdrop))
    except Exception:
        pass


@dataclass(frozen=True)
class _State:
    label: str
    accent: tuple[int, int, int]
    glyph: str


STATES = {
    "listening":    _State("Listening",    (235,  72,  72), "mic"),
    "transcribing": _State("Transcribing", (245, 158,  11), "wave"),
    "enhancing":    _State("Enhancing",    ( 56, 130, 246), "spark"),
    "done":         _State("Ready",        ( 34, 197,  94), "check"),
}

CARD_W, CARD_H = 200, 56
RADIUS = 28
TOP_OFFSET = 18
TRANSP_KEY = "#010203"
CARD_BG = (24, 28, 36, 215)   # slight transparency lets Mica show through
SUPERSAMPLE = 2
ANIM_INTERVAL_MS = 33
LEVEL_BARS = 9


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("seguisb.ttf", "segoeuisb.ttf", "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_glyph(d: ImageDraw.ImageDraw, cx: float, cy: float, kind: str, s: float) -> None:
    col = (255, 255, 255)
    if kind == "mic":
        # Body: rounded rect; stem + base centered around (cx, cy).
        body_top = cy - 9 * s
        body_bot = cy + 3 * s
        d.rounded_rectangle((cx - 5 * s, body_top, cx + 5 * s, body_bot), radius=5 * s, fill=col)
        # U-shaped cradle below the body.
        d.arc((cx - 8 * s, cy - 1 * s, cx + 8 * s, cy + 9 * s), 0, 180,
              fill=col, width=max(2, int(2 * s)))
        d.line((cx, cy + 9 * s, cx, cy + 12 * s), fill=col, width=max(2, int(2 * s)))
        d.line((cx - 5 * s, cy + 12 * s, cx + 5 * s, cy + 12 * s),
               fill=col, width=max(2, int(2 * s)))
    elif kind == "wave":
        for i, h in enumerate((3, 7, 5, 9, 4, 6, 3)):
            x = cx - 12 * s + i * 4 * s
            d.rounded_rectangle((x - 1 * s, cy - h * s, x + 1 * s, cy + h * s),
                                radius=1 * s, fill=col)
    elif kind == "spark":
        pts = [(cx, cy - 10 * s), (cx + 3 * s, cy - 3 * s), (cx + 10 * s, cy),
               (cx + 3 * s, cy + 3 * s), (cx, cy + 10 * s), (cx - 3 * s, cy + 3 * s),
               (cx - 10 * s, cy), (cx - 3 * s, cy - 3 * s)]
        d.polygon(pts, fill=col)
    elif kind == "check":
        d.line((cx - 7 * s, cy + 1 * s, cx - 2 * s, cy + 7 * s),
               fill=col, width=max(3, int(3 * s)))
        d.line((cx - 2 * s, cy + 7 * s, cx + 9 * s, cy - 6 * s),
               fill=col, width=max(3, int(3 * s)))


def _render_card(state: str, dpi_scale: float, level: float | None = None,
                 phase: float = 0.0) -> Image.Image:
    s = STATES.get(state, STATES["listening"])
    px = dpi_scale * SUPERSAMPLE
    w = int(CARD_W * px)
    h = int(CARD_H * px)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=int(RADIUS * px), fill=CARD_BG)

    badge_d = 34 * px
    bx = 9 * px
    by = (h - badge_d) / 2
    # Pulse the badge gently while listening so the user always sees activity.
    pulse = 0.0
    if state == "listening":
        pulse = 0.5 + 0.5 * math.sin(phase * 2 * math.pi)
    tint = tuple(min(255, int(c + 25 * pulse)) for c in s.accent)
    d.ellipse((bx, by, bx + badge_d, by + badge_d), fill=(*tint, 255))
    _draw_glyph(d, bx + badge_d / 2, by + badge_d / 2, s.glyph, px)

    # Compute the area between the badge and the right edge — text/meter
    # are centered inside it so the card looks balanced.
    pad_right = 14 * px
    region_left = bx + badge_d + 10 * px
    region_right = w - pad_right
    region_center_x = (region_left + region_right) / 2

    if state == "listening":
        meter_y = h / 2
        bar_w = 4 * px
        gap = 4 * px
        n = LEVEL_BARS
        max_h = 36 * px
        lvl = max(0.0, min(1.0, level or 0.0))
        total_w = n * bar_w + (n - 1) * gap
        meter_x = region_center_x - total_w / 2
        for i in range(n):
            dist = abs(i - (n - 1) / 2) / ((n - 1) / 2)
            shape = 1.0 - 0.4 * (dist ** 1.5)
            baseline = 0.22 * shape * (0.55 + 0.45 * math.sin(phase * 2 * math.pi + i * 0.55))
            audio = (lvl ** 0.65) * 1.25 * shape
            mag = max(baseline, audio)
            bar_h = max(4 * px, min(max_h, mag * max_h))
            x = meter_x + i * (bar_w + gap)
            d.rounded_rectangle((x, meter_y - bar_h / 2, x + bar_w, meter_y + bar_h / 2),
                                radius=bar_w / 2, fill=(255, 255, 255, 235))
    else:
        font = _load_font(int(15 * px))
        d.text((region_center_x, h / 2), s.label, fill=(240, 243, 250, 255),
               font=font, anchor="mm")

    return img.resize((int(CARD_W * dpi_scale), int(CARD_H * dpi_scale)), Image.LANCZOS)


class Overlay:
    def __init__(self, tk_root: tk.Tk, dpi_scale: float = 1.0) -> None:
        self.root = tk_root
        self.dpi = dpi_scale
        self.win = tk.Toplevel(tk_root)
        self.win.withdraw()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-toolwindow", True)
        self.win.configure(bg=TRANSP_KEY)
        try:
            self.win.attributes("-transparentcolor", TRANSP_KEY)
        except tk.TclError:
            pass
        self.label = tk.Label(self.win, bg=TRANSP_KEY, borderwidth=0, highlightthickness=0)
        self.label.pack()

        self._static = {
            name: ImageTk.PhotoImage(_render_card(name, self.dpi))
            for name in STATES if name != "listening"
        }
        self._live_photo: ImageTk.PhotoImage | None = None
        self._state = "idle"
        self._level = 0.0
        self._level_target = 0.0
        self._anim_id: str | None = None
        self._hide_after_id: str | None = None
        self._t0 = time.time()
        self._position_top_center()
        # Apply Mica after the window is created and mapped.
        self.win.after(0, lambda: apply_mica(self.win.winfo_id()))

    def show(self, state: str) -> None:
        self.root.after(0, lambda: self._show_now(state))

    def show_briefly(self, state: str, ms: int = 700) -> None:
        self.root.after(0, lambda: self._show_briefly_now(state, ms))

    def hide(self) -> None:
        self.root.after(0, self._hide_now)

    def set_level(self, level: float) -> None:
        self._level_target = max(0.0, min(1.0, float(level)))

    def _show_now(self, state: str) -> None:
        if self._hide_after_id is not None:
            self.root.after_cancel(self._hide_after_id)
            self._hide_after_id = None
        self._state = state
        if state == "listening":
            self._level = 0.0
            self._level_target = 0.0
            self._tick()
        else:
            self._stop_anim()
            self._show_static(state)
        self.win.deiconify()
        self.win.lift()

    def _show_briefly_now(self, state: str, ms: int) -> None:
        self._show_now(state)
        self._hide_after_id = self.root.after(ms, self._hide_now)

    def _hide_now(self) -> None:
        self._hide_after_id = None
        self._stop_anim()
        self.win.withdraw()

    def _show_static(self, state: str) -> None:
        img = self._static.get(state)
        if img is None:
            return
        self.label.configure(image=img)
        self.label.image = img

    def _tick(self) -> None:
        if self._state != "listening":
            return
        if self._level_target > self._level:
            self._level += (self._level_target - self._level) * 0.55
        else:
            self._level = self._level * 0.82 + self._level_target * 0.18
        phase = (time.time() - self._t0) * 0.9  # 0.9 Hz pulse
        photo = ImageTk.PhotoImage(_render_card("listening", self.dpi, level=self._level, phase=phase))
        self._live_photo = photo
        self.label.configure(image=photo)
        self.label.image = photo
        self._anim_id = self.root.after(ANIM_INTERVAL_MS, self._tick)

    def _stop_anim(self) -> None:
        if self._anim_id is not None:
            self.root.after_cancel(self._anim_id)
            self._anim_id = None

    def _position_top_center(self) -> None:
        self.win.update_idletasks()
        w = int(CARD_W * self.dpi)
        h = int(CARD_H * self.dpi)
        sw = self.win.winfo_screenwidth()
        x = (sw - w) // 2
        y = int(TOP_OFFSET * self.dpi)
        self.win.geometry(f"{w}x{h}+{x}+{y}")
