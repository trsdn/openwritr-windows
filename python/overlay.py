"""Floating overlay window — Windows counterpart of macOS OverlayPanel.swift.

A borderless always-on-top tkinter window pinned to the top-center of the
primary screen. Mirrors the Mac states: listening (red), transcribing
(orange), enhancing (blue), done (green).
"""
from __future__ import annotations
import tkinter as tk
from dataclasses import dataclass


@dataclass(frozen=True)
class _State:
    label: str
    accent: str       # outer accent for border + badge
    fg: str           # text colour
    glyph: str        # canvas-drawn icon: 'mic' | 'wave' | 'spark' | 'check'


STATES = {
    "listening":    _State("Listening",    "#dc2626", "#ffffff", "mic"),
    "transcribing": _State("Transcribing", "#f59e0b", "#ffffff", "wave"),
    "enhancing":    _State("Enhancing",    "#2563eb", "#ffffff", "spark"),
    "done":         _State("Ready",        "#16a34a", "#ffffff", "check"),
}

WIDTH, HEIGHT = 320, 84
BG_KEY = "#010203"   # rare colour used as transparent key on Windows
BG_FILL = "#1f2430"  # dark glass fill
TOP_OFFSET = 12


class Overlay:
    def __init__(self, tk_root: tk.Tk) -> None:
        self.root = tk_root
        self.win = tk.Toplevel(tk_root)
        self.win.withdraw()
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.attributes("-toolwindow", True)
        # Transparent key colour so the rounded card sits on "nothing".
        self.win.configure(bg=BG_KEY)
        try:
            self.win.attributes("-transparentcolor", BG_KEY)
        except tk.TclError:
            pass

        self.canvas = tk.Canvas(
            self.win, width=WIDTH, height=HEIGHT,
            bg=BG_KEY, highlightthickness=0, bd=0,
        )
        self.canvas.pack()
        self._position_top_center()

        self._hide_after_id: str | None = None
        self._draw("listening")

    # ----- public API (called from any thread via root.after) -----
    def show(self, state: str) -> None:
        self.root.after(0, lambda: self._show_now(state))

    def show_briefly(self, state: str, ms: int = 900) -> None:
        self.root.after(0, lambda: self._show_briefly_now(state, ms))

    def hide(self) -> None:
        self.root.after(0, self._hide_now)

    # ----- internal -----
    def _show_now(self, state: str) -> None:
        if self._hide_after_id is not None:
            self.root.after_cancel(self._hide_after_id)
            self._hide_after_id = None
        self._draw(state)
        self.win.deiconify()
        self.win.lift()

    def _show_briefly_now(self, state: str, ms: int) -> None:
        self._show_now(state)
        self._hide_after_id = self.root.after(ms, self._hide_now)

    def _hide_now(self) -> None:
        self._hide_after_id = None
        self.win.withdraw()

    def _position_top_center(self) -> None:
        self.win.update_idletasks()
        sw = self.win.winfo_screenwidth()
        x = (sw - WIDTH) // 2
        y = TOP_OFFSET
        self.win.geometry(f"{WIDTH}x{HEIGHT}+{x}+{y}")

    # ----- drawing -----
    def _draw(self, state: str) -> None:
        s = STATES.get(state, STATES["listening"])
        c = self.canvas
        c.delete("all")
        # Rounded card (simulated with arcs + rects).
        pad = 6
        x1, y1, x2, y2 = pad, pad, WIDTH - pad, HEIGHT - pad
        self._rounded_rect(x1, y1, x2, y2, r=22, fill=BG_FILL, outline=s.accent, width=2)
        # Translucent accent overlay tint — Tk has no alpha on canvas items, so
        # we paint a slightly lighter band along the top to imply the tint.
        self._rounded_rect(x1 + 2, y1 + 2, x2 - 2, y1 + 26, r=18, fill=s.accent, outline="")
        # Badge circle.
        bx, by, br = 32, HEIGHT // 2, 19
        c.create_oval(bx - br, by - br, bx + br, by + br, fill=s.accent, outline="#ffffff", width=1)
        self._draw_glyph(bx, by, s.glyph)
        # Title text.
        c.create_text(
            64, HEIGHT // 2,
            text=s.label, anchor="w",
            font=("Segoe UI Semibold", 16),
            fill=s.fg,
        )

    def _rounded_rect(self, x1, y1, x2, y2, r=16, **kw):
        c = self.canvas
        c.create_arc(x1, y1, x1 + 2 * r, y1 + 2 * r, start=90, extent=90, style="pieslice", **kw)
        c.create_arc(x2 - 2 * r, y1, x2, y1 + 2 * r, start=0, extent=90, style="pieslice", **kw)
        c.create_arc(x1, y2 - 2 * r, x1 + 2 * r, y2, start=180, extent=90, style="pieslice", **kw)
        c.create_arc(x2 - 2 * r, y2 - 2 * r, x2, y2, start=270, extent=90, style="pieslice", **kw)
        c.create_rectangle(x1 + r, y1, x2 - r, y2, **{k: v for k, v in kw.items() if k != "outline"}, outline="")
        c.create_rectangle(x1, y1 + r, x2, y2 - r, **{k: v for k, v in kw.items() if k != "outline"}, outline="")

    def _draw_glyph(self, cx: int, cy: int, kind: str) -> None:
        c = self.canvas
        col = "#ffffff"
        if kind == "mic":
            c.create_rectangle(cx - 5, cy - 11, cx + 5, cy + 3, fill=col, outline="")
            c.create_arc(cx - 5, cy - 3, cx + 5, cy + 7, start=180, extent=180, style="chord", fill=col, outline="")
            c.create_line(cx, cy + 7, cx, cy + 12, fill=col, width=2)
            c.create_line(cx - 6, cy + 12, cx + 6, cy + 12, fill=col, width=2)
        elif kind == "wave":
            for i, h in enumerate((4, 9, 6, 11, 5, 8, 4)):
                x = cx - 12 + i * 4
                c.create_line(x, cy - h, x, cy + h, fill=col, width=2, capstyle="round")
        elif kind == "spark":
            c.create_line(cx, cy - 11, cx, cy + 11, fill=col, width=2, capstyle="round")
            c.create_line(cx - 11, cy, cx + 11, cy, fill=col, width=2, capstyle="round")
            c.create_line(cx - 7, cy - 7, cx + 7, cy + 7, fill=col, width=2, capstyle="round")
            c.create_line(cx + 7, cy - 7, cx - 7, cy + 7, fill=col, width=2, capstyle="round")
        elif kind == "check":
            c.create_line(cx - 8, cy + 1, cx - 2, cy + 7, fill=col, width=3, capstyle="round")
            c.create_line(cx - 2, cy + 7, cx + 9, cy - 6, fill=col, width=3, capstyle="round")
