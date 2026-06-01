"""Render a hero banner SVG + PNG for the README."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1280, 640


def font(size: int, bold: bool = False):
    names = (["seguisb.ttf", "segoeuib.ttf"] if bold else ["segoeui.ttf"]) + ["arial.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_banner() -> Path:
    # Background: soft radial blue/purple gradient on near-black.
    img = Image.new("RGB", (W, H), (16, 18, 26))
    px = img.load()
    for y in range(H):
        for x in range(W):
            # Two radial sources.
            dx1, dy1 = x - 280, y - 130
            r1 = (dx1 * dx1 + dy1 * dy1) ** 0.5
            dx2, dy2 = x - 1000, y - 520
            r2 = (dx2 * dx2 + dy2 * dy2) ** 0.5
            i1 = max(0, 1 - r1 / 700)
            i2 = max(0, 1 - r2 / 700)
            r = int(16 + 60 * i1 * i1 + 30 * i2 * i2)
            g = int(18 + 70 * i1 * i1 + 22 * i2 * i2)
            b = int(26 + 180 * i1 * i1 + 130 * i2 * i2)
            px[x, y] = (min(r, 255), min(g, 255), min(b, 255))

    d = ImageDraw.Draw(img, "RGBA")

    # Title.
    title = "OpenWritr"
    d.text((90, 188), title, fill=(255, 255, 255), font=font(118, bold=True))
    d.text((92, 312), "for Windows on ARM", fill=(180, 195, 230), font=font(40))
    d.text((92, 384), "Push-to-talk voice-to-text · NVIDIA Parakeet TDT v3 · 40× realtime on Snapdragon X",
           fill=(150, 165, 200), font=font(22))

    # Right-side overlay mockup card.
    card_x, card_y, card_w, card_h = 760, 230, 440, 110
    rad = 32
    # Drop shadow.
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle((card_x + 4, card_y + 16, card_x + card_w + 4, card_y + card_h + 16),
                         radius=rad, fill=(0, 0, 0, 160))
    sh = sh.filter(ImageFilter.GaussianBlur(radius=24))
    img.paste(Image.alpha_composite(img.convert("RGBA"), sh).convert("RGB"))

    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle((card_x, card_y, card_x + card_w, card_y + card_h),
                        radius=rad, fill=(24, 28, 36, 240))
    # Badge.
    badge_d = 72
    bx = card_x + 22
    by = card_y + (card_h - badge_d) // 2
    d.ellipse((bx, by, bx + badge_d, by + badge_d), fill=(235, 72, 72))
    # Mic glyph centered.
    cx, cy = bx + badge_d // 2, by + badge_d // 2
    s = 2.6
    d.rounded_rectangle((cx - 5 * s, cy - 9 * s, cx + 5 * s, cy + 3 * s), radius=5 * s, fill=(255, 255, 255))
    d.arc((cx - 8 * s, cy - 1 * s, cx + 8 * s, cy + 9 * s), 0, 180, fill=(255, 255, 255), width=int(2 * s))
    d.line((cx, cy + 9 * s, cx, cy + 12 * s), fill=(255, 255, 255), width=int(2 * s))
    d.line((cx - 5 * s, cy + 12 * s, cx + 5 * s, cy + 12 * s), fill=(255, 255, 255), width=int(2 * s))

    # Level bars.
    meter_x = bx + badge_d + 28
    meter_y = card_y + card_h // 2
    levels = [0.45, 0.78, 0.55, 0.95, 0.62, 0.40, 0.72, 0.50, 0.30]
    bar_w = 10
    gap = 10
    max_h = 76
    for i, lv in enumerate(levels):
        bh = max(8, int(lv * max_h))
        x = meter_x + i * (bar_w + gap)
        d.rounded_rectangle((x, meter_y - bh // 2, x + bar_w, meter_y + bh // 2),
                            radius=5, fill=(255, 255, 255, 235))

    # Footer tag.
    d.text((90, 528), "Local · Private · Open Source · MIT",
           fill=(120, 140, 180), font=font(20))

    out = OUT_DIR / "hero.png"
    img.save(out, optimize=True)
    print(f"wrote {out}")
    return out


def make_overlay_demo() -> Path:
    """Just the overlay card on a neutral backdrop for inline docs."""
    W2, H2 = 600, 160
    img = Image.new("RGB", (W2, H2), (40, 45, 56))
    d = ImageDraw.Draw(img, "RGBA")
    card_x, card_y, card_w, card_h = 90, 30, 420, 100
    rad = 30
    sh = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle((card_x + 4, card_y + 12, card_x + card_w + 4, card_y + card_h + 12),
                         radius=rad, fill=(0, 0, 0, 170))
    sh = sh.filter(ImageFilter.GaussianBlur(radius=18))
    img = Image.alpha_composite(img.convert("RGBA"), sh).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle((card_x, card_y, card_x + card_w, card_y + card_h),
                        radius=rad, fill=(24, 28, 36, 240))
    badge_d = 64
    bx = card_x + 18
    by = card_y + (card_h - badge_d) // 2
    d.ellipse((bx, by, bx + badge_d, by + badge_d), fill=(235, 72, 72))
    cx, cy = bx + badge_d // 2, by + badge_d // 2
    s = 2.3
    d.rounded_rectangle((cx - 5 * s, cy - 9 * s, cx + 5 * s, cy + 3 * s), radius=5 * s, fill=(255, 255, 255))
    d.arc((cx - 8 * s, cy - 1 * s, cx + 8 * s, cy + 9 * s), 0, 180, fill=(255, 255, 255), width=int(2 * s))
    d.line((cx, cy + 9 * s, cx, cy + 12 * s), fill=(255, 255, 255), width=int(2 * s))
    d.line((cx - 5 * s, cy + 12 * s, cx + 5 * s, cy + 12 * s), fill=(255, 255, 255), width=int(2 * s))
    meter_x = bx + badge_d + 24
    meter_y = card_y + card_h // 2
    levels = [0.4, 0.7, 0.5, 0.9, 0.55, 0.35, 0.65, 0.45, 0.3]
    bar_w = 9
    gap = 8
    max_h = 64
    for i, lv in enumerate(levels):
        bh = max(7, int(lv * max_h))
        x = meter_x + i * (bar_w + gap)
        d.rounded_rectangle((x, meter_y - bh // 2, x + bar_w, meter_y + bh // 2),
                            radius=4, fill=(255, 255, 255, 235))
    out = OUT_DIR / "overlay-listening.png"
    img.save(out, optimize=True)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    make_banner()
    make_overlay_demo()
