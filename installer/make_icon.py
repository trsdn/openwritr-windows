"""
Generate the OpenWritr app icon as a multi-resolution .ico.

A rounded-square badge in the app's blue (the tray idle color 74,144,226),
with a simple white microphone glyph. Emitted at 16/32/48/64/128/256 px so
Windows picks the right size for the taskbar, Start Menu, Alt-Tab, and the
installer.

    python installer/make_icon.py
"""
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "openwritr.ico"
BASE = 256  # design at 256, downscale for the .ico set


def rounded_rect_mask(size, radius):
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return m


def vertical_gradient(size, top, bottom):
    grad = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / (size - 1)
        grad.putpixel((0, y), tuple(
            int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)
        ))
    return grad.resize((size, size))


def make_base():
    s = BASE
    # Blue gradient: brighter top → deeper bottom, around the tray idle blue.
    bg = vertical_gradient(s, (96, 165, 250), (37, 99, 235))
    icon = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    icon.paste(bg, (0, 0), rounded_rect_mask(s, radius=int(s * 0.22)))

    d = ImageDraw.Draw(icon)
    white = (255, 255, 255, 255)

    # Microphone capsule (rounded vertical pill).
    cap_w = int(s * 0.26)
    cap_h = int(s * 0.42)
    cx = s // 2
    cap_top = int(s * 0.16)
    cap_left = cx - cap_w // 2
    cap_right = cx + cap_w // 2
    cap_bottom = cap_top + cap_h
    d.rounded_rectangle([cap_left, cap_top, cap_right, cap_bottom],
                        radius=cap_w // 2, fill=white)

    # U-shaped cradle arc around the lower capsule.
    arc_pad = int(s * 0.06)
    arc_box = [cap_left - arc_pad, cap_top + int(cap_h * 0.30),
               cap_right + arc_pad, cap_bottom + arc_pad]
    d.arc(arc_box, start=20, end=160, fill=white, width=int(s * 0.045))

    # Stem.
    stem_top = cap_bottom + arc_pad
    stem_bottom = int(s * 0.82)
    d.line([(cx, stem_top), (cx, stem_bottom)], fill=white, width=int(s * 0.05))

    # Base foot.
    foot_w = int(s * 0.28)
    d.line([(cx - foot_w // 2, stem_bottom), (cx + foot_w // 2, stem_bottom)],
           fill=white, width=int(s * 0.05))

    return icon


def make_store_assets(base):
    """Generate the PNG asset set the Microsoft Store / MSIX manifest needs.

    Logos with transparent padding around the badge look better in the Store
    and on tiles, so each asset renders the badge at ~80% of the canvas.
    """
    out_dir = Path(__file__).parent / "store-assets"
    out_dir.mkdir(exist_ok=True)
    assets = {
        # name: (canvas_w, canvas_h)
        "Square44x44Logo.png":   (44, 44),
        "Square71x71Logo.png":   (71, 71),
        "Square150x150Logo.png": (150, 150),
        "Square310x310Logo.png": (310, 310),
        "Wide310x150Logo.png":   (310, 150),
        "StoreLogo.png":         (50, 50),
        "SplashScreen.png":      (620, 300),
    }
    for name, (w, h) in assets.items():
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        side = int(min(w, h) * 0.8)
        badge = base.resize((side, side), Image.LANCZOS)
        canvas.paste(badge, ((w - side) // 2, (h - side) // 2), badge)
        canvas.save(out_dir / name, format="PNG")
    print(f"wrote {len(assets)} store assets -> {out_dir}")


def main():
    base = make_base()
    sizes = [16, 32, 48, 64, 128, 256]
    base.save(OUT, format="ICO", sizes=[(n, n) for n in sizes])
    # Also a PNG for README / HF / web use.
    base.save(OUT.with_suffix(".png"), format="PNG")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes) + {OUT.with_suffix('.png').name}")
    make_store_assets(base)


if __name__ == "__main__":
    main()
