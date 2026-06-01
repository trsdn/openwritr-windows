Drop `icon.png` (256×256, transparent) and `icon.ico` here before first build.

Tauri expects these paths from `tauri.conf.json`. We intentionally do not commit
a binary placeholder; the build will fail loudly if the icon is missing, which
is the right signal during scaffolding.
