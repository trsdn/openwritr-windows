"""Standalone settings window — launched as a subprocess from openwritr.py.

Reads current settings.json, presents a Fluent-styled WebView2 dialog, and
writes back on Save. Pure isolation: the main process keeps its Tk loop
clean and pywebview's one-shot start() restriction is bypassed by giving
each settings invocation its own process.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import webview


APPDATA = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "OpenWritr"
SETTINGS_PATH = APPDATA / "settings.json"

DEFAULTS = {
    "hotkey_modifiers": ["ctrl", "shift"],
    "auto_paste": True,
    "overlay": True,
    "sounds": True,
    "min_record_seconds": 0.25,
    "max_record_seconds": 60,
    "enhance": {"provider": "off", "base_url": "https://api.openai.com/v1",
                "api_key": "", "model": "gpt-4o-mini"},
}


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OpenWritr Settings</title>
<style>
  :root {
    --bg: #161a23;
    --bg-2: #20242e;
    --bg-3: #2a2f3a;
    --fg: #e8ecf3;
    --fg-dim: #9aa3b2;
    --accent: #4f8cff;
    --accent-hover: #6aa0ff;
    --border: #353b48;
    --ok: #22c55e;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    color: var(--fg);
    font: 13px "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
    background:
      radial-gradient(900px 500px at 20% -10%, rgba(79,140,255,.10), transparent 60%),
      linear-gradient(180deg, #1c2029, #14171f);
  }
  body { padding: 22px 26px 26px; overflow-y: auto; }
  h1 { font: 600 22px/1.2 "Segoe UI Variable Display","Segoe UI",sans-serif; margin: 0 0 4px; }
  .sub { color: var(--fg-dim); margin-bottom: 18px; }
  .section {
    background: rgba(255,255,255,.03);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 14px;
  }
  .section h2 {
    font: 600 11px/1 "Segoe UI Variable Small","Segoe UI",sans-serif;
    letter-spacing: .08em; text-transform: uppercase;
    color: var(--fg-dim);
    margin: 0 0 12px;
  }
  .row { display: flex; align-items: center; justify-content: space-between;
         padding: 8px 0; border-top: 1px solid rgba(255,255,255,.04); }
  .row:first-of-type { border-top: 0; padding-top: 0; }
  .row label { flex: 1; }
  .hint { color: var(--fg-dim); font-size: 12px; margin-top: 2px; }
  .mods { display: flex; gap: 10px; flex-wrap: wrap; }
  .chip {
    user-select: none; cursor: pointer;
    padding: 7px 14px; border-radius: 999px;
    background: var(--bg-3); border: 1px solid var(--border);
    color: var(--fg); transition: all .14s ease;
    font-weight: 500;
  }
  .chip.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  .chip:hover { border-color: var(--accent); }
  .switch { position: relative; width: 38px; height: 22px; flex: none; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .track { position: absolute; inset: 0; background: var(--bg-3);
           border: 1px solid var(--border); border-radius: 999px;
           transition: background .15s, border-color .15s; cursor: pointer; }
  .track::after { content:""; position: absolute; top: 2px; left: 2px;
                  width: 16px; height: 16px; border-radius: 50%; background: #d5dbe6;
                  transition: transform .15s, background .15s; }
  input:checked + .track { background: var(--accent); border-color: var(--accent); }
  input:checked + .track::after { transform: translateX(16px); background: #fff; }
  select, input[type=text], input[type=password] {
    width: 100%; padding: 9px 11px; margin-top: 4px;
    background: var(--bg-2); color: var(--fg);
    border: 1px solid var(--border); border-radius: 6px;
    font: 13px "Segoe UI Variable Text","Segoe UI",sans-serif;
    outline: none;
    transition: border-color .12s;
  }
  select:focus, input:focus { border-color: var(--accent); }
  .field { margin-top: 10px; }
  .field label { display: block; color: var(--fg-dim); font-size: 12px; margin-bottom: 2px; }
  .footer { display: flex; align-items: center; justify-content: flex-end;
            gap: 10px; margin-top: 14px; }
  button {
    padding: 9px 20px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--bg-3); color: var(--fg); cursor: pointer;
    font: 600 13px "Segoe UI Variable Text","Segoe UI",sans-serif;
    transition: background .14s, border-color .14s;
  }
  button:hover { background: #343a48; }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  button.primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
  .status { font-size: 12px; color: var(--ok); margin-right: auto;
            opacity: 0; transition: opacity .2s; }
  .status.show { opacity: 1; }
</style>
</head>
<body>
  <h1>OpenWritr</h1>
  <div class="sub">Voice-to-text for Windows on ARM</div>

  <div class="section">
    <h2>Hotkey (hold to record)</h2>
    <div class="mods" id="mods"></div>
    <div class="hint" style="margin-top:10px;">… plus Space (always required). Hold the combo to dictate; release to transcribe.</div>
  </div>

  <div class="section">
    <h2>Behaviour</h2>
    <div class="row"><label>Auto-paste at cursor</label>
      <span class="switch"><input type="checkbox" id="auto_paste"><span class="track"></span></span></div>
    <div class="row"><label>Show overlay while recording</label>
      <span class="switch"><input type="checkbox" id="overlay"><span class="track"></span></span></div>
    <div class="row"><label>Play start/stop sounds</label>
      <span class="switch"><input type="checkbox" id="sounds"><span class="track"></span></span></div>
  </div>

  <div class="section">
    <h2>Enhance (punctuation + cleanup)</h2>
    <div class="hint" style="margin-bottom:8px;">Hold the hotkey with Alt also pressed to trigger LLM cleanup after transcription.</div>
    <div class="field">
      <label for="provider">Provider</label>
      <select id="provider">
        <option value="off">Off</option>
        <option value="github_copilot">GitHub Copilot (uses <code>gh auth token</code>)</option>
        <option value="openai_compatible">OpenAI-compatible API</option>
      </select>
    </div>
    <div class="field">
      <label for="base_url">Base URL (OpenAI-compatible only)</label>
      <input type="text" id="base_url" placeholder="https://api.openai.com/v1">
    </div>
    <div class="field">
      <label for="api_key">API key (OpenAI-compatible only)</label>
      <input type="password" id="api_key" placeholder="sk-...">
    </div>
    <div class="field">
      <label for="model">Model</label>
      <input type="text" id="model" placeholder="gpt-4o-mini">
    </div>
  </div>

  <div class="footer">
    <div class="status" id="status">Saved</div>
    <button onclick="window.pywebview.api.close()">Close</button>
    <button class="primary" onclick="save()">Save</button>
  </div>

<script>
  const MODS = ["ctrl","shift","alt","win"];
  async function load() {
    const s = await window.pywebview.api.get_settings();
    const root = document.getElementById("mods");
    root.innerHTML = "";
    const have = new Set(s.hotkey_modifiers || []);
    MODS.forEach(m => {
      const chip = document.createElement("div");
      chip.className = "chip" + (have.has(m) ? " on" : "");
      chip.textContent = m.charAt(0).toUpperCase() + m.slice(1);
      chip.dataset.mod = m;
      chip.onclick = () => chip.classList.toggle("on");
      root.appendChild(chip);
    });
    document.getElementById("auto_paste").checked = s.auto_paste !== false;
    document.getElementById("overlay").checked = s.overlay !== false;
    document.getElementById("sounds").checked = s.sounds !== false;
    const e = s.enhance || {};
    document.getElementById("provider").value = e.provider || "off";
    document.getElementById("base_url").value = e.base_url || "https://api.openai.com/v1";
    document.getElementById("api_key").value = e.api_key || "";
    document.getElementById("model").value = e.model || "gpt-4o-mini";
  }
  async function save() {
    const mods = [...document.querySelectorAll(".chip.on")].map(c => c.dataset.mod);
    const payload = {
      hotkey_modifiers: mods.length ? mods : ["ctrl","shift"],
      auto_paste: document.getElementById("auto_paste").checked,
      overlay: document.getElementById("overlay").checked,
      sounds: document.getElementById("sounds").checked,
      enhance: {
        provider: document.getElementById("provider").value,
        base_url: document.getElementById("base_url").value.trim(),
        api_key: document.getElementById("api_key").value.trim(),
        model: document.getElementById("model").value.trim() || "gpt-4o-mini",
      },
    };
    await window.pywebview.api.save_settings(payload);
    const s = document.getElementById("status");
    s.classList.add("show");
    setTimeout(() => s.classList.remove("show"), 1400);
  }
  window.addEventListener("pywebviewready", load);
</script>
</body>
</html>
"""


def _load() -> dict:
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text("utf-8"))
            merged = {**DEFAULTS, **data}
            merged["enhance"] = {**DEFAULTS["enhance"], **(data.get("enhance") or {})}
            return merged
        except Exception:
            pass
    return {**DEFAULTS, "enhance": dict(DEFAULTS["enhance"])}


def _save(payload: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(payload, indent=2), "utf-8")


class _Api:
    def __init__(self):
        self.window = None

    def get_settings(self):
        return _load()

    def save_settings(self, payload):
        _save(payload)
        return True

    def close(self):
        if self.window is not None:
            self.window.destroy()


def main() -> int:
    api = _Api()
    window = webview.create_window(
        "OpenWritr Settings",
        html=HTML,
        width=560, height=720,
        resizable=True,
        background_color="#14171f",
        js_api=api,
    )
    api.window = window
    webview.start(debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
