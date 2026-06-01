"""Grammar / cleanup enhancement provider.

Two backends, configurable via Settings:
  * github_copilot: read OAuth token from `gh auth token` and call
    https://api.githubcopilot.com/chat/completions
  * openai_compatible: any OpenAI-style /chat/completions endpoint

Falls back to returning the raw transcript on any error.
"""
from __future__ import annotations
import json
import subprocess
import time
from urllib import request as urlrequest

SYSTEM_PROMPT = (
    "You are a transcription cleanup assistant. Fix punctuation, casing, "
    "filler words ('um', 'uh', 'like'), and obvious recognition errors in "
    "the user message. Preserve the original meaning, language, and tone. "
    "Return ONLY the cleaned text — no preamble, no quotes, no commentary."
)


def _gh_token() -> str | None:
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=5,
        )
        token = (out.stdout or "").strip()
        return token or None
    except Exception:
        return None


def _post_chat(url: str, token: str, model: str, text: str, extra_headers: dict | None = None) -> str:
    body = json.dumps({
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    req = urlrequest.Request(url, data=body, headers=headers, method="POST")
    with urlrequest.urlopen(req, timeout=12) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"].strip()


def enhance(text: str, settings: dict) -> str:
    cfg = settings.get("enhance") or {}
    provider = cfg.get("provider", "off")
    if provider == "off" or not text.strip():
        return text
    try:
        if provider == "github_copilot":
            token = _gh_token()
            if not token:
                return text
            return _post_chat(
                "https://api.githubcopilot.com/chat/completions",
                token,
                cfg.get("model") or "gpt-4o-mini",
                text,
                extra_headers={
                    "Copilot-Integration-Id": "vscode-chat",
                    "Editor-Version": "OpenWritr/0.1",
                },
            )
        if provider == "openai_compatible":
            base = (cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
            key = cfg.get("api_key") or ""
            if not key:
                return text
            return _post_chat(
                f"{base}/chat/completions",
                key,
                cfg.get("model") or "gpt-4o-mini",
                text,
            )
    except Exception:
        pass
    return text
