"""Clients for the two hosted APIs the assessment prescribes: Gemini (script, narration) and
Cloudflare Workers AI (keyframes). Shapes follow the official docs (see the spec, section 5);
response parsing is strict, so an unexpected shape raises instead of guessing."""
from __future__ import annotations

import base64
import functools
import http.client
import io
import json
import os
import urllib.error
import urllib.request
import wave

from pipeline.common import ProviderError

TEXT_MODEL = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-tts"
IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
CF_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/" + IMAGE_MODEL
CF_DAILY_LIMIT_CODE = "3036"


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ProviderError(f"{name} is not set; copy .env.example to .env and fill it in")
    return value


def http_post(url: str, headers: dict, body: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode(errors="replace")[:300]
        except (OSError, http.client.HTTPException):
            detail = ""
        quota = exc.code == 429 and CF_DAILY_LIMIT_CODE in detail
        raise ProviderError(f"HTTP {exc.code}: {detail}", status=exc.code, quota=quota,
                            retryable=not quota and exc.code in (429, 500, 502, 503, 504))
    except (OSError, http.client.HTTPException, ValueError) as exc:
        # timeouts, dropped connections and half-read bodies (Python 3.9: socket.timeout is not a TimeoutError)
        raise ProviderError(f"request failed: {exc!r}", retryable=True)
    if not isinstance(data, dict):
        raise ProviderError("response was not a JSON object")
    return data


def strict(fn):
    """A response that is not shaped as documented is a ProviderError, never a traceback."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (AttributeError, TypeError, KeyError, ValueError) as exc:
            raise ProviderError(f"unexpected response shape: {exc!r}")
    return wrapper


def _gemini(body: dict) -> dict:
    return http_post(GEMINI_URL, {"x-goog-api-key": _env("GEMINI_API_KEY"), "Content-Type": "application/json",
                                  "Api-Revision": "2026-05-20"}, body)


def _content(resp: dict, kind: str) -> list:
    return [item for step in resp.get("steps", []) if step.get("type") == "model_output"
            for item in step.get("content", []) if item.get("type") == kind]


@strict
def gemini_json(prompt: str, schema: dict) -> dict:
    resp = _gemini({"model": TEXT_MODEL, "input": prompt,
                    "response_format": {"type": "text", "mime_type": "application/json", "schema": schema}})
    texts = _content(resp, "text")
    if not texts:
        raise ProviderError("Gemini response contained no text")
    try:
        result = json.loads(texts[-1]["text"])
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Gemini returned invalid JSON: {exc}")
    if not isinstance(result, dict):
        raise ProviderError("Gemini returned JSON that is not an object")
    return result


@strict
def gemini_tts(text: str, voice: str) -> bytes:
    """Return a WAV file (24 kHz mono 16-bit) speaking `text`."""
    resp = _gemini({"model": TTS_MODEL,
                    "input": [{"type": "user_input", "content": [{"type": "text", "text": text}]}],
                    "response_format": {"type": "audio"},
                    "generation_config": {"speech_config": [{"voice": voice}]}})
    audio = _content(resp, "audio")
    if not audio or not audio[-1].get("data"):
        raise ProviderError("Gemini response contained no audio")
    data = base64.b64decode(audio[-1]["data"], validate=True)
    try:
        wave.open(io.BytesIO(data)).close()
    except (wave.Error, EOFError):
        raise ProviderError("Gemini audio was not a WAV file")
    return data


@strict
def cloudflare_image(prompt: str, steps: int = 4) -> bytes:
    """Return the image bytes. The documented input has only `prompt` and `steps`: no seed, no size."""
    url = CF_URL.format(account=_env("CLOUDFLARE_ACCOUNT_ID"))
    resp = http_post(url, {"Authorization": f"Bearer {_env('CLOUDFLARE_API_TOKEN')}",
                           "Content-Type": "application/json"}, {"prompt": prompt, "steps": steps})
    if resp.get("success") is False:
        raise ProviderError(f"Cloudflare error: {resp.get('errors')}")
    result = resp.get("result") or {}
    image = (result.get("image") if isinstance(result, dict) else None) or resp.get("image")
    if not image:
        raise ProviderError("Cloudflare response contained no image")
    return base64.b64decode(image, validate=True)
