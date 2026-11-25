"""Clients for the two hosted APIs the assessment prescribes: Gemini (script, narration) and
Cloudflare Workers AI (keyframes). Shapes follow the official docs (see the spec, section 5);
response parsing is strict, so an unexpected shape raises instead of guessing."""
from __future__ import annotations

import base64
import functools
import http.client
import io
import json
import math
import os
import urllib.error
import urllib.request
import uuid
import wave

from pipeline.common import Generated, ProviderError

TEXT_MODEL = "gemini-3.5-flash-lite"   # default text model (EXPERIMENTS.md E2); storyboard text_model selects
TEXT_MODEL_FLASH = "gemini-3.8-flash"
TTS_MODEL = "gemini-3.8-flash-tts"
IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell"
KLEIN_MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
CF_RUN = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/"
CF_URL = CF_RUN + IMAGE_MODEL
CF_DAILY_LIMIT_CODE = "3036"
NEURON_PRICE_USD = 0.011 / 1000   # Workers AI pricing page: $0.011 per 1,000 neurons above the free daily allocation


def _env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ProviderError(f"{name} is not set; copy .env.example to .env and fill it in")
    return value


def http_post(url: str, headers: dict, body: dict, timeout: float = 120) -> dict:
    return _send(url, headers, json.dumps(body).encode(), timeout)[0]


def http_post_form(url: str, headers: dict, fields: dict, files: dict, timeout: float = 120) -> tuple:
    """POST multipart/form-data: text `fields`, then binary `files` ({name: bytes}). Returns (JSON body, headers)."""
    boundary = "heritage-" + uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in fields.items()]
    for name, data in files.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}"\r\n'
                     f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    return _send(url, {**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"}, body, timeout)


def _send(url: str, headers: dict, data: bytes, timeout: float) -> tuple:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            reply_headers = dict(getattr(resp, "headers", None) or {})
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
    return data, reply_headers


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
def gemini_json(prompt: str, schema: dict, model: str = TEXT_MODEL) -> dict:
    resp = _gemini({"model": model, "input": prompt,
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
    return _cf_image(resp)


def _neurons(headers: dict):
    """Neurons the call used, from Cloudflare's cf-ai-neurons response header; None if absent or unreadable."""
    value = next((v for k, v in headers.items() if k.lower() == "cf-ai-neurons"), None)
    try:
        neurons = float(value)
    except (TypeError, ValueError):
        return None
    return neurons if math.isfinite(neurons) and neurons >= 0 else None


def _cf_image(resp: dict) -> bytes:
    if resp.get("success") is False:
        raise ProviderError(f"Cloudflare error: {resp.get('errors')}")
    result = resp.get("result") or {}
    image = (result.get("image") if isinstance(result, dict) else None) or resp.get("image")
    if not image:
        raise ProviderError("Cloudflare response contained no image")
    return base64.b64decode(image, validate=True)


@strict
def cloudflare_klein(prompt: str, width: int, height: int, seed: int, reference) -> Generated:
    """FLUX.2 [klein] 4B: multipart form fields documented in Cloudflare's klein-4b changelog entry. `reference`
    (bytes, each side under 512 px) is sent as input_image_0. Returns the image and the neurons used."""
    url = CF_RUN.format(account=_env("CLOUDFLARE_ACCOUNT_ID")) + KLEIN_MODEL
    resp, headers = http_post_form(url, {"Authorization": f"Bearer {_env('CLOUDFLARE_API_TOKEN')}"},
                                   {"prompt": prompt, "width": width, "height": height, "seed": seed},
                                   {"input_image_0": reference} if reference is not None else {})
    return Generated(_cf_image(resp), _neurons(headers))


def cloudflare_usage(result: Generated) -> dict:
    """runs.jsonl fields for a Cloudflare image call: the neurons it reported and their list price (None if not
    reported). cost_usd stays the amount actually billed (0.0 inside the free allocation)."""
    neurons = result.neurons
    return {"neurons": neurons, "list_price_usd": round(neurons * NEURON_PRICE_USD, 8) if neurons is not None else None}
