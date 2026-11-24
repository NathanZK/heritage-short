"""Placeholder providers for --mock runs and tests: no network, no credentials, deterministic output.
Same interface as pipeline.providers. Media is plainly fake (colour frames, sine tones)."""
from __future__ import annotations

import hashlib
import io
import math
import struct
import wave

TEXT_MODEL = "mock-text"
TTS_MODEL = "mock-tts"
IMAGE_MODEL = "mock-image"


def _digest(*parts) -> bytes:
    return hashlib.sha256("|".join(map(str, parts)).encode()).digest()


def sine_wav(seconds: float, freq: float = 220.0, rate: int = 24000, volume: float = 0.3) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(volume * 32767 * math.sin(2 * math.pi * freq * i / rate)))
                               for i in range(int(seconds * rate))))
    return buf.getvalue()


def gemini_json(prompt: str, schema: dict) -> dict:
    return {"shots": [{
        "id": f"s{i:02d}", "duration": 5.0, "reason": f"Mock reason for shot {i}.",
        "keyframe_prompt": f"Mock keyframe {i}.", "motion_prompt": "slow push in",
        "narration": f"Mock narration line number {i}.", "subtitle": f"Mock narration line number {i}.",
        "music_cue": f"mock cue {i}",
    } for i in range(1, 13)]}


def gemini_tts(text: str, voice: str) -> bytes:
    """Length follows the word count (never above 3 s); pitch follows the text hash."""
    freq = 180 + _digest(text)[0] % 300
    return sine_wav(min(3.0, 0.3 * len(text.split()) + 0.4), freq=freq)


def cloudflare_image(prompt: str, steps: int = 4) -> bytes:
    from PIL import Image, ImageDraw

    digest = _digest(prompt, steps)
    img = Image.new("RGB", (1280, 720), tuple(60 + b % 120 for b in digest[:3]))
    ImageDraw.Draw(img).text((40, 40), f"MOCK {digest.hex()[:8]}", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()
