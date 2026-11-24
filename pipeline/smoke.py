"""Smoke test: one real call per provider, logged to runs.jsonl with its latency.
Run this yourself after setting credentials (make smoke); the tests only use --mock."""
from __future__ import annotations

from pipeline.common import call_model, stage_main
from pipeline.keyframes import STEPS


def run(ctx, sb, args) -> int:
    p = ctx.provider
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    calls = [
        ("gemini", p.TEXT_MODEL, lambda: p.gemini_json("Reply with JSON {\"ok\": true}.", schema)),
        ("gemini", p.TTS_MODEL, lambda: p.gemini_tts("Hello from the heritage short.", sb["voice"])),
        ("cloudflare", p.IMAGE_MODEL, lambda: p.cloudflare_image("a quiet village road at dawn", STEPS)),
    ]
    for provider, model, fn in calls:
        call_model(ctx, stage="smoke", shot=None, provider=provider, model=model, params={}, seed=None, fn=fn)
        print(f"smoke: {provider} {model} ok")
    return 0


def main(argv=None) -> None:
    stage_main(run, __doc__, need_shots=False, only_mode="ignore", argv=argv)


if __name__ == "__main__":
    main()
