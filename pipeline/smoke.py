"""Smoke test: one real call per provider, with the storyboard's configured models (text_model, keyframe_model),
logged to runs.jsonl with its latency. Run this yourself after setting credentials (make smoke); the tests only
use --mock. The klein image is 256x256, the smallest size, to use as few neurons as possible."""
from __future__ import annotations

from pipeline.common import call_model, seed_of, stage_main
from pipeline.keyframes import STEPS, klein
from pipeline.providers import cloudflare_usage
from pipeline.script import text_model_id

SMOKE_PROMPT = "a quiet village road at dawn"


def run(ctx, sb, args) -> int:
    p, text = ctx.provider, text_model_id(ctx, sb)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    if klein(sb):
        image = (p.KLEIN_MODEL, lambda: p.cloudflare_klein(SMOKE_PROMPT, 256, 256, seed_of(sb), None), cloudflare_usage)
    else:
        image = (p.IMAGE_MODEL, lambda: p.cloudflare_image(SMOKE_PROMPT, STEPS), None)
    calls = [
        ("gemini", text, lambda: p.gemini_json("Reply with JSON {\"ok\": true}.", schema, text), None),
        ("gemini", p.TTS_MODEL, lambda: p.gemini_tts("Hello from the heritage short.", sb["voice"]), None),
        ("cloudflare", *image),
    ]
    for provider, model, fn, usage in calls:
        call_model(ctx, stage="smoke", shot=None, provider=provider, model=model, params={}, seed=None, fn=fn,
                   usage=usage)
        print(f"smoke: {provider} {model} ok")
    return 0


def main(argv=None) -> None:
    stage_main(run, __doc__, need_shots=False, only_mode="ignore", argv=argv)


if __name__ == "__main__":
    main()
