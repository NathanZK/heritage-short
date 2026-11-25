"""Stage: keyframes. A character reference sheet, then one keyframe per shot, with Cloudflare Workers AI
(flux-1-schnell). The documented API has no seed and no reference-image input, so nothing here is
conditioned on the reference sheet; it is a golden reference for human review."""
from __future__ import annotations

import io

from pipeline.common import (Manifest, ProviderError, call_model, check_image, commit_bytes, composed_keyframe_prompt,
                             fingerprint, stage_main)

STEPS = 4  # documented default; maximum 8
V = 1


def reference_prompt(sb: dict) -> str:
    return (f"{sb['style']}. Character reference sheet: {sb['character']['description']}. "
            "Front view and side view, neutral background.")


def reference_fp(ctx, sb: dict) -> str:
    return fingerprint(V, "reference", ctx.provider.IMAGE_MODEL, STEPS, reference_prompt(sb))


def keyframe_fp(ctx, sb: dict, shot: dict) -> str:
    return fingerprint(V, "keyframe", ctx.provider.IMAGE_MODEL, STEPS, sb["style"], sb["character"]["description"],
                       shot["keyframe_prompt"])


def to_jpeg(data: bytes) -> bytes:
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except OSError as exc:
        raise ProviderError(f"provider returned an unreadable image: {exc}")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _generate(ctx, manifest, key, shot_label, path, fp, prompt) -> None:
    if manifest.fresh(key, fp, path):
        print(f"keyframes: {key} up to date")
        return
    p = ctx.provider
    data = call_model(ctx, stage="keyframes", shot=shot_label, provider="cloudflare", model=p.IMAGE_MODEL,
                      params={"prompt": prompt, "steps": STEPS}, seed=None,
                      fn=lambda: to_jpeg(p.cloudflare_image(prompt, STEPS)))
    commit_bytes(path, data, validate=check_image)
    manifest.record(key, fp, path)
    print(f"keyframes: generated {key}")


def run(ctx, sb, args) -> int:
    manifest = Manifest(ctx)
    _generate(ctx, manifest, "reference", "reference", ctx.path("keyframes", "reference.jpg"),
              reference_fp(ctx, sb), reference_prompt(sb))
    for shot in filter(ctx.wanted, sb["shots"]):
        _generate(ctx, manifest, f"keyframe:{shot['id']}", shot["id"], ctx.path("keyframes", f"{shot['id']}.jpg"),
                  keyframe_fp(ctx, sb, shot), composed_keyframe_prompt(sb, shot))
    return 0


def main(argv=None) -> None:
    stage_main(run, __doc__, argv=argv)


if __name__ == "__main__":
    main()
