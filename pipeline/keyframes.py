"""Stage: keyframes, with Cloudflare Workers AI.

Default model FLUX.2 [klein] 4B (`keyframe_model: flux-2-klein-4b`): 1280x704 keyframes, a per-shot seed, and,
when `character.reference_image` is set, every shot keyframe conditioned on that approved image (sent as
input_image_0). Without a reference image a character reference sheet is made for human review only.
`keyframe_model: flux-1-schnell` keeps the original path: no seed, no size, no reference image."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

from pipeline.common import (Manifest, PipelineExit, ProviderError, call_model, check_image, commit_bytes,
                             composed_keyframe_prompt, fingerprint, seed_of, shot_seed, stage_main)
from pipeline.providers import cloudflare_usage

STEPS = 4             # flux-1-schnell: documented default; maximum 8 (klein's 4 steps are fixed)
V = 1                 # fingerprint version of schnell keyframes and the schnell sheet
KLEIN_V = 1           # fingerprint version of klein keyframes and the klein sheet
KLEIN_SIZE = (1280, 704)   # the shots stage's Wan size, so keyframes need no crop
REFERENCE_MAX = 511        # Cloudflare: "All input images must be smaller than 512x512"


def model_name(sb: dict) -> str:
    return sb.get("keyframe_model", "flux-2-klein-4b")


def klein(sb: dict) -> bool:
    return model_name(sb) == "flux-2-klein-4b"


def reference_source(ctx, sb: dict):
    """The approved reference image's bytes, or None when the storyboard sets none."""
    rel = sb["character"].get("reference_image")
    if not rel:
        return None
    root = ctx.root.resolve()
    path = (root / rel).resolve()
    if Path(rel).is_absolute() or not path.is_relative_to(root):
        raise PipelineExit(1, f"character.reference_image {rel} must be a path inside the repository "
                              f"(it is uploaded to Cloudflare)")
    try:
        return path.read_bytes()
    except OSError:
        raise PipelineExit(1, f"character.reference_image {rel} cannot be read; add the approved image there")


def reference_input(data: bytes) -> bytes:
    """The reference as sent: unchanged when it is already a plain, upright RGB or greyscale JPEG/PNG under 512 px;
    otherwise turned upright (EXIF orientation), converted to RGB and saved as JPEG, downscaled to fit 511x511."""
    from PIL import Image, ImageOps

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except OSError as exc:
        raise PipelineExit(1, f"character.reference_image is not a readable image: {exc}")
    upright = img.getexif().get(0x0112, 1) == 1
    if max(img.size) <= REFERENCE_MAX and upright and img.format in ("JPEG", "PNG") and img.mode in ("RGB", "L"):
        return data
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((REFERENCE_MAX, REFERENCE_MAX), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def reference_prompt(sb: dict) -> str:
    return (f"{sb['style']}. Character reference sheet: {sb['character']['description']}. "
            "Front view and side view, neutral background.")


def reference_fp(ctx, sb: dict) -> str:
    if klein(sb):
        return fingerprint(KLEIN_V, "reference", ctx.provider.KLEIN_MODEL, KLEIN_SIZE, seed_of(sb), reference_prompt(sb))
    return fingerprint(V, "reference", ctx.provider.IMAGE_MODEL, STEPS, reference_prompt(sb))


def keyframe_fp(ctx, sb: dict, shot: dict) -> str:
    if klein(sb):
        source = reference_source(ctx, sb)
        return fingerprint(KLEIN_V, "keyframe", ctx.provider.KLEIN_MODEL, KLEIN_SIZE, shot_seed(sb, shot),
                           composed_keyframe_prompt(sb, shot), hashlib.sha256(source).hexdigest() if source else None)
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


def _generate(ctx, sb, manifest, key, shot_label, path, fp, prompt, seed, reference=None, reference_sha=None) -> None:
    if manifest.fresh(key, fp, path):
        print(f"keyframes: {key} up to date")
        return
    p = ctx.provider
    if klein(sb):
        params = {"prompt": prompt, "width": KLEIN_SIZE[0], "height": KLEIN_SIZE[1],
                  "reference_sha256": reference_sha}   # of the source file, as in the fingerprint
        result = call_model(ctx, stage="keyframes", shot=shot_label, provider="cloudflare", model=p.KLEIN_MODEL,
                            params=params, seed=seed, usage=cloudflare_usage,
                            fn=lambda: p.cloudflare_klein(prompt, *KLEIN_SIZE, seed, reference))
        data = to_jpeg(result.image)
    else:
        data = call_model(ctx, stage="keyframes", shot=shot_label, provider="cloudflare", model=p.IMAGE_MODEL,
                          params={"prompt": prompt, "steps": STEPS}, seed=None,
                          fn=lambda: to_jpeg(p.cloudflare_image(prompt, STEPS)))
    commit_bytes(path, data, validate=check_image)
    manifest.record(key, fp, path)
    print(f"keyframes: generated {key}")


def run(ctx, sb, args) -> int:
    source = reference_source(ctx, sb)
    if source is not None and not klein(sb):
        raise PipelineExit(1, f"keyframe_model {model_name(sb)} takes no reference image; remove "
                              f"character.reference_image or use flux-2-klein-4b")
    reference = reference_input(source) if source is not None else None
    reference_sha = hashlib.sha256(source).hexdigest() if source is not None else None
    manifest = Manifest(ctx)
    if reference is None:   # an approved reference image replaces the generated sheet
        _generate(ctx, sb, manifest, "reference", "reference", ctx.path("keyframes", "reference.jpg"),
                  reference_fp(ctx, sb), reference_prompt(sb), seed_of(sb))
    for shot in filter(ctx.wanted, sb["shots"]):   # (schnell ignores the seed)
        _generate(ctx, sb, manifest, f"keyframe:{shot['id']}", shot["id"], ctx.path("keyframes", f"{shot['id']}.jpg"),
                  keyframe_fp(ctx, sb, shot), composed_keyframe_prompt(sb, shot), shot_seed(sb, shot),
                  reference, reference_sha)
    return 0


def main(argv=None) -> None:
    stage_main(run, __doc__, argv=argv)


if __name__ == "__main__":
    main()
