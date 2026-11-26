"""Stage: shots. Wan 2.2 TI2V-5B image-to-video through Diffusers on Kaggle's two T4s (kernels/shots_kernel.py, the
setup measured in EXPERIMENTS.md E6).

Only shots whose clip is not fresh are submitted. Each shot asks Wan for just enough frames to cover it, up to 121
(a longer shot holds the clip's last frame). --fallback fills shots Wan did not deliver with an FFmpeg pan-and-zoom
of their keyframe; --fallback-only makes those clips with no Kaggle call at all. A fallback clip is usable and
announced on every run, but it is not submitted again unless you pass --retry-fallbacks, so an unchanged re-run
costs no GPU time."""
from __future__ import annotations

import json
import math

from pipeline import kaggle, keyframes
from pipeline.common import (FPS, Manifest, PipelineExit, ProviderError, commit_bytes, commit_file, commit_with, ffmpeg,
                             fingerprint, log_run, model_record, probe, sha256_file, shot_seed,
                             stage_main, timed)

V = 2
MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
MAX_FRAMES, SIZE = 121, "832x480"   # E6: 121 frames is the most measured on a T4
NORMALISE_V = 1


def wan_frames(seconds: float) -> int:
    """The fewest Wan frames (4k+1, at least 5) covering the shot, capped at MAX_FRAMES."""
    needed = max(1, round(seconds * FPS))
    return min(MAX_FRAMES, max(5, 4 * math.ceil((needed - 1) / 4) + 1))


def clip_fp(ctx, sb: dict, shot: dict, manifest: Manifest) -> str:
    key = f"keyframe:{shot['id']}"
    if not manifest.fresh(key, keyframes.keyframe_fp(ctx, sb, shot), ctx.out / "keyframes" / f"{shot['id']}.jpg"):
        raise PipelineExit(1, f"keyframe for {shot['id']} is missing or stale; run: python -m pipeline.keyframes")
    return fingerprint(V, "shot", MODEL, shot_seed(sb, shot), wan_frames(shot["duration"]), SIZE,
                       shot["motion_prompt"], NORMALISE_V, manifest.sha(key))


def clip_state(ctx, shot_id: str, fp: str):
    """The kind ('real'|'mock'|'fallback') of the clip on disk for this fingerprint, else None."""
    clip, side = ctx.out / "shots" / f"{shot_id}.mp4", ctx.out / "shots" / f"{shot_id}.json"
    if not (clip.exists() and side.exists()):
        return None
    info = json.loads(side.read_text())
    if info.get("fingerprint") != fp or info.get("sha256") != sha256_file(clip):
        return None
    kind = info.get("kind")
    return kind if kind in ("real", "fallback") or (kind == "mock" and ctx.mock) else None


def check_clip(expected: float, tolerance: float):
    def check(path):
        info = probe(path)
        if not any(s["codec_type"] == "video" for s in info["streams"]):
            raise ProviderError("clip has no video stream")
        if abs(float(info["format"]["duration"]) - expected) > tolerance:
            raise ProviderError(f"clip is {info['format']['duration']}s, expected about {expected:.2f}s")
    return check


def ken_burns(keyframe, dst, seconds: float) -> None:
    frames = round(seconds * FPS)
    commit_with(dst, lambda tmp: ffmpeg(
        "-i", keyframe, "-an", "-frames:v", frames, "-f", "mp4", "-vf",
        f"scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,"
        f"zoompan=z='min(zoom+0.0007,1.2)':d={frames}:s=1280x720:fps={FPS},format=yuv420p", tmp),
        check_clip(seconds, 0.3))


def write_sidecar(ctx, shot_id: str, **info) -> None:
    commit_bytes(ctx.path("shots", f"{shot_id}.json"), json.dumps(info, indent=2).encode())


def make_local(ctx, sb, shot, fp, kind, model) -> None:
    clip = ctx.path("shots", f"{shot['id']}.mp4")
    timed(ctx, "shots", shot["id"], "fallback" if kind == "fallback" else "ffmpeg",
          lambda: ken_burns(ctx.out / "keyframes" / f"{shot['id']}.jpg", clip, shot["duration"]))
    write_sidecar(ctx, shot["id"], fingerprint=fp, kind=kind, sha256=sha256_file(clip), model=model,
                  seed=shot_seed(sb, shot), run_ids_logged=[])


def build_job(ctx, sb, todo: list, fps: dict):
    items, inputs = [], {}
    for shot in todo:
        src = ctx.out / "keyframes" / f"{shot['id']}.jpg"
        dst = ctx.path("jobs", "shots", "inputs", f"{shot['id']}.jpg")
        timed(ctx, "shots", shot["id"], "normalise", lambda: commit_with(dst, lambda tmp: ffmpeg(
            "-i", src, "-vf", "scale=1280:704:force_original_aspect_ratio=increase,crop=1280:704",
            "-frames:v", 1, "-q:v", 2, "-f", "image2", tmp)))
        inputs[dst.name] = dst
        items.append({"id": shot["id"], "fingerprint": fps[shot["id"]],
                      "params": {"prompt": shot["motion_prompt"], "seed": shot_seed(sb, shot),
                                 "num_frames": wan_frames(shot["duration"])},
                      "inputs": [dst.name]})
    return items, inputs


def ingest(ctx, res, expected: dict) -> list:
    """Commit accepted clips. Idempotent: the kernel's record is logged once per (run, shot)."""
    done = []
    for item, path in res.accepted:
        shot_id = item["id"]
        clip, side = ctx.path("shots", f"{shot_id}.mp4"), ctx.path("shots", f"{shot_id}.json")
        try:
            commit_file(path, clip, check_clip(item["params"]["num_frames"] / FPS, 0.5))
        except ProviderError as exc:
            res.rejected[shot_id] = str(exc)
            continue
        old = json.loads(side.read_text()) if side.exists() else {}
        logged = old.get("run_ids_logged", []) if old.get("fingerprint") == expected[shot_id] else []
        if res.run_id not in logged:
            log_run(ctx, **model_record(stage="shots", shot=shot_id, provider="kaggle", model=item["model"],
                                        params=item["params"], seed=item["seed"], latency_s=item["latency_s"],
                                        gpu_seconds=item["gpu_seconds"], run_id=res.run_id))
            logged = logged + [res.run_id]
        write_sidecar(ctx, shot_id, fingerprint=expected[shot_id], kind="real", sha256=sha256_file(clip),
                      bytes=clip.stat().st_size, model=item["model"], seed=item["seed"], run_ids_logged=logged)
        done.append(shot_id)
    return done


def unfinished_run_shots(ctx, sb, manifest, skip: set, resubmit: bool) -> tuple:
    """Shots that are part of a Kaggle run still pending, with their current fingerprints. Their results must be
    judged too, whatever --only or the fallback policy says, or paid-for clips would be thrown away."""
    path = ctx.out / "jobs" / "shots" / "pending.json"
    ids = set(json.loads(path.read_text())["item_ids"]) if path.exists() and not resubmit else set()
    extra, fps = [], {}
    for shot in sb["shots"]:
        if shot["id"] in ids and shot["id"] not in skip:
            try:
                fps[shot["id"]] = clip_fp(ctx, sb, shot, manifest)
            except PipelineExit:
                continue   # its keyframe is stale: any result for it would be stale too
            extra.append(shot)
    return extra, fps


def fallback_only(ctx, sb, scope: list, fps: dict) -> None:
    """Pan-and-zoom clips with no Kaggle call. A fresh Wan clip is replaced only for a shot named with --only."""
    for shot in scope:
        state = clip_state(ctx, shot["id"], fps[shot["id"]])
        if state == "fallback" or (state in ("real", "mock") and shot["id"] not in ctx.only):
            continue
        if state in ("real", "mock"):
            print(f"shots: replacing the Wan clip for {shot['id']} with a fallback clip (--fallback-only --only)")
        make_local(ctx, sb, shot, fps[shot["id"]], "fallback", "ffmpeg-zoompan")


def run(ctx, sb, args) -> int:
    manifest = Manifest(ctx)
    scope = list(filter(ctx.wanted, sb["shots"]))
    fps = {s["id"]: clip_fp(ctx, sb, s, manifest) for s in scope}
    only_fallback = getattr(args, "fallback_only", False)
    if only_fallback and (args.fallback or args.retry_fallbacks or args.resubmit):
        raise PipelineExit(1, "--fallback-only makes no Kaggle call; it cannot be combined with --fallback, "
                              "--retry-fallbacks or --resubmit")
    fresh_kind = "mock" if ctx.mock else "real"
    todo = [s for s in scope
            if (state := clip_state(ctx, s["id"], fps[s["id"]])) != fresh_kind
            and not (state == "fallback" and not args.retry_fallbacks)]
    if only_fallback:
        fallback_only(ctx, sb, scope, fps)
    elif todo and ctx.mock:
        for shot in todo:
            make_local(ctx, sb, shot, fps[shot["id"]], "mock", "mock-wan")
            log_run(ctx, **model_record(stage="shots", shot=shot["id"], provider="kaggle", model="mock-wan",
                                        params={"prompt": shot["motion_prompt"]}, seed=shot_seed(sb, shot)))
    elif todo:
        resumed, resumed_fps = unfinished_run_shots(ctx, sb, manifest, {s["id"] for s in todo}, args.resubmit)
        items, inputs = build_job(ctx, sb, todo + resumed, {**fps, **resumed_fps})
        try:
            res = kaggle.collect(ctx, "shots", items, inputs, resubmit=args.resubmit)
        except PipelineExit as exc:
            if not args.fallback:
                raise
            print(f"shots: Kaggle did not deliver ({exc}); falling back where needed")
        else:
            ingested = ingest(ctx, res, {i["id"]: i["fingerprint"] for i in items})
            for shot_id, reason in res.rejected.items():
                print(f"shots: rejected {shot_id}: {reason}")
            if res.done:
                kaggle.log_failures(ctx, "shots", items, ingested, res, MODEL)
                kaggle.finish(ctx, "shots")
                left_out = [s["id"] for s in todo if s["id"] not in res.submitted]
                if left_out:
                    print(f"shots: {', '.join(left_out)} were not part of that Kaggle run; run again to submit them")
        if args.fallback:
            for shot in todo:
                if clip_state(ctx, shot["id"], fps[shot["id"]]) is None:
                    make_local(ctx, sb, shot, fps[shot["id"]], "fallback", "ffmpeg-zoompan")
    elif not ctx.mock and not args.resubmit and (ctx.out / "jobs" / "shots" / "pending.json").exists():
        print("shots: a Kaggle run is pending (outputs/jobs/shots/pending.json); --retry-fallbacks collects it")
    fallbacks = [s["id"] for s in scope if clip_state(ctx, s["id"], fps[s["id"]]) == "fallback"]
    if fallbacks:
        print(f"shots: FALLBACK clips in use (pan-and-zoom, not Wan) for {', '.join(fallbacks)}; "
              f"re-run with --retry-fallbacks to try Wan again")
    unusable = [s["id"] for s in scope if clip_state(ctx, s["id"], fps[s["id"]]) is None]
    if unusable:
        print(f"shots: no usable clip for {', '.join(unusable)}; re-run to resume, or use --fallback")
        return 2
    print(f"shots: {len(scope)} clip(s) ready")
    return 0


def main(argv=None) -> None:
    def extra(p):
        p.add_argument("--fallback", action="store_true", help="pan-and-zoom clips for shots Wan could not make")
        p.add_argument("--retry-fallbacks", action="store_true", help="try Wan again for shots that have a fallback clip")
        p.add_argument("--resubmit", action="store_true", help="discard a pending Kaggle run and submit again")
        p.add_argument("--fallback-only", action="store_true",
                       help="pan-and-zoom clips with no Kaggle call; replaces a Wan clip only for shots named "
                            "with --only")
    stage_main(run, __doc__, extra_args=extra, argv=argv)


if __name__ == "__main__":
    main()
