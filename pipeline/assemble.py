"""Stage: assemble. Cut the film with local FFmpeg and export 9:16 and 16:9 from the same footage.

Order: mix the audio once (narration on the timeline, score ducked under it by sidechain compression,
two-pass loudness normalisation to -14 LUFS), then render each aspect ratio: clips cropped to fill, English
subtitles and the AI end card burned in as PNG overlays, audio muxed. Overlays are used because common
FFmpeg builds lack the `subtitles` and `drawtext` filters. Needs no credentials and makes no model calls.
--radio builds the radio cut instead: the same mix, subtitles and end card over a black picture (no footage), plus
final/radio_cut.srt."""
from __future__ import annotations

import functools
import json
import re
from pathlib import Path

from pipeline import music, narration, shots
from pipeline.common import (DEFAULT_END_CARD, END_CARD_SECONDS, FPS, Manifest, PipelineExit, assessment_note, check_assessment, commit_bytes, commit_with, ffmpeg, fingerprint,
                             run_tool, sha256_file, shot_frames, shot_seconds, shot_starts, stage_main, timed)

V = 1
ASPECTS = {"9x16": (720, 1280), "16x9": (1280, 720)}
FONTS = ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"]
FILTERS = ["sidechaincompress", "loudnorm", "adelay", "amix", "overlay", "zoompan"]
ENCODERS = ["libx264", "aac"]
BITEXACT = ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact", "-map_metadata", "-1"]
DUCK = "sidechaincompress=threshold=0.02:ratio=10:attack=20:release=500"
LOUDNORM = "loudnorm=I=-14:TP=-1:LRA=11"


@functools.lru_cache(maxsize=None)
def available(kind: str) -> frozenset:
    """Names of FFmpeg filters or encoders on this machine."""
    out = run_tool(["ffmpeg", "-hide_banner", f"-{kind}"], capture_output=True, text=True).stdout
    return frozenset(m.group(1) for m in re.finditer(r"^\s[A-Z.]{2,6}\s+(\S+)", out, re.M))


@functools.lru_cache(maxsize=None)
def ffmpeg_version() -> str:
    """First line of `ffmpeg -version`, used so a different FFmpeg rebuilds the outputs."""
    return run_tool(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]


def require_tools() -> None:
    for kind, names in (("filters", FILTERS), ("encoders", ENCODERS)):
        have = available(kind)
        missing = [n for n in names if n not in have]
        if missing:
            raise PipelineExit(1, f"this FFmpeg lacks {', '.join(missing)}; install a full build (brew install ffmpeg)")


def find_font() -> Path:
    for font in FONTS:
        if Path(font).exists():
            return Path(font)
    raise PipelineExit(1, "no TrueType font found; install DejaVu Sans or edit FONTS in pipeline/assemble.py")


def render_text_png(path: Path, size: tuple, text: str, y_frac: float, scale: float, box: bool) -> None:
    from PIL import Image, ImageDraw, ImageFont

    w, h = size
    px = int(min(w, h) * 0.05 * scale)
    font = ImageFont.truetype(str(find_font()), px)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) > w * 0.8 and line:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    line_h = int(px * 1.3)
    top = int(h * y_frac) - len(lines) * line_h // 2
    for i, text_line in enumerate(lines):
        width = draw.textlength(text_line, font=font)
        x, y = (w - width) / 2, top + i * line_h
        if box:
            draw.rounded_rectangle((x - 14, y - 4, x + width + 14, y + line_h - 2), 10, fill=(0, 0, 0, 170))
        draw.text((x, y), text_line, font=font, fill=(255, 255, 255, 255))
    img.save(path)


# ------------------------------------------------------------------ prerequisites

def check_inputs(ctx, sb: dict, draft: bool):
    """Exit 1 naming the stage to run if any input is missing or stale. Returns the score path or None."""
    manifest, problems = Manifest(ctx), []
    stale_clips = [s["id"] for s in sb["shots"] if shots.clip_state(ctx, s["id"], shots.clip_fp(ctx, sb, s, manifest)) is None]
    if stale_clips:
        problems.append(f"clips for {', '.join(stale_clips)} are missing or stale: python -m pipeline.shots")
    stale_voice = stale_narration(ctx, sb, manifest)
    if stale_voice:
        problems.append(f"narration for {', '.join(stale_voice)} is missing or stale: python -m pipeline.narration")
    elif not subtitles_fresh(ctx, sb, manifest):
        problems.append("subtitles.srt is missing or stale: python -m pipeline.narration")
    score = ctx.out / "music" / "score.wav"
    if not music.score_ready(ctx, sb, manifest):
        if not draft:
            problems.append("no current score: python -m pipeline.music (and set music.choice)")
        else:
            print("assemble: --draft: exporting without a score (none is current; see python -m pipeline.music)")
        score = None
    if problems:
        raise PipelineExit(1, "cannot assemble:\n  " + "\n  ".join(problems))
    return score


def stale_narration(ctx, sb: dict, manifest: Manifest) -> list:
    return [s["id"] for s in sb["shots"]
            if not manifest.fresh(f"narration:{s['id']}", narration.narration_fp(ctx, sb, s),
                                  ctx.out / "narration" / f"{s['id']}.wav")]


def narration_durations(ctx, sb: dict) -> dict:
    return {s["id"]: narration.wav_seconds(ctx.out / "narration" / f"{s['id']}.wav") for s in sb["shots"]}


def subtitles_fresh(ctx, sb: dict, manifest: Manifest) -> bool:
    return manifest.fresh("subtitles", narration.subtitles_fp(sb, narration_durations(ctx, sb)),
                          ctx.out / "subtitles.srt")


def check_narration_fits(ctx, sb: dict, draft: bool) -> None:
    """A clip longer than its shot would run into the next shot's voice (which starts 0.2 s into that shot)."""
    spoken = narration_durations(ctx, sb)
    long = [s["id"] for s in sb["shots"] if spoken[s["id"]] > shot_seconds(s)]
    if long and not draft:
        raise PipelineExit(1, f"narration is longer than its shot for {', '.join(long)}; shorten the line "
                              f"or lengthen the shot")
    if long:
        print(f"assemble: warning: narration overruns its shot for {', '.join(long)}")


def fallback_shots(ctx, sb: dict, manifest: Manifest) -> list:
    return [s["id"] for s in sb["shots"]
            if shots.clip_state(ctx, s["id"], shots.clip_fp(ctx, sb, s, manifest)) == "fallback"]


# ------------------------------------------------------------------------ audio

def mix_graph(sb: dict, score, total: float) -> str:
    """The filter graph (ending in the label [pre]): inputs are the narration wavs in shot order, then the score."""
    parts, starts = [], shot_starts(sb)
    for i, shot in enumerate(sb["shots"]):
        parts.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,"
                     f"adelay={round((starts[shot['id']] + narration.SUBTITLE_LEAD) * 1000)}:all=1[a{i}]")
    n = len(sb["shots"])
    labels = "".join(f"[a{i}]" for i in range(n))
    parts.append(f"{labels}amix=inputs={n}:normalize=0,apad=whole_dur={total}[voice]")
    if score:
        parts += ["[voice]asplit=2[vsc][vmix]",
                  f"[{n}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=duration={total},"
                  f"afade=t=out:st={total - 2}:d=2,volume=0.6[mus]",
                  f"[mus][vsc]{DUCK}[duck]", "[vmix][duck]amix=inputs=2:normalize=0[pre]"]
    else:
        parts.append("[voice]anull[pre]")
    return ";".join(parts)


def run_mix(ctx, sb: dict, score, total: float, dst: Path) -> None:
    graph = mix_graph(sb, score, total)
    inputs = []
    for shot in sb["shots"]:
        inputs += ["-i", ctx.out / "narration" / f"{shot['id']}.wav"]
    if score:
        inputs += ["-stream_loop", -1, "-i", score]
    first = ffmpeg(*inputs, "-filter_complex", f"{graph};[pre]{LOUDNORM}:print_format=json[m]", "-map", "[m]",
                   "-t", total, "-f", "null", "-", loglevel="info")
    m = json.loads(re.findall(r"\{[^{}]*\}", first)[-1])
    measured = (f"measured_I={m['input_i']}:measured_LRA={m['input_lra']}:measured_TP={m['input_tp']}:"
                f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    commit_with(dst, lambda tmp: ffmpeg(
        *inputs, "-filter_complex", f"{graph};[pre]{LOUDNORM}:{measured}[mix]", "-map", "[mix]", "-t", total,
        "-ar", 48000, "-ac", 2, *BITEXACT, "-f", "wav", tmp))


# ------------------------------------------------------------------------ video

def fill_crop(shot: dict, w: int, h: int) -> str:
    """Scale a clip to cover w x h, then crop it. portrait_crop_x places the crop where the clip is wider than the
    frame (the 9:16 export): 0 its left edge, 0.5 the centre, 1 its right edge. A 16:9 clip in a 16:9 frame has no
    horizontal slack, so the setting changes nothing there."""
    x = shot.get("portrait_crop_x", 0.5)
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}:(iw-ow)*{x}:(ih-oh)/2"


def render_video(ctx, sb: dict, size: tuple, mix: Path, cues: list, total: float, dst: Path,
                 footage: bool = True) -> None:
    """footage=False (the radio cut): a black picture for the whole length instead of the shots."""
    w, h = size
    overlay_dir = ctx.path("jobs", "assemble", f"overlays-{w}x{h}", ".keep").parent
    args, parts, labels = [], [], []
    for i, shot in enumerate(sb["shots"] if footage else []):
        args += ["-i", ctx.out / "shots" / f"{shot['id']}.mp4"]
        parts.append(f"[{i}:v]{fill_crop(shot, w, h)},setsar=1,fps={FPS},"
                     f"tpad=stop_mode=clone:stop_duration={shot['duration']},trim=end_frame={shot_frames(shot)},"
                     f"setpts=PTS-STARTPTS[v{i}]")
        labels.append(f"[v{i}]")
    n = len(labels)
    args += ["-f", "lavfi", "-t", END_CARD_SECONDS if footage else total, "-i", f"color=c=black:s={w}x{h}:r={FPS}"]
    parts.append(f"[{n}:v]setsar=1,fps={FPS}[vend]")
    parts.append(f"{''.join(labels)}[vend]concat=n={n + 1}:v=1:a=0[base]")
    cards = [(a, b, text, 0.80 if h > w else 0.88, 1.0, True) for a, b, text in cues]
    cards.append((total - END_CARD_SECONDS, total, sb.get("end_card", DEFAULT_END_CARD), 0.5, 1.6, False))
    last = "base"
    for j, (a, b, text, y, scale, box) in enumerate(cards):
        png = overlay_dir / f"o{j}.png"
        render_text_png(png, size, text, y, scale, box)
        args += ["-loop", 1, "-t", total, "-i", png]
        parts.append(f"[{last}][{n + 1 + j}:v]overlay=0:0:enable='between(t,{a:.3f},{b:.3f})'[o{j}]")
        last = f"o{j}"
    args += ["-i", mix]
    parts.append(f"[{last}]format=yuv420p[vout]")
    commit_with(dst, lambda tmp: ffmpeg(
        *args, "-filter_complex", ";".join(parts), "-map", "[vout]", "-map", f"{n + 1 + len(cards)}:a",
        "-c:v", "libx264", "-crf", 20, "-preset", "medium", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        "-t", total, *BITEXACT, "-f", "mp4", tmp))


# ------------------------------------------------------------------------- stage

def run_radio(ctx, sb: dict, manifest: Manifest) -> int:
    stale = stale_narration(ctx, sb, manifest)
    if stale:
        raise PipelineExit(1, f"narration for {', '.join(stale)} is missing or stale: python -m pipeline.narration")
    srt = ctx.out / "subtitles.srt"
    if not srt.exists():
        raise PipelineExit(1, "subtitles.srt is missing: python -m pipeline.narration")
    total = sum(shot_seconds(s) for s in sb["shots"]) + END_CARD_SECONDS
    score = ctx.out / "music" / "score.wav" if music.score_ready(ctx, sb, manifest) else None
    mix, dst, srt_copy = ctx.path("final", "radio_mix.wav"), ctx.path("final", "radio_cut.mp4"), ctx.path("final", "radio_cut.srt")
    fp = fingerprint(V, "radio", [(s["id"], s["duration"], manifest.sha(f"narration:{s['id']}")) for s in sb["shots"]],
                     sha256_file(score) if score else None, DUCK, LOUDNORM, narration.SUBTITLE_LEAD, sha256_file(srt),
                     sb.get("end_card", DEFAULT_END_CARD), sha256_file(find_font()), ffmpeg_version())
    print("assemble: radio cut " + ("with the chosen score" if score else "narration only (no chosen score yet)"))
    if manifest.fresh("final:radio", fp, dst) and srt_copy.exists() and srt_copy.read_bytes() == srt.read_bytes():
        print("assemble: radio cut up to date")
        return 0
    cues = narration.cues(sb, narration_durations(ctx, sb))
    timed(ctx, "assemble", "radio", "ffmpeg", lambda: run_mix(ctx, sb, score, total, mix))
    timed(ctx, "assemble", "radio", "ffmpeg",
          lambda: render_video(ctx, sb, ASPECTS["16x9"], mix, cues, total, dst, footage=False))
    commit_bytes(srt_copy, srt.read_bytes())
    manifest.record("final:radio", fp, dst)
    print(f"assemble: wrote {dst} and {srt_copy.name}")
    return 0


def run(ctx, sb, args) -> int:
    require_tools()
    manifest = Manifest(ctx)
    if args.radio:
        return run_radio(ctx, sb, manifest)
    print(assessment_note(sb))
    check_assessment(sb)
    score = check_inputs(ctx, sb, args.draft)
    check_narration_fits(ctx, sb, args.draft)
    total = sum(shot_seconds(s) for s in sb["shots"]) + END_CARD_SECONDS
    font = sha256_file(find_font())
    clip_shas = [sha256_file(ctx.out / "shots" / f"{s['id']}.mp4") for s in sb["shots"]]
    durations = [(s["id"], s["duration"]) for s in sb["shots"]]
    srt = ctx.out / "subtitles.srt"
    voices = [manifest.sha(f"narration:{s['id']}") for s in sb["shots"]]
    mix, mix_key = ctx.path("final", "mix.wav"), "final:mix"
    mix_fp = fingerprint(V, "mix", voices, durations, sha256_file(score) if score else None, DUCK, LOUDNORM, narration.SUBTITLE_LEAD,
                         ffmpeg_version())
    if not manifest.fresh(mix_key, mix_fp, mix):
        timed(ctx, "assemble", "mix", "ffmpeg", lambda: run_mix(ctx, sb, score, total, mix))
        manifest.record(mix_key, mix_fp, mix)
    cues = narration.cues(sb, narration_durations(ctx, sb))
    for name, size in ASPECTS.items():
        dst = ctx.path("final", f"film_{name}.mp4")
        crops = [(s["id"], s.get("portrait_crop_x", 0.5)) for s in sb["shots"]] if size[0] < size[1] else None
        fp = fingerprint(V, "export", name, clip_shas, sha256_file(mix), sha256_file(srt), sb.get("end_card", DEFAULT_END_CARD),
                         durations, font, ffmpeg_version(), crops)
        if manifest.fresh(f"final:{name}", fp, dst):
            print(f"assemble: {dst.name} up to date")
            continue
        timed(ctx, "assemble", name, "ffmpeg", lambda: render_video(ctx, sb, size, mix, cues, total, dst))
        manifest.record(f"final:{name}", fp, dst)
        print(f"assemble: wrote {dst}")
    if fallbacks := fallback_shots(ctx, sb, manifest):
        print(f"assemble: FALLBACK clips in the exports (pan-and-zoom, not Wan): {', '.join(fallbacks)}")
    return 0


def main(argv=None) -> None:
    def extra(p):
        p.add_argument("--draft", action="store_true",
                       help="export without a chosen score; warn on narration overrun (never skips the assessment rules)")
        p.add_argument("--radio", action="store_true",
                       help="radio cut: narration, the chosen score (if any), burned-in subtitles and the end card "
                            "over a black picture, plus an .srt")
    stage_main(run, __doc__, only_mode="error", extra_args=extra, argv=argv)


if __name__ == "__main__":
    main()
