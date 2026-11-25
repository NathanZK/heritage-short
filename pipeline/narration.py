"""Stage: narration. One Gemini TTS clip per shot, plus the English subtitle file."""
from __future__ import annotations

import wave

from pipeline.common import (Manifest, call_model, check_wav, commit_bytes, fingerprint, shot_seconds, shot_starts, stage_main)

V = 1
SUBTITLE_LEAD = 0.2  # seconds after the shot starts before the line is spoken


def wav_seconds(path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def srt_time(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def narration_fp(ctx, sb: dict, shot: dict) -> str:
    return fingerprint(V, "narration", ctx.provider.TTS_MODEL, sb["voice"], shot["narration"])


def subtitles_fp(sb: dict, durations: dict) -> str:
    return fingerprint(V, "subtitles", [(s["id"], s["duration"], s["subtitle"]) for s in sb["shots"]],
                       [round(d, 4) for d in durations.values()])


def cues(sb: dict, durations: dict) -> list:
    """[(start_s, end_s, text)]: each line starts just after its shot starts and lasts as long as it is spoken."""
    starts, out = shot_starts(sb), []
    for shot in sb["shots"]:
        begin = starts[shot["id"]] + SUBTITLE_LEAD
        end = min(begin + durations[shot["id"]], starts[shot["id"]] + shot_seconds(shot))
        out.append((begin, end, " ".join(shot["subtitle"].split())))
    return out


def build_srt(sb: dict, durations: dict) -> str:
    return "\n".join(f"{n}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n"
                     for n, (a, b, text) in enumerate(cues(sb, durations), 1))


def run(ctx, sb, args) -> int:
    manifest, p = Manifest(ctx), ctx.provider
    for shot in filter(ctx.wanted, sb["shots"]):
        path, fp, key = ctx.path("narration", f"{shot['id']}.wav"), narration_fp(ctx, sb, shot), f"narration:{shot['id']}"
        if manifest.fresh(key, fp, path):
            print(f"narration: {shot['id']} up to date")
            continue
        audio = call_model(ctx, stage="narration", shot=shot["id"], provider="gemini", model=p.TTS_MODEL,
                           params={"voice": sb["voice"], "text": shot["narration"]}, seed=None,
                           fn=lambda: p.gemini_tts(shot["narration"], sb["voice"]))
        commit_bytes(path, audio, validate=check_wav)
        manifest.record(key, fp, path)
        print(f"narration: generated {shot['id']}")
    wavs = {s["id"]: ctx.out / "narration" / f"{s['id']}.wav" for s in sb["shots"]}
    if not all(w.exists() for w in wavs.values()):
        print("narration: subtitles.srt not written (not every shot has narration yet)")
        return 0
    durations = {i: wav_seconds(w) for i, w in wavs.items()}
    srt_path = ctx.path("subtitles.srt")
    fp = subtitles_fp(sb, durations)
    if not manifest.fresh("subtitles", fp, srt_path):
        commit_bytes(srt_path, build_srt(sb, durations).encode())
        manifest.record("subtitles", fp, srt_path)
    return 0


def main(argv=None) -> None:
    stage_main(run, __doc__, argv=argv)


if __name__ == "__main__":
    main()
