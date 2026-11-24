import json
import re
import subprocess
from types import SimpleNamespace

import pytest
from PIL import Image

from pipeline import assemble, keyframes, music, narration, shots
from pipeline.common import Ctx, PipelineExit, StoryboardError, media_seconds, probe, read_runs
from tests.conftest import FIXTURES

NS = SimpleNamespace


def build_all(ctx, sb, with_music=True):
    keyframes.run(ctx, sb, NS())
    narration.run(ctx, sb, NS())
    shots.run(ctx, sb, NS(fallback=False, resubmit=False))
    if with_music:
        music.run(ctx, sb, NS(draft=False, resubmit=False))


@pytest.fixture(scope="module")
def film(tmp_path_factory):
    root = tmp_path_factory.mktemp("film")
    sb = json.loads((FIXTURES / "storyboard_3shot.json").read_text())
    (root / "storyboard.json").write_text(json.dumps(sb))
    ctx = Ctx(root=root, mock=True, sleep=lambda s: None)
    build_all(ctx, sb)
    assemble.run(ctx, sb, NS(draft=True, radio=False))
    return ctx, sb


def streams(path):
    info = probe(path)
    return {s["codec_type"]: s for s in info["streams"]}, float(info["format"]["duration"])


@pytest.mark.parametrize("name,size", [("9x16", (720, 1280)), ("16x9", (1280, 720))])
def test_both_exports_have_the_right_shape(film, name, size):
    ctx, sb = film
    by_type, seconds = streams(ctx.out / "final" / f"film_{name}.mp4")
    video, audio = by_type["video"], by_type["audio"]
    assert (video["width"], video["height"]) == size and video["codec_name"] == "h264"
    assert video["avg_frame_rate"] == "24/1" and video["pix_fmt"] == "yuv420p"
    assert audio["codec_name"] == "aac" and audio["sample_rate"] == "48000"
    assert seconds == pytest.approx(sum(s["duration"] for s in sb["shots"]) + 3.0, abs=0.2)


def frame(path, t, size):
    out = path.with_suffix(f".{t}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                    "-vf", f"scale={size[0]}:{size[1]}", str(out)], check=True)
    return Image.open(out).convert("RGB")


def mean_abs_diff(a, b):
    return sum(abs(x - y) for pa, pb in zip(a.getdata(), b.getdata()) for x, y in zip(pa, pb)) / (len(a.getdata()) * 3)


def test_end_card_is_the_last_frame_with_the_text(film, tmp_path):
    ctx, sb = film
    export = ctx.out / "final" / "film_16x9.mp4"
    last = frame(export, 15.5, (1280, 720))
    overlay = tmp_path / "o.png"
    assemble.render_text_png(overlay, (1280, 720), sb["end_card"], 0.5, 1.6, False)
    expected = Image.alpha_composite(Image.new("RGBA", (1280, 720), "black"), Image.open(overlay)).convert("RGB")
    assert mean_abs_diff(last, expected) < 3
    assert mean_abs_diff(last, Image.new("RGB", (1280, 720), "black")) > mean_abs_diff(last, expected)


def white_pixels_in_subtitle_band(img):
    band = img.crop((0, int(img.height * 0.80), img.width, int(img.height * 0.95))).convert("L")
    return sum(1 for v in band.getdata() if v > 235)


def test_subtitles_are_burned_in_only_while_spoken(film):
    ctx, sb = film
    export = ctx.out / "final" / "film_16x9.mp4"
    speaking = white_pixels_in_subtitle_band(frame(export, 1.0, (1280, 720)))
    silent = white_pixels_in_subtitle_band(frame(export, 3.6, (1280, 720)))
    assert speaking > 300 and speaking > 5 * max(silent, 1)


def integrated_lufs(path):
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                          capture_output=True, text=True)
    return float(re.findall(r"I:\s+(-?\d+\.\d+) LUFS", proc.stderr)[-1])


def test_loudness_is_normalised_to_minus_14(film):
    ctx, _ = film
    assert integrated_lufs(ctx.out / "final" / "film_16x9.mp4") == pytest.approx(-14.0, abs=1.5)


def band_db(path, start, seconds, hz):
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", str(start), "-t", str(seconds), "-i",
                           str(path), "-af", f"bandpass=f={hz}:width_type=h:w=15,astats=metadata=0:reset=0",
                           "-f", "null", "-"], capture_output=True, text=True)
    return float(re.findall(r"RMS level dB: (-?\d+\.\d+)", proc.stderr)[-1])


def test_music_is_ducked_under_the_narration(film):
    ctx, _ = film
    mix = ctx.out / "final" / "mix.wav"
    # mock score is a 110 Hz tone; s01's narration spans about 0.2-2.7 s, then only music until 4.5 s
    while_talking, alone = band_db(mix, 0.6, 1.8, 110), band_db(mix, 3.2, 1.2, 110)
    assert alone - while_talking > 6


def test_assembly_logs_no_model_calls_and_writes_timings(film):
    ctx, _ = film
    assert all(r["stage"] != "assemble" for r in read_runs(ctx.runs_path))
    assert "ffmpeg" in (ctx.out / "timings.jsonl").read_text()


def test_unchanged_assembly_does_nothing(film, monkeypatch):
    ctx, sb = film
    monkeypatch.setattr(assemble, "ffmpeg", lambda *a, **k: pytest.fail("ffmpeg ran"))
    assemble.run(ctx, sb, NS(draft=True, radio=False))


def test_ffmpeg_version_survives_nightly_build_strings(monkeypatch):
    out = type("R", (), {"stdout": "ffmpeg version N-118000-gabcdef1234 Copyright (c) 2000-2026\nbuilt with clang"})()
    monkeypatch.setattr(assemble, "run_tool", lambda *a, **k: out)
    assert assemble.ffmpeg_version.__wrapped__().startswith("ffmpeg version N-118000")


def test_film_length_rule_applies_without_draft(film):
    ctx, sb = film
    with pytest.raises(StoryboardError):
        assemble.run(ctx, sb, NS(draft=False, radio=False))


def fresh_ctx(tmp_path, sb=None, with_music=True):
    sb = sb or json.loads((FIXTURES / "storyboard_3shot.json").read_text())
    (tmp_path / "storyboard.json").write_text(json.dumps(sb))
    ctx = Ctx(root=tmp_path, mock=True, sleep=lambda s: None)
    build_all(ctx, sb, with_music)
    return ctx, sb


def test_missing_or_stale_inputs_exit_1_naming_the_stage(tmp_path):
    ctx, sb = fresh_ctx(tmp_path)
    (ctx.out / "shots" / "s02.mp4").unlink()
    with pytest.raises(PipelineExit) as exc:
        assemble.run(ctx, sb, NS(draft=True, radio=False))
    assert exc.value.code == 1 and "pipeline.shots" in str(exc.value) and "s02" in str(exc.value)


def test_stale_narration_is_refused(tmp_path):
    ctx, sb = fresh_ctx(tmp_path)
    sb["shots"][0]["narration"] = "A different line entirely."
    with pytest.raises(PipelineExit) as exc:
        assemble.run(ctx, sb, NS(draft=True, radio=False))
    assert "pipeline.narration" in str(exc.value)


def test_a_missing_score_is_refused_unless_draft(tmp_path):
    ctx, sb = fresh_ctx(tmp_path, with_music=False)
    with pytest.raises(PipelineExit) as exc:
        assemble.check_inputs(ctx, sb, draft=False)
    assert exc.value.code == 1 and "pipeline.music" in str(exc.value)
    assert assemble.check_inputs(ctx, sb, draft=True) is None


def test_draft_without_a_score_still_exports(tmp_path):
    ctx, sb = fresh_ctx(tmp_path, with_music=False)
    assemble.run(ctx, sb, NS(draft=True, radio=False))
    assert (ctx.out / "final" / "film_9x16.mp4").exists()


def test_long_narration_exits_1_unless_draft(tmp_path):
    ctx, sb = fresh_ctx(tmp_path)
    sb["shots"][0]["duration"] = 1.0
    narration.run(ctx, sb, NS())
    with pytest.raises(PipelineExit) as exc:
        assemble.check_narration_fits(ctx, sb, draft=False)
    assert exc.value.code == 1 and "s01" in str(exc.value)
    assemble.check_narration_fits(ctx, sb, draft=True)


@pytest.mark.parametrize("spoken,too_long", [(4.51, True), (4.49, False)])
def test_narration_may_not_outlast_its_shot_so_voices_never_overlap(storyboard, monkeypatch, spoken, too_long):
    # s01 lasts 4.5 s; the next voice starts 0.2 s into s02, so any clip longer than its shot would overlap it
    monkeypatch.setattr(assemble, "narration_durations", lambda ctx, sb: {"s01": spoken, "s02": 1.0, "s03": 1.0})
    if too_long:
        with pytest.raises(PipelineExit) as exc:
            assemble.check_narration_fits(None, storyboard, draft=False)
        assert exc.value.code == 1 and "s01" in str(exc.value)
    assemble.check_narration_fits(None, storyboard, draft=too_long)


def test_missing_ffmpeg_filter_is_named(tmp_path, monkeypatch):
    ctx, sb = fresh_ctx(tmp_path)
    monkeypatch.setattr(assemble, "available", lambda kind: {"overlay"})
    with pytest.raises(PipelineExit) as exc:
        assemble.run(ctx, sb, NS(draft=True, radio=False))
    assert exc.value.code == 1 and "sidechaincompress" in str(exc.value)


def test_changing_the_end_card_rebuilds_exports_but_not_the_mix(tmp_path):
    ctx, sb = fresh_ctx(tmp_path)
    assemble.run(ctx, sb, NS(draft=True, radio=False))
    mix = (ctx.out / "final" / "mix.wav").stat().st_mtime_ns
    old = (ctx.out / "final" / "film_9x16.mp4").read_bytes()
    sb["end_card"] = "Created with AI"
    assemble.run(ctx, sb, NS(draft=True, radio=False))
    assert (ctx.out / "final" / "mix.wav").stat().st_mtime_ns == mix
    assert (ctx.out / "final" / "film_9x16.mp4").read_bytes() != old


def test_radio_cut_needs_only_narration(tmp_path):
    sb = json.loads((FIXTURES / "storyboard_3shot.json").read_text())
    (tmp_path / "storyboard.json").write_text(json.dumps(sb))
    ctx = Ctx(root=tmp_path, mock=True, sleep=lambda s: None)
    narration.run(ctx, sb, NS())
    assemble.run(ctx, sb, NS(draft=False, radio=True))
    by_type, seconds = streams(ctx.out / "final" / "radio_cut.mp4")
    assert "audio" in by_type and by_type["video"]["width"] == 1280
    assert seconds == pytest.approx(13.5, abs=0.2)
    assert not (ctx.out / "final" / "film_16x9.mp4").exists()


def rms_db(path, start, seconds):
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", str(start), "-t", str(seconds), "-i", str(path),
                           "-af", "astats=metadata=0:reset=0", "-f", "null", "-"], capture_output=True, text=True)
    value = re.findall(r"RMS level dB: (-?\d+\.\d+|-inf)", proc.stderr)[-1]
    return -200.0 if value == "-inf" else float(value)


def test_voice_starts_when_its_subtitle_does(tmp_path):
    sb = json.loads((FIXTURES / "storyboard_3shot.json").read_text())
    (tmp_path / "storyboard.json").write_text(json.dumps(sb))
    ctx = Ctx(root=tmp_path, mock=True, sleep=lambda s: None)
    narration.run(ctx, sb, NS())
    assert f"adelay={round(narration.SUBTITLE_LEAD * 1000)}:all=1[a0]" in assemble.mix_graph(sb, None, 16.5)
    assert narration.cues(sb, {s["id"]: 2.0 for s in sb["shots"]})[0][0] == narration.SUBTITLE_LEAD
    assemble.run(ctx, sb, NS(draft=False, radio=True))
    radio = ctx.out / "final" / "radio_mix.wav"
    assert rms_db(radio, 0.0, 0.15) < -80, "silent until the lead has passed"
    assert rms_db(radio, 0.3, 0.5) > -40, "then the voice"


def test_timeline_stays_in_frame_lockstep_for_awkward_durations(tmp_path):
    sb = json.loads((FIXTURES / "storyboard_3shot.json").read_text())
    base = sb["shots"][0]
    sb["shots"] = [dict(base, id=f"s{i}", duration=4.9, keyframe_prompt=f"scene {i}", narration=f"Line number {i}.",
                        subtitle=f"Line number {i}.") for i in range(5)]
    ctx, sb = fresh_ctx(tmp_path, sb)
    assemble.run(ctx, sb, NS(draft=True, radio=False))
    for name in ("9x16", "16x9"):
        by_type, seconds = streams(ctx.out / "final" / f"film_{name}.mp4")
        assert by_type["video"]["nb_frames"] == str(5 * 118 + 72)
        assert seconds == pytest.approx((5 * 118 + 72) / 24, abs=0.05)
        assert float(by_type["audio"]["duration"]) == pytest.approx(seconds, abs=0.06)


def test_a_shot_longer_than_its_clip_holds_the_last_frame(tmp_path, monkeypatch):
    original = shots.ken_burns
    monkeypatch.setattr(shots, "ken_burns", lambda keyframe, dst, seconds: original(keyframe, dst, min(seconds, 5.04)))
    sb = json.loads((FIXTURES / "storyboard_3shot.json").read_text())
    sb["shots"][0]["duration"] = 8.0
    ctx, sb = fresh_ctx(tmp_path, sb)
    assert media_seconds(ctx.out / "shots" / "s01.mp4") == pytest.approx(5.04, abs=0.1)
    assemble.run(ctx, sb, NS(draft=True, radio=False))
    export = ctx.out / "final" / "film_16x9.mp4"
    assert streams(export)[1] == pytest.approx(8.0 + 4.0 + 5.0 + 3.0, abs=0.1)
    held_a, held_b = frame(export, 6.0, (320, 180)), frame(export, 7.6, (320, 180))
    assert mean_abs_diff(held_a, held_b) < 2, "frozen on the last frame, not black"
    assert mean_abs_diff(held_a, Image.new("RGB", (320, 180), "black")) > 20


def test_assemble_announces_fallback_clips(tmp_path, capsys):
    ctx, sb = fresh_ctx(tmp_path)
    for shot_id in ("s01", "s03"):
        side = ctx.out / "shots" / f"{shot_id}.json"
        side.write_text(json.dumps({**json.loads(side.read_text()), "kind": "fallback"}))
    capsys.readouterr()
    assemble.run(ctx, sb, NS(draft=True, radio=False))
    out = capsys.readouterr().out
    assert "FALLBACK" in out and "s01, s03" in out and "s02" not in out.split("FALLBACK")[1]


def test_assemble_says_when_draft_drops_the_score(tmp_path, capsys):
    ctx, sb = fresh_ctx(tmp_path, with_music=False)
    assemble.run(ctx, sb, NS(draft=True, radio=False))
    assert "without a score" in capsys.readouterr().out
