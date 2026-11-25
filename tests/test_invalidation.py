import hashlib
import json
from types import SimpleNamespace

import pytest

from pipeline import run as pipeline_run
from pipeline.common import Ctx, load_storyboard, read_runs

NS = SimpleNamespace


@pytest.fixture
def project(root):
    ctx = Ctx(root=root, mock=True, sleep=lambda s: None)

    def go():
        pipeline_run.run_all(ctx, load_storyboard(ctx), NS(draft=True, fallback=False, resubmit=False, force=False,
                                                           retry_fallbacks=False))

    def edit(fn):
        sb = json.loads((root / "storyboard.json").read_text())
        fn(sb)
        (root / "storyboard.json").write_text(json.dumps(sb))

    def lines():
        return read_runs(ctx.runs_path)

    def digest(*names):
        return {n: hashlib.sha256((ctx.out / "final" / n).read_bytes()).hexdigest() for n in names}

    ctx.go, ctx.edit, ctx.lines, ctx.digest = go, edit, lines, digest
    return ctx


EXPORTS = ("film_9x16.mp4", "film_16x9.mp4")


def test_one_narration_edit_means_one_tts_call_and_reassembly_costs_no_model_calls(project):
    ctx = project
    ctx.go()
    first_lines, first = len(ctx.lines()), ctx.digest("mix.wav", *EXPORTS)
    assert first_lines == 4 + 3 + 3 + 3  # reference+3 keyframes, 3 narration, 3 shots, 3 candidates

    srt_before = (ctx.out / "subtitles.srt").read_text()
    ctx.edit(lambda sb: sb["shots"][1].update(narration="She carried everything she had in the world."))
    ctx.go()
    assert (ctx.out / "subtitles.srt").read_text() != srt_before, "a longer line shifts the subtitle times"
    new = ctx.lines()[first_lines:]
    assert [(r["stage"], r["shot"], r["status"]) for r in new] == [("narration", "s02", "ok")]
    second = ctx.digest("mix.wav", *EXPORTS)
    assert all(second[k] != first[k] for k in second), "mix and both exports are rebuilt"

    n = len(ctx.lines())
    ctx.go()
    assert len(ctx.lines()) == n and ctx.digest("mix.wav", *EXPORTS) == second

    for name in EXPORTS:
        (ctx.out / "final" / name).unlink()
    ctx.go()
    assert len(ctx.lines()) == n, "reassembly makes no model call"
    assert ctx.digest(*EXPORTS) == {k: second[k] for k in EXPORTS}, "bit-exact rebuild"


@pytest.mark.parametrize("draft", [False, True])
def test_a_film_outside_the_assessment_rules_fails_before_any_model_call(root, draft):
    from pipeline.common import StoryboardError
    ctx = Ctx(root=root, mock=True, sleep=lambda s: None)
    sb = dict(load_storyboard(ctx), assessment_rules=True)
    with pytest.raises(StoryboardError):
        pipeline_run.run_all(ctx, sb, NS(draft=draft, fallback=False, resubmit=False, force=False,
                                         retry_fallbacks=False))
    assert not ctx.runs_path.exists()
    assert not ctx.runs_path.exists() and not (ctx.out / "keyframes").exists()


def test_one_keyframe_prompt_edit_regenerates_one_keyframe_and_one_clip(project):
    ctx = project
    ctx.go()
    n, before = len(ctx.lines()), ctx.digest(*EXPORTS)
    ctx.edit(lambda sb: sb["shots"][0].update(keyframe_prompt="a river at dawn"))
    ctx.go()
    assert [(r["stage"], r["shot"]) for r in ctx.lines()[n:]] == [("keyframes", "s01"), ("shots", "s01")]
    assert all(ctx.digest(k)[k] != before[k] for k in EXPORTS)


def test_style_edit_regenerates_every_keyframe_and_clip_but_not_voice_or_music(project):
    ctx = project
    ctx.go()
    n = len(ctx.lines())
    ctx.edit(lambda sb: sb.update(style="stark black and white"))
    ctx.go()
    stages = sorted((r["stage"] for r in ctx.lines()[n:]))
    assert stages == ["keyframes"] * 4 + ["shots"] * 3


def test_music_choice_is_a_local_step(project):
    ctx = project
    ctx.go()
    n, before = len(ctx.lines()), ctx.digest("mix.wav", *EXPORTS)
    ctx.edit(lambda sb: sb["music"].update(choice={"candidate": 3, "reason": "brightest"}))
    ctx.go()
    assert len(ctx.lines()) == n
    assert ctx.digest("mix.wav")["mix.wav"] != before["mix.wav"]


def test_subtitle_only_edit_makes_no_model_call_and_rebuilds_the_exports(project):
    ctx = project
    ctx.go()
    n, before = len(ctx.lines()), ctx.digest(*EXPORTS)
    ctx.edit(lambda sb: sb["shots"][2].update(subtitle="And the market welcomed her, as always."))
    ctx.go()
    assert len(ctx.lines()) == n
    assert all(ctx.digest(k)[k] != before[k] for k in EXPORTS)


def test_a_duration_edit_costs_no_keyframe_or_voice_call(project):
    ctx = project
    ctx.go()
    n = len(ctx.lines())
    ctx.edit(lambda sb: sb["shots"][0].update(duration=4.0))
    ctx.go()
    assert [r for r in ctx.lines()[n:] if r["stage"] in ("keyframes", "narration")] == []
    # the cue sheet's whole-second times moved (s03 ends at 13 s, not 14 s), so the score candidates are redone
    assert len([r for r in ctx.lines()[n:] if r["stage"] == "music"]) == 3


def test_project_length_and_assessment_switch_are_in_no_fingerprint(project):
    ctx = project
    ctx.go()
    n, before = len(ctx.lines()), ctx.digest("mix.wav", *EXPORTS)
    ctx.edit(lambda sb: sb.update(target_seconds=120))
    ctx.go()
    ctx.edit(lambda sb: sb.update(assessment_rules=False))
    ctx.go()
    assert len(ctx.lines()) == n, "no model call"
    assert ctx.digest("mix.wav", *EXPORTS) == before, "nothing rebuilt"
