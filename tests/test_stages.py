import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from pipeline import keyframes, mock, narration, script
from pipeline.common import Ctx, Generated, Manifest, PipelineExit, ProviderError, StoryboardError, read_runs

NS = SimpleNamespace


def lines(ctx, stage=None):
    runs = read_runs(ctx.runs_path)
    return [r for r in runs if stage is None or r["stage"] == stage]


def load(ctx):
    return json.loads(ctx.storyboard_path.read_text())


def edit(ctx, fn):
    sb = json.loads((ctx.root / "storyboard.json").read_text())
    fn(sb)
    (ctx.root / "storyboard.json").write_text(json.dumps(sb))
    return sb


# ---------------------------------------------------------------- script

def empty(ctx):
    return edit(ctx, lambda sb: sb.update(shots=[]))


def test_script_mock_writes_mock_copy_not_the_real_storyboard(ctx):
    sb = empty(ctx)
    assert script.run(ctx, sb, NS(force=False)) == 0
    assert json.loads((ctx.root / "storyboard.json").read_text())["shots"] == []
    assert len(load(ctx)["shots"]) == 3       # fixture target 16.5 s
    (line,) = lines(ctx)
    assert (line["stage"], line["provider"], line["shot"]) == ("script", "gemini", None)


def test_script_without_a_brief_is_a_clear_error_even_in_mock_mode(ctx):
    sb = edit(ctx, lambda sb: (sb.update(shots=[]), sb.pop("brief")))
    with pytest.raises(PipelineExit) as exc:
        script.run(ctx, sb, NS(force=False))
    assert exc.value.code == 1 and "brief" in str(exc.value)


def test_script_prompt_follows_the_target(storyboard):
    prompt = script.build_prompt(storyboard, 3)
    assert "exactly 3 shots" in prompt and "at most 8 seconds" in prompt and "13.5 seconds" in prompt
    assert "holds its last frame" in prompt and "8 to 12" not in prompt


def test_script_does_nothing_when_shots_exist(ctx, storyboard):
    assert script.run(ctx, storyboard, NS(force=False)) == 0
    assert not ctx.runs_path.exists()


def test_script_force_backs_up_and_replaces(root):
    ctx = Ctx(root=root, provider_override=mock, sleep=lambda s: None)
    sb = dict(json.loads((root / "storyboard.json").read_text()), target_seconds=63)
    script.run(ctx, sb, NS(force=True))
    assert len(json.loads((root / "storyboard.json").read_text())["shots"]) == 12
    (backup,) = root.glob("storyboard.json.bak-*")
    assert len(json.loads(backup.read_text())["shots"]) == 3


def test_script_refuses_placeholder_brief_in_real_mode(root):
    ctx = Ctx(root=root, provider_override=mock)
    sb = json.loads((root / "storyboard.json").read_text())
    sb["shots"], sb["brief"]["premise"] = [], "TODO: the story"
    with pytest.raises(PipelineExit) as exc:
        script.run(ctx, sb, NS(force=False))
    assert exc.value.code == 1 and "brief" in str(exc.value)
    assert not (root / "runs.jsonl").exists()


class Scripted:
    """Fake provider returning queued shot lists."""
    TEXT_MODEL = "fake"

    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def gemini_json(self, prompt, schema, model=None):
        self.prompts.append(prompt)
        return {"shots": self.answers.pop(0)}


def test_script_retries_once_with_the_error_then_fails_without_writing(root):
    bad = [{"id": "s01", "duration": 9, "reason": "r", "keyframe_prompt": "k", "motion_prompt": "m",
            "narration": "n", "subtitle": "s", "music_cue": "c"}]
    fake = Scripted(bad, bad)
    ctx = Ctx(root=root, provider_override=fake, sleep=lambda s: None)
    sb = json.loads((root / "storyboard.json").read_text())
    sb["shots"] = []
    before = (root / "storyboard.json").read_text()
    with pytest.raises(StoryboardError):
        script.run(ctx, sb, NS(force=False))
    assert len(fake.prompts) == 2 and "duration" in fake.prompts[1]
    assert (root / "storyboard.json").read_text() == before


@pytest.mark.parametrize("answer", [{"nope": 1}, {"shots": "text"}, [], None])
def test_script_survives_a_response_without_a_shot_list(root, answer):
    class Odd:
        TEXT_MODEL = "fake"

        def gemini_json(self, prompt, schema, model=None):
            return answer

    ctx = Ctx(root=root, provider_override=Odd(), sleep=lambda s: None)
    sb = json.loads((root / "storyboard.json").read_text())
    sb["shots"] = []
    with pytest.raises(StoryboardError):
        script.run(ctx, sb, NS(force=False))
    assert len(read_runs(ctx.runs_path)) == 2


def test_the_script_stage_has_no_draft_flag(root):
    with pytest.raises(SystemExit) as exc:
        script.main(["--root", str(root), "--draft"])
    assert exc.value.code == 2


# ------------------------------------------------------------- keyframes

def run_keyframes(ctx):
    return keyframes.run(ctx, load(ctx), NS())


def test_keyframes_generate_reference_then_every_shot(ctx):
    run_keyframes(ctx)
    out = ctx.out / "keyframes"
    assert sorted(p.name for p in out.glob("*.jpg")) == ["reference.jpg", "s01.jpg", "s02.jpg", "s03.jpg"]
    runs = lines(ctx, "keyframes")
    assert [r["shot"] for r in runs] == ["reference", "s01", "s02", "s03"]
    assert all(r["seed"] is not None and set(r["params"]) == {"prompt", "width", "height", "reference_sha256"}
               and (r["params"]["width"], r["params"]["height"]) == (1280, 704) for r in runs)   # klein, no reference
    assert Manifest(ctx).fresh("keyframe:s01", keyframes.keyframe_fp(ctx, load(ctx), load(ctx)["shots"][0]),
                               out / "s01.jpg")


def test_keyframes_unchanged_rerun_makes_no_calls(ctx):
    run_keyframes(ctx)
    n = len(lines(ctx))
    run_keyframes(ctx)
    assert len(lines(ctx)) == n


def test_changing_the_character_description_regenerates_every_keyframe(ctx):
    run_keyframes(ctx)
    n = len(lines(ctx))
    edit(ctx, lambda sb: sb["character"].update(description="an elderly man in a grey coat"))
    run_keyframes(ctx)
    assert [r["shot"] for r in lines(ctx)[n:]] == ["reference", "s01", "s02", "s03"]


def test_only_limits_shots_but_still_ensures_the_reference(ctx):
    ctx.only = {"s01"}
    run_keyframes(ctx)
    assert [r["shot"] for r in lines(ctx)] == ["reference", "s01"]


def test_non_jpeg_provider_output_is_stored_as_jpeg(root):
    class Png:
        KLEIN_MODEL = "fake"

        def cloudflare_klein(self, prompt, width, height, seed, reference):
            buf = io.BytesIO()
            Image.new("RGB", (width, height), "red").save(buf, "PNG")
            return Generated(buf.getvalue(), None)

    ctx = Ctx(root=root, provider_override=Png(), sleep=lambda s: None)
    keyframes.run(ctx, load(ctx), NS())
    assert Image.open(ctx.out / "keyframes" / "s01.jpg").format == "JPEG"


def test_failed_generation_keeps_the_previous_keyframe(ctx):
    run_keyframes(ctx)
    old = (ctx.out / "keyframes" / "s02.jpg").read_bytes()

    class Failing:
        KLEIN_MODEL = "mock-klein"

        def cloudflare_klein(self, prompt, width, height, seed, reference):
            raise ProviderError("HTTP 400", status=400)

    edit(ctx, lambda sb: sb["shots"][1].update(keyframe_prompt="changed"))
    ctx.provider_override = Failing()
    with pytest.raises(ProviderError):
        run_keyframes(ctx)
    assert (ctx.out / "keyframes" / "s02.jpg").read_bytes() == old


# ------------------------------------------------------------- narration

def run_narration(ctx):
    return narration.run(ctx, load(ctx), NS())


def test_narration_generates_wavs_and_subtitles(ctx):
    run_narration(ctx)
    assert sorted(p.name for p in (ctx.out / "narration").glob("*.wav")) == ["s01.wav", "s02.wav", "s03.wav"]
    runs = lines(ctx, "narration")
    assert [r["shot"] for r in runs] == ["s01", "s02", "s03"] and runs[0]["params"]["voice"] == "Kore"
    srt = (ctx.out / "subtitles.srt").read_text()
    assert srt.startswith("1\n00:00:00,200 --> ")
    assert "Every morning she walked to the market." in srt
    assert srt.count("-->") == 3


def test_subtitle_cue_times(ctx):
    run_narration(ctx)
    dur = narration.wav_seconds(ctx.out / "narration" / "s02.wav")
    start2 = 4.5 + 0.2
    srt = (ctx.out / "subtitles.srt").read_text()
    assert f"{narration.srt_time(start2)} --> {narration.srt_time(min(start2 + dur, 4.5 + 4.0))}" in srt


def test_narration_is_cached(ctx):
    run_narration(ctx)
    n = len(lines(ctx))
    run_narration(ctx)
    assert len(lines(ctx)) == n


def test_multiline_subtitles_are_flattened(storyboard):
    storyboard["shots"][0]["subtitle"] = "Hello\n\nWorld"
    cues = narration.cues(storyboard, {s["id"]: 2.0 for s in storyboard["shots"]})
    assert cues[0][2] == "Hello World"
    assert "Hello World" in narration.build_srt(storyboard, {s["id"]: 2.0 for s in storyboard["shots"]})


def test_changing_the_voice_regenerates_every_narration_clip(ctx):
    run_narration(ctx)
    n = len(lines(ctx))
    edit(ctx, lambda sb: sb.update(voice="Puck"))
    run_narration(ctx)
    new = lines(ctx)[n:]
    assert [r["shot"] for r in new] == ["s01", "s02", "s03"] and new[0]["params"]["voice"] == "Puck"


def test_a_narration_with_a_different_length_rebuilds_the_subtitle_file(ctx):
    run_narration(ctx)
    before = (ctx.out / "subtitles.srt").read_text()
    edit(ctx, lambda sb: sb["shots"][0].update(narration="Every single morning she walked all the way to the market."))
    run_narration(ctx)
    assert (ctx.out / "subtitles.srt").read_text() != before


def test_partial_run_does_not_write_subtitles(ctx):
    ctx.only = {"s01"}
    run_narration(ctx)
    assert not (ctx.out / "subtitles.srt").exists()


# ------------------------------------------------------------- text model selection

class TextModels:
    TEXT_MODEL, TEXT_MODEL_FLASH = "lite-id", "flash-id"

    def __init__(self):
        self.models = []

    def gemini_json(self, prompt, schema, model="lite-id"):
        self.models.append(model)
        return mock.gemini_json(prompt, schema)


@pytest.mark.parametrize("setting,expected", [(None, "lite-id"), ("gemini-3.5-flash-lite", "lite-id"),
                                              ("gemini-3.8-flash", "flash-id")])
def test_the_script_stage_uses_the_configured_text_model(root, setting, expected):
    fake = TextModels()
    ctx = Ctx(root=root, provider_override=fake, sleep=lambda s: None)
    sb = json.loads((root / "storyboard.json").read_text())
    sb["shots"] = []
    if setting:
        sb["text_model"] = setting
    script.run(ctx, sb, NS(force=False))
    assert fake.models == [expected]
    assert [r["model"] for r in read_runs(ctx.runs_path)] == [expected]


@pytest.mark.parametrize("value,ok", [("gemini-3.5-flash-lite", True), ("gemini-3.8-flash", True),
                                      ("gemini-1.0-pro", False)])
def test_text_model_is_one_of_the_supported_ids(storyboard, value, ok):
    from pipeline.common import validate
    storyboard["text_model"] = value
    if ok:
        validate(storyboard)
    else:
        with pytest.raises(StoryboardError):
            validate(storyboard)
