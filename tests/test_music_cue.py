"""Per-shot music cues: the score's caption is the music prompt plus a timed cue sheet."""
import copy
from types import SimpleNamespace

from pipeline import keyframes, mock, music, narration, script, shots
import pytest

from pipeline.common import Ctx, Manifest, MAX_MUSIC_CAPTION, PipelineExit, StoryboardError, check_film, music_caption, validate

NS = SimpleNamespace


def test_caption_starts_with_the_prompt_and_lists_the_cues_in_order_with_whole_seconds(storyboard):
    # fixture shots: 4.5 s, 4.0 s, 5.0 s
    assert music_caption(storyboard) == (
        "calm acoustic strings, slow tempo Cue sheet: "
        "0-4s soft strings enter; 4-8s gentle melody walks; 8-14s warm swell, resolve")


def test_there_is_one_canonical_caption_function():
    assert not hasattr(music, "caption") and music.music_caption is music_caption


def test_editing_one_cue_changes_only_the_music_fingerprint(ctx, storyboard):
    keyframes.run(ctx, storyboard, NS())
    manifest = Manifest(ctx)
    edited = copy.deepcopy(storyboard)
    edited["shots"][1]["music_cue"] = "drums enter"
    assert music.candidate_fp(edited, 1) != music.candidate_fp(storyboard, 1)
    for before, after in zip(storyboard["shots"], edited["shots"]):
        assert keyframes.keyframe_fp(ctx, edited, after) == keyframes.keyframe_fp(ctx, storyboard, before)
        assert narration.narration_fp(ctx, edited, after) == narration.narration_fp(ctx, storyboard, before)
        assert shots.clip_fp(ctx, edited, after, manifest) == shots.clip_fp(ctx, storyboard, before, manifest)


def test_editing_a_narration_line_leaves_the_music_fingerprint_alone(storyboard):
    edited = copy.deepcopy(storyboard)
    edited["shots"][0]["narration"] = "A different line entirely."
    assert all(music.candidate_fp(edited, n) == music.candidate_fp(storyboard, n) for n in (1, 2, 3))


def test_the_script_asks_for_a_music_cue(storyboard):
    item = script.SHOTS_SCHEMA["properties"]["shots"]["items"]
    assert "music_cue" in item["properties"] and "music_cue" in item["required"]
    assert "music_cue" in script.build_prompt(storyboard)


def test_the_mock_film_validates(storyboard):
    sb = dict(storyboard, shots=mock.gemini_json("", {})["shots"])
    validate(sb)
    check_film(sb)


def twelve_longest_shots(storyboard, prompt_chars, cue_chars):
    sb = copy.deepcopy(storyboard)
    sb["music"]["prompt"] = "p" * prompt_chars
    base = sb["shots"][0]
    sb["shots"] = [dict(base, id=f"s{i:02d}", duration=8.0, music_cue="c" * cue_chars) for i in range(12)]
    return sb


@pytest.mark.parametrize("prompt_chars", [33, 100, 200])
def test_the_cue_length_the_script_asks_for_always_fits_the_caption_limit(storyboard, prompt_chars):
    sb = twelve_longest_shots(storyboard, prompt_chars, 1)
    allowance = script.cue_allowance(sb)
    assert 10 <= allowance <= 80
    assert f"at most {allowance} characters" in script.build_prompt(sb)
    validate(twelve_longest_shots(storyboard, prompt_chars, allowance))            # exactly at the allowance: fits
    assert len(music_caption(twelve_longest_shots(storyboard, prompt_chars, allowance))) <= MAX_MUSIC_CAPTION
    with pytest.raises(StoryboardError):
        validate(twelve_longest_shots(storyboard, prompt_chars, allowance + 3))   # a few characters over: rejected


def test_a_longer_music_prompt_leaves_less_room_per_cue(storyboard):
    short = script.cue_allowance(twelve_longest_shots(storyboard, 40, 1))
    long = script.cue_allowance(twelve_longest_shots(storyboard, 160, 1))
    assert short > long


def test_a_music_prompt_that_leaves_no_room_for_cues_stops_the_script_stage_before_any_call(root, storyboard):
    ctx = Ctx(root=root, mock=True, sleep=lambda s: None)
    sb = copy.deepcopy(storyboard)
    sb["music"]["prompt"] = "p" * 500
    sb["shots"] = []
    with pytest.raises(PipelineExit) as exc:
        script.run(ctx, sb, NS(force=False, draft=False))
    assert exc.value.code == 1 and "music.prompt" in str(exc.value) and "characters" in str(exc.value)
    assert not ctx.runs_path.exists()
