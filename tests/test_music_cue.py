"""Per-shot music cues: the score's caption is the music prompt plus a timed cue sheet."""
import copy
from types import SimpleNamespace

from pipeline import keyframes, mock, music, narration, script, shots
import pytest

from pipeline.common import Ctx, Manifest, MAX_MUSIC_CAPTION, PipelineExit, StoryboardError, check_assessment, music_caption, validate

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
    assert "music_cue" in script.build_prompt(storyboard, 3)


def test_the_mock_film_validates(storyboard):
    sb = dict(storyboard, shots=mock.gemini_json("", {})["shots"])
    validate(sb)
    check_assessment(dict(sb, assessment_rules=True))


def test_a_longer_music_prompt_leaves_less_room_per_cue(storyboard):
    def room(chars):
        sb = copy.deepcopy(storyboard)
        sb["music"]["prompt"], sb["target_seconds"] = "p" * chars, 120
        return script.cue_allowance(sb, script.planned_shots(sb))
    assert room(40) > room(160)


def test_a_music_prompt_that_leaves_no_room_for_cues_stops_the_script_stage_before_any_call(root, storyboard):
    ctx = Ctx(root=root, mock=True, sleep=lambda s: None)
    sb = copy.deepcopy(storyboard)
    sb["music"]["prompt"] = "p" * 500
    sb["shots"] = []
    with pytest.raises(PipelineExit) as exc:
        script.run(ctx, sb, NS(force=False))
    assert exc.value.code == 1 and "music.prompt" in str(exc.value) and "characters" in str(exc.value)
    assert not ctx.runs_path.exists()
