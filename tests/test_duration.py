"""Film length is project configuration (target_seconds); the 10 Academy rules are opt-in (assessment_rules).
The script stage plans the shot count from the target and refuses impossible requests before any model call."""
import copy
import json
import math
from types import SimpleNamespace

import pytest

from pipeline import mock, script
from pipeline.common import Ctx, END_CARD_SECONDS, MAX_MUSIC_CAPTION, PipelineExit, StoryboardError, music_caption, \
    read_runs

NS = SimpleNamespace
ASSESSMENT_SENTENCE = "This film must meet the brief's rules: 8 to 12 shots and 60 to 90 seconds"


def board(storyboard, target, rules=None, prompt=None):
    sb = copy.deepcopy(storyboard)
    sb["target_seconds"], sb["shots"] = target, []
    if rules is not None:
        sb["assessment_rules"] = rules
    if prompt is not None:
        sb["music"]["prompt"] = prompt
    return sb


def shots(durations, cue="c"):
    return [{"id": f"s{i:02d}", "duration": d, "reason": "r", "keyframe_prompt": "k", "motion_prompt": "m",
             "narration": "n", "subtitle": "s", "music_cue": cue} for i, d in enumerate(durations, 1)]


class Recorder:
    """Fake provider that records every call and answers with queued shot lists."""
    TEXT_MODEL = "fake"

    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    def gemini_json(self, prompt, schema):
        self.prompts.append(prompt)
        return {"shots": self.answers.pop(0)}


def refused(root, sb):
    """Run the script stage; return the exit message after checking that no call and no log line happened."""
    fake = Recorder()
    ctx = Ctx(root=root, provider_override=fake, sleep=lambda s: None)
    with pytest.raises(PipelineExit) as exc:
        script.run(ctx, sb, NS(force=False))
    assert exc.value.code == 1 and fake.prompts == [] and not ctx.runs_path.exists()
    return str(exc.value)


# ------------------------------------------------------------------ planning

@pytest.mark.parametrize("target,rules,n", [
    (16.5, None, 3), (18, None, 3), (15.5, None, 3),      # 15.5: S = 12.5, half rounds up
    (4, None, 1), (63, None, 12), (90, False, 17), (90, True, 12), (120, None, 23),
    (18, True, 8),                                       # clamp; the stage itself rejects this target first
])
def test_planned_shots_follow_the_target(storyboard, target, rules, n):
    assert script.planned_shots(board(storyboard, target, rules)) == n


@pytest.mark.parametrize("s,tol", [(1, 1.0), (15, 1.0), (20, 1.0), (57, 2.85), (87, 4.35)])
def test_tolerance_is_a_second_or_five_percent(s, tol):
    assert script.tolerance(s) == pytest.approx(tol)


def test_the_cue_allowance_shrinks_as_the_target_grows(storyboard):
    allowances = [script.cue_allowance(sb, script.planned_shots(sb))
                  for sb in (board(storyboard, t) for t in (18, 40, 63, 90, 120))]
    assert allowances == sorted(allowances, reverse=True) and allowances[0] > allowances[-1]


def front_loaded(n, total):
    """n shots adding up to `total`: as many 8 s shots first as fit, which widens the cue sheet's timestamps."""
    out, left = [], total
    for i in range(n):
        d = max(0.05, min(8.0, left - 0.05 * (n - i - 1)))
        out.append(round(d, 3))
        left -= d
    return out


@pytest.mark.parametrize("prompt_chars", [1, 33, 100])
def test_a_draft_that_respects_the_allowance_always_fits_the_caption(storyboard, prompt_chars):
    checked = 0
    for half_seconds in range(20, 301):
        sb = board(storyboard, half_seconds / 2, prompt="p" * prompt_chars)
        n, s = script.planned_shots(sb), sb["target_seconds"] - END_CARD_SECONDS
        allowance = script.cue_allowance(sb, n)
        if allowance < 1:
            continue
        for total in (s - script.tolerance(s), s, s + script.tolerance(s)):
            if 0 < total <= 8 * n:
                sb["shots"] = shots(front_loaded(n, total), cue="c" * allowance)
                assert len(music_caption(sb)) <= MAX_MUSIC_CAPTION, (sb["target_seconds"], total, n, allowance)
                checked += 1
    assert checked > 300


# --------------------------------------------------- feasibility before any call

def test_assessment_targets_from_60_to_90_are_always_draftable(storyboard):
    for quarter in range(240, 361):
        sb = board(storyboard, quarter / 4, rules=True)
        n, s = script.planned_shots(sb), sb["target_seconds"] - END_CARD_SECONDS
        tol = script.tolerance(s)
        assert 8 <= n <= 12 and s / n <= 8
        assert 57 <= s <= 87 and s - tol <= s <= s + tol
        assert script.check_draftable(sb)[0] == n


@pytest.mark.parametrize("target,n", [(60, 11), (60.5, 12), (90, 12)])
def test_assessment_boundaries_are_draftable(storyboard, target, n):
    assert script.check_draftable(board(storyboard, target, rules=True)) == (
        n, script.cue_allowance(board(storyboard, target, rules=True), n))


@pytest.mark.parametrize("target", [59.9, 90.1, 30, 18])
def test_assessment_rejects_targets_outside_60_to_90_before_any_call(root, storyboard, target):
    message = refused(root, board(storyboard, target, rules=True))
    assert "target_seconds" in message and "60-90" in message and "assessment_rules" in message


def test_without_assessment_rules_the_same_targets_are_draftable(storyboard):
    for target in (18, 30, 59.9, 90.1, 120):
        script.check_draftable(board(storyboard, target))


@pytest.mark.parametrize("target,rules,prompt_chars", [(300, None, 100), (63, True, 300), (90, True, 300)])
def test_a_target_the_music_caption_cannot_describe_is_refused_before_any_call(root, storyboard, target, rules,
                                                                               prompt_chars):
    sb = board(storyboard, target, rules, prompt="p" * prompt_chars)
    message = refused(root, sb)
    n = script.planned_shots(sb)
    for part in (f"target_seconds {target}", f"{n} shots", "characters per music cue", "512", "music.prompt"):
        assert part in message


def test_shots_longer_than_eight_seconds_are_refused_before_any_call(root, storyboard, monkeypatch):
    monkeypatch.setattr(script, "planned_shots", lambda sb: 1)
    message = refused(root, board(storyboard, 30))
    assert "needs 1 shots of more than 8 s" in message


# ---------------------------------------------------------------- prompt

def test_the_prompt_states_the_plan(storyboard):
    sb = board(storyboard, 18)
    allowance = script.cue_allowance(sb, 3)
    prompt = script.build_prompt(sb, 3)
    for phrase in ("Write exactly 3 shots.", "Each duration is at most 8 seconds.", "holds its last frame",
                   "The durations must add up to 15.0 seconds.", f"at most {allowance} characters"):
        assert phrase in prompt
    assert ASSESSMENT_SENTENCE not in prompt


def test_the_assessment_sentence_appears_only_with_the_rules_on(storyboard):
    assert ASSESSMENT_SENTENCE not in script.build_prompt(board(storyboard, 63, rules=False), 12)
    assert ASSESSMENT_SENTENCE in script.build_prompt(board(storyboard, 63, rules=True), 12)


# ---------------------------------------------------------------- output check

def run_with(root, sb, *answers):
    fake = Recorder(*answers)
    (root / "storyboard.json").write_text(json.dumps(sb))
    ctx = Ctx(root=root, provider_override=fake, sleep=lambda s: None)
    return ctx, fake


def written(root):
    return json.loads((root / "storyboard.json").read_text())["shots"]


def test_a_draft_with_the_wrong_shot_count_is_retried_with_the_reason(root, storyboard):
    sb = board(storyboard, 18)
    ctx, fake = run_with(root, sb, shots([5, 5, 5, 0.5]), shots([5, 5, 5]))
    assert script.run(ctx, sb, NS(force=False)) == 0
    assert len(fake.prompts) == 2 and "4 shots" in fake.prompts[1] and "exactly 3" in fake.prompts[1]
    assert [s["duration"] for s in written(root)] == [5, 5, 5]


def test_a_draft_outside_the_tolerance_twice_is_not_written(root, storyboard):
    sb = board(storyboard, 18)
    ctx, fake = run_with(root, sb, shots([5, 5, 7]), shots([4, 4, 4]))     # 17 s and 12 s for 15 +- 1 s
    with pytest.raises(StoryboardError) as exc:
        script.run(ctx, sb, NS(force=False))
    assert len(fake.prompts) == 2 and "15.0" in str(exc.value)
    assert written(root) == []


def test_a_draft_inside_the_tolerance_is_accepted(root, storyboard):
    sb = board(storyboard, 18)
    ctx, _ = run_with(root, sb, shots([5, 5, 6]))                          # 16 s: the edge of 15 +- 1 s
    assert script.run(ctx, sb, NS(force=False)) == 0 and len(written(root)) == 3


def test_with_the_rules_on_a_draft_inside_the_tolerance_but_under_60_s_is_retried(root, storyboard):
    sb = board(storyboard, 60, rules=True)                                  # n 11, 57 +- 2.85 s
    short = shots([5.0] * 11)                                               # 55 s: inside tolerance, film 58 s
    ctx, fake = run_with(root, sb, short, shots([5.25] * 11))              # then 57.75 s, film 60.75 s
    assert script.run(ctx, sb, NS(force=False)) == 0
    assert "assessment_rules" in fake.prompts[1] and len(written(root)) == 11


def test_with_the_right_count_the_output_check_reports_schema_problems(root, storyboard):
    sb = board(storyboard, 18)
    bad = shots([9, 3, 3])                                                  # right count, invalid duration
    ctx, fake = run_with(root, sb, bad, bad)
    with pytest.raises(StoryboardError) as exc:
        script.run(ctx, sb, NS(force=False))
    assert "shots/0/duration" in str(exc.value) and "exactly" not in str(exc.value)


# ---------------------------------------------------------------- mock provider

def test_the_mock_follows_the_planned_count_total_and_cue_length(storyboard):
    sb = board(storyboard, 18)
    allowance = script.cue_allowance(sb, 3)
    answer = mock.gemini_json(script.build_prompt(sb, 3), script.SHOTS_SCHEMA)["shots"]
    assert len(answer) == 3 and sum(s["duration"] for s in answer) == pytest.approx(15.0)
    assert all(len(s["music_cue"]) <= allowance for s in answer)


def test_the_mock_falls_back_to_twelve_five_second_shots():
    answer = mock.gemini_json("Reply with JSON {\"ok\": true}.", {})["shots"]
    assert [s["duration"] for s in answer] == [5.0] * 12


def test_mock_script_on_an_18_second_target_writes_three_shots(root, storyboard):
    sb = board(storyboard, 18)
    (root / "storyboard.json").write_text(json.dumps(sb))
    ctx = Ctx(root=root, mock=True, sleep=lambda s: None)
    assert script.run(ctx, sb, NS(force=False)) == 0
    drafted = json.loads((root / "outputs" / "mock" / "storyboard.json").read_text())["shots"]
    assert len(drafted) == 3 and abs(sum(s["duration"] for s in drafted) - 15) <= script.tolerance(15)
    assert [r["stage"] for r in read_runs(ctx.runs_path)] == ["script"]


def test_the_planning_constant_is_wans_clip_length():
    assert script.CLIP_SECONDS == 5 and math.isclose(script.tolerance(100), 5.0)


# ---------------------------------------------------------------- review follow-ups

def test_a_wrong_shot_count_is_reported_before_the_schema(root, storyboard):
    sb = board(storyboard, 18)
    many = shots([1.25] * 12, cue="c" * 80)                                  # too many shots, cues overflow 512
    ctx, fake = run_with(root, sb, many, shots([5, 5, 5]))
    assert script.run(ctx, sb, NS(force=False)) == 0
    assert "12 shots; write exactly 3" in fake.prompts[1] and "512" not in fake.prompts[1]


def test_a_shot_shorter_than_one_frame_is_rejected(root, storyboard):
    sb = board(storyboard, 4)                                                # 1 s of shots, 1 s tolerance
    tiny = shots([0.01])
    ctx, fake = run_with(root, sb, tiny, tiny)
    with pytest.raises(StoryboardError) as exc:
        script.run(ctx, sb, NS(force=False))
    assert "s01" in str(exc.value) and "frame" in str(exc.value) and written(root) == []
