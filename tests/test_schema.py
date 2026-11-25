import copy

import pytest

from pipeline.common import StoryboardError, check_assessment, validate


def invalid(sb, **kw):
    with pytest.raises(StoryboardError) as exc:
        validate(sb, **kw)
    return str(exc.value)


def test_fixture_is_valid(storyboard):
    validate(storyboard)


@pytest.mark.parametrize("mutate,expect", [
    (lambda sb: sb.pop("voice"), "voice"),
    (lambda sb: sb.update(extra=1), "extra"),
    (lambda sb: sb["shots"][0].update(id="Bad Id"), "shots/0/id"),
    (lambda sb: sb["shots"][0].update(duration=0), "shots/0/duration"),
    (lambda sb: sb["shots"][0].update(duration=8.5), "shots/0/duration"),
    (lambda sb: sb.update(voice="Nobody"), "voice"),
    (lambda sb: sb["shots"][1].update(id="s01"), "unique"),
    (lambda sb: sb.update(end_card="Hello"), "end_card"),
    (lambda sb: sb["shots"][0].update(keyframe_prompt="x" * 1000, ), None),
])
def test_invalid_storyboards_list_the_path(storyboard, mutate, expect):
    sb = copy.deepcopy(storyboard)
    mutate(sb)
    if expect is None:
        sb["style"] = "y" * 500
        sb["character"]["description"] = "z" * 1000
        expect = "2048"
    assert expect in invalid(sb)


@pytest.mark.parametrize("mutate", [
    lambda shot: shot.pop("music_cue"),
    lambda shot: shot.update(music_cue=""),
    lambda shot: shot.update(music_cue="x" * 81),
])
def test_every_shot_needs_a_short_music_cue(storyboard, mutate):
    mutate(storyboard["shots"][1])
    message = invalid(storyboard)
    assert "shots/1" in message and "music_cue" in message


def test_a_music_cue_of_eighty_characters_is_valid(storyboard):
    storyboard["shots"][1]["music_cue"] = "x" * 80
    validate(storyboard)


def test_the_composed_music_caption_is_limited_to_512_characters(storyboard):
    storyboard["music"]["prompt"] = "p" * 450
    message = invalid(storyboard)
    assert "music" in message and "512" in message


def test_all_errors_are_listed_together(storyboard):
    storyboard.pop("voice")
    storyboard["shots"][0]["duration"] = 9
    message = invalid(storyboard)
    assert "voice" in message and "shots/0/duration" in message


def test_length_is_measured_in_whole_frames_like_the_film_itself():
    short = {"assessment_rules": True,
             "shots": [{"duration": 7.145417}] * 7 + [{"duration": 7.10375}]}   # 60.12 s of seconds, 59.958 s of frames
    with pytest.raises(StoryboardError):
        check_assessment(short)
    check_assessment({"assessment_rules": True,
                      "shots": [{"duration": 7.25}] * 11 + [{"duration": 7.27}]})  # 90.02 s of seconds, exactly 90.0 of frames


def test_long_shots_up_to_eight_seconds_are_valid(storyboard):
    storyboard["shots"][0]["duration"] = 8
    validate(storyboard)


def test_non_script_stages_need_a_shot(storyboard):
    storyboard["shots"] = []
    validate(storyboard, need_shots=False)
    assert "shot" in invalid(storyboard, need_shots=True)


def make_film(n, each, rules=True):
    return {"assessment_rules": rules, "shots": [{"duration": each} for _ in range(n)]}


@pytest.mark.parametrize("n,each,ok", [
    (12, 5.0, True),     # 63 s
    (8, 5.0, False),     # 43 s: too short
    (8, 7.125, True),    # exactly 60 s with the fewest shots
    (8, 8.0, True),      # 67 s
    (12, 4.75, True),    # exactly 60 s with the most shots
    (12, 4.5, False),    # 57 s
    (12, 7.25, True),    # exactly 90 s
    (12, 7.5, False),    # 93 s: too long
    (11, 5.0, False),    # 58 s
    (13, 5.0, False),    # too many shots
    (7, 8.0, False),     # too few shots
])
def test_film_length_and_shot_count(n, each, ok):
    if ok:
        check_assessment(make_film(n, each))
    else:
        with pytest.raises(StoryboardError):
            check_assessment(make_film(n, each))


@pytest.mark.parametrize("rules", [False, None])
def test_without_assessment_rules_any_shot_count_and_length_is_accepted(rules):
    film = make_film(3, 5.0, rules)
    if rules is None:
        film.pop("assessment_rules")
    check_assessment(film)
    check_assessment(make_film(40, 8.0, rules))


def test_assessment_problems_are_listed_together_with_the_way_out():
    with pytest.raises(StoryboardError) as exc:
        check_assessment(make_film(3, 5.0))
    message = str(exc.value)
    assert "3 shots" in message and "length" in message
    assert message.endswith("set assessment_rules to false to experiment outside these rules")


# ------------------------------------------------------- project length and assessment rules

def test_target_seconds_is_required(storyboard):
    storyboard.pop("target_seconds", None)
    assert "target_seconds" in invalid(storyboard)


@pytest.mark.parametrize("value,ok", [(4, True), (16.5, True), (300, True), (3.9, False), (0, False), ("18", False)])
def test_target_seconds_leaves_at_least_one_second_of_shots(storyboard, value, ok):
    storyboard["target_seconds"] = value
    if ok:
        validate(storyboard)
    else:
        assert "target_seconds" in invalid(storyboard)


@pytest.mark.parametrize("value,ok", [(True, True), (False, True), ("yes", False), (1, False)])
def test_assessment_rules_is_an_optional_boolean(storyboard, value, ok):
    validate(storyboard)                      # absent is valid
    storyboard["assessment_rules"] = value
    if ok:
        validate(storyboard)
    else:
        assert "assessment_rules" in invalid(storyboard)
