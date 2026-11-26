import json
from types import SimpleNamespace

import pytest

from pipeline import keyframes, kaggle, mock, music, shots
from pipeline.common import Ctx, Manifest, PipelineExit, media_seconds, read_runs

NS = SimpleNamespace


def sb_of(ctx):
    return json.loads((ctx.root / "storyboard.json").read_text())


def edit(ctx, fn):
    sb = sb_of(ctx)
    fn(sb)
    (ctx.root / "storyboard.json").write_text(json.dumps(sb))
    return sb


def runs(ctx, stage=None):
    return [r for r in read_runs(ctx.runs_path) if stage is None or r["stage"] == stage]


def shots_args(**kw):
    return NS(**{"fallback": False, "resubmit": False, "retry_fallbacks": False, "fallback_only": False, **kw})


@pytest.fixture
def real(root, fake_kaggle):
    """Real-mode context: mock media providers, fake kaggle executable."""
    ctx = Ctx(root=root, provider_override=mock, sleep=lambda s: None, poll_s=10, remote_timeout_s=100)
    keyframes.run(ctx, sb_of(ctx), NS())
    return ctx


def sidecar(ctx, shot_id):
    return json.loads((ctx.out / "shots" / f"{shot_id}.json").read_text())


# ---------------------------------------------------------------- shots, mock

def test_shots_need_current_keyframes(ctx):
    with pytest.raises(PipelineExit) as exc:
        shots.run(ctx, sb_of(ctx), shots_args())
    assert exc.value.code == 1 and "keyframes" in str(exc.value)
    keyframes.run(ctx, sb_of(ctx), NS())
    edit(ctx, lambda sb: sb["shots"][0].update(keyframe_prompt="changed but not regenerated"))
    with pytest.raises(PipelineExit) as exc:
        shots.run(ctx, sb_of(ctx), shots_args())
    assert "s01" in str(exc.value)


def test_mock_shots_are_cached_and_follow_changes(ctx):
    keyframes.run(ctx, sb_of(ctx), NS())
    assert shots.run(ctx, sb_of(ctx), shots_args()) == 0
    assert sorted(p.name for p in (ctx.out / "shots").glob("*.mp4")) == ["s01.mp4", "s02.mp4", "s03.mp4"]
    assert sidecar(ctx, "s01")["kind"] == "mock"
    assert media_seconds(ctx.out / "shots" / "s01.mp4") == pytest.approx(4.5, abs=0.1)
    n = len(runs(ctx, "shots"))
    assert n == 3 and all(r["provider"] == "kaggle" and r["model"] == "mock-wan" for r in runs(ctx, "shots"))
    shots.run(ctx, sb_of(ctx), shots_args())
    assert len(runs(ctx, "shots")) == n
    edit(ctx, lambda sb: sb["shots"][1].update(motion_prompt="orbit slowly"))
    shots.run(ctx, sb_of(ctx), shots_args())
    assert [r["shot"] for r in runs(ctx, "shots")[n:]] == ["s02"]


def test_a_regenerated_keyframe_regenerates_its_clip(ctx):
    keyframes.run(ctx, sb_of(ctx), NS())
    shots.run(ctx, sb_of(ctx), shots_args())
    n = len(runs(ctx, "shots"))
    edit(ctx, lambda sb: sb["shots"][2].update(keyframe_prompt="a different market"))
    keyframes.run(ctx, sb_of(ctx), NS())
    shots.run(ctx, sb_of(ctx), shots_args())
    assert [r["shot"] for r in runs(ctx, "shots")[n:]] == ["s03"]


# ---------------------------------------------------------------- shots, real mode

def test_only_stale_shots_go_to_kaggle_with_normalised_keyframes(real, fake_kaggle):
    ctx = real
    assert shots.run(ctx, sb_of(ctx), shots_args()) == 0
    assert fake_kaggle.count("pushes") == 1
    job = json.loads((fake_kaggle.state_dir / "dataset" / "job.json").read_text())
    assert [i["id"] for i in job["items"]] == ["s01", "s02", "s03"]
    item = job["items"][0]
    assert item["params"]["prompt"].endswith("slow push in") and isinstance(item["params"]["seed"], int)
    from PIL import Image
    assert Image.open(fake_kaggle.state_dir / "dataset" / "s01.jpg").size == (1280, 704)
    assert sidecar(ctx, "s01")["kind"] == "real"
    lines = runs(ctx, "shots")
    assert len(lines) == 3 and lines[0]["provider"] == "kaggle" and lines[0]["model"] == "fake-model"
    assert lines[0]["run_id"].startswith("shots-") and lines[0]["gpu_seconds"] == 1.5
    assert not (ctx.out / "jobs" / "shots" / "pending.json").exists()
    assert shots.run(ctx, sb_of(ctx), shots_args()) == 0
    assert fake_kaggle.count("pushes") == 1, "nothing stale, nothing submitted"


def test_one_changed_shot_submits_only_that_shot(real, fake_kaggle):
    ctx = real
    shots.run(ctx, sb_of(ctx), shots_args())
    edit(ctx, lambda sb: sb["shots"][1].update(motion_prompt="orbit slowly"))
    shots.run(ctx, sb_of(ctx), shots_args())
    assert fake_kaggle.count("pushes") == 2
    job = json.loads((fake_kaggle.state_dir / "dataset" / "job.json").read_text())
    assert [i["id"] for i in job["items"]] == ["s02"]
    assert [r["shot"] for r in runs(ctx, "shots")][3:] == ["s02"]


def test_only_flag_limits_the_job(real, fake_kaggle):
    real.only = {"s03"}
    shots.run(real, sb_of(real), shots_args())
    job = json.loads((fake_kaggle.state_dir / "dataset" / "job.json").read_text())
    assert [i["id"] for i in job["items"]] == ["s03"]


def test_ingesting_the_same_results_twice_logs_once(real, fake_kaggle):
    ctx = real
    sb = sb_of(ctx)
    manifest = Manifest(ctx)
    fps = {"s01": shots.clip_fp(ctx, sb, sb["shots"][0], manifest)}
    items, inputs = shots.build_job(ctx, sb, [sb["shots"][0]], fps)
    res = kaggle.collect(ctx, "shots", items, inputs)
    expected = {"s01": items[0]["fingerprint"]}
    shots.ingest(ctx, res, expected)
    shots.ingest(ctx, res, expected)
    assert len(runs(ctx, "shots")) == 1


def test_failed_kernel_without_fallback_exits_2_and_lists_missing(real, fake_kaggle, capsys):
    fake_kaggle.scenario(mode="fail")
    assert shots.run(real, sb_of(real), shots_args()) == 2
    assert "s01" in capsys.readouterr().out
    assert not list((real.out / "shots").glob("*.mp4"))


def test_fallback_fills_gaps_and_a_rerun_makes_no_new_call(real, fake_kaggle, capsys):
    ctx = real
    fake_kaggle.scenario(mode="fail")
    assert shots.run(ctx, sb_of(ctx), NS(fallback=True, resubmit=False, retry_fallbacks=False)) == 0
    assert sidecar(ctx, "s02")["kind"] == "fallback"
    assert media_seconds(ctx.out / "shots" / "s02.mp4") == pytest.approx(4.0, abs=0.1)
    timings = [json.loads(l) for l in (ctx.out / "timings.jsonl").read_text().splitlines()]
    assert sum(t["op"] == "fallback" for t in timings) == 3
    assert fake_kaggle.count("pushes") == 1
    capsys.readouterr()
    assert shots.run(ctx, sb_of(ctx), shots_args()) == 0
    assert fake_kaggle.count("pushes") == 1, "an unchanged re-run submits nothing"
    out = capsys.readouterr().out
    assert "FALLBACK" in out and "s01, s02, s03" in out and "--retry-fallbacks" in out


def test_fallback_clips_are_usable_even_when_kaggle_is_unreachable(real, fake_kaggle, monkeypatch):
    fake_kaggle.scenario(mode="fail")
    shots.run(real, sb_of(real), NS(fallback=True, resubmit=False, retry_fallbacks=False))
    monkeypatch.delenv("KAGGLE_USERNAME")
    assert shots.run(real, sb_of(real), shots_args()) == 0


def test_retry_fallbacks_asks_wan_again_and_a_real_clip_replaces_the_fallback(real, fake_kaggle):
    ctx = real
    fake_kaggle.scenario(mode="fail")
    shots.run(ctx, sb_of(ctx), NS(fallback=True, resubmit=False, retry_fallbacks=False))
    fake_kaggle.scenario(mode="ok", running_polls=0)
    assert shots.run(ctx, sb_of(ctx), NS(fallback=True, resubmit=False, retry_fallbacks=True)) == 0
    assert fake_kaggle.count("pushes") == 2 and sidecar(ctx, "s02")["kind"] == "real"


def test_fallback_also_covers_a_timeout_and_an_unreachable_kaggle(real, fake_kaggle, monkeypatch):
    fake_kaggle.scenario(mode="never")
    assert shots.run(real, sb_of(real), NS(fallback=True, resubmit=False, retry_fallbacks=False)) == 0
    assert (real.out / "jobs" / "shots" / "pending.json").exists()
    monkeypatch.delenv("KAGGLE_USERNAME")
    edit(real, lambda sb: sb["shots"][0].update(motion_prompt="new"))
    with pytest.raises(PipelineExit):
        shots.run(real, sb_of(real), shots_args())
    assert shots.run(real, sb_of(real), NS(fallback=True, resubmit=False, retry_fallbacks=False)) == 0


def error_lines(ctx, stage):
    return [r for r in runs(ctx, stage) if r["status"] == "error"]


def test_a_failed_kernel_is_logged_item_by_item(real, fake_kaggle):
    fake_kaggle.scenario(mode="fail")
    shots.run(real, sb_of(real), shots_args())
    lines = error_lines(real, "shots")
    assert sorted(r["shot"] for r in lines) == ["s01", "s02", "s03"]
    assert all(r["provider"] == "kaggle" and r["run_id"].startswith("shots-") and r["error"] for r in lines)
    assert lines[0]["model"] == shots.MODEL and lines[0]["seed"] is not None


def test_a_kernel_that_reports_an_error_for_one_item_logs_the_reason(real, fake_kaggle):
    fake_kaggle.scenario(mode="kernel_error")
    assert shots.run(real, sb_of(real), shots_args()) == 2
    assert all("CUDA out of memory" in r["error"] for r in error_lines(real, "shots"))


def test_a_partial_run_logs_one_success_and_the_failures(real, fake_kaggle):
    fake_kaggle.scenario(mode="partial")
    shots.run(real, sb_of(real), shots_args())
    lines = runs(real, "shots")
    assert [(r["shot"], r["status"]) for r in lines] == [("s01", "ok"), ("s02", "error"), ("s03", "error")]


def test_nothing_is_logged_as_failed_while_the_run_may_still_be_pending(real, fake_kaggle):
    fake_kaggle.scenario(mode="wrong_run")
    assert shots.run(real, sb_of(real), shots_args()) == 2
    assert error_lines(real, "shots") == []
    assert (real.out / "jobs" / "shots" / "pending.json").exists()


def test_a_clip_that_is_too_short_is_rejected_and_never_committed(real, fake_kaggle):
    fake_kaggle.scenario(mode="short")
    assert shots.run(real, sb_of(real), shots_args()) == 2
    assert not list((real.out / "shots").glob("*.mp4"))
    assert any("clip is" in r["error"] for r in error_lines(real, "shots"))


def test_real_runs_ignore_mock_sidecars(real, fake_kaggle):
    ctx = real
    sb = sb_of(ctx)
    fp = shots.clip_fp(ctx, sb, sb["shots"][0], Manifest(ctx))
    (ctx.out / "shots").mkdir(parents=True, exist_ok=True)
    (ctx.out / "shots" / "s01.mp4").write_bytes(b"x")
    from pipeline.common import sha256_file
    (ctx.out / "shots" / "s01.json").write_text(json.dumps({"fingerprint": fp, "kind": "mock",
                                                             "sha256": sha256_file(ctx.out / "shots" / "s01.mp4")}))
    shots.run(ctx, sb, shots_args())
    job = json.loads((fake_kaggle.state_dir / "dataset" / "job.json").read_text())
    assert "s01" in [i["id"] for i in job["items"]]


# ---------------------------------------------------------------- shots, Wan frames and fallbacks

@pytest.mark.parametrize("seconds,frames", [(5.0, 121), (4.0, 97), (4.5, 109), (8.0, 121), (5.05, 121),
                                            (0.1, 5), (1 / 24, 5)])
def test_wan_frames_cover_the_shot_up_to_the_measured_121(seconds, frames):
    assert shots.wan_frames(seconds) == frames
    assert frames % 4 == 1 and frames <= 121


def test_each_shot_asks_wan_for_its_own_frame_count_and_the_e6_model(real, fake_kaggle):
    assert shots.run(real, sb_of(real), shots_args()) == 0
    job = json.loads((fake_kaggle.state_dir / "dataset" / "job.json").read_text())
    assert [i["params"]["num_frames"] for i in job["items"]] == [109, 97, 121]   # 4.5 s, 4.0 s, 5.0 s
    assert shots.MODEL == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert media_seconds(real.out / "shots" / "s02.mp4") == pytest.approx(97 / 24, abs=0.05)
    assert sidecar(real, "s02")["kind"] == "real"


def test_a_duration_change_resubmits_only_when_the_wan_frame_count_changes(real, fake_kaggle):
    shots.run(real, sb_of(real), shots_args())
    edit(real, lambda sb: sb["shots"][2].update(duration=6.0))   # 5.0 s and 6.0 s both need the full 121 frames
    shots.run(real, sb_of(real), shots_args())
    assert fake_kaggle.count("pushes") == 1
    edit(real, lambda sb: sb["shots"][1].update(duration=4.5))   # 97 -> 109 frames
    shots.run(real, sb_of(real), shots_args())
    job = json.loads((fake_kaggle.state_dir / "dataset" / "job.json").read_text())
    assert fake_kaggle.count("pushes") == 2 and [i["id"] for i in job["items"]] == ["s02"]


def test_fallback_after_a_partial_run_keeps_wan_clips_and_never_logs_a_fallback_as_a_success(real, fake_kaggle,
                                                                                         capsys):
    fake_kaggle.scenario(mode="partial")
    assert shots.run(real, sb_of(real), shots_args(fallback=True)) == 0
    assert [sidecar(real, s)["kind"] for s in ("s01", "s02", "s03")] == ["real", "fallback", "fallback"]
    assert sidecar(real, "s02")["model"] == "ffmpeg-zoompan"
    lines = runs(real, "shots")
    assert [(r["shot"], r["status"]) for r in lines] == [("s01", "ok"), ("s02", "error"), ("s03", "error")]
    assert "FALLBACK clips in use (pan-and-zoom, not Wan) for s02, s03" in capsys.readouterr().out


def test_fallback_only_makes_pan_and_zoom_clips_without_any_kaggle_call(real, fake_kaggle, capsys):
    assert shots.run(real, sb_of(real), shots_args(fallback_only=True)) == 0
    assert fake_kaggle.calls == []
    assert [sidecar(real, s)["kind"] for s in ("s01", "s02", "s03")] == ["fallback"] * 3
    assert runs(real, "shots") == []
    assert "FALLBACK" in capsys.readouterr().out
    assert shots.run(real, sb_of(real), shots_args()) == 0
    assert fake_kaggle.calls == [], "fallback clips are kept until --retry-fallbacks"


def test_fallback_only_keeps_a_wan_clip_unless_the_shot_is_named(real, fake_kaggle, capsys):
    shots.run(real, sb_of(real), shots_args())
    wan_sha = sidecar(real, "s01")["sha256"]
    shots.run(real, sb_of(real), shots_args(fallback_only=True))
    assert [sidecar(real, s)["kind"] for s in ("s01", "s02", "s03")] == ["real"] * 3
    assert sidecar(real, "s01")["sha256"] == wan_sha
    capsys.readouterr()
    real.only = {"s01"}
    assert shots.run(real, sb_of(real), shots_args(fallback_only=True)) == 0
    assert sidecar(real, "s01")["kind"] == "fallback" and sidecar(real, "s02")["kind"] == "real"
    assert "replacing the Wan clip for s01" in capsys.readouterr().out
    assert fake_kaggle.count("pushes") == 1


def test_fallback_only_cannot_be_combined_with_kaggle_options(real, fake_kaggle):
    for extra in ({"fallback": True}, {"retry_fallbacks": True}, {"resubmit": True}):
        with pytest.raises(PipelineExit):
            shots.run(real, sb_of(real), shots_args(fallback_only=True, **extra))
    assert fake_kaggle.calls == []


def test_the_shots_command_line_offers_fallback_only(real, monkeypatch):
    seen = {}
    monkeypatch.setattr(shots, "run", lambda ctx, sb, args: seen.setdefault("args", args) and 0)
    with pytest.raises(SystemExit):
        shots.main(["--root", str(real.root), "--fallback-only"])
    assert seen["args"].fallback_only is True


# ---------------------------------------------------------------- music

def choose(ctx, candidate=2, reason="warmest strings"):
    edit(ctx, lambda sb: sb["music"].update(choice={"candidate": candidate, "reason": reason}))


def test_mock_music_makes_candidates_and_picks_one(ctx):
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 0
    assert sorted(p.name for p in (ctx.out / "music" / "candidates").glob("*.wav")) == [
        "candidate_1.wav", "candidate_2.wav", "candidate_3.wav"]
    score = json.loads((ctx.out / "music" / "score.json").read_text())
    assert score["candidate"] == 1 and score["reason"] == "mock" and (ctx.out / "music" / "score.wav").exists()
    n = len(runs(ctx, "music"))
    assert n == 3
    music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False))
    assert len(runs(ctx, "music")) == n


def test_music_real_flow_waits_for_a_choice(real, fake_kaggle, capsys):
    ctx = real
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 3
    assert "music.choice" in capsys.readouterr().out
    job = json.loads((fake_kaggle.state_dir / "dataset" / "job.json").read_text())
    assert [i["id"] for i in job["items"]] == ["candidate_1", "candidate_2", "candidate_3"]
    assert job["items"][0]["params"] == {"caption": "calm acoustic strings, slow tempo Cue sheet: 0-4s soft "
                                        "strings enter; 4-8s gentle melody walks; 8-14s warm swell, resolve",
                                        "duration": 20.0,
                                        "seed": 42}
    assert [i["params"]["seed"] for i in job["items"]] == [42, 43, 44]
    assert len(runs(ctx, "music")) == 3
    assert not (ctx.out / "music" / "score.wav").exists()
    assert music.run(ctx, sb_of(ctx), NS(draft=True, resubmit=False)) == 0
    assert fake_kaggle.count("pushes") == 1


def test_choosing_copies_a_candidate_locally_with_no_model_call(real, fake_kaggle):
    ctx = real
    music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False))
    n = len(runs(ctx))
    choose(ctx, 2)
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 0
    score = json.loads((ctx.out / "music" / "score.json").read_text())
    assert (score["candidate"], score["reason"]) == (2, "warmest strings")
    assert (ctx.out / "music" / "score.wav").read_bytes() == (ctx.out / "music" / "candidates" /
                                                                "candidate_2.wav").read_bytes()
    choose(ctx, 3, "brighter")
    music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False))
    assert json.loads((ctx.out / "music" / "score.json").read_text())["candidate"] == 3
    assert len(runs(ctx)) == n and fake_kaggle.count("pushes") == 1


def test_changed_prompt_regenerates_candidates_and_invalidates_the_choice(real, fake_kaggle, capsys):
    ctx = real
    choose(ctx, 2)
    music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False))
    edit(ctx, lambda sb: sb["music"].update(prompt="fast drums"))
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 3
    assert fake_kaggle.count("pushes") == 2 and "choose again" in capsys.readouterr().out
    choose(ctx, 1, "now this one")
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 0
    assert fake_kaggle.count("pushes") == 2


def test_a_duration_change_regenerates_music_only_when_the_bucket_or_cue_times_move(real, fake_kaggle):
    ctx = real
    music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False))
    edit(ctx, lambda sb: sb["shots"][2].update(duration=5.2))   # cue sheet still 8-14s, same 20 s bucket
    music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False))
    assert fake_kaggle.count("pushes") == 1
    edit(ctx, lambda sb: sb["shots"][0].update(duration=3.4))   # moves the cue times: the caption changes
    music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False))
    assert fake_kaggle.count("pushes") == 2
    edit(ctx, lambda sb: [sh.update(duration=1.0) for sh in sb["shots"]])   # 3 s + 3 s end card: 10 s bucket
    music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False))
    assert fake_kaggle.count("pushes") == 3


def test_failed_music_run_exits_2(real, fake_kaggle):
    fake_kaggle.scenario(mode="fail")
    assert music.run(real, sb_of(real), NS(draft=False, resubmit=False)) == 2


def test_music_failures_are_logged_per_candidate(real, fake_kaggle):
    fake_kaggle.scenario(mode="fail")
    assert music.run(real, sb_of(real), NS(draft=False, resubmit=False)) == 2
    lines = error_lines(real, "music")
    assert len(lines) == 3 and all(r["shot"] is None and r["model"] == music.MODEL for r in lines)


def test_candidates_that_are_too_short_are_rejected(real, fake_kaggle):
    fake_kaggle.scenario(mode="short")
    assert music.run(real, sb_of(real), NS(draft=False, resubmit=False)) == 2
    assert not list((real.out / "music" / "candidates").glob("*.wav"))


def test_ingesting_music_twice_logs_once_and_lists_the_run_once(real, fake_kaggle):
    ctx = real
    sb = sb_of(ctx)
    items = [{"id": f"candidate_{n}", "fingerprint": music.candidate_fp(sb, n), "inputs": [],
              "params": {"caption": "c", "duration": 20.0, "seed": 42 + n - 1}} for n in (1, 2, 3)]
    res = kaggle.collect(ctx, "music", items, {})
    manifest = Manifest(ctx)
    music.ingest(ctx, sb, res, manifest)
    music.ingest(ctx, sb, res, manifest)
    assert len(runs(ctx, "music")) == 3
    assert json.loads((ctx.out / "music" / "candidates.json").read_text())["run_ids_logged"] == [res.run_id]


def test_a_choice_goes_stale_when_the_chosen_candidate_is_regenerated_with_different_audio(real, fake_kaggle, capsys):
    ctx = real
    fake_kaggle.scenario(vary=True)
    choose(ctx, 2, "warmest strings")
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 0
    before = (ctx.out / "music" / "score.wav").read_bytes()
    (ctx.out / "music" / "candidates" / "candidate_3.wav").unlink()   # a lost file: all three are regenerated
    capsys.readouterr()
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 3
    assert fake_kaggle.count("pushes") == 2 and "choose again" in capsys.readouterr().out
    assert (ctx.out / "music" / "score.wav").read_bytes() == before, "the old score is not silently replaced"
    assert not music.score_ready(ctx, sb_of(ctx), Manifest(ctx))
    choose(ctx, 2, "warmest strings, second listen")
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 0
    assert music.score_ready(ctx, sb_of(ctx), Manifest(ctx))
    assert (ctx.out / "music" / "score.wav").read_bytes() != before


def test_the_mock_auto_choice_follows_regenerated_candidates(ctx, capsys):
    music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False))
    edit(ctx, lambda sb: [s.update(duration=1.0) for s in sb["shots"]])   # film length bucket 20 s -> 10 s
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 0
    assert "choose again" not in capsys.readouterr().out
    assert music.score_ready(ctx, sb_of(ctx), Manifest(ctx))
    assert music.run(ctx, sb_of(ctx), NS(draft=True, resubmit=False)) == 0


def test_resuming_after_an_edit_keeps_the_paid_for_clips_that_still_match(real, fake_kaggle):
    ctx = real
    fake_kaggle.scenario(mode="never")
    assert shots.run(ctx, sb_of(ctx), NS(fallback=True, resubmit=False, retry_fallbacks=False)) == 0
    edit(ctx, lambda sb: sb["shots"][0].update(motion_prompt="orbit slowly"))   # only s01 changes
    fake_kaggle.scenario(mode="ok", running_polls=0)
    shots.run(ctx, sb_of(ctx), NS(fallback=True, resubmit=False, retry_fallbacks=False))
    assert sidecar(ctx, "s02")["kind"] == "real" and sidecar(ctx, "s03")["kind"] == "real"
    assert sidecar(ctx, "s01")["kind"] == "fallback", "s01's result belongs to the old prompt"
    assert fake_kaggle.count("pushes") == 1
    assert [r["shot"] for r in error_lines(ctx, "shots")] == ["s01"]


def test_only_does_not_throw_away_the_rest_of_a_resumed_run(real, fake_kaggle):
    ctx = real
    fake_kaggle.scenario(mode="never")
    shots.run(ctx, sb_of(ctx), NS(fallback=True, resubmit=False, retry_fallbacks=False))
    fake_kaggle.scenario(mode="ok", running_polls=0)
    ctx.only = {"s01"}
    shots.run(ctx, sb_of(ctx), NS(fallback=True, resubmit=False, retry_fallbacks=True))
    assert all(sidecar(ctx, i)["kind"] == "real" for i in ("s01", "s02", "s03"))


def test_a_music_run_that_delivers_on_resume_is_logged(real, fake_kaggle):
    ctx = real
    fake_kaggle.scenario(mode="no_results")
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 2
    assert runs(ctx, "music") == []
    fake_kaggle.scenario(mode="ok", running_polls=0)
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 3
    assert [r["status"] for r in runs(ctx, "music")] == ["ok", "ok", "ok"] and fake_kaggle.count("pushes") == 1


def test_a_new_choice_made_in_the_same_edit_as_a_regeneration_is_not_applied_blind(real, fake_kaggle, capsys):
    ctx = real
    fake_kaggle.scenario(vary=True)
    choose(ctx, 2, "warmest strings")
    music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False))
    before = (ctx.out / "music" / "score.wav").read_bytes()
    edit(ctx, lambda sb: sb["music"].update(prompt="fast drums", choice={"candidate": 3, "reason": "brighter"}))
    capsys.readouterr()
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 3
    assert "choose again" in capsys.readouterr().out
    assert (ctx.out / "music" / "score.wav").read_bytes() == before
    choose(ctx, 3, "brighter, after listening")
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 0
    assert json.loads((ctx.out / "music" / "score.json").read_text())["candidate"] == 3


def test_a_pending_run_is_announced_when_every_shot_already_has_a_fallback(real, fake_kaggle, capsys):
    ctx = real
    fake_kaggle.scenario(mode="never")
    shots.run(ctx, sb_of(ctx), NS(fallback=True, resubmit=False, retry_fallbacks=False))
    assert (ctx.out / "jobs" / "shots" / "pending.json").exists()
    capsys.readouterr()
    assert shots.run(ctx, sb_of(ctx), shots_args()) == 0
    out = capsys.readouterr().out
    assert "shots: a Kaggle run is pending (outputs/jobs/shots/pending.json); --retry-fallbacks collects it" in out


def test_no_pending_note_without_a_pending_run(real, fake_kaggle, capsys):
    ctx = real
    shots.run(ctx, sb_of(ctx), shots_args())
    capsys.readouterr()
    shots.run(ctx, sb_of(ctx), shots_args())
    assert "is pending" not in capsys.readouterr().out


def test_a_choice_made_before_listening_stays_stale_on_a_plain_rerun(real, fake_kaggle, capsys):
    ctx = real
    fake_kaggle.scenario(vary=True)
    choose(ctx, 2, "warmest strings")
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 0
    before = (ctx.out / "music" / "score.wav").read_bytes()
    edit(ctx, lambda sb: sb["music"].update(prompt="fast drums", choice={"candidate": 3, "reason": "brighter"}))
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 3
    info = json.loads((ctx.out / "music" / "score.json").read_text())
    assert info["awaiting_choice"] == {"candidate": 3, "reason": "brighter"}
    assert info["candidate"] == 2 and info["reason"] == "warmest strings"     # the old score's record is kept
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 3    # plain re-run: still stale
    assert music.run(ctx, sb_of(ctx), NS(draft=True, resubmit=False)) == 0     # --draft continues without it
    assert (ctx.out / "music" / "score.wav").read_bytes() == before
    choose(ctx, 3, "brighter, after listening")
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 0
    info = json.loads((ctx.out / "music" / "score.json").read_text())
    assert info["candidate"] == 3 and "awaiting_choice" not in info


def test_the_first_generation_without_a_score_writes_no_score_record(real, fake_kaggle):
    ctx = real
    choose(ctx, 2, "picked before hearing")
    assert music.run(ctx, sb_of(ctx), NS(draft=False, resubmit=False)) == 0
    assert json.loads((ctx.out / "music" / "score.json").read_text())["candidate"] == 2
