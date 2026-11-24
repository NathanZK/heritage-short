import json
import os
import threading

import pytest

from pipeline import assemble, common
from pipeline.common import (Ctx, Manifest, PipelineExit, ProviderError, call_model, commit_bytes, fingerprint,
                             load_env, log_run, read_runs, redact)


def test_fingerprint_is_stable_and_order_independent():
    assert fingerprint({"a": 1, "b": [1, 2]}) == fingerprint({"b": [1, 2], "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})
    assert len(fingerprint("x")) == 16


def test_timeline_is_frame_aligned():
    sb = {"shots": [{"id": "a", "duration": 4.9}, {"id": "b", "duration": 4.0}, {"id": "c", "duration": 5.0}]}
    assert common.shot_frames(sb["shots"][0]) == 118 and common.shot_seconds(sb["shots"][0]) == 118 / 24
    starts = common.shot_starts(sb)
    assert starts == {"a": 0.0, "b": 118 / 24, "c": 118 / 24 + 4.0}


def test_shot_seed_follows_the_spec_formula_and_survives_inserting_shots():
    import zlib
    sb = {"seed": 7}
    assert common.shot_seed(sb, {"id": "s01"}) == (7 + zlib.crc32(b"s01")) % 2**31
    assert common.shot_seed(sb, {"id": "s01"}) != common.shot_seed(sb, {"id": "s02"})
    assert common.shot_seed({"seed": 8}, {"id": "s01"}) == common.shot_seed(sb, {"id": "s01"}) + 1


def test_ctx_separates_mock_from_real(tmp_path):
    real, mock = Ctx(root=tmp_path), Ctx(root=tmp_path, mock=True)
    assert real.out == tmp_path / "outputs" and real.runs_path == tmp_path / "runs.jsonl"
    assert mock.out == tmp_path / "outputs" / "mock" and mock.runs_path == tmp_path / "outputs" / "mock" / "runs.jsonl"


def test_mock_storyboard_prefers_mock_copy(tmp_path):
    (tmp_path / "storyboard.json").write_text('{"shots": []}')
    mock = Ctx(root=tmp_path, mock=True)
    assert mock.storyboard_path == tmp_path / "storyboard.json"
    (tmp_path / "outputs" / "mock").mkdir(parents=True)
    (tmp_path / "outputs" / "mock" / "storyboard.json").write_text("{}")
    assert mock.storyboard_path == tmp_path / "outputs" / "mock" / "storyboard.json"
    assert Ctx(root=tmp_path).storyboard_path == tmp_path / "storyboard.json"


def test_mock_copy_never_shadows_a_storyboard_that_has_shots(tmp_path):
    (tmp_path / "storyboard.json").write_text(json.dumps({"shots": [{"id": "s01"}]}))
    (tmp_path / "outputs" / "mock").mkdir(parents=True)
    (tmp_path / "outputs" / "mock" / "storyboard.json").write_text(json.dumps({"shots": [{"id": "old"}]}))
    assert Ctx(root=tmp_path, mock=True).storyboard_path == tmp_path / "storyboard.json"


def test_commit_writes_atomically_and_keeps_old_on_invalid(tmp_path):
    target = tmp_path / "out.bin"
    commit_bytes(target, b"good")
    assert target.read_bytes() == b"good"

    def reject(path):
        raise ProviderError("invalid output")

    with pytest.raises(ProviderError):
        commit_bytes(target, b"bad", validate=reject)
    assert target.read_bytes() == b"good"
    assert [p.name for p in tmp_path.iterdir()] == ["out.bin"]


def test_cleanup_removes_orphan_temp_files(ctx):
    orphan = ctx.path("narration", ".s01.wav.tmp-123")
    orphan.write_bytes(b"x")
    common.cleanup_tmp(ctx)
    assert not orphan.exists()


def test_manifest_freshness_needs_fingerprint_file_and_hash(ctx):
    out = ctx.path("keyframes", "s01.jpg")
    commit_bytes(out, b"image")
    m = Manifest(ctx)
    assert not m.fresh("k", "fp1", out)
    m.record("k", "fp1", out)
    assert Manifest(ctx).fresh("k", "fp1", out)
    assert not Manifest(ctx).fresh("k", "fp2", out)
    out.write_bytes(b"tampered")
    assert not Manifest(ctx).fresh("k", "fp1", out)
    out.unlink()
    assert not Manifest(ctx).fresh("k", "fp1", out)


def test_corrupt_manifest_exits_1(ctx):
    ctx.path("manifest.json").write_text("{not json")
    with pytest.raises(PipelineExit) as exc:
        Manifest(ctx)
    assert exc.value.code == 1


def test_lock_refuses_second_holder(ctx):
    with common.lock(ctx):
        with pytest.raises(PipelineExit) as exc:
            with common.lock(ctx):
                pass
        assert exc.value.code == 1
    with common.lock(ctx):
        pass


def test_runs_log_is_jsonl_and_tolerates_a_torn_line(ctx):
    log_run(ctx, stage="script", shot=None)
    log_run(ctx, stage="narration", shot="s01")
    with ctx.runs_path.open("a") as f:
        f.write('{"stage": "torn')
    runs = read_runs(ctx.runs_path)
    assert [r["stage"] for r in runs] == ["script", "narration"]
    assert all("ts" in r for r in runs)


def test_runs_log_appends_whole_lines_concurrently(ctx):
    def writer(i):
        for j in range(25):
            log_run(ctx, stage="s", shot=f"{i}-{j}", pad="x" * 200)
    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(read_runs(ctx.runs_path)) == 100


def test_redact_hides_secret_values(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-value")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct123456")
    text = "url /accounts/acct123456/ai failed with key secret-gemini-value Bearer abcdefghijklmnopqrstuvwxyz"
    cleaned = redact(text)
    assert "secret-gemini-value" not in cleaned and "acct123456" not in cleaned and "abcdefghijkl" not in cleaned
    assert cleaned.count("***") == 3


def test_redact_hides_google_style_keys_even_when_not_in_the_environment():
    assert redact("failed: AIza" + "A" * 35 + " end") == "failed: *** end"


def test_load_env_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# c\nFOO_KEY=from-file\nBAR_KEY='quoted'\n")
    monkeypatch.setenv("FOO_KEY", "real")
    monkeypatch.delenv("BAR_KEY", raising=False)
    load_env(env)
    assert os.environ["FOO_KEY"] == "real" and os.environ["BAR_KEY"] == "quoted"
    monkeypatch.delenv("BAR_KEY")


def call(ctx, fn, **kw):
    return call_model(ctx, stage="keyframes", shot="s01", provider="cloudflare", model="m",
                      params={"prompt": "p"}, seed=None, fn=fn, **kw)


def test_call_model_logs_one_line_per_attempt_and_returns(ctx):
    assert call(ctx, lambda: b"img") == b"img"
    (line,) = read_runs(ctx.runs_path)
    assert line["status"] == "ok" and line["attempt"] == 1 and line["seed"] is None
    assert line["cost_usd"] == 0.0 and line["list_price_usd"] is None
    assert {"stage", "shot", "provider", "model", "params", "latency_s"} <= set(line)


def test_retries_three_attempts_with_backoff(ctx):
    delays, attempts = [], []
    ctx.sleep = delays.append

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ProviderError("HTTP 503", retryable=True, status=503)
        return "ok"

    assert call(ctx, flaky) == "ok"
    assert delays == [2, 4]
    assert [r["status"] for r in read_runs(ctx.runs_path)] == ["error", "error", "ok"]
    assert [r["attempt"] for r in read_runs(ctx.runs_path)] == [1, 2, 3]


def test_gives_up_after_three_attempts_and_flags_429_as_quota(ctx):
    def always():
        raise ProviderError("HTTP 429", retryable=True, status=429)

    with pytest.raises(ProviderError) as exc:
        call(ctx, always)
    assert exc.value.quota
    assert len(read_runs(ctx.runs_path)) == 3


def test_non_retryable_and_quota_codes_are_not_retried(ctx):
    def quota():
        raise ProviderError("daily allocation 3036", quota=True, status=429)

    with pytest.raises(ProviderError):
        call(ctx, quota)
    assert len(read_runs(ctx.runs_path)) == 1

    def bad():
        raise ProviderError("HTTP 400", status=400)

    with pytest.raises(ProviderError):
        call(ctx, bad)
    assert len(read_runs(ctx.runs_path)) == 2


def test_error_text_is_redacted_in_the_log(ctx, monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok-very-secret")

    def bad():
        raise ProviderError("failed with tok-very-secret", status=400)

    with pytest.raises(ProviderError):
        call(ctx, bad)
    assert "tok-very-secret" not in ctx.runs_path.read_text()


def test_stage_main_exit_codes(root, capsys):
    ran = []

    def stage(ctx, sb, args):
        ran.append(1)
        return 0

    bad = json.loads((root / "storyboard.json").read_text())
    bad.pop("voice")
    (root / "storyboard.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit) as exc:
        common.stage_main(stage, "t", argv=["--mock", "--root", str(root)])
    assert exc.value.code == 1 and not ran and "voice" in capsys.readouterr().err


def test_stage_main_rejects_unknown_only_ids_and_propagates_codes(root):
    with pytest.raises(SystemExit) as exc:
        common.stage_main(lambda c, s, a: 0, "t", argv=["--mock", "--root", str(root), "--only", "nope"])
    assert exc.value.code == 1
    with pytest.raises(SystemExit) as exc:
        common.stage_main(lambda c, s, a: 3, "t", argv=["--mock", "--root", str(root)])
    assert exc.value.code == 3


def test_stage_main_maps_quota_to_2(root):
    def stage(ctx, sb, args):
        raise ProviderError("quota", quota=True)

    with pytest.raises(SystemExit) as exc:
        common.stage_main(stage, "t", argv=["--mock", "--root", str(root)])
    assert exc.value.code == 2


@pytest.mark.parametrize("call", [
    lambda: common.ffmpeg("-version"),
    lambda: common.probe("x.mp4"),
    lambda: assemble.available.__wrapped__("filters"),
    lambda: assemble.ffmpeg_version.__wrapped__(),
])
def test_a_missing_ffmpeg_is_a_clean_exit_1_not_a_traceback(monkeypatch, tmp_path, call):
    monkeypatch.setenv("PATH", str(tmp_path))   # an empty directory: no ffmpeg, no ffprobe
    with pytest.raises(PipelineExit) as exc:
        call()
    assert exc.value.code == 1 and "not found" in str(exc.value) and "install FFmpeg" in str(exc.value)
