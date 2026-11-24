import json

import pytest

from pipeline import kaggle
from pipeline.common import Ctx, PipelineExit, read_runs


@pytest.fixture
def kctx(root, fake_kaggle):
    return Ctx(root=root, sleep=lambda s: None, poll_s=10, remote_timeout_s=100)


def job(ctx, ids=("s01", "s02"), stage="shots"):
    inputs = {f"{i}.jpg": _write(ctx, f"{i}.jpg") for i in ids}
    items = [{"id": i, "fingerprint": f"fp-{i}", "params": {"prompt": "p", "seed": 1, "duration": 3.0},
              "inputs": [f"{i}.jpg"]} for i in ids]
    return stage, items, inputs


def _write(ctx, name):
    p = ctx.root / "src" / name
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(b"image-" + name.encode())
    return p


@pytest.mark.parametrize("text,expected", [
    ('user/k has status "KernelWorkerStatus.RUNNING"', "running"),
    ('user/k has status "KernelWorkerStatus.COMPLETE"', "finished"),
    ("complete", "finished"),
    ('status "KernelWorkerStatus.ERROR"', "failed"),
    ("Failed", "failed"),
    ("cancelled", "failed"),
    ("incomplete upload, still queued", "running"),
    ("completed with error", "failed"),
    ("queued", "running"),
])
def test_status_classification(text, expected):
    assert kaggle.classify(text) == expected


def test_happy_path_runs_the_documented_commands(kctx, fake_kaggle):
    stage, items, inputs = job(kctx)
    res = kaggle.collect(kctx, stage, items, inputs)
    assert sorted(i["id"] for i, _ in res.accepted) == ["s01", "s02"] and not res.rejected
    calls = fake_kaggle.calls
    assert calls[0].startswith("datasets version -p ") and f"-m {res.run_id}" in calls[0]
    push = [c for c in calls if c.startswith("kernels push")][0]
    assert "--accelerator NvidiaTeslaT4" in push and "-t 100" in push
    assert sum(c.startswith("kernels status tester/heritage-short-shots") for c in calls) == 3
    assert any(c.startswith("kernels output tester/heritage-short-shots") and "-o" in c for c in calls)
    assert (kctx.out / "jobs" / "shots" / "pending.json").exists(), "kept until the caller has ingested"
    kaggle.finish(kctx, stage)
    assert not (kctx.out / "jobs" / "shots" / "pending.json").exists()


def test_dataset_and_kernel_folders_follow_the_spec(kctx, fake_kaggle):
    stage, items, inputs = job(kctx)
    stale = kctx.out / "jobs" / "shots" / "dataset"
    stale.mkdir(parents=True)
    (stale / "leftover.txt").write_text("old run")
    kaggle.collect(kctx, stage, items, inputs)
    dataset = fake_kaggle.state_dir / "dataset"
    assert sorted(p.name for p in dataset.iterdir()) == ["dataset-metadata.json", "job.json", "s01.jpg", "s02.jpg"]
    meta = json.loads((dataset / "dataset-metadata.json").read_text())
    assert meta["id"] == "tester/heritage-short-shots-jobs" and 6 <= len(meta["title"]) <= 50
    job_json = json.loads((dataset / "job.json").read_text())
    assert [i["id"] for i in job_json["items"]] == ["s01", "s02"] and job_json["stage"] == "shots"
    kernel = json.loads((fake_kaggle.state_dir / "kernel.json").read_text())["meta"]
    assert kernel["id"] == "tester/heritage-short-shots" and kernel["title"] == "heritage-short-shots"
    assert kernel["kernel_type"] == "script" and kernel["enable_gpu"] and kernel["enable_internet"]
    assert kernel["machine_shape"] == "NvidiaTeslaT4" and kernel["is_private"]
    assert kernel["dataset_sources"] == ["tester/heritage-short-shots-jobs"]
    kernel_dir = kctx.out / "jobs" / "shots" / "kernel"
    code = (kernel_dir / kernel["code_file"]).read_text()
    assert f'RUN_ID = "{job_json["run_id"]}"' in code and 'EXPECTED_IDS = ["s01", "s02"]' in code


def test_no_secret_reaches_the_dataset_or_kernel_folder(kctx, fake_kaggle):
    stage, items, inputs = job(kctx)
    kaggle.collect(kctx, stage, items, inputs)
    for path in (kctx.out / "jobs").rglob("*"):
        if path.is_file() and path.suffix != ".jpg":
            assert "kaggle-key-secret-9876" not in path.read_text()


@pytest.mark.parametrize("mode,reason", [("wrong_run", "run_id"), ("bad_sha", "sha256"),
                                         ("bad_fingerprint", "fingerprint")])
def test_untrusted_results_are_rejected(kctx, fake_kaggle, mode, reason):
    fake_kaggle.scenario(mode=mode)
    stage, items, inputs = job(kctx)
    res = kaggle.collect(kctx, stage, items, inputs)
    assert not res.accepted and sorted(res.rejected) == ["s01", "s02"]
    assert all(reason in r for r in res.rejected.values())


def test_stale_incoming_files_are_cleared_first(kctx, fake_kaggle):
    fake_kaggle.scenario(mode="no_results")
    leftover = kctx.out / "_incoming" / "shots" / "results.json"
    leftover.parent.mkdir(parents=True)
    leftover.write_text(json.dumps({"run_id": "old", "items": []}))
    stage, items, inputs = job(kctx)
    res = kaggle.collect(kctx, stage, items, inputs)
    assert not leftover.exists() and not res.accepted


def test_unparseable_results_count_as_no_results(kctx, fake_kaggle):
    fake_kaggle.scenario(mode="bad_json")
    stage, items, inputs = job(kctx)
    res = kaggle.collect(kctx, stage, items, inputs)
    assert not res.accepted


def test_a_stray_results_json_inside_the_output_is_ignored(kctx, fake_kaggle):
    fake_kaggle.scenario(junk=True)
    stage, items, inputs = job(kctx)
    res = kaggle.collect(kctx, stage, items, inputs)
    assert sorted(i["id"] for i, _ in res.accepted) == ["s01", "s02"]


@pytest.mark.parametrize("body", [[], "text", {"items": None}, {"items": "x"}, {"items": [1, None]}])
def test_results_with_the_wrong_shape_count_as_no_results(kctx, fake_kaggle, body):
    fake_kaggle.scenario(mode="bad_shape", body=body)
    stage, items, inputs = job(kctx)
    res = kaggle.collect(kctx, stage, items, inputs)
    assert not res.accepted and not res.delivered


def test_failures_are_logged_only_for_items_that_were_in_the_run(kctx):
    items = [{"id": "s01", "params": {"seed": 1}}, {"id": "s02", "params": {"seed": 2}}]
    res = kaggle.Collected("r1", submitted=["s02"], state="failed")
    kaggle.log_failures(kctx, "shots", items, [], res, "model")
    assert [r["shot"] for r in read_runs(kctx.runs_path)] == ["s02"]


def test_partial_output_of_a_failed_run_is_kept(kctx, fake_kaggle):
    fake_kaggle.scenario(mode="partial")
    stage, items, inputs = job(kctx)
    res = kaggle.collect(kctx, stage, items, inputs)
    assert [i["id"] for i, _ in res.accepted] == ["s01"] and not res.rejected


def test_failed_run_without_results_marks_everything_missing(kctx, fake_kaggle):
    fake_kaggle.scenario(mode="fail")
    stage, items, inputs = job(kctx)
    res = kaggle.collect(kctx, stage, items, inputs)
    assert not res.accepted and not res.rejected


def test_a_finished_run_without_results_keeps_pending_so_a_rerun_resumes_it(kctx, fake_kaggle):
    fake_kaggle.scenario(mode="no_results")
    stage, items, inputs = job(kctx)
    res = kaggle.collect(kctx, stage, items, inputs)
    assert not res.accepted and not res.done
    assert (kctx.out / "jobs" / "shots" / "pending.json").exists()
    fake_kaggle.scenario(mode="ok", running_polls=0)
    res = kaggle.collect(kctx, stage, items, inputs)
    assert fake_kaggle.count("pushes") == 1 and res.done and len(res.accepted) == 2


def test_results_from_another_run_keep_pending(kctx, fake_kaggle):
    fake_kaggle.scenario(mode="wrong_run")
    stage, items, inputs = job(kctx)
    res = kaggle.collect(kctx, stage, items, inputs)
    assert not res.done and (kctx.out / "jobs" / "shots" / "pending.json").exists()


def test_a_failed_run_is_done_even_without_results(kctx, fake_kaggle):
    fake_kaggle.scenario(mode="fail")
    stage, items, inputs = job(kctx)
    assert kaggle.collect(kctx, stage, items, inputs).done


def test_timeout_keeps_pending_and_a_rerun_resumes_without_pushing_again(kctx, fake_kaggle):
    fake_kaggle.scenario(mode="never")
    stage, items, inputs = job(kctx)
    with pytest.raises(PipelineExit) as exc:
        kaggle.collect(kctx, stage, items, inputs)
    assert exc.value.code == 2 and "pending" in str(exc.value).lower()
    pending = json.loads((kctx.out / "jobs" / "shots" / "pending.json").read_text())
    assert pending["kernel"] == "tester/heritage-short-shots" and pending["item_ids"] == ["s01", "s02"]
    assert fake_kaggle.count("pushes") == 1
    fake_kaggle.scenario(mode="ok", running_polls=0)
    res = kaggle.collect(kctx, stage, items, inputs)
    assert fake_kaggle.count("pushes") == 1 and fake_kaggle.count("datasets") == 1
    assert sorted(i["id"] for i, _ in res.accepted) == ["s01", "s02"]


def test_resubmit_discards_a_pending_run(kctx, fake_kaggle):
    fake_kaggle.scenario(mode="never")
    stage, items, inputs = job(kctx)
    with pytest.raises(PipelineExit):
        kaggle.collect(kctx, stage, items, inputs)
    fake_kaggle.scenario(mode="ok", running_polls=0)
    kaggle.collect(kctx, stage, items, inputs, resubmit=True)
    assert fake_kaggle.count("pushes") == 2


def test_three_failed_status_polls_abort(kctx, fake_kaggle):
    fake_kaggle.scenario(status_breaks=True)
    stage, items, inputs = job(kctx)
    with pytest.raises(PipelineExit) as exc:
        kaggle.collect(kctx, stage, items, inputs)
    assert exc.value.code == 1 and sum(c.startswith("kernels status") for c in fake_kaggle.calls) == 3


def test_missing_cli_and_username_are_clear_errors(kctx, monkeypatch):
    stage, items, inputs = job(kctx)
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(PipelineExit) as exc:
        kaggle.collect(kctx, stage, items, inputs)
    assert exc.value.code == 1 and "kaggle" in str(exc.value).lower()
    monkeypatch.delenv("KAGGLE_USERNAME")
    with pytest.raises(PipelineExit) as exc:
        kaggle.collect(kctx, stage, items, inputs)
    assert "KAGGLE_USERNAME" in str(exc.value)


def test_cli_errors_are_redacted(kctx, fake_kaggle):
    fake_kaggle.scenario(leak=True)
    stage, items, inputs = job(kctx)
    with pytest.raises(PipelineExit) as exc:
        kaggle.collect(kctx, stage, items, inputs)
    assert exc.value.code == 1 and "auth failed" in str(exc.value) and "kaggle-key-secret-9876" not in str(exc.value)


def test_setup_creates_both_private_datasets(kctx, fake_kaggle):
    kaggle.setup(kctx)
    creates = [c for c in fake_kaggle.calls if c.startswith("datasets create")]
    assert len(creates) == 2 and not any("--public" in c or " -u" in c for c in creates)


def judged(tmp_path, file):
    incoming = tmp_path / "incoming"
    incoming.mkdir(exist_ok=True)
    (incoming / "ok.bin").write_bytes(b"data")
    (tmp_path / "outside.bin").write_bytes(b"data")
    digest = kaggle.sha256_file(tmp_path / "outside.bin")
    item = {"id": "s01", "status": "ok", "fingerprint": "fp", "file": file, "sha256": digest}
    return kaggle.judge(item, {"s01": "fp"}, "r1", incoming, "r1")


@pytest.mark.parametrize("name", ["../outside.bin", "sub/ok.bin", 5, None, "", "..", "/etc/hosts"])
def test_result_file_names_must_be_plain_names(tmp_path, name):
    path, reason = judged(tmp_path, name)
    assert path is None and "file name" in reason


def test_a_symlink_in_the_output_is_refused(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (tmp_path / "secret.bin").write_bytes(b"data")
    (incoming / "link.bin").symlink_to(tmp_path / "secret.bin")
    item = {"id": "s01", "status": "ok", "fingerprint": "fp", "file": "link.bin",
            "sha256": kaggle.sha256_file(tmp_path / "secret.bin")}
    path, reason = kaggle.judge(item, {"s01": "fp"}, "r1", incoming, "r1")
    assert path is None and "symlink" in reason


def test_a_plain_file_in_the_output_is_accepted(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "ok.bin").write_bytes(b"data")
    item = {"id": "s01", "status": "ok", "fingerprint": "fp", "file": "ok.bin",
            "sha256": kaggle.sha256_file(incoming / "ok.bin")}
    assert kaggle.judge(item, {"s01": "fp"}, "r1", incoming, "r1") == (incoming / "ok.bin", None)
