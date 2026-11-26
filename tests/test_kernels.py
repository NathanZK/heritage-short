import importlib.util
import json
import sys
from pathlib import Path

import pytest

from pipeline import kaggle, mock
from tests.conftest import REPO

UV_RUN = [sys.executable, "-m", "uv", "run", "python"]


def load(stage):
    spec = importlib.util.spec_from_file_location(f"{stage}_kernel", REPO / "kernels" / f"{stage}_kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["shots", "music"])
def stage(request):
    return request.param


def test_templates_compile_and_render_with_a_run_id(stage, tmp_path):
    rendered = kaggle.render_kernel(REPO / "kernels" / f"{stage}_kernel.py", "run-123", ["a", "b"])
    compile(rendered, stage, "exec")
    assert 'RUN_ID = "run-123"' in rendered and 'EXPECTED_IDS = ["a", "b"]' in rendered
    assert 'RUN_ID = "__RUN_ID__"' not in rendered


def write_job(root, run_id="r1", ids=("s01",), name="ds"):
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job.json").write_text(json.dumps({"run_id": run_id, "stage": "x", "items": [{"id": i} for i in ids]}))
    return folder


def configured(stage, run_id="r1", ids=("s01",)):
    module = load(stage)
    module.RUN_ID, module.EXPECTED_IDS = run_id, list(ids)
    return module


def test_kernel_finds_only_the_job_that_matches_its_run_id(stage, tmp_path):
    module = configured(stage)
    write_job(tmp_path, run_id="other", name="old")
    write_job(tmp_path, run_id="r1", ids=("zzz",), name="wrong-items")
    sleeps = []
    with pytest.raises(SystemExit):
        module.find_job(tmp_path, wait_s=30, poll_s=10, sleep=sleeps.append)
    assert sleeps == [10, 10, 10]
    good = write_job(tmp_path, name="new")
    folder, job = module.find_job(tmp_path, wait_s=30, poll_s=10, sleep=lambda s: None)
    assert folder == good and job["run_id"] == "r1"


def test_kernel_waits_for_the_dataset_to_appear(stage, tmp_path):
    module = configured(stage)
    calls = []

    def sleep(s):
        calls.append(s)
        write_job(tmp_path)

    folder, _ = module.find_job(tmp_path, wait_s=60, poll_s=5, sleep=sleep)
    assert calls == [5] and folder.name == "ds"


# Shots kernel tests: tests/test_shots_kernel.py


# ----------------------------------------------------------------- music

def music_job(seconds=70.0):
    return {"run_id": "r1", "stage": "music", "items": [
        {"id": f"candidate_{n}", "fingerprint": "fp", "params": {"caption": "calm strings", "duration": seconds,
                                                                 "seed": 42 + n - 1}} for n in (1, 2, 3)]}


def test_ace_step_parameters_follow_the_documented_api():
    module = load("music")
    assert module.build_generation(music_job()) == {
        "params": {"task_type": "text2music", "caption": "calm strings", "lyrics": "[Instrumental]",
                   "instrumental": True, "duration": 70.0, "inference_steps": 8, "shift": 3.0},
        "config": {"batch_size": 3, "use_random_seed": False, "seeds": [42, 43, 44], "audio_format": "wav"}}


def test_music_runner_is_valid_python():
    compile(load("music").RUNNER, "runner", "exec")


def test_music_main_names_candidates_in_returned_order(tmp_path):
    module = configured("music", ids=("candidate_1", "candidate_2", "candidate_3"))
    inputs, work = tmp_path / "in", tmp_path / "work"
    folder = inputs / "ds"
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(json.dumps(music_job()))

    def run(cmd, cwd=None, **kw):
        if cmd[:5] == UV_RUN:
            args = json.loads((Path(cwd) / cmd[-1]).read_text())
            paths = []
            for n in range(3):
                p = Path(args["save_dir"]) / f"gen-{n}.wav"
                p.write_bytes(mock.sine_wav(0.2, rate=8000, freq=100 + n))
                paths.append(str(p))
            Path(args["paths_file"]).write_text(json.dumps(paths))
        return type("R", (), {"returncode": 0})()

    module.main(root=inputs, work=work, scratch=tmp_path / "scratch", run=run, sleep=lambda s: None, install=False)
    results = json.loads((work / "results.json").read_text())
    assert [i["id"] for i in results["items"]] == ["candidate_1", "candidate_2", "candidate_3"]
    assert all(i["status"] == "ok" and (work / i["file"]).exists() for i in results["items"])
    assert results["items"][1]["seed"] == 43
    assert (work / "candidate_1.wav").read_bytes() != (work / "candidate_2.wav").read_bytes()


def test_music_runner_uses_a_non_interactive_matplotlib_backend(tmp_path, monkeypatch):
    """Kaggle sets MPLBACKEND to the Jupyter inline backend, which ACE-Step's import of matplotlib rejects
    (seen in the owner's ace-step-music-test notebook); the runner must get Agg, and keep the rest of the env."""
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    monkeypatch.setenv("HERITAGE_KEEP_ME", "1")
    module = configured("music", ids=("candidate_1", "candidate_2", "candidate_3"))
    inputs = tmp_path / "in" / "ds"
    inputs.mkdir(parents=True)
    (inputs / "job.json").write_text(json.dumps(music_job()))
    seen = {}

    def run(cmd, cwd=None, env=None, **kw):
        if cmd[:5] == UV_RUN:
            seen.update(env or {})
        return type("R", (), {"returncode": 1})()

    module.main(root=tmp_path / "in", work=tmp_path / "work", scratch=tmp_path / "scratch", run=run,
                sleep=lambda s: None, install=False)
    assert seen.get("MPLBACKEND") == "Agg" and seen.get("HERITAGE_KEEP_ME") == "1"


class Done:
    returncode = 0


def realistic_runner(commands):
    """Behaves like subprocess in the ways that matter: cwd must exist, clones create directories."""
    def run(cmd, cwd=None, **kw):
        assert cwd and Path(cwd).is_dir(), f"cwd {cwd} does not exist for {cmd}"
        commands.append((cmd, Path(cwd)))
        if cmd[:2] == ["git", "clone"]:
            repo = Path(cwd) / cmd[2].rsplit("/", 1)[1]
            repo.mkdir()
            (repo / "requirements.txt").write_text("torch>=2.4.0\n")
        if cmd[:5] == UV_RUN:
            args = json.loads((Path(cwd) / cmd[-1]).read_text())
            paths = []
            for n in range(3):
                p = Path(args["save_dir"]) / f"gen-{n}.wav"
                p.write_bytes(mock.sine_wav(0.2, rate=8000, freq=100 + n))
                paths.append(str(p))
            Path(args["paths_file"]).write_text(json.dumps(paths))
        return Done()
    return run


def test_ace_step_installs_in_scratch_and_only_deliverables_reach_the_output_folder(tmp_path):
    module = configured("music", ids=("candidate_1", "candidate_2", "candidate_3"))
    inputs, work, scratch = tmp_path / "in", tmp_path / "work", tmp_path / "scratch"
    (inputs / "ds").mkdir(parents=True)
    (inputs / "ds" / "job.json").write_text(json.dumps(music_job()))
    commands = []
    module.main(root=inputs, work=work, scratch=scratch, run=realistic_runner(commands), sleep=lambda s: None)
    assert sorted(p.name for p in work.iterdir()) == ["candidate_1.wav", "candidate_2.wav", "candidate_3.wav",
                                                       "results.json"]
    assert (scratch / "ACE-Step-1.5").is_dir()
    assert all(cwd == scratch or scratch in cwd.parents for _, cwd in commands)


def test_kernel_output_flows_through_judge_and_ingest(tmp_path, clip_fixture, ctx):
    """The kernels' own results.json must satisfy the code that ingests it."""
    from pipeline import kaggle, music, shots
    from tests.test_shots_kernel import FakeWan, wan_job
    module = configured("shots")
    work = tmp_path / "work"
    module.main(root=wan_job(tmp_path, ids=("s01",), num_frames=121), work=work, scratch=tmp_path / "s",
                run=lambda *a, **k: Done(), sleep=lambda s: None, backend=FakeWan(clip=clip_fixture))
    results = json.loads((work / "results.json").read_text())
    path, reason = kaggle.judge(results["items"][0], {"s01": "fp-s01"}, "r1", work, results["run_id"])
    assert path and reason is None
    res = kaggle.Collected("r1", accepted=[(results["items"][0], path)])
    assert shots.ingest(ctx, res, {"s01": "fp-s01"}) == ["s01"]

    music_module = configured("music", ids=("candidate_1", "candidate_2", "candidate_3"))
    mwork = tmp_path / "mwork"
    (tmp_path / "min" / "ds").mkdir(parents=True)
    mjob = music_job(2.0)
    (tmp_path / "min" / "ds" / "job.json").write_text(json.dumps(mjob))
    music_module.main(root=tmp_path / "min", work=mwork, scratch=tmp_path / "ms", run=realistic_runner([]),
                      sleep=lambda s: None, install=False)
    mres = json.loads((mwork / "results.json").read_text())
    accepted = [(i, kaggle.judge(i, {x["id"]: "fp" for x in mjob["items"]}, "r1", mwork, mres["run_id"])[0])
                for i in mres["items"]]
    assert all(p for _, p in accepted)
    sb = json.loads((ctx.root / "storyboard.json").read_text())
    music.ingest(ctx, sb, kaggle.Collected("r1", accepted=accepted), music.Manifest(ctx))


def test_ace_step_setup_uses_this_python_for_uv_so_path_does_not_matter(tmp_path):
    module = configured("music", ids=("candidate_1", "candidate_2", "candidate_3"))
    (tmp_path / "in" / "ds").mkdir(parents=True)
    (tmp_path / "in" / "ds" / "job.json").write_text(json.dumps(music_job()))
    commands = []
    module.main(root=tmp_path / "in", work=tmp_path / "work", scratch=tmp_path / "scratch",
                run=realistic_runner(commands), sleep=lambda s: None)
    assert [cmd for cmd, _ in commands] == [
        [sys.executable, "-m", "pip", "install", "uv"],
        ["git", "clone", "https://github.com/ACE-Step/ACE-Step-1.5"],
        [sys.executable, "-m", "uv", "sync"],
        UV_RUN + ["runner.py", "args.json"]]
    source = (REPO / "kernels" / "music_kernel.py").read_text()
    assert "curl" not in source and "| sh" not in source


@pytest.mark.parametrize("failing,step", [("pip", "pip install uv"), ("git", "git clone"), ("sync", "uv sync")])
def test_a_failed_ace_step_setup_step_reports_every_item_and_generates_nothing(tmp_path, failing, step):
    module = configured("music", ids=("candidate_1", "candidate_2", "candidate_3"))
    (tmp_path / "in" / "ds").mkdir(parents=True)
    (tmp_path / "in" / "ds" / "job.json").write_text(json.dumps(music_job()))
    commands, real_run = [], realistic_runner([])

    def run(cmd, cwd=None, **kw):
        commands.append(cmd)
        if failing in cmd:
            return type("R", (), {"returncode": 7, "stdout": "", "stderr": "boom: no space left"})()
        return real_run(cmd, cwd=cwd, **kw)

    module.main(root=tmp_path / "in", work=tmp_path / "work", scratch=tmp_path / "scratch", run=run,
                sleep=lambda s: None)
    results = json.loads((tmp_path / "work" / "results.json").read_text())
    assert [i["id"] for i in results["items"]] == ["candidate_1", "candidate_2", "candidate_3"]
    for item in results["items"]:
        assert item["status"] == "error" and step in item["error"] and "boom: no space left" in item["error"]
        assert item["fingerprint"] == "fp"
    assert not any(cmd[:5] == UV_RUN for cmd in commands)
    assert sorted(p.name for p in (tmp_path / "work").iterdir()) == ["results.json"]
