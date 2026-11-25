import hashlib
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


# ---------------------------------------------------------------- shots

PIP = [sys.executable, "-m", "pip"]


def test_wan_command_uses_exactly_the_documented_flags():
    module = load("shots")
    item = {"id": "s01", "params": {"prompt": "slow push in", "seed": 99}}
    assert module.build_generate_cmd(item, "/in/s01.jpg", "/out/s01.mp4") == [
        sys.executable, "generate.py", "--task", "ti2v-5B", "--size", "1280*704", "--ckpt_dir", "./Wan2.2-TI2V-5B",
        "--offload_model", "True", "--convert_model_dtype", "--t5_cpu", "--image", "/in/s01.jpg",
        "--prompt", "slow push in", "--base_seed", "99", "--frame_num", "121", "--save_file", "/out/s01.mp4"]


def test_wan_install_steps_and_flash_attn_fallback(tmp_path):
    module = load("shots")
    calls = []
    (tmp_path / "Wan2.2").mkdir()
    (tmp_path / "Wan2.2" / "requirements.txt").write_text("torch>=2.4.0\nflash_attn\neasydict\n")

    def run(cmd, cwd=None, **kw):
        calls.append(cmd)
        return type("R", (), {"returncode": 1 if cmd[:5] == PIP + ["install", "-r"] and len(
            [c for c in calls if c[:5] == PIP + ["install", "-r"]]) == 1 else 0})()

    assert module.install_wan(run, tmp_path) == ""
    assert calls[0] == ["git", "clone", "https://github.com/Wan-Video/Wan2.2"]
    assert ["huggingface-cli", "download", "Wan-AI/Wan2.2-TI2V-5B", "--local-dir", "./Wan2.2-TI2V-5B"] in calls
    assert PIP + ["install", "flash_attn"] in calls
    assert "flash_attn" not in (tmp_path / "Wan2.2" / "requirements-noflash.txt").read_text()


def fake_wan(commands):
    def run(cmd, cwd=None, **kw):
        commands.append(cmd)
        if "--save_file" in cmd:
            out = Path(cmd[cmd.index("--save_file") + 1])
            if "bad" in out.name:
                return type("R", (), {"returncode": 1})()
            out.write_bytes(b"video-bytes-" + out.name.encode())
        return type("R", (), {"returncode": 0})()
    return run


def test_shots_main_writes_results_after_every_item(tmp_path):
    module = configured("shots", ids=("s01", "bad"))
    inputs, work = tmp_path / "in", tmp_path / "work"
    folder = write_job(inputs, ids=("s01", "bad"))
    job = json.loads((folder / "job.json").read_text())
    for it in job["items"]:
        it.update(fingerprint=f"fp-{it['id']}", inputs=[f"{it['id']}.jpg"], params={"prompt": "p", "seed": 5})
        (folder / f"{it['id']}.jpg").write_bytes(b"img")
    (folder / "job.json").write_text(json.dumps(job))
    commands = []
    module.main(root=inputs, work=work, scratch=tmp_path / "scratch", run=fake_wan(commands), sleep=lambda s: None,
                install=False)
    results = json.loads((work / "results.json").read_text())
    assert results["run_id"] == "r1" and results["stage"] == "shots"
    ok, failed = results["items"]
    assert ok["id"] == "s01" and ok["status"] == "ok" and ok["fingerprint"] == "fp-s01" and ok["seed"] == 5
    assert ok["sha256"] == hashlib.sha256((work / "s01.mp4").read_bytes()).hexdigest()
    assert failed["id"] == "bad" and failed["status"] == "error"
    assert len(commands) == 2


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
        if "--save_file" in cmd:
            Path(cmd[cmd.index("--save_file") + 1]).write_bytes(b"video")
        return Done()
    return run


def test_wan_installs_in_scratch_and_only_deliverables_reach_the_output_folder(tmp_path):
    module = configured("shots")
    inputs, work, scratch = tmp_path / "in", tmp_path / "work", tmp_path / "scratch"
    folder = write_job(inputs)
    job = json.loads((folder / "job.json").read_text())
    job["items"][0].update(fingerprint="fp", inputs=["s01.jpg"], params={"prompt": "p", "seed": 1})
    (folder / "job.json").write_text(json.dumps(job))
    (folder / "s01.jpg").write_bytes(b"img")
    commands = []
    module.main(root=inputs, work=work, scratch=scratch, run=realistic_runner(commands), sleep=lambda s: None)
    assert sorted(p.name for p in work.iterdir()) == ["results.json", "s01.mp4"]
    assert (scratch / "Wan2.2").is_dir()
    assert all(cwd == scratch or scratch in cwd.parents for _, cwd in commands)


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
    module = configured("shots")
    inputs, work = tmp_path / "in", tmp_path / "work"
    folder = write_job(inputs)
    job = json.loads((folder / "job.json").read_text())
    job["items"][0].update(fingerprint="fp-s01", inputs=["s01.jpg"], params={"prompt": "p", "seed": 5})
    (folder / "job.json").write_text(json.dumps(job))
    (folder / "s01.jpg").write_bytes(b"img")

    def run(cmd, cwd=None, **kw):
        if "--save_file" in cmd:
            import shutil
            shutil.copyfile(clip_fixture, cmd[cmd.index("--save_file") + 1])
        return Done()

    module.main(root=inputs, work=work, scratch=tmp_path / "s", run=run, sleep=lambda s: None, install=False)
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


def wan_job(tmp_path):
    inputs = tmp_path / "in"
    folder = write_job(inputs, ids=("s01", "s02"))
    job = json.loads((folder / "job.json").read_text())
    for it in job["items"]:
        it.update(fingerprint=f"fp-{it['id']}", inputs=[f"{it['id']}.jpg"], params={"prompt": "p", "seed": 1})
        (folder / f"{it['id']}.jpg").write_bytes(b"img")
    (folder / "job.json").write_text(json.dumps(job))
    return inputs


def test_wan_setup_runs_the_documented_steps_with_this_python_so_path_does_not_matter(tmp_path):
    module = configured("shots", ids=("s01", "s02"))
    commands = []
    module.main(root=wan_job(tmp_path), work=tmp_path / "work", scratch=tmp_path / "scratch",
                run=realistic_runner(commands), sleep=lambda s: None)
    setup = [cmd for cmd, _ in commands if "--save_file" not in cmd]
    assert setup == [
        ["git", "clone", "https://github.com/Wan-Video/Wan2.2"],
        PIP + ["install", "-r", "requirements.txt"],
        PIP + ["install", "huggingface_hub[cli]"],
        ["huggingface-cli", "download", "Wan-AI/Wan2.2-TI2V-5B", "--local-dir", "./Wan2.2-TI2V-5B"]]
    generated = [cmd for cmd, _ in commands if "--save_file" in cmd]
    assert len(generated) == 2 and all(cmd[0] == sys.executable for cmd in generated)


@pytest.mark.parametrize("failing,step", [
    (lambda c: c[:2] == ["git", "clone"], "git clone"),
    (lambda c: "-r" in c, "pip install requirements"),
    (lambda c: "huggingface_hub[cli]" in c, "pip install huggingface_hub"),
    (lambda c: c[0] == "huggingface-cli", "download weights"),
])
def test_a_failed_wan_setup_step_reports_every_shot_and_generates_nothing(tmp_path, failing, step):
    module = configured("shots", ids=("s01", "s02"))
    commands, real_run = [], realistic_runner([])

    def run(cmd, cwd=None, **kw):
        commands.append(cmd)
        if failing(cmd):
            return type("R", (), {"returncode": 9, "stdout": "", "stderr": "boom: no space left"})()
        return real_run(cmd, cwd=cwd, **kw)

    module.main(root=wan_job(tmp_path), work=tmp_path / "work", scratch=tmp_path / "scratch", run=run,
                sleep=lambda s: None)
    results = json.loads((tmp_path / "work" / "results.json").read_text())
    assert [i["id"] for i in results["items"]] == ["s01", "s02"]
    for item in results["items"]:
        assert item["status"] == "error" and step in item["error"] and "boom: no space left" in item["error"]
        assert item["fingerprint"] == f"fp-{item['id']}"
    assert not any("--save_file" in cmd for cmd in commands)
    assert sorted(p.name for p in (tmp_path / "work").iterdir()) == ["results.json"]


def test_a_failing_optional_flash_attn_install_does_not_stop_wan_setup(tmp_path):
    module = configured("shots", ids=("s01", "s02"))
    commands, real_run = [], realistic_runner([])
    first_requirements = []

    def run(cmd, cwd=None, **kw):
        commands.append(cmd)
        if cmd[:5] == PIP + ["install", "-r"] and not first_requirements:
            first_requirements.append(1)
            return type("R", (), {"returncode": 1, "stdout": "", "stderr": "flash_attn build failed"})()
        if cmd == PIP + ["install", "flash_attn"]:
            return type("R", (), {"returncode": 1, "stdout": "", "stderr": "no wheel for sm75"})()
        return real_run(cmd, cwd=cwd, **kw)

    module.main(root=wan_job(tmp_path), work=tmp_path / "work", scratch=tmp_path / "scratch", run=run,
                sleep=lambda s: None)
    results = json.loads((tmp_path / "work" / "results.json").read_text())
    assert [i["status"] for i in results["items"]] == ["ok", "ok"]
