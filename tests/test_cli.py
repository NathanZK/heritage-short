import json
import os
import subprocess
import sys

from tests.conftest import REPO


def cli(module, root, *args, env=None):
    return subprocess.run([sys.executable, "-m", f"pipeline.{module}", "--root", str(root), *args],
                          capture_output=True, text=True, cwd=REPO, env={**os.environ, **(env or {})})


def runs(root):
    path = root / "outputs" / "mock" / "runs.jsonl"
    return len(path.read_text().splitlines()) if path.exists() else 0


def test_run_mock_draft_end_to_end_and_second_run_is_free(root):
    first = cli("run", root, "--mock", "--draft")
    assert first.returncode == 0, first.stderr
    final = root / "outputs" / "mock" / "final"
    assert sorted(p.name for p in final.glob("film_*.mp4")) == ["film_16x9.mp4", "film_9x16.mp4"]
    n = runs(root)
    assert n == 13
    assert cli("run", root, "--mock", "--draft").returncode == 0 and runs(root) == n
    assert not (root / "runs.jsonl").exists(), "mock never touches the real log"
    assert not (root / "outputs" / "manifest.json").exists()


def test_exit_codes(root):
    assert cli("narration", root, "--mock", "--only", "nope").returncode == 1
    assert cli("assemble", root, "--mock", "--only", "s01").returncode == 1
    out = cli("keyframes", root, "--mock", "--only", "s01")
    assert out.returncode == 0 and "s01" in out.stdout
    before = runs(root)
    bad = json.loads((root / "storyboard.json").read_text())
    bad["voice"] = "Nobody"
    (root / "storyboard.json").write_text(json.dumps(bad))
    result = cli("run", root, "--mock", "--draft")
    assert result.returncode == 1 and "voice" in result.stderr and runs(root) == before


def test_assemble_without_prerequisites_names_the_stage(root):
    result = cli("assemble", root, "--mock", "--draft")
    assert result.returncode == 1 and "pipeline.keyframes" in result.stderr


def test_a_second_process_is_refused_while_the_lock_is_held(root):
    from pipeline.common import Ctx, lock
    with lock(Ctx(root=root, mock=True)):
        result = cli("narration", root, "--mock")
    assert result.returncode == 1 and "another pipeline process" in result.stderr


def test_script_mock_does_not_overwrite_the_real_storyboard(root):
    sb = json.loads((root / "storyboard.json").read_text())
    sb["shots"] = []
    (root / "storyboard.json").write_text(json.dumps(sb))
    assert cli("script", root, "--mock", "--draft").returncode == 0
    assert json.loads((root / "storyboard.json").read_text())["shots"] == []
    assert len(json.loads((root / "outputs" / "mock" / "storyboard.json").read_text())["shots"]) == 12
    assert cli("run", root, "--mock", "--draft").returncode == 0, "run continues from the mock storyboard"


def test_smoke_mock(root):
    assert cli("smoke", root, "--mock").returncode == 0 and runs(root) == 3


def test_real_run_without_credentials_names_the_variable(root):
    env = {"GEMINI_API_KEY": "", "CLOUDFLARE_API_TOKEN": "", "CLOUDFLARE_ACCOUNT_ID": ""}
    result = cli("keyframes", root, env=env)
    assert result.returncode == 1 and "CLOUDFLARE_ACCOUNT_ID" in result.stderr and ".env.example" in result.stderr
    assert not (root / "outputs" / "keyframes").exists() or not list((root / "outputs" / "keyframes").glob("*.jpg"))
