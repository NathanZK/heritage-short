import os
import re
import shutil
import subprocess
import sys

import pytest

from tests.conftest import REPO

PATTERNS = [r"AIza[0-9A-Za-z_-]{35}", r"Bearer\s+[A-Za-z0-9._-]{20,}", r"KAGGLE_KEY=(?!your-)[A-Za-z0-9]{8,}",
            r"(CLOUDFLARE_API_TOKEN|GEMINI_API_KEY)=(?!your-)[A-Za-z0-9_-]{8,}"]


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def test_env_is_ignored_and_the_example_has_placeholders_only():
    assert git("check-ignore", "-q", ".env").returncode == 0
    for line in (REPO / ".env.example").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            key, _, value = line.partition("=")
            assert value == "" or value.startswith("your-"), f"{key} must be a placeholder"
    names = {l.split("=")[0] for l in (REPO / ".env.example").read_text().splitlines() if "=" in l and l[0] != "#"}
    assert {"GEMINI_API_KEY", "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN", "KAGGLE_USERNAME",
            "KAGGLE_API_TOKEN"} <= names


def test_runs_log_is_tracked_and_not_ignored():
    assert git("check-ignore", "--no-index", "-q", "runs.jsonl").returncode == 1
    assert git("ls-files", "--error-unmatch", "runs.jsonl").returncode == 0


def test_generated_outputs_are_ignored_by_default():
    for path in ["outputs/manifest.json", "outputs/final/film_9x16.mp4", "outputs/keyframes/s01.jpg",
                 "outputs/narration/s01.wav", "outputs/timings.jsonl", "outputs/subtitles.srt"]:
        assert git("check-ignore", "--no-index", "-q", path).returncode == 0, path
    assert git("check-ignore", "--no-index", "-q", "outputs/.gitkeep").returncode == 1, "the folder stays"


def test_generated_and_private_paths_are_ignored():
    for path in ["outputs/jobs/x", "outputs/_incoming/x", "outputs/mock/x", "outputs/.lock", "storyboard.json.bak-1",
                 ".outputs.tmp-1"]:
        assert git("check-ignore", "-q", path).returncode == 0, path


def test_no_key_shaped_strings_in_tracked_or_new_files():
    files = git("ls-files", "-co", "--exclude-standard").stdout.split()
    hits = []
    for name in files:
        if name.startswith("tests/") or not (REPO / name).is_file():
            continue
        try:
            text = (REPO / name).read_text()
        except UnicodeDecodeError:
            continue
        hits += [f"{name}: {m.group(0)[:12]}" for p in PATTERNS for m in re.finditer(p, text)]
    assert not hits, hits


def test_make_film_runs_the_pipeline():
    out = subprocess.run(["make", "-n", "film"], cwd=REPO, capture_output=True, text=True).stdout
    assert "-m pipeline.run" in out
    assert "-m pytest" in subprocess.run(["make", "-n", "test"], cwd=REPO, capture_output=True, text=True).stdout


def test_a_fresh_copy_runs_make_film_and_tests(tmp_path):
    if os.environ.get("HERITAGE_NESTED"):
        pytest.skip("already inside the fresh-copy run")
    copy = tmp_path / "clone"
    shutil.copytree(REPO, copy, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".pytest_cache", ".venv", "mock", "runs.jsonl", ".DS_Store"))
    for stale in (copy / "outputs").glob("*"):   # a clean clone has no generated outputs
        if stale.name != ".gitkeep":
            shutil.rmtree(stale) if stale.is_dir() else stale.unlink()
    env = {**os.environ, "HERITAGE_NESTED": "1"}
    film = subprocess.run(["make", f"PY={sys.executable}", "film", "PIPELINE_FLAGS=--mock --draft"], cwd=copy,
                          capture_output=True, text=True, env=env)
    assert film.returncode == 0, film.stdout + film.stderr
    assert (copy / "outputs" / "mock" / "final" / "film_9x16.mp4").exists()
    tests = subprocess.run(["make", f"PY={sys.executable}", "test", "PYTEST_ARGS=tests/test_core.py tests/test_schema.py"],
                           cwd=copy, capture_output=True, text=True, env=env)
    assert tests.returncode == 0, tests.stdout[-800:] + tests.stderr[-800:]
