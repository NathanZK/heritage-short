import json
import socket
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
FIXTURES = REPO / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise AssertionError("test attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", guard)


@pytest.fixture
def storyboard():
    return json.loads((FIXTURES / "storyboard_3shot.json").read_text())


@pytest.fixture
def root(tmp_path, storyboard):
    (tmp_path / "storyboard.json").write_text(json.dumps(storyboard))
    return tmp_path


@pytest.fixture
def ctx(root):
    from pipeline.common import Ctx
    return Ctx(root=root, mock=True, sleep=lambda s: None)


@pytest.fixture(scope="session")
def clip_fixture(tmp_path_factory):
    """A real 121-frame 1280x704 clip, like Wan's output, made once per test session."""
    import subprocess
    clip = tmp_path_factory.mktemp("clip") / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x704:rate=24",
                    "-frames:v", "121", "-pix_fmt", "yuv420p", str(clip)], check=True)
    return clip


@pytest.fixture(scope="session")
def short_clip_fixture(tmp_path_factory):
    clip = tmp_path_factory.mktemp("short") / "short.mp4"
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x704:rate=24",
                    "-frames:v", "24", "-pix_fmt", "yuv420p", str(clip)], check=True)
    return clip


@pytest.fixture
def fake_kaggle(tmp_path, monkeypatch, clip_fixture, short_clip_fixture):
    """Put a fake `kaggle` executable on PATH. Returns an object to pick the scenario and read what happened."""
    import os
    bin_dir, state = tmp_path / "bin", tmp_path / "kaggle-state"
    bin_dir.mkdir()
    state.mkdir()
    exe = bin_dir / "kaggle"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FIXTURES.parent / "fake_kaggle.py"}" "$@"\n')
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_KAGGLE_STATE", str(state))
    monkeypatch.setenv("FAKE_KAGGLE_CLIP", str(clip_fixture))
    monkeypatch.setenv("FAKE_KAGGLE_SHORT_CLIP", str(short_clip_fixture))
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "kaggle-key-secret-9876")

    class Fake:
        def scenario(self, **kw):
            monkeypatch.setenv("FAKE_KAGGLE_SCENARIO", json.dumps(kw))

        def count(self, name):
            f = state / name
            return int(f.read_text()) if f.exists() else 0

        @property
        def calls(self):
            f = state / "calls.log"
            return f.read_text().splitlines() if f.exists() else []

        state_dir = state
    return Fake()
