"""Per-shot horizontal framing of the 9:16 export (`portrait_crop_x`), so a subject off-centre in a landscape shot
stays in the vertical frame. Ported from the framing fix found in the end-to-end integration test (FAILURES 25)."""
import json
import subprocess
from types import SimpleNamespace

import pytest
from PIL import Image

from pipeline import assemble
from pipeline.common import Ctx, StoryboardError, validate
from tests.conftest import FIXTURES
from tests.test_assemble import fresh_ctx

NS = SimpleNamespace


# ------------------------------------------------------------- 9:16 framing

@pytest.mark.parametrize("value,ok", [(0, True), (0.5, True), (1, True), (0.65, True),
                                      (-0.1, False), (1.5, False), ("left", False)])
def test_portrait_crop_x_is_an_optional_fraction(storyboard, value, ok):
    validate(storyboard)                                    # absent is valid
    storyboard["shots"][0]["portrait_crop_x"] = value
    if ok:
        validate(storyboard)
    else:
        with pytest.raises(StoryboardError) as exc:
            validate(storyboard)
        assert "portrait_crop_x" in str(exc.value)


def test_fill_crop_defaults_to_the_centre_and_follows_the_shot():
    assert assemble.fill_crop({}, 720, 1280) == assemble.fill_crop({"portrait_crop_x": 0.5}, 720, 1280)
    assert "crop=720:1280:(iw-ow)*0.65:(ih-oh)/2" in assemble.fill_crop({"portrait_crop_x": 0.65}, 720, 1280)
    assert assemble.fill_crop({}, 720, 1280).startswith("scale=720:1280:force_original_aspect_ratio=increase,")


def left_red_right_blue(path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=640x720:d=1",
                    "-f", "lavfi", "-i", "color=c=blue:s=640x720:d=1", "-filter_complex", "hstack", "-frames:v", "1",
                    str(path)], check=True)


def cropped(src, dst, shot, size):
    w, h = size
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", assemble.fill_crop(shot, w, h),
                    "-frames:v", "1", str(dst)], check=True)
    img = Image.open(dst).convert("RGB")
    red = sum(1 for r, g, b in img.getdata() if r > 200 and b < 60)
    blue = sum(1 for r, g, b in img.getdata() if b > 200 and r < 60)
    return red / (w * h), blue / (w * h), img.tobytes()


def test_the_9x16_crop_moves_with_portrait_crop_x_and_16x9_does_not(tmp_path):
    src = tmp_path / "src.png"
    left_red_right_blue(src)
    red, blue, _ = cropped(src, tmp_path / "a.png", {"portrait_crop_x": 0}, (720, 1280))
    assert red > 0.95 and blue < 0.01, "0 keeps the left of the frame"
    red, blue, _ = cropped(src, tmp_path / "b.png", {"portrait_crop_x": 1}, (720, 1280))
    assert blue > 0.95 and red < 0.01, "1 keeps the right of the frame"
    red, blue, _ = cropped(src, tmp_path / "c.png", {}, (720, 1280))
    assert 0.4 < red < 0.6 and 0.4 < blue < 0.6, "default stays centred"
    *_, left = cropped(src, tmp_path / "d.png", {"portrait_crop_x": 0}, (1280, 720))
    *_, right = cropped(src, tmp_path / "e.png", {"portrait_crop_x": 1}, (1280, 720))
    assert left == right, "16:9 fills the frame without a horizontal crop, so the setting changes nothing"


def test_render_video_uses_each_shots_crop(tmp_path, monkeypatch):
    sb = json.loads((FIXTURES / "storyboard_3shot.json").read_text())
    sb["shots"][0]["portrait_crop_x"] = 0.65
    ctx = Ctx(root=tmp_path, mock=True)
    seen = []
    monkeypatch.setattr(assemble, "commit_with", lambda dst, fn, validate=None: seen.append(fn))
    monkeypatch.setattr(assemble, "ffmpeg", lambda *args, **kw: seen.append(args))
    assemble.render_video(ctx, sb, (720, 1280), tmp_path / "mix.wav", [], 16.5, tmp_path / "out.mp4")
    seen[0](tmp_path / "tmp.mp4")
    graph = seen[1][seen[1].index("-filter_complex") + 1]
    assert "[0:v]" + assemble.fill_crop(sb["shots"][0], 720, 1280) in graph
    assert "[1:v]" + assemble.fill_crop(sb["shots"][1], 720, 1280) in graph


def test_changing_a_crop_rebuilds_only_the_9x16_export(tmp_path):
    ctx, sb = fresh_ctx(tmp_path)
    assemble.run(ctx, sb, NS(draft=True, radio=False))
    final = ctx.out / "final"
    before = {n: (final / n).stat().st_mtime_ns for n in ("film_9x16.mp4", "film_16x9.mp4", "mix.wav")}
    sb["shots"][0]["portrait_crop_x"] = 0.9
    assemble.run(ctx, sb, NS(draft=True, radio=False))
    assert (final / "film_9x16.mp4").stat().st_mtime_ns != before["film_9x16.mp4"], "9:16 is re-rendered"
    assert (final / "film_16x9.mp4").stat().st_mtime_ns == before["film_16x9.mp4"], "16:9 is not"
    assert (final / "mix.wav").stat().st_mtime_ns == before["mix.wav"], "nor is the mix"
