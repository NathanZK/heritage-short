"""Keyframes with FLUX.2 [klein] 4B and an approved character reference image (default), or FLUX.1 schnell."""
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from pipeline import keyframes, mock
from pipeline.common import Ctx, Generated, Manifest, PipelineExit, ProviderError, composed_keyframe_prompt, \
    read_runs, validate

NS = SimpleNamespace
REF = "refs/character.jpg"


def jpeg(size, colour="red"):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "JPEG")
    return buf.getvalue()


class Recorder:
    """Fake klein provider: records each call and reports usage."""
    KLEIN_MODEL = "fake-klein"
    IMAGE_MODEL = "fake-schnell"

    def __init__(self, neurons=50.0):
        self.calls, self.neurons = [], neurons

    def cloudflare_klein(self, prompt, width, height, seed, reference):
        self.calls.append({"prompt": prompt, "width": width, "height": height, "seed": seed, "reference": reference})
        return Generated(jpeg((width, height)), self.neurons)

    def cloudflare_image(self, prompt, steps=4):
        self.calls.append({"prompt": prompt, "steps": steps, "schnell": True})
        return jpeg((1024, 1024))


def write_board(root, mutate=None):
    sb = json.loads((root / "storyboard.json").read_text())
    if mutate:
        mutate(sb)
    (root / "storyboard.json").write_text(json.dumps(sb))
    return sb


def with_reference(root, size=(344, 511), colour="red"):
    (root / "refs").mkdir(exist_ok=True)
    (root / REF).write_bytes(jpeg(size, colour))
    return write_board(root, lambda sb: sb["character"].update(reference_image=REF))


def run(root, provider, sb=None):
    ctx = Ctx(root=root, provider_override=provider, sleep=lambda s: None)
    keyframes.run(ctx, sb or json.loads((root / "storyboard.json").read_text()), NS())
    return ctx


# ------------------------------------------------------------- schema

def test_storyboard_accepts_the_keyframe_model_and_reference(storyboard):
    validate(dict(storyboard, keyframe_model="flux-2-klein-4b"))
    validate(dict(storyboard, keyframe_model="flux-1-schnell"))
    storyboard["character"]["reference_image"] = REF
    validate(storyboard)


@pytest.mark.parametrize("mutate", [lambda sb: sb.update(keyframe_model="dall-e"),
                                    lambda sb: sb["character"].update(reference_image=""),
                                    lambda sb: sb["character"].update(reference_image=7)])
def test_storyboard_rejects_bad_keyframe_settings(storyboard, mutate):
    from pipeline.common import StoryboardError
    mutate(storyboard)
    with pytest.raises(StoryboardError):
        validate(storyboard)


# ------------------------------------------------------------- klein with a reference

def test_shots_use_the_reference_at_1280x704_with_their_seeds_and_no_sheet(root):
    sb = with_reference(root)
    fake = Recorder()
    ctx = run(root, fake)
    assert [c["width"] for c in fake.calls] == [1280] * 3 and {c["height"] for c in fake.calls} == {704}
    assert all(c["reference"] == (root / REF).read_bytes() for c in fake.calls)
    from pipeline.common import shot_seed
    assert [c["seed"] for c in fake.calls] == [shot_seed(sb, s) for s in sb["shots"]]
    assert [c["prompt"] for c in fake.calls] == [composed_keyframe_prompt(sb, s) for s in sb["shots"]]
    assert sorted(p.name for p in (ctx.out / "keyframes").glob("*.jpg")) == ["s01.jpg", "s02.jpg", "s03.jpg"]
    assert Image.open(ctx.out / "keyframes" / "s01.jpg").size == (1280, 704)


def test_the_composed_prompt_names_the_character_in_image_0(root):
    sb = with_reference(root)
    prompt = composed_keyframe_prompt(sb, sb["shots"][0])
    assert prompt.startswith(f"{sb['character']['name']} is the person in image 0 ({sb['character']['description']}).")
    assert sb["shots"][0]["keyframe_prompt"] in prompt and prompt.endswith(f"Style: {sb['style']}.")


def test_runs_log_the_neurons_and_request(root):
    with_reference(root)
    ctx = run(root, Recorder(neurons=107.8))
    lines = [r for r in read_runs(ctx.runs_path) if r["stage"] == "keyframes"]
    assert [r["shot"] for r in lines] == ["s01", "s02", "s03"]
    assert all(r["neurons"] == 107.8 and r["model"] == "fake-klein" and r["status"] == "ok" for r in lines)
    assert lines[0]["params"]["width"] == 1280 and lines[0]["params"]["height"] == 704
    assert len(lines[0]["params"]["reference_sha256"]) == 64 and lines[0]["seed"] is not None


def test_a_large_reference_is_downscaled_below_512(root):
    with_reference(root, size=(1024, 1500))
    fake = Recorder()
    run(root, fake)
    sent = Image.open(io.BytesIO(fake.calls[0]["reference"]))
    assert max(sent.size) <= 511 and sent.size == (349, 511)


def test_a_missing_reference_file_stops_before_any_call(root):
    write_board(root, lambda sb: sb["character"].update(reference_image="refs/nope.jpg"))
    fake = Recorder()
    with pytest.raises(PipelineExit) as exc:
        run(root, fake)
    assert exc.value.code == 1 and "refs/nope.jpg" in str(exc.value) and fake.calls == []


def test_schnell_with_a_reference_stops_before_any_call(root):
    with_reference(root)
    write_board(root, lambda sb: sb.update(keyframe_model="flux-1-schnell"))
    fake = Recorder()
    with pytest.raises(PipelineExit) as exc:
        run(root, fake)
    assert "flux-1-schnell" in str(exc.value) and "reference_image" in str(exc.value) and fake.calls == []


# ------------------------------------------------------------- cache

def test_an_unchanged_rerun_makes_no_call(root):
    with_reference(root)
    run(root, Recorder())
    again = Recorder()
    run(root, again)
    assert again.calls == []


def test_new_reference_bytes_regenerate_every_shot(root):
    with_reference(root)
    run(root, Recorder())
    with_reference(root, colour="blue")
    again = Recorder()
    run(root, again)
    assert len(again.calls) == 3


def test_one_prompt_edit_regenerates_one_keyframe(root):
    with_reference(root)
    run(root, Recorder())
    sb = write_board(root, lambda sb: sb["shots"][1].update(keyframe_prompt="Almaz crushes beans in a mortar"))
    again = Recorder()
    run(root, again)
    assert [c["prompt"] for c in again.calls] == [composed_keyframe_prompt(sb, sb["shots"][1])]


def test_removing_the_reference_changes_the_fingerprint(root):
    sb = with_reference(root)
    ctx = run(root, Recorder())
    before = keyframes.keyframe_fp(ctx, sb, sb["shots"][0])
    sb2 = dict(sb, character=dict(sb["character"]))
    del sb2["character"]["reference_image"]
    assert keyframes.keyframe_fp(ctx, sb2, sb["shots"][0]) != before


def test_the_seed_and_size_are_in_the_fingerprint(root, monkeypatch):
    sb = with_reference(root)
    ctx = Ctx(root=root, provider_override=Recorder())
    fp = keyframes.keyframe_fp(ctx, sb, sb["shots"][0])
    assert keyframes.keyframe_fp(ctx, dict(sb, seed=7), sb["shots"][0]) != fp
    monkeypatch.setattr(keyframes, "KLEIN_SIZE", (1024, 768))
    assert keyframes.keyframe_fp(ctx, sb, sb["shots"][0]) != fp


# ------------------------------------------------------------- without a reference, and schnell

def test_schnell_keeps_its_request_and_fingerprints(root):
    sb = write_board(root, lambda sb: sb.update(keyframe_model="flux-1-schnell"))
    fake = Recorder()
    ctx = run(root, fake)
    assert len(fake.calls) == 4 and all(c.get("schnell") for c in fake.calls)
    from pipeline.common import fingerprint
    assert keyframes.keyframe_fp(ctx, sb, sb["shots"][0]) == fingerprint(
        1, "keyframe", "fake-schnell", keyframes.STEPS, sb["style"], sb["character"]["description"],
        sb["shots"][0]["keyframe_prompt"])
    assert all("neurons" not in r for r in read_runs(ctx.runs_path))


def test_mock_mode_works_with_a_reference(root):
    with_reference(root)
    ctx = Ctx(root=root, mock=True, sleep=lambda s: None)
    keyframes.run(ctx, json.loads((root / "storyboard.json").read_text()), NS())
    assert Image.open(ctx.out / "keyframes" / "s03.jpg").size == (1280, 704)
    assert Manifest(ctx).data["outputs"].keys() >= {"keyframe:s01", "keyframe:s02", "keyframe:s03"}


# ------------------------------------------------------------- reference normalisation and containment (review)

def save_reference(root, img, fmt="JPEG", **kw):
    (root / "refs").mkdir(exist_ok=True)
    img.save(root / REF, fmt, **kw)
    return write_board(root, lambda sb: sb["character"].update(reference_image=REF))


def sent(root):
    fake = Recorder()
    run(root, fake)
    return fake.calls[0]["reference"]


def test_a_plain_small_rgb_jpeg_is_sent_byte_for_byte(root):
    with_reference(root, size=(344, 511))
    assert sent(root) == (root / REF).read_bytes()


def test_exif_rotation_is_applied_before_sending(root):
    exif = Image.Exif()
    exif[0x0112] = 6                       # stored landscape, shown portrait
    save_reference(root, Image.new("RGB", (1500, 1000), "red"), exif=exif.tobytes())
    out = Image.open(io.BytesIO(sent(root)))
    assert out.size == (341, 511) and out.getexif().get(0x0112) in (None, 1)


def test_a_small_rotated_reference_is_also_normalised(root):
    exif = Image.Exif()
    exif[0x0112] = 6
    save_reference(root, Image.new("RGB", (300, 200), "red"), exif=exif.tobytes())
    out = Image.open(io.BytesIO(sent(root)))
    assert out.size == (200, 300) and out.format == "JPEG"


@pytest.mark.parametrize("mode,fmt", [("CMYK", "JPEG"), ("RGBA", "PNG"), ("P", "PNG")])
def test_unusual_modes_are_sent_as_rgb_jpeg(root, mode, fmt):
    save_reference(root, Image.new(mode, (300, 400)), fmt)
    out = Image.open(io.BytesIO(sent(root)))
    assert (out.format, out.mode, out.size) == ("JPEG", "RGB", (300, 400))


@pytest.mark.parametrize("path", ["/etc/hosts", "../outside.jpg", "refs/../../outside.jpg"])
def test_a_reference_outside_the_repository_is_refused_before_any_call(root, path):
    (root.parent / "outside.jpg").write_bytes(jpeg((100, 100)))
    write_board(root, lambda sb: sb["character"].update(reference_image=path))
    fake = Recorder()
    with pytest.raises(PipelineExit) as exc:
        run(root, fake)
    assert "inside the repository" in str(exc.value) and fake.calls == []


def test_the_logged_reference_hash_matches_the_cache_input(root):
    import hashlib
    with_reference(root, size=(1024, 1500))                 # downscaled before sending
    ctx = run(root, Recorder())
    logged = [r for r in read_runs(ctx.runs_path) if r["stage"] == "keyframes"][0]["params"]["reference_sha256"]
    assert logged == hashlib.sha256((root / REF).read_bytes()).hexdigest()
