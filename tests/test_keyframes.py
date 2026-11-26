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


def test_the_composed_prompt_presents_image_0_as_identity_guidance(root):
    sb = with_reference(root)
    c, shot = sb["character"], sb["shots"][0]
    prompt = composed_keyframe_prompt(sb, shot)
    assert prompt == (f"Image 0 is the visual reference for {c['name']}: keep the same faces, ages, builds and clothing "
                      f"as in image 0. {c['description']}. {shot['keyframe_prompt']}. Style: {sb['style']}.")
    assert "is the person in image 0" not in prompt


def test_a_group_reference_reads_grammatically_and_keeps_the_ages():
    sb = {"style": "realism", "character": {
        "name": "the three children", "reference_image": REF,
        "description": "the narrator, a boy of about twelve; his sister, a girl of about eleven; their cousin, "
                       "a boy of about eight, visibly younger and smaller"},
          "shots": []}
    prompt = composed_keyframe_prompt(sb, {"keyframe_prompt": "the three children on the patio"})
    assert prompt.startswith("Image 0 is the visual reference for the three children: keep the same faces, ages, "
                             "builds and clothing as in image 0.")
    assert "cousin, a boy of about eight, visibly younger and smaller" in prompt


def test_a_shot_without_the_character_gets_no_reference_and_no_character_description(root):
    sb = with_reference(root)
    shot = dict(sb["shots"][0], character_reference=False)
    prompt = composed_keyframe_prompt(sb, shot)
    assert "image 0" not in prompt and sb["character"]["description"] not in prompt
    assert prompt == f"{sb['style']}. {shot['keyframe_prompt']}"


def test_a_shot_without_the_character_keeps_the_setting(root):
    sb = dict(with_reference(root), setting="An ordinary family home with a patio in front of it")
    shot = dict(sb["shots"][0], character_reference=False)
    assert composed_keyframe_prompt(sb, shot) == f"{sb['style']}. {sb['setting']}. {shot['keyframe_prompt']}"


def test_the_setting_field_does_not_change_shots_that_show_the_character(root):
    sb = with_reference(root)
    before = [composed_keyframe_prompt(sb, s) for s in sb["shots"]]
    assert [composed_keyframe_prompt(dict(sb, setting="a patio"), s) for s in sb["shots"]] == before
    del sb["character"]["reference_image"]
    plain = [composed_keyframe_prompt(sb, s) for s in sb["shots"]]
    assert [composed_keyframe_prompt(dict(sb, setting="a patio"), s) for s in sb["shots"]] == plain


def test_the_schema_accepts_an_optional_setting(storyboard):
    from pipeline.common import StoryboardError
    validate(dict(storyboard, setting="An ordinary family home with a patio"))
    with pytest.raises(StoryboardError):
        validate(dict(storyboard, setting=""))


# ------------------------------------------------------------- per-shot reference

def opt_out(root, *ids):
    return write_board(root, lambda sb: [s.update(character_reference=False) for s in sb["shots"] if s["id"] in ids])


def test_the_schema_accepts_only_a_boolean_character_reference(storyboard):
    from pipeline.common import StoryboardError
    storyboard["shots"][0]["character_reference"] = False
    validate(storyboard)
    storyboard["shots"][0]["character_reference"] = "no"
    with pytest.raises(StoryboardError):
        validate(storyboard)


def test_only_shots_that_use_the_reference_are_sent_it(root):
    with_reference(root)
    sb = opt_out(root, "s02")
    fake = Recorder()
    run(root, fake)
    sent = {s["id"]: c["reference"] for s, c in zip(sb["shots"], fake.calls)}
    assert sent["s01"] == (root / REF).read_bytes() and sent["s03"] == (root / REF).read_bytes()
    assert sent["s02"] is None
    assert len(fake.calls) == 3, "no reference sheet either: an approved reference exists"


def test_runs_record_no_reference_for_an_opted_out_shot(root):
    with_reference(root)
    opt_out(root, "s02")
    ctx = run(root, Recorder())
    lines = {r["shot"]: r for r in read_runs(ctx.runs_path) if r["stage"] == "keyframes"}
    assert lines["s02"]["params"]["reference_sha256"] is None
    assert len(lines["s01"]["params"]["reference_sha256"]) == 64


def test_a_new_reference_regenerates_only_the_shots_that_use_it(root):
    with_reference(root)
    opt_out(root, "s02")
    run(root, Recorder())
    with_reference(root, colour="blue")
    again = Recorder()
    run(root, again)
    assert len(again.calls) == 2 and all(c["reference"] is not None for c in again.calls)


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
