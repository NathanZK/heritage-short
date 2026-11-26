"""The Wan shots kernel (kernels/shots_kernel.py): the E6 Diffusers setup on Kaggle's two T4s.

The GPU code only runs on Kaggle, so these tests check its pure parts and drive `main` with a fake backend that
stands in for the loaded models. Nothing here imports torch or diffusers, or calls Kaggle."""
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

from tests.test_kernels import configured, write_job

PIP = [sys.executable, "-m", "pip"]


class Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class FakeWan:
    """Stands in for the loaded models. `fail` maps a shot's prompt to the exception its render raises."""
    def __init__(self, clip=None, fail=None, partial=False, load_error=None):
        self.clip, self.fail, self.partial, self.load_error = clip, fail or {}, partial, load_error
        self.calls = []

    def __call__(self, scratch, log=print):
        self.calls.append(("load", Path(scratch)))
        if self.load_error:
            raise self.load_error
        return self

    def encode(self, prompts):
        self.calls.append(("encode", list(prompts)))

    def render(self, image, params, out):
        self.calls.append(("render", params["prompt"], params["num_frames"], params["seed"]))
        if params["prompt"] in self.fail:
            if self.partial:
                Path(out).write_bytes(b"half a clip")
            raise self.fail[params["prompt"]]
        if self.clip:
            shutil.copyfile(self.clip, out)
        else:
            Path(out).write_bytes(b"clip-" + params["prompt"].encode())
        return {"decode": "untiled", "num_frames": params["num_frames"]}


def wan_job(tmp_path, ids=("s01", "s02"), num_frames=121):
    inputs = tmp_path / "in"
    folder = write_job(inputs, ids=ids)
    job = json.loads((folder / "job.json").read_text())
    for n, it in enumerate(job["items"]):
        it.update(fingerprint=f"fp-{it['id']}", inputs=[f"{it['id']}.jpg"],
                  params={"prompt": f"prompt {it['id']}", "seed": 100 + n, "num_frames": num_frames})
        Image.new("RGB", (1280, 704), (200, 120, 40)).save(folder / f"{it['id']}.jpg")
    (folder / "job.json").write_text(json.dumps(job))
    return inputs


def run_main(tmp_path, backend, ids=("s01", "s02"), run=None, num_frames=121):
    module = configured("shots", ids=ids)
    commands = []

    def default_run(cmd, cwd=None, **kw):
        commands.append((cmd, cwd))
        return Proc()
    module.main(root=wan_job(tmp_path, ids=ids, num_frames=num_frames), work=tmp_path / "work",
                scratch=tmp_path / "scratch", run=run or default_run, sleep=lambda s: None, backend=backend)
    return json.loads((tmp_path / "work" / "results.json").read_text()), commands


# ------------------------------------------------------------ the measured E6 configuration

def test_the_kernel_uses_the_measured_e6_settings():
    module = configured("shots")
    assert module.MODEL == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert (module.WIDTH, module.HEIGHT, module.FPS) == (832, 480, 24)
    assert (module.STEPS, module.GUIDANCE, module.MAX_FRAMES) == (20, 5.0, 121)
    assert module.PACKAGES == ["diffusers==0.43.0", "transformers==5.20.2", "ftfy"]


def test_models_are_placed_as_in_e6_with_no_flash_attention_or_bf16_video_model():
    module = configured("shots")
    load = inspect.getsource(module.load_wan)
    assert 'WanTransformer3DModel.from_pretrained(ckpt, subfolder="transformer", torch_dtype=torch.float16)' in load
    assert 'transformer.to("cuda:0")' in load
    assert 'UMT5EncoderModel.from_pretrained(ckpt, subfolder="text_encoder", torch_dtype=torch.bfloat16)' in load
    assert 'text_encoder.to("cuda:1")' in load
    assert 'AutoencoderKLWan.from_pretrained(ckpt, subfolder="vae", torch_dtype=torch.float16)' in load
    assert 'vae.to("cuda:1")' in load
    source = Path(module.__file__).read_text().lower()
    assert "flash_attn" not in source and "flash_attention" not in source and "generate.py" not in source
    assert source.count("torch.bfloat16") == 2, "bfloat16 only for loading and running the text encoder"
    assert "enable_model_cpu_offload" not in source and "enable_group_offload" not in source


def test_decoding_uses_the_full_vae_untiled_first_and_tiles_only_after_running_out_of_memory():
    module = configured("shots")
    render = inspect.getsource(module.load_wan)
    assert render.index("vae.decode") < render.index("OutOfMemoryError") < render.index("vae.enable_tiling()")
    assert "vae.disable_tiling()" in render, "the next shot tries untiled again"
    assert 'output_type="latent"' in render and "taehv" not in render.lower()


def test_the_keyframe_is_cropped_and_resized_exactly_as_in_e6(tmp_path):
    module = configured("shots")
    src = tmp_path / "k.jpg"
    Image.new("RGB", (1280, 704), "white").save(src)
    image, box = module.prepare_image(src)
    assert image.size == (832, 480) and box == (30, 0, 1250, 704)


@pytest.mark.parametrize("frames,ok", [(121, True), (97, True), (5, True), (1, False), (120, False),
                                       (125, False), ("121", False)])
def test_only_wan_frame_counts_up_to_the_measured_121_are_accepted(frames, ok):
    module = configured("shots")
    if ok:
        assert module.check_frames(frames) == frames
    else:
        with pytest.raises(ValueError):
            module.check_frames(frames)


def test_the_kernel_imports_without_torch_or_diffusers(monkeypatch):
    for name in ("torch", "diffusers", "transformers"):
        monkeypatch.setitem(sys.modules, name, None)   # any top-level import of these now fails
    assert configured("shots").MODEL


# ------------------------------------------------------------ main: setup, per-shot results, failures

def test_setup_installs_the_pinned_packages_with_this_python_in_scratch(tmp_path):
    results, commands = run_main(tmp_path, FakeWan())
    assert [cmd for cmd, _ in commands] == [PIP + ["install", "-q", "diffusers==0.43.0", "transformers==5.20.2",
                                                   "ftfy"]]
    assert Path(commands[0][1]) == tmp_path / "scratch"


def test_models_load_once_and_every_prompt_is_encoded_before_any_shot_renders(tmp_path):
    backend = FakeWan()
    run_main(tmp_path, backend)
    kinds = [c[0] for c in backend.calls]
    assert kinds == ["load", "encode", "render", "render"]
    assert backend.calls[1] == ("encode", ["prompt s01", "prompt s02"])


def test_each_shot_uses_its_own_prompt_seed_and_frame_count(tmp_path):
    backend = FakeWan()
    run_main(tmp_path, backend, ids=("s01",), num_frames=97)
    assert backend.calls[-1] == ("render", "prompt s01", 97, 100)


def test_results_describe_each_clip_and_only_deliverables_reach_the_output_folder(tmp_path):
    results, _ = run_main(tmp_path, FakeWan())
    work = tmp_path / "work"
    assert results["run_id"] == "r1" and results["stage"] == "shots"
    assert sorted(p.name for p in work.iterdir()) == ["results.json", "s01.mp4", "s02.mp4"]
    first = results["items"][0]
    assert first["status"] == "ok" and first["file"] == "s01.mp4" and first["fingerprint"] == "fp-s01"
    assert first["sha256"] == hashlib.sha256((work / "s01.mp4").read_bytes()).hexdigest()
    assert first["model"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers" and first["seed"] == 100
    assert first["params"]["num_frames"] == 121 and first["decode"] == "untiled"
    assert isinstance(first["latency_s"], float) and first["gpu_seconds"] == first["latency_s"]
    assert isinstance(results["setup_s"], float)


def test_results_are_written_after_every_shot(tmp_path):
    module = configured("shots", ids=("s01", "s02"))
    seen = []

    class Watching(FakeWan):
        def render(self, image, params, out):
            seen.append(json.loads((tmp_path / "work" / "results.json").read_text())["items"])
            return super().render(image, params, out)
    module.main(root=wan_job(tmp_path), work=tmp_path / "work", scratch=tmp_path / "scratch",
                run=lambda *a, **k: Proc(), sleep=lambda s: None, backend=Watching())
    assert [len(items) for items in seen] == [0, 1]


def test_a_failing_shot_is_reported_its_partial_file_removed_and_the_next_shot_still_runs(tmp_path):
    backend = FakeWan(fail={"prompt s01": RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")}, partial=True)
    results, _ = run_main(tmp_path, backend)
    bad, good = results["items"]
    assert bad["status"] == "error" and "RuntimeError" in bad["error"] and "out of memory" in bad["error"]
    assert bad["fingerprint"] == "fp-s01" and "file" not in bad
    assert good["status"] == "ok"
    assert sorted(p.name for p in (tmp_path / "work").iterdir()) == ["results.json", "s02.mp4"]


def test_nan_latents_are_an_error_not_a_clip(tmp_path):
    backend = FakeWan(fail={"prompt s02": FloatingPointError("NaN/Inf in latents at step 3")})
    results, _ = run_main(tmp_path, backend)
    assert [i["status"] for i in results["items"]] == ["ok", "error"]
    assert "NaN" in results["items"][1]["error"]


def test_a_shot_whose_frame_count_is_invalid_is_refused_before_rendering(tmp_path):
    backend = FakeWan()
    results, _ = run_main(tmp_path, backend, ids=("s01",), num_frames=120)
    assert results["items"][0]["status"] == "error" and "frames" in results["items"][0]["error"]
    assert not [c for c in backend.calls if c[0] == "render"]


def test_a_failed_package_install_reports_every_shot_and_loads_nothing(tmp_path):
    backend = FakeWan()
    results, _ = run_main(tmp_path, backend, run=lambda cmd, cwd=None, **kw: Proc(1, stderr="no space left"))
    assert [i["id"] for i in results["items"]] == ["s01", "s02"]
    for item in results["items"]:
        assert item["status"] == "error" and "pip install" in item["error"] and "no space left" in item["error"]
        assert item["fingerprint"] == f"fp-{item['id']}"
    assert backend.calls == []
    assert sorted(p.name for p in (tmp_path / "work").iterdir()) == ["results.json"]


def test_a_failure_while_loading_the_models_reports_every_shot(tmp_path):
    backend = FakeWan(load_error=RuntimeError("needs 2 GPUs, found 1"))
    results, _ = run_main(tmp_path, backend)
    assert all(i["status"] == "error" and "needs 2 GPUs" in i["error"] and "loading" in i["error"]
               for i in results["items"])
    assert sorted(p.name for p in (tmp_path / "work").iterdir()) == ["results.json"]
