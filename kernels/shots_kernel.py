"""Kaggle kernel: Wan 2.2 TI2V-5B image-to-video through Hugging Face Diffusers on Kaggle's two T4s.

Pushed by pipeline/kaggle.py as a single generated file (RUN_ID and EXPECTED_IDS are filled in at push time). It
reproduces the configuration measured in EXPERIMENTS.md E6 ("Run 1"):

- the video model in float16, resident on cuda:0, with no CPU offloading;
- the UMT5 text encoder in bfloat16 on cuda:1, used once for every prompt and then freed;
- the Wan VAE in float16 on cuda:1, decoding untiled (tiled only after running out of memory, and recorded);
- 832x480, 24 fps, 20 steps, guidance 5.0, the shot's seed and up to 121 frames;
- PyTorch's own attention (Diffusers' default), so it needs no extra attention library on a T4.

The models load once per run; each shot's clip and results.json are written as soon as it is done. Torch and
Diffusers are imported only inside load_wan, so the tests can drive main() with a fake backend."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RUN_ID = "__RUN_ID__"
EXPECTED_IDS = []
MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WIDTH, HEIGHT, FPS = 832, 480, 24
STEPS, GUIDANCE, MAX_FRAMES = 20, 5.0, 121
# The versions E6 ran with (its experiment.json); torch is Kaggle's own (2.11.0+cu130 in E6).
PACKAGES = ["diffusers==0.43.0", "transformers==5.20.2", "ftfy"]
PIP = [sys.executable, "-m", "pip"]   # this interpreter's pip, so PATH does not decide which Python gets the packages
# Wan's default negative prompt, as used in E5 and E6.
NEGATIVE_PROMPT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
                   "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
                   "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


def find_job(root="/kaggle/input", wait_s=600, poll_s=15, sleep=time.sleep):
    """Find the job.json for THIS run (the dataset version may take a while to appear)."""
    waited = 0
    while True:
        for path in sorted(Path(root).rglob("job.json")):
            job = json.loads(path.read_text())
            if job.get("run_id") == RUN_ID and [i["id"] for i in job["items"]] == EXPECTED_IDS:
                return path.parent, job
        if waited >= wait_s:
            sys.exit(f"no job.json with run_id {RUN_ID} under {root}")
        sleep(poll_s)
        waited += poll_s


def check_frames(frames):
    """Wan needs 4k+1 frames; more than E6's 121 has not been tried on a T4."""
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 5 or frames > MAX_FRAMES \
            or frames % 4 != 1:
        raise ValueError(f"num_frames must be 4k+1 between 5 and {MAX_FRAMES}, got {frames!r}")
    return frames


def prepare_image(path):
    """As in E6: centre-crop the keyframe to 832:480, then resize it to 832x480 (Lanczos)."""
    from PIL import Image
    src = Image.open(path).convert("RGB")
    w, h = src.size
    if w * HEIGHT >= h * WIDTH:
        tw = round(h * WIDTH / HEIGHT)
        box = ((w - tw) // 2, 0, (w - tw) // 2 + tw, h)
    else:
        th = round(w * HEIGHT / WIDTH)
        box = (0, (h - th) // 2, w, (h - th) // 2 + th)
    return src.crop(box).resize((WIDTH, HEIGHT), Image.LANCZOS), box


def setup_failed(proc) -> str:
    """'' when a setup command succeeded, else its exit code and the tail of its output."""
    if proc.returncode == 0:
        return ""
    tail = ((getattr(proc, "stderr", "") or "") + (getattr(proc, "stdout", "") or "")).strip()[-300:]
    return f"exited with {proc.returncode}" + (f": {tail}" if tail else "")


def install_packages(run, scratch) -> str:
    problem = setup_failed(run(PIP + ["install", "-q"] + PACKAGES, cwd=str(scratch), capture_output=True, text=True))
    return f"setup step 'pip install {' '.join(PACKAGES)}' {problem}" if problem else ""


def describe(exc) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def load_wan(scratch, log=print):
    """Download the checkpoint and load it exactly as E6 did. Returns an engine with encode() and render().
    Runs only on Kaggle (needs two CUDA GPUs)."""
    import gc

    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
    from diffusers.utils import export_to_video
    from huggingface_hub import snapshot_download
    from transformers import UMT5EncoderModel

    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"needs 2 GPUs, found {torch.cuda.device_count()}")
    ckpt = str(Path(scratch) / "Wan2.2-TI2V-5B-Diffusers")
    log(f"downloading {MODEL}")
    snapshot_download(MODEL, local_dir=ckpt)

    class Pipe(WanImageToVideoPipeline):
        @property
        def _execution_device(self):
            return torch.device("cuda:0")   # the video model's GPU; the text encoder and VAE stay on cuda:1

    log("loading the models")
    transformer = WanTransformer3DModel.from_pretrained(ckpt, subfolder="transformer", torch_dtype=torch.float16)
    transformer.to("cuda:0")
    gc.collect()
    text_encoder = UMT5EncoderModel.from_pretrained(ckpt, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    text_encoder.to("cuda:1")
    gc.collect()
    vae = AutoencoderKLWan.from_pretrained(ckpt, subfolder="vae", torch_dtype=torch.float16)
    vae.to("cuda:1")
    pipe = Pipe.from_pretrained(ckpt, transformer=transformer, text_encoder=text_encoder, vae=vae,
                                torch_dtype=torch.float16)
    del text_encoder
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to("cuda:1", torch.float16)
    std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to("cuda:1", torch.float16)

    def free():
        gc.collect()
        torch.cuda.empty_cache()

    class Engine:
        def __init__(self):
            self.embeds = {}

        def encode(self, prompts):
            """Every prompt once, on cuda:1; then the text encoder is freed to leave room for decoding."""
            with torch.no_grad():
                for prompt in dict.fromkeys(prompts):
                    pe, npe = pipe.encode_prompt(prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
                                                 do_classifier_free_guidance=True, num_videos_per_prompt=1,
                                                 max_sequence_length=512, device=torch.device("cuda:1"),
                                                 dtype=torch.bfloat16)
                    bad = bool(torch.isnan(pe).any()) or bool(torch.isnan(npe).any())
                    self.embeds[prompt] = None if bad else (pe.to("cuda:0", torch.float16),
                                                            npe.to("cuda:0", torch.float16))
            pipe.text_encoder = None
            free()

        def render(self, image_path, params, out):
            frames = check_frames(params["num_frames"])
            embeds = self.embeds.get(params["prompt"])
            if embeds is None:
                raise FloatingPointError("NaN in the prompt embeddings")
            image, _ = prepare_image(image_path)

            def on_step(p, i, t, kw):
                lat = kw["latents"]
                if bool(torch.isnan(lat).any()) or bool(torch.isinf(lat).any()):
                    raise FloatingPointError(f"NaN/Inf in latents at step {i}")
                return kw

            try:
                latents = pipe(image=image, prompt_embeds=embeds[0], negative_prompt_embeds=embeds[1], height=HEIGHT,
                               width=WIDTH, num_frames=frames, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                               generator=torch.Generator(device="cpu").manual_seed(int(params["seed"])),
                               output_type="latent", callback_on_step_end=on_step,
                               callback_on_step_end_tensor_inputs=["latents"]).frames
                lat = latents.detach().to("cuda:1", torch.float16) * std + mean
                del latents
                method, video = "untiled", None
                try:
                    with torch.no_grad():
                        video = vae.decode(lat, return_dict=False)[0]
                except torch.cuda.OutOfMemoryError:
                    method = "tiled after running out of memory"
                if video is None:
                    # Outside the except block, so the failed attempt's tensors can be freed before retrying.
                    log("full VAE untiled: out of memory; decoding tiled (recorded)")
                    free()
                    vae.enable_tiling()
                    try:
                        with torch.no_grad():
                            video = vae.decode(lat, return_dict=False)[0]
                    finally:
                        vae.disable_tiling()
                arr = pipe.video_processor.postprocess_video(video.float(), output_type="np")[0]
                del video, lat
                if not np.isfinite(arr).all():
                    raise FloatingPointError("NaN/Inf in the decoded frames")
                export_to_video(list(arr), str(out), fps=FPS)
                return {"decode": method, "num_frames": len(arr)}
            finally:
                free()

    engine = Engine()
    log("models loaded")
    return engine


def write_results(work, results):
    tmp = Path(work) / ".results.json.tmp"
    tmp.write_text(json.dumps(results, indent=2))
    os.replace(tmp, Path(work) / "results.json")


def main(root="/kaggle/input", work="/kaggle/working", scratch=None, run=subprocess.run, sleep=time.sleep,
         install=True, backend=None):
    """`work` is what Kaggle returns, so it gets only results.json and the clips; packages and weights go in
    `scratch`. `backend` stands in for load_wan in tests."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    scratch = Path(scratch or Path(tempfile.gettempdir()) / "heritage-short")
    scratch.mkdir(parents=True, exist_ok=True)
    folder, job = find_job(root, sleep=sleep)
    items = job["items"]
    results = {"run_id": RUN_ID, "stage": "shots", "items": []}
    write_results(work, results)

    def fail_all(message):
        results["items"] = [{"id": item["id"], "status": "error", "fingerprint": item["fingerprint"],
                             "error": message} for item in items]
        write_results(work, results)

    start = time.monotonic()
    problem = install_packages(run, scratch) if install else ""
    if problem:
        return fail_all(problem)
    try:
        engine = (backend or load_wan)(scratch, log=print)
        engine.encode([item["params"]["prompt"] for item in items])
    except Exception as exc:
        return fail_all(f"loading Wan failed: {describe(exc)}")
    results["setup_s"] = round(time.monotonic() - start, 1)
    write_results(work, results)
    for item in items:
        out = work / f"{item['id']}.mp4"
        record = {"id": item["id"], "fingerprint": item.get("fingerprint"), "seed": item.get("params", {}).get("seed"),
                  "model": MODEL, "params": item.get("params", {})}
        start = time.monotonic()
        try:
            check_frames(item["params"].get("num_frames"))
            info = engine.render(folder / item["inputs"][0], item["params"], out)
            if not out.is_file():
                raise RuntimeError("no clip was written")
            record.update(status="ok", file=out.name, sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                          bytes=out.stat().st_size, decode=info.get("decode"))
        except Exception as exc:
            out.unlink(missing_ok=True)
            record.update(status="error", error=describe(exc))
        seconds = round(time.monotonic() - start, 1)
        record.update(latency_s=seconds, gpu_seconds=seconds)
        print(f"{item['id']}: {record['status']} in {seconds} s" + (f" ({record['error']})" if "error" in record else ""),
              flush=True)
        results["items"].append(record)
        write_results(work, results)


if __name__ == "__main__":
    main()
