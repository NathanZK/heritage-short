"""Kaggle kernel: Wan 2.2 TI2V-5B image-to-video, one documented `generate.py` call per shot.

Pushed by pipeline/kaggle.py as a single generated file (RUN_ID and EXPECTED_IDS are filled in at push
time). It only runs on Kaggle (T4 x2, internet on); the tests cover its pure functions. T4 compatibility
is UNVERIFIED (spec A9): the Wan README states a 24 GB minimum for this command."""
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
MODEL = "Wan-AI/Wan2.2-TI2V-5B"
WAN_REPO = "https://github.com/Wan-Video/Wan2.2"
CKPT = "./Wan2.2-TI2V-5B"
PIP = [sys.executable, "-m", "pip"]   # this interpreter's pip, so PATH does not decide which Python gets the packages


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


def build_generate_cmd(item, image, save_file, ckpt=CKPT):
    return [sys.executable, "generate.py", "--task", "ti2v-5B", "--size", "1280*704", "--ckpt_dir", ckpt,
            "--offload_model", "True", "--convert_model_dtype", "--t5_cpu", "--image", str(image),
            "--prompt", item["params"]["prompt"], "--base_seed", str(item["params"]["seed"]),
            "--frame_num", "121", "--save_file", str(save_file)]


def setup_failed(proc) -> str:
    """'' when a setup command succeeded, else its exit code and the tail of its output."""
    if proc.returncode == 0:
        return ""
    tail = ((getattr(proc, "stderr", "") or "") + (getattr(proc, "stdout", "") or "")).strip()[-300:]
    return f"exited with {proc.returncode}" + (f": {tail}" if tail else "")


def install_wan(run, work):
    """Set up Wan; returns '' or a message naming the step that failed. A flash_attn failure is not a failure:
    it is optional, and the README says to install it last."""
    repo = Path(work) / "Wan2.2"

    def step(name, cmd, cwd):
        problem = setup_failed(run(cmd, cwd=str(cwd), capture_output=True, text=True))
        return f"setup step '{name}' {problem}" if problem else ""

    problem = step("git clone", ["git", "clone", WAN_REPO], work)
    if not problem and setup_failed(run(PIP + ["install", "-r", "requirements.txt"], cwd=str(repo),
                                        capture_output=True, text=True)):
        lines = (repo / "requirements.txt").read_text().splitlines()
        (repo / "requirements-noflash.txt").write_text("\n".join(l for l in lines if "flash_attn" not in l))
        problem = step("pip install requirements", PIP + ["install", "-r", "requirements-noflash.txt"], repo)
        if not problem:
            run(PIP + ["install", "flash_attn"], cwd=str(repo), capture_output=True, text=True)
    for name, cmd in (("pip install huggingface_hub", PIP + ["install", "huggingface_hub[cli]"]),
                      ("download weights", ["huggingface-cli", "download", "Wan-AI/Wan2.2-TI2V-5B",
                                            "--local-dir", CKPT])):
        problem = problem or step(name, cmd, repo)
    return problem


def write_results(work, results):
    tmp = Path(work) / ".results.json.tmp"
    tmp.write_text(json.dumps(results, indent=2))
    os.replace(tmp, Path(work) / "results.json")


def main(root="/kaggle/input", work="/kaggle/working", scratch=None, run=subprocess.run, sleep=time.sleep,
         install=True):
    """`work` is what Kaggle returns, so it gets only results.json and the clips; the repo and weights go in `scratch`."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    scratch = Path(scratch or Path(tempfile.gettempdir()) / "heritage-short")
    scratch.mkdir(parents=True, exist_ok=True)
    folder, job = find_job(root, sleep=sleep)
    results = {"run_id": RUN_ID, "stage": "shots", "items": []}
    write_results(work, results)
    problem = install_wan(run, scratch) if install else ""
    if problem:
        results["items"] = [{"id": item["id"], "status": "error", "fingerprint": item["fingerprint"], "error": problem}
                            for item in job["items"]]
        write_results(work, results)
        return
    for item in job["items"]:
        out = work / f"{item['id']}.mp4"
        start = time.monotonic()
        code = run(build_generate_cmd(item, folder / item["inputs"][0], out), cwd=str(scratch / "Wan2.2")).returncode
        seconds = round(time.monotonic() - start, 1)
        record = {"id": item["id"], "fingerprint": item["fingerprint"], "seed": item["params"]["seed"],
                  "model": MODEL, "params": item["params"], "latency_s": seconds, "gpu_seconds": seconds}
        if code == 0 and out.exists():
            record.update(status="ok", file=out.name, sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                          bytes=out.stat().st_size)
        else:
            record.update(status="error", error=f"generate.py exited with {code}")
        results["items"].append(record)
        write_results(work, results)


if __name__ == "__main__":
    main()
