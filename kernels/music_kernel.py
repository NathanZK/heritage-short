"""Kaggle kernel: ACE-Step 1.5 text-to-music, three candidates in one documented `generate_music` call.

Pushed by pipeline/kaggle.py as a single generated file (RUN_ID and EXPECTED_IDS are filled in at push
time). It only runs on Kaggle; the tests cover its pure functions. Works from the ACE-Step docs'
Quick Start: uv, language-model handler, turbo DiT. uv is pip-installed and run as `python -m uv`, so PATH does not
matter (unverified on Kaggle, README M5). Kaggle T4 behaviour is UNVERIFIED (spec A10)."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RUN_ID = "__RUN_ID__"
EXPECTED_IDS = []
MODEL = "ACE-Step/Ace-Step1.5"
REPO = "https://github.com/ACE-Step/ACE-Step-1.5"

# Runs inside the ACE-Step project environment (`uv run python runner.py args.json`).
RUNNER = '''
import json, sys
from pathlib import Path
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

args = json.load(open(sys.argv[1]))
root = Path.cwd()
dit_handler, llm_handler = AceStepHandler(), LLMHandler()
dit_handler.initialize_service(project_root=str(root), config_path="acestep-v15-turbo", device="cuda")
for backend in ("vllm", "pt"):
    try:
        llm_handler.initialize(checkpoint_dir=str(root / "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B",
                               backend=backend, device="cuda")
        print("language-model backend:", backend)
        break
    except Exception as exc:
        print("backend", backend, "failed:", exc)
else:
    raise SystemExit("no language-model backend worked")
result = generate_music(dit_handler, llm_handler, GenerationParams(**args["params"]),
                        GenerationConfig(**args["config"]), save_dir=args["save_dir"])
if not result.success:
    raise SystemExit(str(result.error))
json.dump([audio["path"] for audio in result.audios], open(args["paths_file"], "w"))
'''


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


def build_generation(job):
    first, seeds = job["items"][0]["params"], [i["params"]["seed"] for i in job["items"]]
    return {"params": {"task_type": "text2music", "caption": first["caption"], "lyrics": "[Instrumental]",
                       "instrumental": True, "duration": first["duration"], "inference_steps": 8, "shift": 3.0},
            "config": {"batch_size": len(seeds), "use_random_seed": False, "seeds": seeds, "audio_format": "wav"}}


def write_results(work, results):
    tmp = Path(work) / ".results.json.tmp"
    tmp.write_text(json.dumps(results, indent=2))
    os.replace(tmp, Path(work) / "results.json")


def setup_failed(proc) -> str:
    """'' when a setup command succeeded, else its exit code and the tail of its output."""
    if proc.returncode == 0:
        return ""
    tail = ((getattr(proc, "stderr", "") or "") + (getattr(proc, "stdout", "") or "")).strip()[-300:]
    return f"exited with {proc.returncode}" + (f": {tail}" if tail else "")


def setup_ace_step(run, scratch, project):
    """Set up ACE-Step; returns '' or a message naming the step that failed."""
    steps = [("pip install uv", [sys.executable, "-m", "pip", "install", "uv"], scratch),
             ("git clone", ["git", "clone", REPO], scratch),
             ("uv sync", [sys.executable, "-m", "uv", "sync"], project)]
    for name, cmd, cwd in steps:
        problem = setup_failed(run(cmd, cwd=str(cwd), capture_output=True, text=True))
        if problem:
            return f"setup step '{name}' {problem}"
    return ""


def main(root="/kaggle/input", work="/kaggle/working", scratch=None, run=subprocess.run, sleep=time.sleep,
         install=True):
    """`work` is what Kaggle returns, so it gets only results.json and the candidates; the repo, venv and weights go in `scratch`."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    scratch = Path(scratch or Path(tempfile.gettempdir()) / "heritage-short")
    scratch.mkdir(parents=True, exist_ok=True)
    folder, job = find_job(root, sleep=sleep)
    results = {"run_id": RUN_ID, "stage": "music", "items": []}
    write_results(work, results)
    project = scratch / "ACE-Step-1.5"
    problem = setup_ace_step(run, scratch, project) if install else ""
    if problem:
        results["items"] = [{"id": item["id"], "status": "error", "fingerprint": item["fingerprint"], "error": problem}
                            for item in job["items"]]
        write_results(work, results)
        return
    project.mkdir(parents=True, exist_ok=True)
    generated, paths_file = scratch / "generated", scratch / "paths.json"
    generated.mkdir(exist_ok=True)
    (project / "runner.py").write_text(RUNNER)
    args = {**build_generation(job), "save_dir": str(generated), "paths_file": str(paths_file)}
    (project / "args.json").write_text(json.dumps(args))
    start = time.monotonic()
    code = run([sys.executable, "-m", "uv", "run", "python", "runner.py", "args.json"], cwd=str(project)).returncode
    seconds = round(time.monotonic() - start, 1)
    paths = json.loads(paths_file.read_text()) if code == 0 and paths_file.exists() else []
    for item, path in zip(job["items"], paths):
        out = work / f"{item['id']}.wav"
        shutil.copyfile(path, out)
        results["items"].append({
            "id": item["id"], "status": "ok", "fingerprint": item["fingerprint"], "file": out.name,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "bytes": out.stat().st_size,
            "seed": item["params"]["seed"], "model": MODEL, "params": item["params"],
            "latency_s": seconds, "gpu_seconds": seconds})
    for item in job["items"][len(paths):]:
        results["items"].append({"id": item["id"], "status": "error", "fingerprint": item["fingerprint"],
                                 "error": f"generation exited with {code}"})
    write_results(work, results)


if __name__ == "__main__":
    main()
