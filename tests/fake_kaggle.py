"""A stand-in `kaggle` executable for tests. Behaviour is chosen by FAKE_KAGGLE_SCENARIO (json) and state
lives in FAKE_KAGGLE_STATE. The status text is INVENTED (the real strings are undocumented, check M4)."""
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
state = Path(os.environ["FAKE_KAGGLE_STATE"])
scenario = {"mode": "ok", "running_polls": 2, **json.loads(os.environ.get("FAKE_KAGGLE_SCENARIO", "{}"))}
args = sys.argv[1:]


def opt(flag):
    return args[args.index(flag) + 1] if flag in args else None


def bump(name):
    f = state / name
    n = int(f.read_text()) + 1 if f.exists() else 1
    f.write_text(str(n))
    return n


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


cmd = " ".join(args[:2])
(state / "calls.log").open("a").write(" ".join(args) + "\n")
if cmd in ("datasets version", "datasets create"):
    if scenario.get("leak"):
        fail("auth failed key=" + os.environ["KAGGLE_KEY"])
    src = Path(opt("-p"))
    if not (src / "dataset-metadata.json").exists():
        fail("dataset-metadata.json missing")
    shutil.rmtree(state / "dataset", ignore_errors=True)
    shutil.copytree(src, state / "dataset")
    bump("datasets")
elif cmd == "kernels push":
    src = Path(opt("-p"))
    meta = json.loads((src / "kernel-metadata.json").read_text())
    code = (src / meta["code_file"]).read_text()
    (state / "kernel.json").write_text(json.dumps({
        "meta": meta, "run_id": re.search(r'^RUN_ID = "(.*)"$', code, re.M).group(1), "args": args}))
    (state / "polls").unlink(missing_ok=True)
    bump("pushes")
elif cmd == "kernels status":
    if scenario.get("status_breaks"):
        fail("Kaggle is down")
    n = bump("polls")
    if scenario["mode"] == "never" or n <= scenario["running_polls"]:
        print('user/kernel has status "KernelWorkerStatus.RUNNING"')
    elif scenario["mode"] in ("fail", "partial"):
        print('user/kernel has status "KernelWorkerStatus.ERROR"')
    else:
        print('user/kernel has status "KernelWorkerStatus.COMPLETE"')
elif cmd == "kernels output":
    dest = Path(opt("-p"))
    dest.mkdir(parents=True, exist_ok=True)
    mode = scenario["mode"]
    if mode in ("fail", "no_results"):
        sys.exit(0)
    if scenario.get("junk"):
        (dest / "repo").mkdir()
        (dest / "repo" / "results.json").write_text(json.dumps({"run_id": "someone-else", "items": []}))
    if mode == "bad_json":
        (dest / "results.json").write_text("{not json")
        sys.exit(0)
    if mode == "bad_shape":
        (dest / "results.json").write_text(json.dumps(scenario.get("body", [])))
        sys.exit(0)
    job = json.loads((state / "dataset" / "job.json").read_text())
    run_id = json.loads((state / "kernel.json").read_text())["run_id"]
    items = job["items"][:1] if mode == "partial" else job["items"]
    if mode == "kernel_error":
        results = {"run_id": json.loads((state / "kernel.json").read_text())["run_id"], "stage": job["stage"],
                   "items": [{"id": i["id"], "status": "error", "fingerprint": i["fingerprint"],
                              "error": "RuntimeError: CUDA out of memory"} for i in items]}
        (dest / "results.json").write_text(json.dumps(results))
        sys.exit(0)
    results = {"run_id": "someone-else" if mode == "wrong_run" else run_id, "stage": job["stage"], "items": []}
    for item in items:
        if job["stage"] == "shots":
            out = dest / f"{item['id']}.mp4"
            src = os.environ["FAKE_KAGGLE_SHORT_CLIP" if mode == "short" else "FAKE_KAGGLE_CLIP"]
            frames = item["params"].get("num_frames")
            if mode == "short" or not frames:
                shutil.copyfile(src, out)
            else:   # like the kernel: exactly the frames asked for
                import subprocess
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-frames:v", str(frames), "-c:v",
                                "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", str(out)], check=True)
        else:
            from pipeline import mock
            out = dest / f"{item['id']}.wav"
            seconds = 1.0 if mode == "short" else item["params"]["duration"]
            out.write_bytes(mock.sine_wav(seconds, rate=8000, freq=100 + 10 * bump("pushes_seen") if scenario.get("vary") else 110))
        results["items"].append({
            "id": item["id"], "status": "ok", "fingerprint": "stale" if mode == "bad_fingerprint" else item["fingerprint"],
            "file": out.name, "sha256": "0" * 64 if mode == "bad_sha" else sha(out), "bytes": out.stat().st_size,
            "seed": item["params"].get("seed"), "model": "fake-model", "params": item["params"], "latency_s": 1.5,
            "gpu_seconds": 1.5})
    (dest / "results.json").write_text(json.dumps(results))
else:
    fail(f"unsupported fake command: {args}")
