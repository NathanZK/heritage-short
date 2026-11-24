"""Remote execution on Kaggle through the documented `kaggle` CLI (datasets version, kernels push,
status, output). A stage with stale items builds a job dataset, pushes a kernel bound to that job by
a run id, polls it, downloads its output and returns only results that match what was asked for."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pipeline.common import ROOT, Ctx, PipelineExit, log_run, model_record, redact, sha256_file

ACCELERATOR = "NvidiaTeslaT4"
KERNELS = ROOT / "kernels"
FAIL_WORDS = {"error", "fail", "failed", "cancel", "cancelled", "canceled"}


@dataclass
class Collected:
    run_id: str
    accepted: list = field(default_factory=list)   # [(result item, Path to downloaded file)]; anything else is missing
    rejected: dict = field(default_factory=dict)   # id -> reason
    submitted: list = field(default_factory=list)  # ids that were in the Kaggle run (a resumed run's, not today's)
    state: str = "finished"                        # what `kernels status` said last: finished | failed
    delivered: bool = False                        # results.json for THIS run_id came back

    @property
    def done(self) -> bool:
        """The run is over: it delivered results, or Kaggle says it failed. Otherwise keep pending.json and resume."""
        return self.delivered or self.state == "failed"


def cli(*args) -> str:
    try:
        proc = subprocess.run(["kaggle", *map(str, args)], capture_output=True, text=True)
    except FileNotFoundError:
        raise PipelineExit(1, "the `kaggle` CLI was not found; install it (it needs Python 3.11+): pipx install kaggle")
    if proc.returncode:
        raise PipelineExit(1, redact(f"kaggle {args[0]} {args[1]} failed: {(proc.stderr or proc.stdout).strip()[-400:]}"))
    return proc.stdout


def classify(status_text: str) -> str:
    """Plain-text matcher (the real status strings are undocumented, assumption A5)."""
    words = set(re.findall(r"[a-z]+", status_text.lower()))
    if words & FAIL_WORDS:
        return "failed"
    return "finished" if "complete" in words or "completed" in words else "running"


def owner() -> str:
    name = os.environ.get("KAGGLE_USERNAME", "")
    if not name:
        raise PipelineExit(1, "KAGGLE_USERNAME is not set; see .env.example")
    return name


def _fresh_dir(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def build_dataset(ctx: Ctx, stage: str, run_id: str, items: list, inputs: dict) -> Path:
    """Rebuilt from empty each time, from an allowlist: job.json, dataset-metadata.json and the named inputs."""
    folder = _fresh_dir(ctx.path("jobs", stage, "dataset"))
    (folder / "job.json").write_text(json.dumps({"run_id": run_id, "stage": stage, "items": items}, indent=2))
    write_dataset_metadata(folder, stage)
    for name, src in inputs.items():
        shutil.copyfile(src, folder / name)
    return folder


def write_dataset_metadata(folder: Path, stage: str) -> None:
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": f"heritage-short-{stage}-jobs", "id": f"{owner()}/heritage-short-{stage}-jobs",
        "licenses": [{"name": "other"}]}, indent=2))


def render_kernel(template: Path, run_id: str, ids: list) -> str:
    code = template.read_text()
    code = re.sub(r'^RUN_ID = ".*"$', f'RUN_ID = "{run_id}"', code, flags=re.M)
    return re.sub(r"^EXPECTED_IDS = .*$", f"EXPECTED_IDS = {json.dumps(ids)}", code, flags=re.M)


def build_kernel(ctx: Ctx, stage: str, run_id: str, ids: list) -> Path:
    folder = _fresh_dir(ctx.path("jobs", stage, "kernel"))
    code_file = f"{stage}_kernel.py"
    (folder / code_file).write_text(render_kernel(KERNELS / code_file, run_id, ids))
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{owner()}/heritage-short-{stage}", "title": f"heritage-short-{stage}", "code_file": code_file,
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "machine_shape": ACCELERATOR,
        "dataset_sources": [f"{owner()}/heritage-short-{stage}-jobs"]}, indent=2))
    return folder


def poll(ctx: Ctx, ref: str) -> str:
    waited, failures = 0.0, 0
    while True:
        try:
            state = classify(cli("kernels", "status", ref))
            failures = 0
        except PipelineExit:
            failures += 1
            if failures >= 3:
                raise
            state = "running"
        if state != "running":
            return state
        if waited >= ctx.remote_timeout_s:
            raise PipelineExit(2, "timed out waiting for the kernel; it may still be running on Kaggle. "
                                  "outputs/jobs/<stage>/pending.json is kept, so re-running resumes it.")
        ctx.sleep(ctx.poll_s)
        waited += ctx.poll_s


def judge(item: dict, expected: dict, run_id: str, incoming: Path, result_run_id: str) -> tuple:
    """Return (path, None) when the kernel's record for this item is trustworthy, else (None, reason)."""
    if result_run_id != run_id:
        return None, "run_id does not match this run"
    if item.get("status") != "ok":
        return None, f"kernel reported {item.get('error') or item.get('status')}"
    if item.get("fingerprint") != expected[item["id"]]:
        return None, "fingerprint does not match the current inputs"
    name = item.get("file")
    if not isinstance(name, str) or not name or name in (".", "..") or name != Path(name).name:
        return None, "result file name is not a plain file name"
    path = incoming / name
    if path.is_symlink():
        return None, "result file is a symlink"
    if not path.is_file() or sha256_file(path) != item.get("sha256"):
        return None, "file missing or sha256 mismatch"
    return path, None


def collect(ctx: Ctx, stage: str, items: list, inputs: dict, resubmit: bool = False) -> Collected:
    expected = {i["id"]: i["fingerprint"] for i in items}
    pending_path = ctx.path("jobs", stage, "pending.json")
    ref = f"{owner()}/heritage-short-{stage}"
    if resubmit:
        pending_path.unlink(missing_ok=True)
    submitted = list(expected)
    if pending_path.exists():
        pending = json.loads(pending_path.read_text())
        run_id, submitted = pending["run_id"], pending["item_ids"]
        print(f"{stage}: resuming kernel run {run_id}")
    else:
        run_id = f"{stage}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        dataset = build_dataset(ctx, stage, run_id, items, inputs)
        kernel = build_kernel(ctx, stage, run_id, list(expected))
        cli("datasets", "version", "-p", dataset, "-m", run_id)
        cli("kernels", "push", "-p", kernel, "--accelerator", ACCELERATOR, "-t", int(ctx.remote_timeout_s))
        pending_path.write_text(json.dumps({
            "run_id": run_id, "kernel": ref, "dataset": f"{ref}-jobs", "item_ids": list(expected),
            "pushed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
    state = poll(ctx, ref)
    incoming = _fresh_dir(ctx.path("_incoming", stage))
    cli("kernels", "output", ref, "-p", incoming, "-o")
    found = incoming / "results.json"
    found = found if found.exists() else next(incoming.rglob("results.json"), None)
    try:
        results = json.loads(found.read_text()) if found else {}
    except json.JSONDecodeError:
        print(f"{stage}: results.json from Kaggle is not valid JSON; treating the run as having no results")
        results = {}
    if not isinstance(results, dict) or not isinstance(results.get("items"), list):
        results = {}
    results = {"run_id": results.get("run_id"), "items": [i for i in results.get("items", []) if isinstance(i, dict)]}
    out = Collected(run_id, submitted=submitted, state=state, delivered=results["run_id"] == run_id)
    returned = {i["id"]: i for i in results["items"] if i.get("id") in expected}
    for item_id in returned:
        path, reason = judge(returned[item_id], expected, run_id, found.parent, results["run_id"])
        if path:
            out.accepted.append((returned[item_id], path))
        else:
            out.rejected[item_id] = reason
    if not out.done:
        print(f"{stage}: Kaggle reported {state} but returned no results for run {run_id}; the run stays pending, so "
              f"re-running resumes it (use --resubmit to start over)")
    return out


def log_failures(ctx: Ctx, stage: str, items: list, ingested: list, res: Collected, model: str) -> None:
    """One runs.jsonl error line for every submitted item that did not come back usable."""
    for item in items:
        if item["id"] in res.submitted and item["id"] not in ingested:
            reason = res.rejected.get(item["id"]) or f"no result returned (run state: {res.state})"
            log_run(ctx, **model_record(stage=stage, shot=item["id"] if stage == "shots" else None, provider="kaggle",
                                        model=model, params=item["params"], seed=item["params"].get("seed"),
                                        status="error", error=redact(reason), run_id=res.run_id))


def finish(ctx: Ctx, stage: str) -> None:
    """Call after ingesting: the run is done, so the next run submits fresh instead of resuming."""
    ctx.path("jobs", stage, "pending.json").unlink(missing_ok=True)


def setup(ctx: Ctx) -> None:
    """One-time: create the two private job datasets (kaggle datasets create is private by default)."""
    for stage in ("shots", "music"):
        folder = _fresh_dir(ctx.path("jobs", stage, "setup"))
        (folder / "job.json").write_text(json.dumps({"run_id": "setup", "stage": stage, "items": []}))
        write_dataset_metadata(folder, stage)
        cli("datasets", "create", "-p", folder)
        print(f"kaggle: created private dataset {owner()}/heritage-short-{stage}-jobs")


def main(argv=None) -> None:
    import argparse
    from pipeline.common import load_env

    parser = argparse.ArgumentParser(description="Kaggle helper: `setup` creates the two private job datasets.")
    parser.add_argument("command", choices=["setup"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    load_env(args.root / ".env")
    try:
        setup(Ctx(root=args.root))
    except PipelineExit as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
