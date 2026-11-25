"""Shared plumbing: storyboard validation, fingerprints and manifest, atomic commits,
lock, runs.jsonl, retries, redaction, FFmpeg helpers and the stage command-line wrapper."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "storyboard.schema.json"
END_CARD_SECONDS = 3.0
ASSESSMENT_SHOTS = (8, 12)          # 10 Academy rules, checked only when storyboard.json sets assessment_rules
ASSESSMENT_SECONDS = (60.0, 90.0)   # film length including the end card
MAX_MUSIC_CAPTION = 512  # ACE-Step's documented caption limit
SECRET_ENV = ("GEMINI_API_KEY", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "KAGGLE_API_TOKEN", "KAGGLE_KEY")
RETRY_DELAYS = (2, 4)
FPS = 24
DEFAULT_SEED = 42
DEFAULT_END_CARD = "Made with AI"


class StoryboardError(Exception):
    pass


class PipelineExit(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = False, quota: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable, self.quota, self.status = retryable, quota, status


@dataclass
class Ctx:
    root: Path = ROOT
    mock: bool = False
    only: set = field(default_factory=set)
    sleep: object = time.sleep
    poll_s: float = field(default_factory=lambda: float(os.environ.get("HERITAGE_POLL_S", 30)))
    remote_timeout_s: float = field(default_factory=lambda: float(os.environ.get("HERITAGE_REMOTE_TIMEOUT_S", 10800)))
    provider_override: object = None

    @property
    def out(self) -> Path:
        return self.root / "outputs" / ("mock" if self.mock else "")

    @property
    def runs_path(self) -> Path:
        return self.out / "runs.jsonl" if self.mock else self.root / "runs.jsonl"

    @property
    def storyboard_path(self) -> Path:
        """Mock runs draft their own shots into outputs/mock/ and use them only while the real storyboard has none."""
        real, mock_copy = self.root / "storyboard.json", self.root / "outputs" / "mock" / "storyboard.json"
        if self.mock and mock_copy.exists() and not json.loads(real.read_text())["shots"]:
            return mock_copy
        return real

    @property
    def provider(self):
        if self.provider_override:
            return self.provider_override
        if self.mock:
            from pipeline import mock
            return mock
        from pipeline import providers
        return providers

    def path(self, *parts: str) -> Path:
        p = self.out.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def wanted(self, shot: dict) -> bool:
        return not self.only or shot["id"] in self.only


# ------------------------------------------------------------------ environment

def load_env(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file without overriding the real environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def redact(text: str) -> str:
    for name in SECRET_ENV:
        value = os.environ.get(name, "")
        if len(value) >= 4:
            text = text.replace(value, "***")
    text = re.sub(r"Bearer\s+[A-Za-z0-9._-]{12,}", "Bearer ***", text)
    return re.sub(r"AIza[0-9A-Za-z_-]{35}", "***", text)


# ------------------------------------------------------------------- storyboard

def composed_keyframe_prompt(sb: dict, shot: dict) -> str:
    return f"{sb['style']}. {sb['character']['description']}. {shot['keyframe_prompt']}"


def music_caption(sb: dict) -> str:
    """The score's caption: the music prompt, then a cue sheet of the per-shot cues at whole-second times."""
    starts = shot_starts(sb)
    cues = [f"{round(starts[s['id']])}-{round(starts[s['id']] + shot_seconds(s))}s {s['music_cue']}" for s in sb["shots"]]
    return f"{sb['music']['prompt']} Cue sheet: " + "; ".join(cues)


def load_storyboard(ctx: Ctx) -> dict:
    return json.loads(ctx.storyboard_path.read_text())


def validate(sb: dict, need_shots: bool = True, schema_path: Path = SCHEMA_PATH) -> None:
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(json.loads(schema_path.read_text()))
    errors = [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
              for e in sorted(validator.iter_errors(sb), key=lambda e: [str(p) for p in e.absolute_path])]
    if not errors:
        shots = sb["shots"]
        if len({s["id"] for s in shots}) != len(shots):
            errors.append("shots: ids must be unique")
        if need_shots and not shots:
            errors.append("shots: at least one shot is required (run the script stage first)")
        if not re.search(r"\bAI\b", sb.get("end_card", DEFAULT_END_CARD)):
            errors.append("end_card: must contain the word 'AI'")
        for i, shot in enumerate(shots):
            if len(composed_keyframe_prompt(sb, shot)) > 2048:
                errors.append(f"shots/{i}/keyframe_prompt: composed prompt exceeds 2048 characters")
        caption = len(music_caption(sb))
        if caption > MAX_MUSIC_CAPTION:
            errors.append(f"music: composed caption (prompt + cue sheet) is {caption} characters; "
                          f"the limit is {MAX_MUSIC_CAPTION}")
    if errors:
        raise StoryboardError("storyboard.json is invalid:\n  " + "\n  ".join(errors))


def assessment_on(sb: dict) -> bool:
    return bool(sb.get("assessment_rules"))


def check_assessment(sb: dict) -> None:
    """The 10 Academy rules (8-12 shots, 60-90 s including the end card), only when the storyboard opts in.
    No flag skips them; the shots are measured in whole frames, like the film itself."""
    if not assessment_on(sb):
        return
    (lo_n, hi_n), (lo_s, hi_s) = ASSESSMENT_SHOTS, ASSESSMENT_SECONDS
    n, total = len(sb["shots"]), (sum(shot_frames(s) for s in sb["shots"]) + END_CARD_SECONDS * FPS) / FPS
    problems = []
    if not lo_n <= n <= hi_n:
        problems.append(f"{n} shots (the brief asks for {lo_n}-{hi_n})")
    if not lo_s <= total <= hi_s:
        problems.append(f"length {total:.2f}s including the {END_CARD_SECONDS:.0f}s end card "
                        f"(the brief asks for {lo_s:.0f}-{hi_s:.0f}s)")
    if problems:
        raise StoryboardError("shots: " + "; ".join(problems)
                              + "; set assessment_rules to false to experiment outside these rules")


def assessment_note(sb: dict) -> str:
    return "assessment rules: " + ("on" if assessment_on(sb) else "off (shot count and length not checked)")


def seed_of(sb: dict) -> int:
    return sb.get("seed", DEFAULT_SEED)


def shot_seed(sb: dict, shot: dict) -> int:
    return (seed_of(sb) + zlib.crc32(shot["id"].encode())) % 2**31


def shot_frames(shot: dict) -> int:
    return round(shot["duration"] * FPS)


def shot_seconds(shot: dict) -> float:
    """A shot's length on the 24 fps timeline: whole frames, so picture, voice and subtitles never drift apart."""
    return shot_frames(shot) / FPS


def shot_starts(sb: dict) -> dict:
    starts, t = {}, 0.0
    for shot in sb["shots"]:
        starts[shot["id"]] = t
        t += shot_seconds(shot)
    return starts


# ------------------------------------------------------ fingerprints, commits, manifest

def fingerprint(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tmp_for(dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    return dst.with_name(f".{dst.name}.tmp-{os.getpid()}")


def commit_with(dst: Path, write, validate=None) -> None:
    """`write(tmp)` fills a temp file in the same directory; it replaces `dst` only after it validates."""
    tmp = _tmp_for(dst)
    try:
        write(tmp)
        with open(tmp, "rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        if validate:
            validate(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)


def commit_bytes(dst: Path, data: bytes, validate=None) -> None:
    commit_with(dst, lambda tmp: tmp.write_bytes(data), validate)


def commit_file(src: Path, dst: Path, validate=None) -> None:
    commit_with(dst, lambda tmp: shutil.copyfile(src, tmp), validate)


def cleanup_tmp(ctx: Ctx) -> None:
    for tmp in ctx.out.rglob(".*.tmp-*"):
        tmp.unlink(missing_ok=True)


def check_image(path: Path) -> None:
    from PIL import Image

    try:
        with Image.open(path) as img:
            img.load()
            if min(img.size) < 256:
                raise ProviderError(f"image too small: {img.size}")
    except OSError as exc:
        raise ProviderError(f"not a valid image: {exc}")


def check_wav(path: Path) -> None:
    try:
        with wave.open(str(path)) as w:
            if w.getnframes() == 0:
                raise ProviderError("empty audio")
    except (wave.Error, EOFError) as exc:
        raise ProviderError(f"not a valid wav: {exc}")


def run_tool(cmd: list, **kw) -> subprocess.CompletedProcess:
    """subprocess.run for FFmpeg's tools: a missing tool is a clean exit 1, not a traceback."""
    try:
        return subprocess.run(cmd, **kw)
    except FileNotFoundError:
        raise PipelineExit(1, f"{cmd[0]} not found; install FFmpeg (e.g. brew install ffmpeg)")


def probe(path: Path) -> dict:
    proc = run_tool(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                           str(path)], capture_output=True, text=True)
    if proc.returncode:
        raise ProviderError(f"unreadable media: {redact(proc.stderr.strip()[-200:])}")
    return json.loads(proc.stdout)


def media_seconds(path: Path) -> float:
    return float(probe(path)["format"]["duration"])


class Manifest:
    """outputs/manifest.json: key -> fingerprint of the inputs, plus the sha256 of the output."""

    def __init__(self, ctx: Ctx):
        self.path = ctx.path("manifest.json")
        try:
            self.data = json.loads(self.path.read_text()) if self.path.exists() else {"version": 1, "outputs": {}}
        except json.JSONDecodeError:
            raise PipelineExit(1, f"{self.path} is corrupt; delete it to rebuild (this regenerates every output)")

    def fresh(self, key: str, fp: str, path: Path) -> bool:
        entry = self.data["outputs"].get(key)
        return bool(entry and entry["fingerprint"] == fp and path.exists() and sha256_file(path) == entry["sha256"])

    def sha(self, key: str) -> str:
        return self.data["outputs"][key]["sha256"]

    def record(self, key: str, fp: str, path: Path) -> None:
        self.data["outputs"][key] = {"fingerprint": fp, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        commit_bytes(self.path, json.dumps(self.data, indent=2, sort_keys=True).encode())


@contextlib.contextmanager
def lock(ctx: Ctx):
    fd = os.open(ctx.path(".lock"), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise PipelineExit(1, "another pipeline process is using this outputs/ directory")
    try:
        yield
    finally:
        os.close(fd)


# --------------------------------------------------------------- logging and retries

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": _now(), **record}, ensure_ascii=False, sort_keys=True) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def log_run(ctx: Ctx, **fields) -> None:
    _append(ctx.runs_path, fields)


def log_timing(ctx: Ctx, stage: str, item: str, op: str, latency_s: float) -> None:
    _append(ctx.path("timings.jsonl"), {"stage": stage, "item": item, "op": op, "latency_s": round(latency_s, 3)})


def read_runs(path: Path) -> list:
    runs = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return runs


def model_record(*, stage, shot, provider, model, params, seed, attempt=1, status="ok", latency_s=0.0,
                 cost_usd=0.0, list_price_usd=None, **extra) -> dict:
    return dict(stage=stage, shot=shot, provider=provider, model=model, params=params, seed=seed, attempt=attempt,
                status=status, latency_s=round(latency_s, 3), cost_usd=cost_usd, list_price_usd=list_price_usd, **extra)


def call_model(ctx: Ctx, *, stage: str, shot, provider: str, model: str, params: dict, seed, fn,
               list_price_usd=None):
    """Run one model call: 3 attempts, delays 2 s then 4 s, one runs.jsonl line per attempt."""
    attempts = len(RETRY_DELAYS) + 1
    for attempt in range(1, attempts + 1):
        start = time.monotonic()
        base = dict(stage=stage, shot=shot, provider=provider, model=model, params=params, seed=seed,
                    attempt=attempt, cost_usd=0.0, list_price_usd=list_price_usd)
        try:
            result = fn()
        except ProviderError as exc:
            log_run(ctx, **base, status="error", error=redact(str(exc)), latency_s=round(time.monotonic() - start, 3))
            if not exc.retryable or attempt == attempts:
                exc.quota = exc.quota or exc.status == 429
                raise
            ctx.sleep(RETRY_DELAYS[attempt - 1])
        else:
            log_run(ctx, **base, status="ok", latency_s=round(time.monotonic() - start, 3))
            return result


# ----------------------------------------------------------------------- FFmpeg

def ffmpeg(*args, loglevel: str = "error") -> str:
    """Run ffmpeg; return its stderr."""
    proc = run_tool(["ffmpeg", "-y", "-hide_banner", "-loglevel", loglevel, *map(str, args)],
                          capture_output=True, text=True)
    if proc.returncode:
        raise ProviderError(f"ffmpeg failed: {redact(proc.stderr.strip()[-500:])}")
    return proc.stderr


def timed(ctx: Ctx, stage: str, item: str, op: str, fn):
    start = time.monotonic()
    result = fn()
    log_timing(ctx, stage, item, op, time.monotonic() - start)
    return result


# ------------------------------------------------------------------------- CLI

def stage_main(stage_fn, description: str, *, need_shots: bool = True, only_mode: str = "shots",
               extra_args=None, argv=None) -> None:
    """Parse flags, validate the storyboard, take the lock, run the stage, map errors to exit codes."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mock", action="store_true", help="placeholder providers; no network, no credentials")
    parser.add_argument("--only", nargs="+", default=[], metavar="SHOT_ID", help="limit to these shots")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    if extra_args:
        extra_args(parser)
    args = parser.parse_args(argv)
    load_env(args.root / ".env")
    ctx = Ctx(root=args.root, mock=args.mock, only=set(args.only))
    try:
        code = run_stage(ctx, stage_fn, args, need_shots=need_shots, only_mode=only_mode)
    except PipelineExit as exc:
        code, message = exc.code, str(exc)
    except StoryboardError as exc:
        code, message = 1, str(exc)
    except ProviderError as exc:
        code, message = (2 if exc.quota else 1), str(exc)
    else:
        message = ""
    if message:
        print(f"error: {redact(message)}", file=sys.stderr)
    raise SystemExit(code or 0)


def run_stage(ctx: Ctx, stage_fn, args, *, need_shots=True, only_mode="shots") -> int:
    sb = load_storyboard(ctx)
    validate(sb, need_shots=need_shots)
    if ctx.only:
        if only_mode == "error":
            raise PipelineExit(1, "--only is not supported by this stage (it needs every shot)")
        if only_mode == "ignore":
            print("note: --only is ignored by this stage")
            ctx.only = set()
        unknown = ctx.only - {s["id"] for s in sb["shots"]}
        if unknown:
            raise PipelineExit(1, f"unknown shot id(s): {', '.join(sorted(unknown))}")
    with lock(ctx):
        cleanup_tmp(ctx)
        return stage_fn(ctx, sb, args) or 0
