"""Stage: music. One original score with ACE-Step 1.5, run on Kaggle (see kernels/music_kernel.py).

The kernel makes three candidates. Listen, then pick one by setting music.choice {candidate, reason} in
storyboard.json; the stage copies it to outputs/music/score.wav (a local step, no model call)."""
from __future__ import annotations

import json
import math

from pipeline import kaggle
from pipeline.common import (END_CARD_SECONDS, Manifest, ProviderError, check_wav, commit_bytes, commit_file,
                             fingerprint, log_run, media_seconds, model_record, music_caption, seed_of, stage_main)
from pipeline.mock import sine_wav

V = 1
MODEL = "ACE-Step/Ace-Step1.5"
CANDIDATES = 3


def film_seconds(sb: dict) -> float:
    return sum(s["duration"] for s in sb["shots"]) + END_CARD_SECONDS


def bucket_seconds(sb: dict) -> float:
    return float(math.ceil(film_seconds(sb) / 10) * 10)


def candidate_fp(sb: dict, n: int) -> str:
    seed = seed_of(sb)
    return fingerprint(V, "music", MODEL, music_caption(sb), bucket_seconds(sb), seed + n - 1)


def candidate_path(ctx, n: int):
    return ctx.path("music", "candidates", f"candidate_{n}.wav")


def check_candidate(minimum: float):
    def check(path):
        check_wav(path)
        if media_seconds(path) < minimum:
            raise ProviderError(f"candidate is shorter than the requested {minimum + 1:.0f}s")
    return check


def ingest(ctx, sb, res, manifest) -> list:
    """Commit accepted candidates; returns the ids that were ingested. Idempotent per run."""
    index = ctx.path("music", "candidates.json")
    old = json.loads(index.read_text()) if index.exists() else {}
    logged, done = old.get("run_ids_logged", []), []
    for item, path in res.accepted:
        n = int(item["id"].rsplit("_", 1)[1])
        dst = candidate_path(ctx, n)
        try:
            commit_file(path, dst, check_candidate(bucket_seconds(sb) - 1))
        except ProviderError as exc:
            res.rejected[item["id"]] = str(exc)
            continue
        manifest.record(f"music:candidate_{n}", candidate_fp(sb, n), dst)
        done.append(item["id"])
        if res.run_id not in logged:
            log_run(ctx, **model_record(stage="music", shot=None, provider="kaggle", model=item["model"],
                                        params=item["params"], seed=item["seed"], latency_s=item["latency_s"],
                                        gpu_seconds=item["gpu_seconds"], run_id=res.run_id))
    if done and res.run_id not in logged:
        commit_bytes(index, json.dumps({"run_ids_logged": logged + [res.run_id]}).encode())
    return done


def make_candidates(ctx, sb, manifest, args) -> int:
    """Generate the candidates (all three, they come from one batch). Returns 0 or an exit code."""
    if ctx.mock:
        for n in range(1, CANDIDATES + 1):
            dst = candidate_path(ctx, n)
            commit_bytes(dst, sine_wav(2.0, freq=110.0 * n, rate=8000, volume=0.2))
            manifest.record(f"music:candidate_{n}", candidate_fp(sb, n), dst)
            log_run(ctx, **model_record(stage="music", shot=None, provider="kaggle", model="mock-ace-step",
                                        params={"candidate": n}, seed=seed_of(sb) + n - 1))
        return 0
    items = [{"id": f"candidate_{n}", "fingerprint": candidate_fp(sb, n), "inputs": [],
              "params": {"caption": music_caption(sb), "duration": bucket_seconds(sb),
                         "seed": seed_of(sb) + n - 1}} for n in range(1, CANDIDATES + 1)]
    res = kaggle.collect(ctx, "music", items, {}, resubmit=args.resubmit)
    ingested = ingest(ctx, sb, res, manifest)
    for item_id, reason in res.rejected.items():
        print(f"music: rejected {item_id}: {reason}")
    if res.done:
        kaggle.log_failures(ctx, "music", items, ingested, res, MODEL)
        kaggle.finish(ctx, "music")
    missing = [n for n in range(1, CANDIDATES + 1)
               if not manifest.fresh(f"music:candidate_{n}", candidate_fp(sb, n), candidate_path(ctx, n))]
    if missing:
        print(f"music: candidates {missing} were not delivered; re-run to resume")
        return 2
    return 0


def candidates_fingerprint(sb: dict) -> str:
    return fingerprint(*[candidate_fp(sb, n) for n in range(1, CANDIDATES + 1)])


def current_choice(ctx, sb: dict):
    """The owner's choice; a mock run picks candidate 1 on its own."""
    return sb["music"].get("choice") or ({"candidate": 1, "reason": "mock"} if ctx.mock else None)


def score_ready(ctx, sb: dict, manifest: Manifest) -> bool:
    """True when score.wav is the current choice made from the current candidates."""
    choice = current_choice(ctx, sb)
    info_path = ctx.out / "music" / "score.json"
    if not choice or not info_path.exists():
        return False
    info = json.loads(info_path.read_text())
    fp = fingerprint(V, "score", manifest.data["outputs"].get(f"music:candidate_{choice['candidate']}", {}).get("sha256"))
    return ((info.get("candidate"), info.get("reason")) == (choice["candidate"], choice["reason"])
            and info.get("candidates_fingerprint") == candidates_fingerprint(sb)
            and manifest.fresh("score", fp, ctx.out / "music" / "score.wav"))


def run(ctx, sb, args) -> int:
    manifest = Manifest(ctx)
    regenerated = False
    if not all(manifest.fresh(f"music:candidate_{n}", candidate_fp(sb, n), candidate_path(ctx, n))
               for n in range(1, CANDIDATES + 1)):
        code = make_candidates(ctx, sb, manifest, args)
        if code:
            return code
        regenerated = True
    choice = current_choice(ctx, sb)
    if not choice:
        print("music: listen to outputs/music/candidates/ and set music.choice {candidate, reason} in storyboard.json")
        if args.draft:
            print("music: --draft: continuing without a score")
            return 0
        return 3
    score, info_path = ctx.path("music", "score.wav"), ctx.path("music", "score.json")
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    cands_fp = candidates_fingerprint(sb)
    key = f"music:candidate_{choice['candidate']}"
    same_choice = (info.get("candidate"), info.get("reason")) == (choice["candidate"], choice["reason"])
    changed = info.get("candidates_fingerprint") != cands_fp or info.get("candidate_sha256") != manifest.sha(key)
    auto = ctx.mock and not sb["music"].get("choice")
    pick = {"candidate": choice["candidate"], "reason": choice["reason"]}
    if info and not auto and (regenerated or (same_choice and changed) or info.get("awaiting_choice") == pick):
        if info.get("awaiting_choice") != pick:   # remember the choice made before listening, so it stays stale
            commit_bytes(info_path, json.dumps({**info, "awaiting_choice": pick}, indent=2).encode())
        print("music: the candidates were regenerated, so an earlier choice no longer applies; listen, then choose "
              "again (change music.choice candidate or reason)")
        return 0 if args.draft else 3
    if not score_ready(ctx, sb, manifest):
        commit_file(candidate_path(ctx, choice["candidate"]), score, check_wav)
        manifest.record("score", fingerprint(V, "score", manifest.sha(key)), score)
        commit_bytes(info_path, json.dumps({
            "candidates_fingerprint": cands_fp, "candidate": choice["candidate"], "reason": choice["reason"],
            "candidate_sha256": manifest.sha(key), "sha256": manifest.sha("score")}, indent=2).encode())
    print(f"music: score is candidate {choice['candidate']} ({choice['reason']})")
    return 0


def main(argv=None) -> None:
    def extra(p):
        p.add_argument("--draft", action="store_true", help="continue without a score when none is chosen")
        p.add_argument("--resubmit", action="store_true", help="discard a pending Kaggle run and submit again")
    stage_main(run, __doc__, only_mode="ignore", extra_args=extra, argv=argv)


if __name__ == "__main__":
    main()
