"""Run the whole pipeline: script (if there are no shots yet), keyframes, narration, shots, music, assemble.

    make film                                    # real providers (needs credentials; see README)
    make film PIPELINE_FLAGS="--mock --draft"    # placeholder providers, no credentials
Each stage skips work whose inputs have not changed. The run stops at the first stage that cannot finish and
exits with that stage's code: 2 = incomplete but resumable (quota, remote work pending), 3 = choose a score."""
from __future__ import annotations

from types import SimpleNamespace

from pipeline import assemble, keyframes, music, narration, script, shots
from pipeline.common import check_film, load_storyboard, stage_main, validate


def run_all(ctx, sb, args) -> int:
    if args.force or not sb["shots"]:
        print("run: script")
        if code := script.run(ctx, sb, SimpleNamespace(force=args.force, draft=args.draft)):
            return code
        sb = load_storyboard(ctx)
    validate(sb)
    if not args.draft:
        check_film(sb)
    stages = [
        ("keyframes", keyframes.run, SimpleNamespace()),
        ("narration", narration.run, SimpleNamespace()),
        ("shots", shots.run, SimpleNamespace(fallback=args.fallback, resubmit=args.resubmit, retry_fallbacks=args.retry_fallbacks)),
        ("music", music.run, SimpleNamespace(draft=args.draft, resubmit=args.resubmit)),
    ]
    for name, stage, stage_args in stages:
        print(f"run: {name}")
        if code := stage(ctx, sb, stage_args):
            return code
    if ctx.only:
        print("run: assemble skipped (--only builds only part of the film)")
        return 0
    print("run: assemble")
    return assemble.run(ctx, sb, SimpleNamespace(draft=args.draft, radio=False))


def main(argv=None) -> None:
    def extra(p):
        p.add_argument("--draft", action="store_true", help="skip the 8-12 shot / 60-90 s rules; allow no score")
        p.add_argument("--fallback", action="store_true", help="pan-and-zoom clips for shots Wan could not make")
        p.add_argument("--retry-fallbacks", action="store_true", help="try Wan again for shots that have a fallback clip")
        p.add_argument("--force", action="store_true", help="redraft the shots with the script stage")
        p.add_argument("--resubmit", action="store_true", help="discard a pending Kaggle run and submit again")
    stage_main(run_all, __doc__, need_shots=False, extra_args=extra, argv=argv)


if __name__ == "__main__":
    main()
