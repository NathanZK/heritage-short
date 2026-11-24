"""Stage: script. Draft the storyboard's shots from its brief with the Gemini API."""
from __future__ import annotations

import copy
import json
import shutil
import time

from pipeline.common import (END_CARD_SECONDS, MAX_MUSIC_CAPTION, MAX_SECONDS, MAX_SHOTS, MIN_SECONDS, MIN_SHOTS,
                             PipelineExit, StoryboardError, call_model, check_film, commit_bytes, music_caption,
                             stage_main, validate)

_STR = {"type": "string"}
SHOTS_SCHEMA = {
    "type": "object",
    "properties": {"shots": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": _STR, "duration": {"type": "number"}, "reason": _STR, "keyframe_prompt": _STR,
                       "motion_prompt": _STR, "narration": _STR, "subtitle": _STR, "music_cue": _STR},
        "required": ["id", "duration", "reason", "keyframe_prompt", "motion_prompt", "narration", "subtitle",
                     "music_cue"]}}},
    "required": ["shots"],
}


MIN_CUE = 8   # characters: below this a music cue says nothing


def cue_allowance(sb: dict) -> int:
    """How many characters each music_cue may use so that even a film of MAX_SHOTS longest shots keeps the whole
    ACE-Step caption (music.prompt plus the timed cue sheet) within its limit."""
    worst = {"music": sb["music"], "shots": [{"id": f"s{i:02d}", "duration": 8, "music_cue": ""}
                                              for i in range(MAX_SHOTS)]}
    return min(80, (MAX_MUSIC_CAPTION - len(music_caption(worst))) // MAX_SHOTS)


def build_prompt(sb: dict, problem: str | None = None) -> str:
    b = sb["brief"]
    prompt = (
        f"You are drafting the storyboard for a short film titled '{sb['title']}'.\n"
        f"Premise: {b['premise']}\nAudience: {b['audience']}\nIntent: {b['intent']}\n"
        f"Style (added to every image prompt later; do not repeat it): {sb['style']}\n"
        f"Main character (described later; do not repeat it): {sb['character']['description']}\n\n"
        f"Write {MIN_SHOTS} to {MAX_SHOTS} shots. Each duration is at most 8 seconds. The image-to-video model makes "
        f"clips of about 5 seconds and a longer shot holds its last frame, so use shots of about 5 seconds when you "
        f"write {MAX_SHOTS}, and longer ones when you write fewer. The durations must add up to between "
        f"{MIN_SECONDS - END_CARD_SECONDS:.0f} and {MAX_SECONDS - END_CARD_SECONDS:.0f} seconds. "
        f"For each shot give: id (s01, s02, ...), duration, reason (why this shot exists), keyframe_prompt (what "
        f"the still image shows), motion_prompt (camera and subject movement), narration (one line in "
        f"{sb['language']}, short enough to speak within the duration), subtitle (the same line in English) and "
        f"music_cue (what the score does under this shot, a few words, at most {cue_allowance(sb)} characters).")
    return prompt + (f"\n\nYour previous answer was rejected: {problem}\nFix this." if problem else "")


def run(ctx, sb, args) -> int:
    if sb["shots"] and not args.force:
        print(f"script: storyboard already has {len(sb['shots'])} shots; use --force to redraft")
        return 0
    brief = sb.get("brief")
    if not brief or (not ctx.mock and brief["premise"].startswith("TODO")):
        raise PipelineExit(1, "fill in 'brief' (and style, character, music) in storyboard.json before drafting")
    if cue_allowance(sb) < MIN_CUE:
        raise PipelineExit(1, f"music.prompt is too long: only {cue_allowance(sb)} characters per music cue would "
                              f"fit in the {MAX_MUSIC_CAPTION}-character ACE-Step caption; shorten it")
    provider = ctx.provider
    problem = None
    for _ in range(2):
        result = call_model(ctx, stage="script", shot=None, provider="gemini", model=provider.TEXT_MODEL,
                            params={"retry_reason": problem} if problem else {}, seed=None,
                            fn=lambda: provider.gemini_json(build_prompt(sb, problem), SHOTS_SCHEMA))
        draft = copy.deepcopy(sb)
        draft["shots"] = result.get("shots") if isinstance(result, dict) else None
        try:
            if not isinstance(draft["shots"], list):
                raise StoryboardError("the response had no 'shots' list")
            validate(draft)
            if not args.draft:
                check_film(draft)
            break
        except StoryboardError as exc:
            problem = str(exc)
    else:
        raise StoryboardError(f"the drafted storyboard was rejected twice:\n{problem}")
    target = ctx.root / "outputs" / "mock" / "storyboard.json" if ctx.mock else ctx.root / "storyboard.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if sb["shots"]:
        shutil.copyfile(ctx.storyboard_path, target.with_name(f"{target.name}.bak-{int(time.time())}"))
    commit_bytes(target, (json.dumps(draft, indent=2, ensure_ascii=False) + "\n").encode(),
                 validate=lambda p: validate(json.loads(p.read_text())))
    print(f"script: wrote {len(draft['shots'])} shots, {sum(s['duration'] for s in draft['shots']):.0f}s "
          f"(+{END_CARD_SECONDS:.0f}s end card)")
    return 0


def main(argv=None) -> None:
    def extra(p):
        p.add_argument("--force", action="store_true", help="redraft over existing shots (backs up storyboard.json)")
        p.add_argument("--draft", action="store_true", help="skip the shot-count and length rules")
    stage_main(run, __doc__, need_shots=False, only_mode="ignore", extra_args=extra, argv=argv)


if __name__ == "__main__":
    main()
