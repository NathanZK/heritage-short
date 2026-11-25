"""Stage: script. Draft the storyboard's shots from its brief with the Gemini API."""
from __future__ import annotations

import copy
import json
import math
import shutil
import time

from pipeline.common import (ASSESSMENT_SECONDS, ASSESSMENT_SHOTS, END_CARD_SECONDS, MAX_MUSIC_CAPTION,
                             FPS, PipelineExit, StoryboardError, assessment_on, call_model, check_assessment,
                             commit_bytes, shot_frames, shot_seconds, stage_main, validate)

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

TEXT_MODELS = {"gemini-3.5-flash-lite": "TEXT_MODEL", "gemini-3.8-flash": "TEXT_MODEL_FLASH"}  # storyboard text_model


def text_model_id(ctx, sb: dict) -> str:
    """The provider's id for the storyboard's text_model (default gemini-3.5-flash-lite)."""
    return getattr(ctx.provider, TEXT_MODELS[sb.get("text_model", "gemini-3.5-flash-lite")])


CLIP_SECONDS = 5        # Wan's clip is about 5 s: the length a planned shot aims for
MAX_SHOT_SECONDS = 8    # the schema's limit on shots[].duration
MIN_CUE = 8             # characters: below this a music cue says nothing
ASSESSMENT_SENTENCE = (f"This film must meet the brief's rules: {ASSESSMENT_SHOTS[0]} to {ASSESSMENT_SHOTS[1]} shots "
                       f"and {ASSESSMENT_SECONDS[0]:.0f} to {ASSESSMENT_SECONDS[1]:.0f} seconds including the "
                       f"{END_CARD_SECONDS:.0f}-second end card.")


def shot_total(sb: dict) -> float:
    """Seconds the shots must fill: the target minus the end card."""
    return sb["target_seconds"] - END_CARD_SECONDS


def planned_shots(sb: dict) -> int:
    """Shots of about CLIP_SECONDS that fill the target (half rounds up); 8-12 under the assessment rules."""
    n = max(1, math.floor(shot_total(sb) / CLIP_SECONDS + 0.5))
    return min(max(n, ASSESSMENT_SHOTS[0]), ASSESSMENT_SHOTS[1]) if assessment_on(sb) else n


def tolerance(seconds: float) -> float:
    """How far a draft's shots may miss their total: a second, or 5 % of a longer total."""
    return max(1.0, 0.05 * seconds)


def cue_allowance(sb: dict, n: int) -> int:
    """Characters each music_cue may use so that any draft of n shots within the tolerance keeps the ACE-Step
    caption (music.prompt, then "a-bs cue" entries joined by "; ") within its limit. Every entry is costed with
    the widest timestamps such a draft can reach."""
    s = shot_total(sb)
    width = len(str(math.ceil(s + tolerance(s))))
    fixed = len(sb["music"]["prompt"]) + len(" Cue sheet: ") + n * (2 * width + 3) + 2 * (n - 1)
    return min(80, (MAX_MUSIC_CAPTION - fixed) // n)


def _t(sb: dict) -> str:
    return f"{sb['target_seconds']:g}"


def check_draftable(sb: dict) -> tuple:
    """Refuse, before any model call, a target no draft can meet. Returns (shots, cue allowance)."""
    lo, hi = ASSESSMENT_SECONDS
    if assessment_on(sb) and not lo <= sb["target_seconds"] <= hi:
        raise PipelineExit(1, f"target_seconds is {_t(sb)}; the assessment rules need {lo:.0f}-{hi:.0f} s "
                              f"(set assessment_rules to false to experiment)")
    n = planned_shots(sb)
    if n * MAX_SHOT_SECONDS < shot_total(sb):
        raise PipelineExit(1, f"target_seconds {_t(sb)} needs {n} shots of more than {MAX_SHOT_SECONDS} s; "
                              f"shorten target_seconds")
    allowance = cue_allowance(sb, n)
    if allowance < MIN_CUE:
        raise PipelineExit(1, f"target_seconds {_t(sb)} plans {n} shots, leaving {allowance} characters per music "
                              f"cue in the {MAX_MUSIC_CAPTION}-character ACE-Step caption (at least {MIN_CUE} "
                              f"needed); shorten music.prompt or target_seconds")
    return n, allowance


def build_prompt(sb: dict, n: int, problem: str | None = None) -> str:
    b = sb["brief"]
    prompt = (
        f"You are drafting the storyboard for a short film titled '{sb['title']}'.\n"
        f"Premise: {b['premise']}\nAudience: {b['audience']}\nIntent: {b['intent']}\n"
        f"Style (added to every image prompt later; do not repeat it): {sb['style']}\n"
        f"Main character (described later; do not repeat it): {sb['character']['description']}\n\n"
        f"Write exactly {n} shots. Each duration is at most {MAX_SHOT_SECONDS} seconds. The image-to-video model "
        f"makes clips of about {CLIP_SECONDS} seconds and a longer shot holds its last frame, so prefer shots of "
        f"about {CLIP_SECONDS} seconds. The durations must add up to {shot_total(sb):.1f} seconds. "
        f"For each shot give: id (s01, s02, ...), duration, reason (why this shot exists), keyframe_prompt (what "
        f"the still image shows), motion_prompt (camera and subject movement), narration (one line in "
        f"{sb['language']}, short enough to speak within the duration), subtitle (the same line in English) and "
        f"music_cue (what the score does under this shot, a few words, at most {cue_allowance(sb, n)} characters).")
    if assessment_on(sb):
        prompt += " " + ASSESSMENT_SENTENCE
    return prompt + (f"\n\nYour previous answer was rejected: {problem}\nFix this." if problem else "")


def target_mismatch(sb: dict) -> str | None:
    """A note when the shots no longer fill target_seconds (it is read only when drafting), else None."""
    s, total = shot_total(sb), sum(shot_seconds(shot) for shot in sb["shots"])
    if abs(total - s) <= tolerance(s):
        return None
    return (f"note: the shots add up to {total:.1f} s, but target_seconds {_t(sb)} asks for {s:.1f} s; "
            f"the shots win (use --force to redraft from the target)")


def check_draft(draft: dict, n: int) -> None:
    """The output check, in order: a shot list, the planned count (before the schema, so a draft with too many
    shots is told so rather than that its music caption is too long), the schema, at least one frame per shot,
    the total, the assessment rules."""
    if not isinstance(draft["shots"], list):
        raise StoryboardError("the response had no 'shots' list")
    if len(draft["shots"]) != n:
        raise StoryboardError(f"the draft has {len(draft['shots'])} shots; write exactly {n}")
    validate(draft)
    short = [shot["id"] for shot in draft["shots"] if shot_frames(shot) < 1]
    if short:
        raise StoryboardError(f"shots {', '.join(short)} are shorter than one frame (1/{FPS} s)")
    s, total = shot_total(draft), sum(shot_seconds(shot) for shot in draft["shots"])
    if abs(total - s) > tolerance(s):
        raise StoryboardError(f"the shots add up to {total:.1f} s; they must add up to {s:.1f} s "
                              f"(within {tolerance(s):.1f} s)")
    check_assessment(draft)


def run(ctx, sb, args) -> int:
    if sb["shots"] and not args.force:
        print(f"script: storyboard already has {len(sb['shots'])} shots; use --force to redraft")
        return 0
    brief = sb.get("brief")
    if not brief or (not ctx.mock and brief["premise"].startswith("TODO")):
        raise PipelineExit(1, "fill in 'brief' (and style, character, music) in storyboard.json before drafting")
    n, _ = check_draftable(sb)
    provider, model = ctx.provider, text_model_id(ctx, sb)
    problem = None
    for _ in range(2):
        result = call_model(ctx, stage="script", shot=None, provider="gemini", model=model,
                            params={"retry_reason": problem} if problem else {}, seed=None,
                            fn=lambda: provider.gemini_json(build_prompt(sb, n, problem), SHOTS_SCHEMA, model))
        draft = copy.deepcopy(sb)
        draft["shots"] = result.get("shots") if isinstance(result, dict) else None
        try:
            check_draft(draft, n)
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
    stage_main(run, __doc__, need_shots=False, only_mode="ignore", extra_args=extra, argv=argv)


if __name__ == "__main__":
    main()
