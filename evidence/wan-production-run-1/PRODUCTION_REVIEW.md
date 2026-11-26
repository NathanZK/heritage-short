# The Tree That Was Always There: Wan production film, review note

**Status:** assembled and technically valid, **with known visual defects** in four shots and weaker results in two
more. It is not visually flawless. All 11 shots are real Wan 2.2 TI2V-5B image-to-video clips; no fallback footage
is used, and no shot was retried or regenerated.

## Outputs (`outputs/wan-production-film/` in the separate working copy used for the Wan run; git-ignored, not in this repository)

| File | sha256 |
|---|---|
| `the_tree_wan_16x9.mp4` | `dda029d3ecd28c7ecad79c65d01495cb4ff4b7f68f29ba3cc5c37d1a8b9bc3bf` |
| `the_tree_wan_9x16.mp4` | `4d14c82f76f8414eaa63b775a58afb01afeae4ca663423e99faabd9c8c046482` |
| `the_tree_wan_en.srt` | `b99c1848613fb4523acad0c8ebb26d5e189ce86c80d1e4e20c09f0f4136e9cd1` |

These are byte-identical copies of `outputs/final/film_16x9.mp4` and `film_9x16.mp4` as written by the pipeline. The
fallback (FFmpeg) film is in a separate working copy and is unchanged.

## How it was made

1. **Wan run (measured):** `python -m pipeline.shots` (no fallback flags), run `shots-20261126T100121Z-c65e05`,
   submitted 2026-11-26T10:01:20Z, ingested 12:03:53Z (2 h 2 min 33 s). Kernel: setup 382.5 s, each shot 621.8–625.0 s,
   7,245.7 s in all; Kaggle GPU quota 1.63 h → 3.65 h (2.02 h). All 11 shots `ok`, untiled decode, first attempt.
   Each shot was conditioned on its own approved Klein keyframe (`outputs/keyframes/<id>.jpg`) and its `motion_prompt`
   through `WanImageToVideoPipeline`; 832×480, 121 frames, 20 steps, guidance 5.0, the shot's seed.
2. **Assembly (local, no model call):** `python -m pipeline.assemble` at the repository root of a separate working copy (2026-11-26T12:07Z):
   mix 3.87 s, 9:16 22.76 s, 16:9 23.26 s. Then the exports were copied here.

Input hashes: `outputs/experiments/wan-production-run-1/assembly_inputs.sha256` (the 11 clips, 11 narration WAVs,
`subtitles.srt`, `score.wav`, `storyboard.json`). Output hashes: `evidence/wan-production-run-1/assembly_outputs.sha256`. The clips are
byte-identical to what Kaggle delivered (`evidence/wan-production-run-1/clips.sha256`); the narration, subtitles and score are byte-identical to the
approved English assets.

## Validation (measured)

| Check | 16:9 | 9:16 |
|---|---|---|
| Size, frame rate | 1280×720, 24 fps | 720×1280, 24 fps |
| Duration, frames | 79.958 s, 1,919 frames | 79.958 s, 1,919 frames |
| Full decode | clean | clean |
| Audio | AAC, 48 kHz stereo | AAC, 48 kHz stereo |
| Loudness | -14.0 LUFS, LRA 5.9 LU, peak -0.9 dBFS | same |
| Assessment rules (8–12 shots, 60–90 s) | met | met |

- The film mix (`outputs/final/mix.wav`) is byte-identical to the approved radio cut's mix, and both exports decode to
  the same audio: same narration, score, ducking, end fade and loudness settings.
- Subtitles: all 11 burned in and legible at each cue in both formats (frames in
  `outputs/experiments/wan-production-run-1/review/assembled/`). "Made with AI" end card visible for the final 3 s.
- No new `runs.jsonl` line during assembly.

## Known visual defects

Times are in the film. Each Wan clip is 5.04 s; every shot is longer, so the clip's **last frame is held** for the
rest of the shot (0.79 to 2.79 s), which keeps any late defect on screen.

| Shot | Time | Defect | Severity |
|---|---|---|---|
| s02 | 7.79–13.62 | A person walks out of the house in a shot meant to show nobody, and ends as a close-up of an adult woman (held 0.79 s) | story-breaking |
| s06 | 34.21–41.29 | Late in the clip, extra children appear, the three identities break up and the sister's face is visibly distorted; held 2.04 s. In 9:16 the cousin is mostly hidden behind her at the end | story-breaking |
| s09 | 56.71–62.96 | A yellow jug with pink contents appears on the stump; a dark shape and a skin-coloured shape intrude at the right edge (held 1.21 s) | story-breaking |
| s10 | 62.96–70.79 | The camera pushes in and the children become different, younger, lighter-skinned faces; the cousin's face changes in the close-up; held 2.79 s | story-breaking |
| s11 | 70.79–76.96 | A large blue-and-white object sweeps in from the right mid-shot | serious |
| s05 | 27.79–34.21 | The children's faces drift by the end (the sister most); in 9:16 the cousin is cut out at the right edge (no crop set) | moderate |
| s07 | 41.29–49.00 | Stable; the narrator still sits in mid-air (from the keyframe); in 9:16 the cousin is cut out (no crop set) | minor |
| s01, s03, s04, s08 | — | Story-accurate; s04 pushes into the tree and ends soft | acceptable |

**9:16 framing:** `portrait_crop_x` keeps the cousin in frame in s08 throughout and in s06 and s10 at the start and
middle; the narrator is outside the vertical frame in s06, s08 and s10 by design (a 9:16 window covers about 32% of
the clip). 480p clips are soft in 9:16.

**Cause:** the late-clip drift seen in E6 and E10, across a whole film; not diagnosed further.

## Remaining blockers before this is a finished film

- The four story-breaking shots (s02, s06, s09, s10) and s11 need a decision: regenerate, shorten what is used of
  each clip before the drift, or cut the shot to its keyframe still.
- `portrait_crop_x` for s05 and s07 so the cousin stays in 9:16.
- Nothing here is committed; the run's `runs.jsonl` lines (11 Wan calls) are in this worktree only.
