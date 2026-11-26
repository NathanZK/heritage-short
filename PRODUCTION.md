# Production walkthrough: The Tree That Was Always There (English radio cut)

How the English radio cut of "The Tree That Was Always There" was made on 2026-11-26, stage by stage, with the
commands that were run and what each needs. Everything marked **measured** comes from `runs.jsonl`,
`outputs/timings.jsonl` or a probe of the output files. Everything marked **estimate** is not a measurement. The
detailed records are in `EXPERIMENTS.md` (E7 narration, E8 score and radio cut), `FAILURES.md` (28 to 30) and
`COSTS.md`.

No keyframes, moving shots or picture exports were made for this film yet. The deliverable is the radio cut: the
film's narration, score, English subtitles and end card over a black picture.

## Overview

| Stage | What it made | Runs where | Needs | Model calls |
|---|---|---|---|---|
| Story and storyboard | `storyboard.json`, `assets/narration_en.txt` | written by hand | nothing | none (the script stage was not used) |
| Narration | `outputs/narration/s01.wav` … `s11.wav`, `outputs/subtitles.srt` | Gemini API | `GEMINI_API_KEY` | 18 attempts: 11 ok, 7 refused (rate limit) |
| Music candidates | `outputs/music/candidates/candidate_1.wav` … `candidate_3.wav` | Kaggle (T4 GPU) | Kaggle CLI, `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN` (or `KAGGLE_KEY`) | 1 Kaggle run, 3 candidates |
| Score choice | `outputs/music/score.wav`, `score.json` | local | nothing | none |
| Radio cut | `outputs/final/radio_cut.mp4`, `radio_cut.srt`, `radio_mix.wav` | local (FFmpeg) | `ffmpeg`, a TrueType font | none |

`outputs/` is git-ignored, so a clone of the repository has the storyboard and the run log but not the audio. See
"Rebuilding the radio cut without model calls" below.

## Setup

Requirements and credentials are as in `README.md` ("Requirements", "Credentials"):

- Python 3.9 or newer, then `make install` (or `make install-dev` to run the tests).
- `ffmpeg` and `ffprobe` with `libx264`, `aac`, `sidechaincompress`, `loudnorm`, `adelay`, `amix` and `overlay`.
- For the narration: `GEMINI_API_KEY` in `.env` (copy `.env.example`; `.env` is git-ignored).
- For the music only: the `kaggle` CLI (Python 3.11 or newer, installed separately; `FAILURES.md` 21), Kaggle
  credentials in `.env`, and the one-time `make kaggle-setup`, which creates the private job datasets.

## 1. Story and narration

The story is the filmmaker's own childhood memory: summers at an aunt's house in Nazreth (now Adama), Ethiopia,
with a sister and a cousin, climbing a small, ordinary tree in the patio after breakfast, until one morning its
branches had been cut off and only the stump remained.

- The English narration was drafted with an AI assistant from the filmmaker's account, under the filmmaker's rules:
  an adult looking back, understated, no invented reason for the cut, no dialogue, no tree species or fruit, and the
  uncertainty about why kept as it is. The filmmaker chose a male narrator voice and asked for the narrator and the
  cousin to be shown as boys, and approved the finished radio cut.
- The text is in `assets/narration_en.txt` (the same lines are each shot's `narration` and `subtitle` in
  `storyboard.json`).

## 2. Storyboard

`storyboard.json` was written by hand against `schema/storyboard.schema.json`; the script stage (Gemini) was not
used, so that nothing in the story would be invented. Every stage validates it before it runs.

- Title "The Tree That Was Always There", English, voice `Algieba`, `seed` 1013, end card "Made with AI".
- `assessment_rules: true`: 11 shots and 79.96 s including the 3 s end card (the 10 Academy rules ask for 8 to 12
  shots and 60 to 90 s).
- `character` describes the setting and the three children once (so every keyframe prompt carries it), without a
  reference image; `style` asks for grounded cinematic realism. Neither has been used yet: no keyframes were made.
- `music.prompt` and one `music_cue` per shot make the score's caption (481 of ACE-Step's 512 characters).
- The shot durations were set after the narration existed, from each line's measured length: spoken length + the
  0.2 s lead + a pause of 2.0 to 3.2 s, rounded to whole frames (a one-off calculation, not a pipeline command).
  Changing durations does not regenerate narration (its cache key is the model, voice and text only).

| Shot | Starts (s) | Length (s) | Beat | Music cue |
|---|---|---|---|---|
| s01 | 0.00 | 7.792 | Establish a summer morning in Nazreth (Adama), Ethiopia. | warm, familiar opening |
| s02 | 7.79 | 5.833 | Introduce the aunt's house and its patio. | gentle and warm |
| s03 | 13.62 | 6.667 | The small, ordinary tree in the middle of the patio. | soft melody enters |
| s04 | 20.29 | 7.5 | Its ordinariness: nothing remarkable about it. | light, unhurried |
| s05 | 27.79 | 6.417 | Sent outside after breakfast while the room is cleaned. | a little movement |
| s06 | 34.21 | 7.083 | Taking turns climbing the tree. | light, playful plucking |
| s07 | 41.29 | 7.708 | The quiet pleasure of the routine. | warm, settled |
| s08 | 49.00 | 7.708 | One morning, the branches are cut off. | sparse, falling quiet |
| s09 | 56.71 | 6.25 | Only the stump remains. | near silence |
| s10 | 62.96 | 7.833 | Quiet shock and sadness; the reason is not known. | sparse, restrained |
| s11 | 70.79 | 6.167 | The empty space where the tree stood. | quiet resolution, fade |
| end card | 76.96 | 3.0 | "Made with AI" | (score already silent) |

## 3. Narration (Gemini TTS)

```bash
python -m pipeline.narration
```

- Model `gemini-3.8-flash-tts`, voice `Algieba` (from `storyboard.json`), one call per shot, plain text (the stage
  sends no style or pace instruction). Output: 24 kHz mono WAVs, then `outputs/subtitles.srt` once every line
  exists, each cue timed to its spoken line (shot start + 0.2 s, for the line's length).
- **Measured:** 11 lines, 3.26 to 5.21 s each, 47.51 s of speech; 2.8 to 4.1 words per second; peaks -1.0 to
  -2.0 dBFS; 62.5 s of request latency in all; free tier (`cost_usd` 0.0).
- The free tier allowed 3 TTS requests a minute (`FAILURES.md` 28). The first run stopped at s08; the rest were made
  with `python -m pipeline.narration --only s08 s09 s10`, then `--only s11` a minute later, and a final plain run
  that found every line cached and wrote `subtitles.srt`. The 7 refused attempts are in `runs.jsonl`.

## 4. Music candidates (ACE-Step 1.5 on Kaggle)

```bash
HERITAGE_REMOTE_TIMEOUT_S=3600 python -m pipeline.music
```

- The stage pushes a private Kaggle job and kernel (`kernels/music_kernel.py`, unchanged since `EXPERIMENTS.md` E4):
  ACE-Step 1.5 turbo, 8 steps, instrumental, one batch of 3 candidates with seeds `seed`, `seed`+1 and `seed`+2
  (1013, 1014, 1015), each as long as the film rounded up to 10 s (80 s). Caption: `music.prompt` followed by the
  cue sheet (each shot's cue with its start and end second).
- It ends with exit code 3: listen, then choose.
- **Measured:** run `music-20261126T011125Z-6f544b`, first attempt; three 80.0 s, 48 kHz stereo WAVs; runner time
  105.2 s; about 256 s from submission to ingest; Kaggle GPU quota 1.58 h → 1.63 h (`kaggle quota`). ACE-Step logged
  `use_lm=False` (`FAILURES.md` 22).

## 5. Choosing the score

The three candidates were compared from measurements, not by listening (no playback was available at the time):
integrated loudness, 2 s level envelopes against the shot timeline, silences and spectrograms, all with local FFmpeg
(a one-off analysis, not a pipeline command; the figures are in `EXPERIMENTS.md` E8). Candidate 3 (seed 1015) was the
only one whose shape follows the story: fuller in the opening, a dip just before the discovery at 49 s, sparse after
it, near silence from about 72 s. The choice is recorded in `storyboard.json`:

```json
"choice": {"candidate": 3, "reason": "Chosen from measured levels and spectrograms, not by listening (seed 1015): …"}
```

then, locally and with no model call:

```bash
python -m pipeline.music
```

copies `candidate_3.wav` byte for byte to `outputs/music/score.wav` (sha256 `ed20d747…5c29`) and records it in
`score.json`. The filmmaker later approved the radio cut that uses it.

## 6. Radio cut (local FFmpeg)

```bash
python -m pipeline.assemble --radio
```

- **Mix** (the same as the film's): each line placed at its shot's start + 0.2 s; the score at 0.6 volume, ducked
  under the voice by sidechain compression (threshold 0.02, ratio 10, attack 20 ms, release 500 ms), faded out over
  the last 2 s; then two-pass `loudnorm` to -14 LUFS, -1 dBTP.
- **Picture:** 1280×720, 24 fps, black; each subtitle burned in as a PNG overlay while its line is spoken; the 3 s
  "Made with AI" end card at the end. H.264 video, 192 kb/s AAC audio.
- **Also written:** `outputs/final/radio_cut.srt` (a copy of `subtitles.srt`) and `radio_mix.wav`.
- **Measured:** 79.96 s, 1,919 frames; decodes cleanly; -14.0 LUFS, loudness range 5.9 LU, sample peak -1.0 dBFS,
  true peak -0.9 dBFS; the voice 8.8 to 42 dB above the ducked score while speaking, the score 6 to 15 dB louder in
  the pauses; every subtitle visible from 80 ms after its cue starts to 80 ms before it ends. Assembly took 3.98 s
  (mix) and 14.49 s (picture).

## Rebuilding the radio cut without model calls

`assemble --radio` makes no model calls and needs no credentials. In a clone, put these git-ignored files from the
production working copy under `outputs/`, then run the command above:

- `outputs/manifest.json`
- `outputs/narration/s01.wav` … `s11.wav`
- `outputs/subtitles.srt`
- `outputs/music/score.wav` and `outputs/music/score.json`

**Measured (2026-11-26):** a copy of this commit's files plus exactly those inputs, with no `.env` and an empty
environment, rebuilt `radio_cut.mp4` and `radio_cut.srt` byte-identical to the approved cut (FFmpeg 9.0.1) and added
nothing to `runs.jsonl`. With a different FFmpeg version or font, expect a valid but not byte-identical file.

Do not run the narration or music stage to "rebuild": without the cached files they would make new model calls, and
new candidates would need a new choice.

## Known limitations

- The score has faded out by about 72 to 73 s, so the last line ends over silence and the end card is silent
  (`EXPERIMENTS.md` E8).
- The narration is brisk (2.8 to 4.1 words per second); the TTS stage has no pace control (E7).
- `loudnorm` falls back to dynamic mode because of the true-peak limit: the gain varies by about 5 dB across the cut,
  and the encoded true peak is -0.9 dBTP, just above the -1 target (`FAILURES.md` 30).
- The narration stage does not pace its calls for the free tier's per-minute limit (`FAILURES.md` 28).
- The candidates were compared by measurement; only the filmmaker's approval of the finished cut involved listening.
- No keyframes, moving shots or picture exports exist for this film yet.
