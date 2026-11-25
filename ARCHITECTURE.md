# Architecture

The implementation was built from an approved implementation spec. That spec was a working document and is not part
of this repository (the "spec section" numbers below refer to it). It records the design as it was approved; where
the code differs from it, the code and this document govern, and every difference is listed under
"Differences from the approved spec" below.

## Stages and contracts

| Stage | Reads | Writes | Runs on |
|---|---|---|---|
| script | `brief`, `target_seconds`, style, character | `shots` in `storyboard.json` | Gemini `gemini-3.8-flash` |
| keyframes | style, character, shot prompts | `outputs/keyframes/{reference,<id>}.jpg` | Cloudflare `flux-1-schnell` |
| narration | shot narration, voice | `outputs/narration/<id>.wav`, `subtitles.srt` | Gemini `gemini-3.8-flash-tts` |
| shots | keyframes, motion prompts | `outputs/shots/<id>.mp4` + sidecar | Kaggle, Wan 2.2 TI2V-5B |
| music | music prompt, per-shot `music_cue`s and their times, film length | three candidates, then `score.wav` | Kaggle, ACE-Step 1.5 |
| assemble | all of the above | `outputs/final/*` | local FFmpeg |

`storyboard.json` is validated against `schema/storyboard.schema.json` before every stage. Exit codes: 0 done,
1 error, 2 incomplete but resumable, 3 awaiting the music choice.

## Film length and assessment rules

| Kind | What | Where |
|---|---|---|
| capability | a shot is at most 8 s; 3 s end card; composed keyframe prompt <= 2048; music caption <= 512 | schema, `validate()` |
| project | `target_seconds`: film length including the end card; read only by the script stage | `storyboard.json` |
| assessment | `assessment_rules: true`: 8-12 shots and 60-90 s, frame-accurate | `check_assessment()` |

- The script stage plans `max(1, floor((target_seconds - 3) / 5 + 0.5))` shots (8-12 under the assessment rules)
  and accepts a draft with exactly that many shots, each at least one frame, whose total is within max(1 s, 5 %) of
  `target_seconds - 3`. After drafting the shots win: `run` only prints a note when they no longer fill the target.
- Before any Gemini call it refuses: a target outside 60-90 s under the assessment rules; a plan needing shots over
  8 s; and a plan leaving fewer than 8 characters per music cue. The cue allowance costs every cue-sheet entry with
  the widest timestamps a draft within the tolerance can reach, so an accepted draft always fits the caption.
- `check_assessment()` runs in the script output check, at the start of `run` and in `assemble`, whatever the
  flags; `assemble --radio` (narration over black, never a film export) does not run it.
- Neither `target_seconds` nor `assessment_rules` is in any fingerprint: changing them regenerates nothing.

## Caching and invalidation

Rule: **input fingerprints decide whether a provider is called; content hashes decide whether downstream
outputs are stale.** Image and audio APIs are not deterministic, so a regenerated output with identical inputs must
still invalidate what depends on it.

- A fingerprint is a short sha256 of the canonical JSON of an output's inputs (model id, parameters, prompts).
- `outputs/manifest.json` stores fingerprint, sha256 and size per output. An output is fresh only if the
  fingerprint matches *and* the file's sha256 matches. A replaced or corrupt file is stale.
- A shot clip's fingerprint includes the sha256 of its keyframe, so a new keyframe makes a new clip.
- Exports hash their clips, mix, subtitles and end card. Changing one narration line makes one TTS call, then
  rebuilds the mix and both exports locally with no further model call (tested in `tests/test_invalidation.py`).
- Changing `style` or `character.description` regenerates every keyframe and clip, on purpose.
- The music candidates' fingerprint holds the caption: `music.prompt` plus a cue sheet of every shot's `music_cue`
  with its whole-second start and end. Editing a cue (or a duration that moves those whole seconds) regenerates the
  three candidates and makes the music choice stale (exit 3 until you choose again). Narration edits do not touch it.

## Failure handling

- Retries: 3 attempts per call, 2 s then 4 s apart, for HTTP 429 (except Cloudflare's daily-limit code 3036),
  5xx and network errors; one `runs.jsonl` line per attempt.
- Writes go to a temp file in the same directory, are fsynced and validated (image decodes, wav opens, clip probes),
  then renamed. A failed or invalid generation never replaces a good output. Orphan temp files are removed at start.
- One process per `outputs/` tree (`outputs/.lock`). `runs.jsonl` lines are single `O_APPEND` writes.
- Errors, CLI output and params are redacted of secret values before they are logged or printed.
- **Accepted risk:** a crash after a paid response but before it is committed loses that response, and a timed-out
  request may already have used quota. Neither provider documents idempotency keys.

## Remote work on Kaggle

For a stage with stale items the pipeline builds a private job dataset (allowlist: `job.json` and inputs), versions
it, pushes a kernel whose code embeds a fresh `run_id`, polls `kaggle kernels status`, downloads the output and trusts
only items whose `run_id`, fingerprint and sha256 match. The kernel finds its own job by `run_id`, so it cannot read an
older dataset version. `outputs/jobs/<stage>/pending.json` records the run so a re-run resumes instead of pushing a
duplicate; it is deleted only once the run has delivered results for its own `run_id` (or Kaggle says it failed).
Result file names must be plain names and symlinks are refused. Every item that was in the run and did not come
back usable is logged to `runs.jsonl` as an error line. A resumed run is judged against the current fingerprints of all
its shots, so clips already paid for are kept even if you edited other shots in the meantime. Both kernels check every setup step (a failing step is reported on every item in `results.json` with its output
tail, and nothing is generated) and run pip with the kernel's own Python. The kernels install repos and weights outside `/kaggle/working`, so only
`results.json` and the outputs come back. Polling that times out does not cancel the Kaggle run (no documented cancel): cancel it in the Kaggle UI.

## Timeline and shot length

Everything is on a 24 fps frame grid: a shot lasts `round(duration * 24)` frames, and the voice, the subtitles and the
end card are placed from cumulative frame counts, so odd durations such as 4.9 s cannot drift picture against sound.
A shot is at most 8 s; Wan's clip is about 5 s (121 frames), so a longer shot holds the clip's last frame. The voice
starts 0.2 s into its shot, exactly when its subtitle appears. A narration clip may not be longer than its shot, so
it always ends before the next shot's voice begins.

The score follows this timeline through the cue sheet in its caption (for example `0-5s quiet strings; 5-10s
swell`). The cuts themselves fall at the storyboard durations and are **not** snapped to musical beats: the cue sheet
asks ACE-Step for the music's shape, and ACE-Step does not guarantee it.

## Fallback clips

`--fallback` fills a shot Wan could not make with an FFmpeg pan-and-zoom of its keyframe. The sidecar says
`kind: fallback`, the shots stage and the assembly both name these shots on every run, and they are not submitted to
Kaggle again unless you pass `--retry-fallbacks` (so an unchanged re-run costs no GPU time).

## Known limits

- The keyframe API has no seed and no reference image: no consistency guarantee, and keyframes are not reproducible
  by re-generation (reproducibility comes from cached outputs).
- Wan on a Kaggle T4 is unverified; `--fallback` makes pan-and-zoom clips instead, clearly marked and retried later.
- Landscape clips are centre-cropped to 9:16: side content is lost.
- Subtitles and end card are PNG overlays (the stock FFmpeg lacks `subtitles` and `drawtext`); Latin text only.
- Cuts are not beat-snapped; the score's shape comes only from the cue sheet in the caption.
- One score's caption bounds the film's length: about 135 s with a very short `music.prompt`, less as it grows.
- The radio cut has no placeholder music: narration over black only.
- `list_price_usd` is always `null`; list-price equivalents in `COSTS.md` are filled in by hand.
- Keyframes are not conditioned on the reference sheet (the documented API has no reference-image input).
- The smoke test (`make smoke`) calls Gemini and Cloudflare only; it does not cover Kaggle.
- The ACE-Step kernel installs uv with pip and runs it as `python -m uv`; unverified on Kaggle (README M5).

## Differences from the approved spec

| # | Spec section | The spec says | The code does |
|---|---|---|---|
| 1 | 2, 8 | shot `duration` <= 5 s | <= 8 s; a shot longer than Wan's ~5 s clip holds the clip's last frame |
| 2 | 2, 3 | film-length rules (8-12 shots, 60-90 s) checked only by `assemble` and the script output check, skipped by `--draft` | opt-in through `assessment_rules`; when on, checked by the script output check, at the start of `run` and by `assemble`, and `--draft` does not skip them; lengths are measured in whole 24 fps frames |
| 3 | 3 | git-ignore `outputs/jobs/`, `_incoming/`, `mock/`, `.lock`; other outputs kept for inspection | all of `outputs/*` is git-ignored (except `.gitkeep`); `runs.jsonl` stays tracked |
| 4 | 3, 6.4, 8 | a fallback clip is never fresh and is retried on every run | a fallback clip is usable and announced on every run; it is retried only with the new flag `--retry-fallbacks` |
| 5 | 5.3 step 8 | delete `pending.json` once every item is ingested or rejected | `pending.json` is kept unless the run delivered results for its own `run_id` or Kaggle reports it failed; a resumed run is judged against the current fingerprints |
| 6 | 3.1 | `runs.jsonl` holds model calls only | it also holds an error line for every Kaggle item that was in a run and did not come back usable |
| 7 | 4, 5.6 | mix/export fingerprints use shot start offsets, a mix-constants version, the FFmpeg major version and an overlay renderer version; narration at the shot start | voice and subtitle start at shot start + 0.2 s; trims are in whole frames; fingerprints use the full first line of `ffmpeg -version`; there is no overlay renderer version (the stage `V` constant covers it) |
| 8 | 5.3 step 5 | a kernel that cannot find its job writes an error `results.json` | it exits without writing `results.json` (the run then counts as failed or not delivered) |
| 9 | 1, 5.6 | section 1: radio cut with an optional score; 5.6: narration only | narration only, no placeholder music |
| 10 | 5.6 | narration may run past its shot by up to 0.3 s | no tolerance: a narration clip longer than its shot is an error (a warning under `--draft`) |
| 11 | 3 | (not specified) | `run --only` skips assembly, because the exports need every shot |
| 12 | 3 | `--force` backup is `storyboard.json.bak-<UTC timestamp>` | the suffix is epoch seconds |
| 13 | 5.6 | score ducked under the narration | the score is also attenuated by `volume=0.6` before ducking |
| 14 | 3 | flag list | `--resubmit` (shots, music, run) and `--retry-fallbacks` (shots, run) are not in the spec's flag list |
| 15 | 2, 4, 5.5 | one `music.prompt` is the ACE-Step caption | every shot has a required `music_cue`; the caption is `music.prompt` plus a timed cue sheet, limited to 512 characters |
| 16 | 5.5 | uv installed with the documented curl installer | uv is installed with `pip` and run as `python -m uv`; a failed setup step reports every item as an error and generates nothing |
| 17 | 3.1 | `score.json` is `{candidates_fingerprint, candidate, candidate_sha256, reason, sha256}` | it also records `awaiting_choice` (the choice that was rejected as stale), so a choice made before listening stays stale on a plain re-run |
| 18 | 2, 3, 5.1 | the script stage drafts 8-12 shots for 60-90 s | it drafts from the required `target_seconds` (shots of about 5 s) and refuses impossible targets before calling Gemini; the script stage has no `--draft` |
