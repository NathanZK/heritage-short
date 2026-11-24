# heritage-short

A small studio in code that turns a written storyboard into a 60-90 second short film (9:16 and 16:9),
narrated, with an original score ducked under the voice, burned-in English subtitles and a visible
"Made with AI" end card. Built for the 10 Academy AI-Content Generation challenge.

One file, `storyboard.json`, is the contract between every stage. Each stage reads it and writes its output,
so changing one shot regenerates only that shot.

```
storyboard.json -> script -> keyframes -> narration -> shots -> music -> assemble -> 9:16 and 16:9 films
                   Gemini    Cloudflare    Gemini      Kaggle    Kaggle    FFmpeg
                             FLUX schnell  TTS         Wan 2.2   ACE-Step
```

## Try it now, no credentials

```bash
python3 -m venv .venv && . .venv/bin/activate
make install-dev
make film PIPELINE_FLAGS="--mock --draft"      # placeholder media, a few seconds to a minute
make test
```

`--mock` uses fake providers and writes only under `outputs/mock/`. It proves the plumbing (validation, caching,
retries, logging, assembly), **not** the quality of any model. Real generation is a separate step (below).

## Requirements

| Need | Why | Notes |
|---|---|---|
| Python 3.9+ | the pipeline | `jsonschema`, `Pillow` (installed by `make install`) |
| `ffmpeg` + `ffprobe` | assembly | needs the `libx264` and `aac` encoders and the filters `sidechaincompress`, `loudnorm`, `adelay`, `amix`, `overlay`, `zoompan`; `brew install ffmpeg` has them |
| a TrueType font | subtitles and end card | macOS and DejaVu locations are searched (`FONTS` in `pipeline/assemble.py`) |
| `kaggle` CLI | shots and music (real runs only) | needs Python 3.11+, so install it separately: `pipx install kaggle` |
| `pytest` | tests | `make install-dev` |

## Credentials (real runs only)

Copy `.env.example` to `.env` and fill it in. `.env` is git-ignored; never commit it. Values are read from the
environment first, then `.env`.

| Variable | For | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | script and narration (Gemini API) | Google AI Studio, "Get API key" |
| `CLOUDFLARE_ACCOUNT_ID` | keyframes | Cloudflare dashboard, Workers AI page or the account's overview |
| `CLOUDFLARE_API_TOKEN` | keyframes | Cloudflare dashboard, My Profile, API Tokens: a token with the **Workers AI** permission |
| `KAGGLE_USERNAME` | job datasets and kernels | your Kaggle username |
| `KAGGLE_API_TOKEN` (or `KAGGLE_KEY`) | Kaggle CLI | Kaggle, Settings, API: create a token; or run `kaggle auth login` instead |
| `HERITAGE_POLL_S`, `HERITAGE_REMOTE_TIMEOUT_S` | optional | Kaggle polling interval (default 30) and longest wait per run (default 10800) |

**Kaggle and Colab secrets:** none are needed or used. Nothing secret is sent to a kernel (Wan and ACE-Step
weights are public), and Kaggle Secrets cannot be set from the CLI. Colab is not used.

**Privacy:** `runs.jsonl` is tracked (the assessment reads it as evidence) and records your narration text and
prompts, so review `git diff runs.jsonl` for sensitive content before every commit, and before pushing to a public
repository. `outputs/` (your media) stays git-ignored; add only what you choose with `git add -f outputs/<path>`.
Free-tier Gemini prompts and outputs may be used by Google to improve its products. Keep private
family details out of the storyboard text, or use a paid key. Keyframes you send to Kaggle sit in a *private*
dataset; confirm that on kaggle.com and delete old dataset versions when you like.

## Running the real pipeline

1. Fill in `storyboard.json`: `title`, `brief` (premise, audience, intent), `style`, `character`, `music.prompt`.
   Leave `shots` empty to have the script stage draft 8-12 shots, or write them yourself (schema:
   `schema/storyboard.schema.json`). Every shot needs a `reason` and a `music_cue`: what the score does under that
   shot, in a few words (80 characters allowed). The music stage sends ACE-Step one caption, `music.prompt` followed
   by a cue sheet of every shot's cue with its start and end second, and that whole caption is limited to 512
   characters (checked when the storyboard is validated). So the room per cue shrinks as `music.prompt` grows: with a
   100-character prompt and 12 shots it is 24 characters. The script stage works the number out and tells the model;
   if you write the shots yourself, keep the cues short, and shorten the prompt if validation says the caption is too
   long.
   A film has 8-12 shots that add up to 60-90 s *including* the 3 s end card. A shot is at most 8 s. Wan makes
   clips of about 5 s, so a longer shot holds its last frame: prefer shots of about 5 s.
2. Check each provider once, cheaply (one call each, logged to `runs.jsonl` with latency): `make smoke`.
3. One-time Kaggle setup: `make kaggle-setup` creates two private datasets for the job files. Kaggle may ask you
   to verify your phone before it allows GPU and internet in notebooks.
4. `make film`. Each stage skips what has not changed. It stops, and says what to do, when:
   - exit **2**: incomplete but resumable (quota used up, or Kaggle work still pending: re-run to resume);
   - exit **3**: listen to `outputs/music/candidates/` and set `music.choice` `{candidate, reason}` in `storyboard.json`;
   - exit **1**: an error (invalid storyboard, missing credential, missing prerequisite).
5. Outputs: `outputs/final/film_9x16.mp4`, `film_16x9.mp4`; a radio cut (narration over black, for your first
   audience test) with `python -m pipeline.assemble --radio` after the narration stage.

Run one stage with `python -m pipeline.<script|keyframes|narration|shots|music|assemble>`. Flags: `--mock`,
`--only SHOT_ID...` (per-shot stages), `--draft` (skip the 8-12 shot and 60-90 s rules, allow no score),
`--fallback` (shots only: FFmpeg pan-and-zoom for shots Wan could not make; announced on every run and in the
assembly, and not submitted again unless you pass `--retry-fallbacks`),
`--force` (script: redraft, backing up `storyboard.json`), `--resubmit` (discard a pending Kaggle run).

The score follows the storyboard's timeline through the cue sheet, but cuts are **not** snapped to beats: the cuts
fall at the storyboard durations, and the cue sheet only asks ACE-Step for the music's shape, which it does not
guarantee. Editing a `music_cue`, or a duration that moves a cue's whole-second times, regenerates the three
candidates and makes your choice stale.

To regenerate one shot: edit its prompt or narration in `storyboard.json` and run `make film`. `runs.jsonl` then
shows new calls only for that shot: one TTS call for a narration edit; one keyframe call plus one Wan item for a
keyframe-prompt edit. Mixing and assembly are local and make no model call.

## Mocked testing versus real generation

| | `make test`, `--mock` | real run |
|---|---|---|
| credentials | none | Gemini, Cloudflare, Kaggle |
| network | blocked in tests | yes |
| media | fake (colour frames, sine tones) | real models |
| log | `outputs/mock/runs.jsonl` | `runs.jsonl` |
| proves | plumbing, caching, exit codes, FFmpeg output | nothing until you run it |

No test calls a provider, Kaggle, Wan or ACE-Step. Generation quality, GPU behaviour and quotas are yours to check.

## What is verified and what is not

Verified against official documentation on 2026-11-25: Gemini Interactions API endpoint, `gemini-3.8-flash` and
`gemini-3.8-flash-tts` ids, TTS request and audio location (24 kHz WAV); Cloudflare `flux-1-schnell` request
fields (`prompt`, `steps` only) and daily limit error; Wan `generate.py` flags; ACE-Step Python API fields; Kaggle CLI
commands and metadata fields.

**Assumptions and manual checks** (also see `FAILURES.md`). The implementation was built from an approved spec that
is not part of this repository; where the code differs from it, the code and `ARCHITECTURE.md` govern. See
"Differences from the approved spec" in `ARCHITECTURE.md`.

| # | Unverified | Your check |
|---|---|---|
| M1 | raw Gemini text-response shape, 429 body, Amharic/English narration quality | `make smoke`, then listen |
| M2 | Cloudflare response envelope and image size; the API has **no seed and no reference-image input** | `make smoke`; see keyframes |
| M3 | Wan on a Kaggle T4 (docs state 24 GB minimum): time, memory, bf16, flash_attn | `python -m pipeline.shots --only <id>`; use `--fallback` if it fails |
| M4 | Kaggle: mount paths, `status` text, output of failed runs, quota, phone verification | the first real shots run |
| M5 | ACE-Step on T4, candidate order; uv installed with pip and run as `python -m uv` in the kernel; how well the cue sheet in the caption is followed | the first real music run |
| M6 | character consistency with text-only conditioning | review the keyframes; log it in `EXPERIMENTS.md` |
| M7 | privacy: Gemini free tier, Kaggle dataset visibility | decide before using real family details |

## Known limitations

- The radio cut is narration over black only; it has no placeholder music.
- `list_price_usd` is always `null` in `runs.jsonl`, so list-price equivalents in `COSTS.md` must be filled in by hand.
- Keyframes are not conditioned on the reference sheet (the documented keyframe API takes no reference image).
- `make smoke` checks Gemini and Cloudflare only; it does not cover Kaggle.
- Cuts are not snapped to musical beats (see above).

Character consistency is **not guaranteed**: the documented keyframe API takes no seed and no reference image, so
every keyframe repeats the same style and character text and hopes. The reference sheet is for your eyes only.

## Layout

```
storyboard.json          the contract          schema/   JSON Schema, validated before every run
pipeline/                one module per stage, common.py, providers.py, kaggle.py, mock.py, run.py, smoke.py
kernels/                 the two scripts that run on Kaggle (Wan 2.2, ACE-Step)
outputs/                 generated media, manifest.json, timings.jsonl (git-ignored; kept locally so a run can be inspected)
runs.jsonl               one line per model-call attempt: model, params, seed, latency, cost (tracked: review before committing)
tests/                   pytest suite, no credentials
DESIGN.md ARCHITECTURE.md EXPERIMENTS.md FAILURES.md COSTS.md ETHICS.md
```
