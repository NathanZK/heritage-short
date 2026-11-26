# heritage-short

A small studio in code that turns a written storyboard into a short film of the length you set (9:16 and 16:9),
narrated, with an original score ducked under the voice, burned-in English subtitles and a visible
"Made with AI" end card. Built for the 10 Academy AI-Content Generation challenge.

One file, `storyboard.json`, is the contract between every stage. Each stage reads it and writes its output,
so changing one shot regenerates only that shot.

```
storyboard.json -> script -> keyframes -> narration -> shots -> music -> assemble -> 9:16 and 16:9 films
                   Gemini    Cloudflare    Gemini      Kaggle    Kaggle    FFmpeg
                             FLUX.2 klein  TTS         Wan 2.2   ACE-Step
```

**Production walkthrough:** how the English radio cut of "The Tree That Was Always There" was made, and how to
rebuild it without model calls: [`PRODUCTION.md`](PRODUCTION.md).

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
| `kaggle` CLI | shots and music (real runs only) | needs Python 3.11+, so install it separately: `pipx install kaggle`, or `uv tool install kaggle --with six` (without `six`, 2.2.5 failed to start; `FAILURES.md` 21) |
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

1. Fill in `storyboard.json`: `title`, `target_seconds`, `brief` (premise, audience, intent), `style`, `character`,
   `music.prompt`. `target_seconds` is the film's length *including* the 3 s end card (at least 4). Leave `shots`
   empty to have the script stage draft them, or write them yourself (schema: `schema/storyboard.schema.json`).
   The script stage plans shots of about 5 s (Wan makes clips of about 5 s; a shot is at most 8 s and a longer one
   holds its last frame): `target_seconds: 18` gives 3 shots, 63 gives 12. It asks Gemini for exactly that many,
   adding up to the target minus the end card, and accepts a draft within max(1 s, 5 %) of that total. Changing
   `target_seconds` later does not redraft: `run` prints a note, and `python -m pipeline.script --force` redrafts. Every shot needs a `reason` and a `music_cue`: what the score does under that
   shot, in a few words (80 characters allowed). The music stage sends ACE-Step one caption, `music.prompt` followed
   by a cue sheet of every shot's cue with its start and end second, and that whole caption is limited to 512
   characters (checked when the storyboard is validated). So the room per cue shrinks as `music.prompt` and the
   number of shots grow, and this bounds how long one score can be: about 135 s with a very short prompt, 110 s with
   100 characters, 90 s with 200. The script stage works the room out, tells the model, and refuses a target with
   fewer than 8 characters per cue before calling Gemini. If you write the shots yourself, keep the cues short, and
   shorten the prompt if validation says the caption is too long.
   **Assessment rules (10 Academy deliverable):** set `"assessment_rules": true` to require 8-12 shots and 60-90 s
   including the end card. The script stage then refuses a `target_seconds` outside 60-90 before any call and drafts
   8-12 shots; `run` and `assemble` refuse a storyboard that breaks the rules, and no flag skips them. Without it
   (the default), any length and shot count the limits above allow is accepted; `run` and `assemble` print
   `assessment rules: on` or `off` so an unchecked export is never silent.
2. Check each provider once, cheaply (one call each, logged to `runs.jsonl` with latency): `make smoke`.
3. One-time Kaggle setup: `make kaggle-setup` creates two private datasets for the job files. Kaggle may ask you
   to verify your phone before it allows GPU and internet in notebooks.
4. `make film`. Each stage skips what has not changed. It stops, and says what to do, when:
   - exit **2**: incomplete but resumable (quota used up, or Kaggle work still pending: re-run to resume);
   - exit **3**: listen to `outputs/music/candidates/` and set `music.choice` `{candidate, reason}` in `storyboard.json`;
   - exit **1**: an error (invalid storyboard, missing credential, missing prerequisite).
5. Outputs: `outputs/final/film_9x16.mp4`, `film_16x9.mp4`; a radio cut with `python -m pipeline.assemble --radio`
   after the narration stage (for your first audience test): the narration and, once chosen, the ducked score, with
   burned-in English subtitles and the 3 s "Made with AI" end card over a black picture, plus
   `outputs/final/radio_cut.srt`.

Run one stage with `python -m pipeline.<script|keyframes|narration|shots|music|assemble>`. Flags: `--mock`,
`--only SHOT_ID...` (per-shot stages), `--draft` (`run`, `music`, `assemble`: continue without a chosen score,
and warn instead of failing when narration overruns its shot; it never skips the assessment rules),
`--fallback` (shots only: FFmpeg pan-and-zoom for shots Wan could not make; announced on every run and in the
assembly, and not submitted again unless you pass `--retry-fallbacks`. It is applied after a Kaggle attempt),
`--fallback-only` (shots only: pan-and-zoom clips with no Kaggle call at all; a Wan clip is replaced only for shots
named with `--only`; it cannot be combined with `--fallback`, `--retry-fallbacks` or `--resubmit`),
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

Verified against official documentation on 2026-11-25: Gemini Interactions API endpoint, `gemini-3.5-flash-lite`
(default text model), `gemini-3.8-flash` and `gemini-3.8-flash-tts` ids, TTS request and audio location (24 kHz WAV); Cloudflare `flux-1-schnell` request
fields (`prompt`, `steps` only) and daily limit error; Cloudflare `flux-2-klein-4b` multipart fields (`prompt`,
`width`, `height`, `seed`, `input_image_0`; input images under 512x512) from Cloudflare's klein-4b changelog entry; Wan `generate.py` flags; ACE-Step Python API fields; Kaggle CLI
commands and metadata fields.

**Assumptions and manual checks** (also see `FAILURES.md`). The implementation was built from an approved spec that
is not part of this repository; where the code differs from it, the code and `ARCHITECTURE.md` govern. See
"Differences from the approved spec" in `ARCHITECTURE.md`.

| # | Unverified | Your check |
|---|---|---|
| M1 | raw Gemini text-response shape, 429 body, Amharic/English narration quality | `make smoke`, then listen |
| M2 | Cloudflare response envelope and image size; `flux-1-schnell` has **no seed and no reference-image input** (`flux-2-klein-4b` has both) | `make smoke`; see keyframes |
| M3 | Wan on Kaggle T4s: **answered.** The pipeline's kernel cannot run there (FAILURES 15). Diffusers can: on two T4s, with the video model resident on one, a 5 s, 480p shot took 591.7 s after 325.0 s of setup (`EXPERIMENTS.md` E6). This is the production approach, with the fallback per shot, and the shots kernel now implements it (`kernels/shots_kernel.py`): a one-shot pipeline smoke test ran it on Kaggle (`EXPERIMENTS.md` E10). Still open: identity drift late in a clip, and whether other shots behave like s01; no full film has been made with it yet | `python -m pipeline.shots --only <one shot>` first; review every Wan clip |
| M4 | Kaggle: **partly verified** by the music run and the Wan experiments: `kaggle quota` shows 30 GPU hours a week; `status` reports `KernelWorkerStatus.RUNNING` and `KernelWorkerStatus.COMPLETE`; job files are found under `/kaggle/input`. Still open: the output of a failed batch run, the session limit, phone verification | the first real pipeline shots run |
| M5 | ACE-Step on T4 and the uv setup: **verified** by the first real run (`EXPERIMENTS.md` E4). Still open: candidate order versus seeds, how well the cue sheet is followed, and why the loaded language model went unused (`use_lm=False`) | listen to the candidates; check ACE-Step's logged `use_lm` decision |
| M6 | character consistency across shots with a reference image (one image tested: `EXPERIMENTS.md` E1) | review the keyframes; log it in `EXPERIMENTS.md` |
| M7 | privacy: Gemini free tier, Kaggle dataset visibility | decide before using real family details |

## Known limitations

- The radio cut uses the chosen score when there is one; before a choice it is narration only, with no placeholder music.
- `list_price_usd` in `runs.jsonl` is filled only for FLUX.2 klein calls (Cloudflare reports their neurons; $0.011 per
  1,000). Gemini and Kaggle lines keep `null`, so their list-price equivalents in `COSTS.md` are filled in by hand.
- With `flux-1-schnell`, keyframes are conditioned on text only (no seed, no reference image).
- `make smoke` checks Gemini and Cloudflare only; it does not cover Kaggle.
- Moving shots: the shots kernel runs Wan 2.2 through Diffusers on two Kaggle T4s, as measured in `EXPERIMENTS.md`
  E6 and in the pipeline smoke test E10 (one shot): about 10 min per 121-frame shot plus about 5 min of setup per
  session. Each shot asks for just enough frames to cover it, up to 121; a longer shot holds the clip's last frame.
  On the shots tested the face changed late in the clip, and the 9:16 crop of a 480p clip is soft.
- The 9:16 export is a centre crop unless a shot sets `portrait_crop_x` (0 left edge, 0.5 centre, 1 right edge);
  side content outside the window is lost (`FAILURES.md` 25).
- Cuts are not snapped to musical beats (see above).

**Models.** The script stage's text model is `gemini-3.5-flash-lite` by default; set `"text_model":
"gemini-3.8-flash"` in `storyboard.json` to use Flash instead (`EXPERIMENTS.md` E2 explains the default). `make smoke`
uses the storyboard's text and keyframe models.

**Keyframes and the character.** The default keyframe model is `flux-2-klein-4b` (`keyframe_model` in
`storyboard.json`; `flux-1-schnell` remains available). It makes 1280x704 keyframes with a per-shot seed. Set
`character.reference_image` to an approved image of the character (a path in the repository, such as
`assets/character_reference.jpg`) and every shot keyframe is conditioned on it, except shots marked
`"character_reference": false` (shots without the character): those get neither the image nor the character
description, only `style`, the optional top-level `setting` and the shot's own prompt. It must be inside the repository (it
is uploaded to Cloudflare) and must stay there, because later stages check keyframes against it. Images over 511 px,
rotated (EXIF) or not plain RGB JPEG/PNG are converted before sending. A storyboard without `keyframe_model` uses
klein, so switching an older project regenerates its keyframes and clips. With a reference image, no reference sheet is generated; without one, a sheet is made for your eyes
only. Changing the reference image regenerates every keyframe that uses it (and so its clip). Cloudflare takes input
images under 512 px, so a wide reference is shrunk and small faces lose detail: crop it to the character
(`FAILURES.md` 33). `runs.jsonl` records the neurons
each klein call used (`neurons`, from Cloudflare's `cf-ai-neurons` header). Character consistency is still **not
guaranteed**: review the keyframes.

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
