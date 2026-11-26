# Costs

Measured values only. Rows are filled as stages are measured; fill the rest from `runs.jsonl` (latency,
`gpu_seconds`, `neurons`) and `outputs/timings.jsonl` (local FFmpeg).

| Stage | Calls | Latency (s) | GPU seconds | Free tier used | List-price equivalent |
|---|---|---|---|---|---|
| script | 4: `gemini-3.8-flash` 3 errors (two timeouts, one HTTP 503; FAILURES 13), then `gemini-3.5-flash-lite` 1 ok | Flash-Lite 4.79; the failed Flash attempts 120.32, 3.65 and 120.27 | n/a | Gemini free tier (`cost_usd` 0.0) | not assumed (`null`) |
| keyframes | 9 ok: `flux-1-schnell` 4 (superseded) and `flux-2-klein-4b` 5 (3 shots, then s02 and s03 again) | schnell 2.39–9.87 (17.3 in all); klein 5.04–7.60 (32.5 in all) | n/a | klein 107.8 neurons per image, 539.0 in all, within the 10,000 a day; schnell's were not logged (neurons logging came with klein) | klein $0.0011858 per image where logged (2 lines, $0.0023716); the 3 earlier klein lines are `null`, and at the same rate all 5 come to $0.0059 (arithmetic) |
| narration | 3 ok (`gemini-3.8-flash-tts`) | 40.20, 4.74, 7.16 (52.1 in all) | n/a | Gemini free tier (`cost_usd` 0.0) | not assumed (`null`) |
| shots (Wan 2.2, pipeline kernel) | not run (cannot run on a T4; FAILURES 15); production uses the E6 setup once the shots stage implements it | | | | |
| shots (Wan 2.2 Diffusers experiment, E5; 1 clip, outside the pipeline) | 1 Kaggle run | 2415.2 in the notebook: denoising 1501.9 (50 × 30.04), VAE decode 410.8, VAE encode 5.0, export 4.2, setup 476.9, other about 16 | not recorded separately | Kaggle GPU quota 0.38 h → 1.05 h (`kaggle quota`; about 0.67 h) | n/a (no GPU list price assumed) |
| shots (Wan 2.2 Diffusers, two T4s, E6 "Run 1"; 1 shot, outside the pipeline) | 1 Kaggle run | 939.9 in the notebook: setup 325.0 once, then per shot 591.7 (denoising 529.2 = 20 × 26.46, full VAE float16 decode 53.3, text encoding 5.7, export 3.5); plus a comparison decode with TAEW2.2-Super and saving latents, about 7; other about 16 | not recorded separately | Kaggle GPU quota 1.05 h → 1.32 h (`kaggle quota`; about 0.27 h) | n/a (no GPU list price assumed) |
| shots (FFmpeg fallback, per shot) | local, no model call | 0.54–0.59 per 5 s clip, plus 0.03–0.08 to normalise the keyframe (integration test, `timings.jsonl` of its isolated root) | n/a | n/a | n/a |
| music (ACE-Step) | 1 Kaggle run, 1 batch call (3 candidates) | 99.1 runner; about 215 end to end | 99.1 (runner time, recorded by the kernel; excludes pip, git clone and `uv sync`, and is repeated on each of the 3 candidate lines in `runs.jsonl`, so do not sum it) | Kaggle GPU quota 0.33 h → 0.38 h (`kaggle quota`; about 0.05 h) | n/a (no GPU list price assumed) |
| assemble | n/a | 18 s film, integration test: mix 0.91, 9:16 export 2.73, 16:9 export 2.41 (0.59, 3.07 and 2.82 after the framing and loudness fixes, which are not on `main`; FAILURES 25–26); radio cut 0.91 in all (main working copy, `outputs/timings.jsonl`) | n/a | n/a | n/a |

**"The Tree That Was Always There" (2026-11-26; measured):**

| Stage | Calls | Latency (s) | GPU seconds | Free tier used | List-price equivalent |
|---|---|---|---|---|---|
| narration (English) | 18 `gemini-3.8-flash-tts` attempts: 11 ok, 7 refused (HTTP 429, per-minute limit; FAILURES 28) | 62.5 for the 11 lines; 3.26–5.21 s of speech each | n/a | Gemini free tier (`cost_usd` 0.0) | not assumed (`null`) |
| music (ACE-Step) | 1 Kaggle run, 1 batch call (3 candidates, 80 s each) | 105.2 runner; about 256 from submission to ingest | 105.2 (runner time; repeated on each of the 3 lines in `runs.jsonl`, so do not sum it) | Kaggle GPU quota 1.58 h → 1.63 h (`kaggle quota`; about 0.05 h) | n/a (no GPU list price assumed) |
| assemble (radio cut) | local, no model call | mix 3.98, picture with subtitles and end card 14.49 (`outputs/timings.jsonl`) | n/a | n/a | n/a |

Documented free allowances (checked 2026-11-25, re-check before relying on them):

- Cloudflare Workers AI: 10,000 neurons per day, reset 00:00 UTC. The docs give per-tile and per-step prices for
  `flux-1-schnell` but no per-image formula, so images per day is not stated here until measured.
- Gemini API: Flash and TTS models are free of charge on the free tier; the request-rate limits are shown only in AI Studio.
- Kaggle GPU quota: `kaggle quota` (2026-11-26) shows 30.00 GPU hours a week, refreshing 2026-11-28 00:00 UTC; it
  reports hours used, rounded to 0.01 h. Kaggle's own documentation pages could not be read.

`list_price_usd` is `null` in `runs.jsonl` wherever no documented price can be applied. It is set for FLUX.2 klein
calls from the neurons Cloudflare reports ($0.011 per 1,000 neurons; free within 10,000 a day). No GPU list price is assumed.

**Film estimates for Wan (not measured).** From E6, with setup once per GPU session and 591.7 s per shot: about 84 min
for 8 shots and 124 min for 12; about 108 and 159 min if about 30% of shots are regenerated once. At roughly the
E6 rate of quota against notebook time (0.27 h for 15.7 min), that is about 1.4 to 2.1 GPU hours of the 30 a week, or
1.9 to 2.7 with regenerations. The figures assume one GPU session for the whole film; every extra session repeats
the setup.
