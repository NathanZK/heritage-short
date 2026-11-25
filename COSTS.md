# Costs

Measured values only. Rows are filled as stages are measured; fill the rest from `runs.jsonl` (latency,
`gpu_seconds`, `neurons`) and `outputs/timings.jsonl` (local FFmpeg).

| Stage | Calls | Latency (s) | GPU seconds | Free tier used | List-price equivalent |
|---|---|---|---|---|---|
| script | | | n/a | | |
| keyframes | | | n/a | | |
| narration | | | n/a | | |
| shots (Wan 2.2, pipeline kernel) | not run (cannot run on a T4; FAILURES 15) | | | | |
| shots (Wan 2.2 Diffusers experiment, E5; 1 clip, outside the pipeline) | 1 Kaggle run | 2415.2 in the notebook: denoising 1501.9 (50 × 30.04), VAE decode 410.8, VAE encode 5.0, export 4.2, setup 476.9, other about 16 | not recorded separately | Kaggle GPU quota 0.38 h → 1.05 h (`kaggle quota`; about 0.67 h) | n/a (no GPU list price assumed) |
| shots (production: FFmpeg fallback) | local, no model call | not yet measured | n/a | n/a | n/a |
| music (ACE-Step) | 1 Kaggle run, 1 batch call (3 candidates) | 99.1 runner; about 215 end to end | 99.1 (runner time, recorded by the kernel; excludes pip, git clone and `uv sync`, and is repeated on each of the 3 candidate lines in `runs.jsonl`, so do not sum it) | Kaggle GPU quota 0.33 h → 0.38 h (`kaggle quota`; about 0.05 h) | n/a (no GPU list price assumed) |
| assemble | n/a | | n/a | n/a | n/a |

Documented free allowances (checked 2026-11-25, re-check before relying on them):

- Cloudflare Workers AI: 10,000 neurons per day, reset 00:00 UTC. The docs give per-tile and per-step prices for
  `flux-1-schnell` but no per-image formula, so images per day is not stated here until measured.
- Gemini API: Flash and TTS models are free of charge on the free tier; the request-rate limits are shown only in AI Studio.
- Kaggle GPU quota: `kaggle quota` (2026-11-26) shows 30.00 GPU hours a week, refreshing 2026-11-28 00:00 UTC; it
  reports hours used, rounded to 0.01 h. Kaggle's own documentation pages could not be read.

`list_price_usd` is `null` in `runs.jsonl` wherever no documented price can be applied. It is set for FLUX.2 klein
calls from the neurons Cloudflare reports ($0.011 per 1,000 neurons; free within 10,000 a day). No GPU list price is assumed.
