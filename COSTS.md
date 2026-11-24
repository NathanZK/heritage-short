# Costs

Measured values only. Nothing below has been measured yet: no real run has happened. Fill this in from
`runs.jsonl` (latency, `gpu_seconds`) and `outputs/timings.jsonl` (local FFmpeg) after your first real runs.

| Stage | Calls | Latency (s) | GPU seconds | Free tier used | List-price equivalent |
|---|---|---|---|---|---|
| script | | | n/a | | |
| keyframes | | | n/a | | |
| narration | | | n/a | | |
| shots (Wan 2.2) | | | | | |
| music (ACE-Step) | | | | | |
| assemble | n/a | | n/a | n/a | n/a |

Documented free allowances (checked 2026-11-25, re-check before relying on them):

- Cloudflare Workers AI: 10,000 neurons per day, reset 00:00 UTC. The docs give per-tile and per-step prices for
  `flux-1-schnell` but no per-image formula, so images per day is not stated here until measured.
- Gemini API: Flash and TTS models are free of charge on the free tier; the request-rate limits are shown only in AI Studio.
- Kaggle GPU quota: the brief says about 30 GPU hours a week; the official Kaggle pages could not be read, so unverified.

`list_price_usd` is `null` in `runs.jsonl` wherever no documented price can be applied. No GPU list price is assumed.
