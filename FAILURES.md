# Failures

What broke or surprised us, what we tried, and what we did. Entries 1-9 were found while reading the official
documentation before writing any integration (2026-11-25). Real run failures get added below as they happen.

1. **Gemini docs moved.** `ai.google.dev` redirects to `aistudio.google.com/docs`, and the default API is now
   `POST /v1beta/interactions` (generateContent is legacy). *Did:* implemented Interactions; kept the parser strict.
   Unverified: the raw text-response shape (spec A1).
2. **No `gemini-3.8-flash-lite`.** The brief mentions Flash-Lite; the text Flash-Lite model is `gemini-3.5-flash-lite`.
   *Did:* no Flash-Lite model is used (no silent substitution).
3. **Cloudflare docs contradict themselves.** The `flux-1-schnell` input schema has `additionalProperties: false`
   with only `prompt` and `steps`, yet the page's examples pass `seed`. *Did:* send only the documented fields, so
   there is **no seed**; recorded as a limit. A test validates our request against a stored copy of the schema.
4. **No reference-image input.** FLUX.2 klein is documented as `multipart` with no field names. *Did:* did not use it.
   Character consistency is text-only and unguaranteed.
5. **Wan TI2V-5B memory.** The README states a 24 GB minimum for the documented command; a Kaggle T4 has 16 GB.
   No T4 or 2-GPU recipe exists. *Did:* kept the documented command and offload flags; `--fallback` is the
   assessment's pan-and-zoom. Unresolved until you run it (M3).
6. **Wan TI2V-5B via diffusers.** The model card shows text-to-video only; no image-to-video example exists.
   *Did:* used the documented `generate.py --image` path, one call per shot (no batch mode).
7. **Kaggle docs unreadable.** `kaggle.com/docs/*` pages render with JavaScript and returned only titles, and
   GitHub returned 503 to the page fetcher. *Did:* read `Kaggle/kaggle-cli` docs through the GitHub API. Quota,
   session limit, mount paths and `status` strings remain undocumented (M4).
8. **Local FFmpeg has no `subtitles` or `drawtext` filter.** *Did:* burn text in as PNG overlays made with Pillow.
9. **Kaggle CLI needs Python 3.11+; this machine has 3.9.** *Did:* the CLI is a separate install; the pipeline only
   calls it as a subprocess.
10. **Praxis was not active in Claude Code** (files copied, not registered), so early work skipped its approval
    gates. *Did:* stopped, archived the code-first draft, registered it through the marketplace, redid the work
    spec-first and test-first.
11. **The shot-length rule could not be met (found in review).** With shots capped at 5 s, 11 shots make 58 s, so
    only exactly 12 shots could satisfy "8-12 shots, 60-90 s". *Did:* allowed shots up to 8 s; Wan's clip is ~5 s, so
    the last frame is held for the remainder.
12. **Network timeouts escaped the retry loop on Python 3.9 (found in review).** `socket.timeout` is not a
    `TimeoutError` there, so a slow response skipped the retry and left no log line. *Did:* any `OSError` or
    `http.client` error is now retried and logged.
13. **`gemini-3.8-flash` over capacity (first real script run, 2026-11-25).** Three attempts: two 120 s timeouts and
    one HTTP 503 "experiencing high demand"; the stage wrote nothing. *Did:* switched the default text model to
    `gemini-3.5-flash-lite`, which drafted on the first attempt; Flash stays selectable (`EXPERIMENTS.md` E2, issue #6).
