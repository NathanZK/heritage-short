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
14. **ACE-Step import failed in a Kaggle notebook (owner's `ace-step-music-test`, 2026-11-24; from that notebook's
    saved output).** Kaggle sets `MPLBACKEND=module://matplotlib_inline.backend_inline`; ACE-Step imports matplotlib
    (via torchmetrics, per the traceback), which rejected it with a `ValueError`. With `MPLBACKEND=Agg` the same
    session loaded ACE-Step 1.5 on a T4 and generated audio. *Did:* the music kernel runs ACE-Step with
    `MPLBACKEND=Agg` (regression test in `tests/test_kernels.py`). The first pipeline music job (2026-11-25) then ran
    without the error; whether Kaggle sets the same variable for batch script kernels was not checked, since the fix
    makes it irrelevant.
15. **The pipeline's Wan kernel cannot run on a Kaggle T4 (found by reading Wan 2.2's source, 2026-11-25; not run).**
    Wan's video model calls `flash_attention` directly and asserts FlashAttention 2 is installed (`model.py`,
    `attention.py`); FlashAttention 2 needs Ampere or newer, and a T4 is Turing. bfloat16 is hard-coded, and Wan's
    README gives a 22.8 GB peak for our flags against the T4's 16 GB. This resolves entry 5. Entry 6 is also out of
    date: Diffusers' `WanImageToVideoPipeline` does image-to-video with TI2V-5B. *Did:* did not submit the kernel,
    to save quota; tested the Diffusers route instead, which worked but slowly (`EXPERIMENTS.md` E5); the film uses
    the FFmpeg fallback (`EXPERIMENTS.md`, "Production decision").
16. **A stale mock storyboard broke mock runs (first Stage 1 work, 2026-11-25).** `python -m pipeline.script --mock`
    failed storyboard validation. `outputs/mock/storyboard.json`, left by an earlier mock run, predated a schema change,
    and mock runs read that copy whenever the real `storyboard.json` has no shots (`Ctx.storyboard_path` in
    `pipeline/common.py`). *Did:* deleted `outputs/mock/` once (it is git-ignored and rebuilt); PR #5 notes this for
    existing working copies. *Status:* open; nothing checks the mock copy's version, so an old working copy can hit
    this again after a schema change. Fresh clones are unaffected.
17. **Built-in assessment rules blocked the three-shot test (2026-11-25).** 8–12 shots and 60–90 s were enforced as
    universal limits, so the script stage could not draft the first real 3-shot, 18 s test; the only way round was
    `--draft`, which skipped the rules without a trace. *Did:* added `target_seconds` and an opt-in
    `assessment_rules` (PR #5). *Status:* fixed.
18. **`flux-1-schnell` keyframes did not keep the character (first real keyframe run, 2026-11-25).** Entries 3 and 4
    in practice: with no seed or reference image, the woman's face changed between shots, and the generated reference
    sheet showed two women. *Did:* switched keyframes to FLUX.2 [klein] 4B with an approved reference image
    (`EXPERIMENTS.md` E1, E3; PR #7). *Status:* the face is consistent across the three Klein keyframes. Still open:
    the jebena's long neck and handleless cups are not followed from text (E3).
19. **The Klein reference image broke the clean-clone check (found before PR #7).** The approved reference was first
    under git-ignored `outputs/`, so a fresh clone could not run `make film`, and the clean-clone hygiene test failed.
    *Did:* moved it to the tracked `assets/character_reference.jpg`. *Status:* fixed (PR #7).
20. **Reference-image handling gaps (found in review of PR #7, not in a run).** EXIF rotation was ignored, a path
    outside the repository was accepted, and CMYK, alpha and palette images were not converted. *Did:* fixed all three
    with tests before merging. *Status:* fixed (PR #7).
21. **The Kaggle CLI would not start (setup for the music stage, 2026-11-25).** `kaggle` 2.2.5 installed with
    `uv tool install kaggle` failed at import with `ModuleNotFoundError: No module named 'six'`. *Did:*
    `uv tool install kaggle --with six --force`; the CLI then authenticated and worked for every run since. *Status:*
    worked around locally. Why `six` was missing from that environment was not investigated, and whether the README's
    `pipx install kaggle` hits the same error was not checked.
22. **ACE-Step loaded its language model but did not use it (first real music run, 2026-11-25).** The kernel log
    shows `use_lm=False`, so the caption went to the music model verbatim (`EXPERIMENTS.md` E4). *Status:* open; the
    cause is not established (README M5). The three candidates were generated and candidate 3 was chosen.
23. **`--fallback` submits the Wan kernel first (found while preparing the first production decision, 2026-11-26).**
    In `pipeline/shots.py` the fallback is applied only after the Kaggle job fails to deliver, so a fallback-only run
    would still push the kernel that cannot run on a T4 (entry 15). *Did:* documented a workaround,
    `KAGGLE_USERNAME= python -m pipeline.shots --fallback`, checked on a scratch copy (no Kaggle CLI call; three
    `kind: fallback` clips). *Status:* open; no fallback-only option exists.
24. **Wan through Diffusers on one T4 was slow (`EXPERIMENTS.md` E5, 2026-11-25).** 30.04 s per step for an 81-frame
    clip, and 410.8 s to decode with the float32 tiled VAE: about 40 GPU-session minutes for one 3.4 s clip. *Did:*
    Run 1 (E6) kept the video model on its own GPU and decoded with a float16 VAE: 0.22 s per frame and step against
    0.37 s, and a 53.3 s decode. The offloading and the float32 tiled VAE are the likely causes, but Run 1 also
    changed the GPU count, frames and steps, so the cause was not isolated. *Status:* superseded by the E6 setup,
    which is still slower than hoped (entry 27).
25. **The 9:16 export cut off the character (end-to-end integration test, 2026-11-25).** Every 16:9 shot was
    centre-cropped for 9:16: in s01 the woman's face was cut off at the right edge, and in s02 it was partly at the
    edge. *Did:* added an optional per-shot `portrait_crop_x` (0 to 1; default 0.5, the old centre crop), with
    regression tests. In the integration test's copy of the storyboard, 0.65 for s01 and 0.62 for s02 kept her face in
    frame for every shot (s02's pestle mostly leaves the frame), and the 16:9 frames stayed bit-identical. *Status:*
    fixed on the branch `fix/vertical-framing-loudnorm-linear`, which is not committed or merged; on `main` the
    9:16 export is still a centre crop, and the real `storyboard.json` has no crop values.
26. **Loudness normalisation went dynamic (same integration test).** The two-pass `loudnorm` asked for `linear=true`,
    but pass 1 measured a loudness range of 18.30 LU against the target of 11. FFmpeg's `loudnorm` only scales
    linearly when the target range covers the input's, so it switched to dynamic mode: the gain varied from +1.0 dB
    under the first line to +6.4 dB on the end card. *Did:* pass 2 targets a range of at least the measured one
    (11 to 50) and warns if the result is still not linear. On the same material the gain became a constant +2.45 dB,
    at -14.0 LUFS and -1.6 dBTP. *Status:* fixed on the same uncommitted branch as entry 25; not on `main`.
27. **Wan Run 1 meets the quality bar only partly (`EXPERIMENTS.md` E6, 2026-11-25).** On the one shot tested, the
    woman's face changed in the last second (darker skin, more hair, no earring), the camera drifted instead of making
    the prompted push-in, and the 9:16 crop of the 480p clip is soft (edge measure 2.76 against 4.62 for the fallback).
    It also took 591.7 s per shot, against the 2–5 min aimed for. *Did:* adopted Wan as the primary approach with the
    pan-and-zoom as the per-shot fallback, and made every Wan clip subject to the filmmaker's review
    (`EXPERIMENTS.md`, "Production decision (revised after E6, 2026-11-26)"). *Status:* open.

## Status of the early entries (2026-11-26)

The entries above are kept as written. Where later evidence changed them:

- **3:** still true of `flux-1-schnell`; the default keyframe model is now FLUX.2 [klein] 4B, which takes a seed.
- **4:** superseded. Cloudflare's klein-4b changelog entry documents the multipart fields, and keyframes now use a
  reference image (`input_image_0`; PR #7, entry 18).
- **5 and 6:** resolved by entry 15 and by E5 and E6: Wan's own command cannot run on a T4; Diffusers can.
- **7:** partly answered by real runs. `kaggle quota` reports the weekly GPU hours (30 h); `kaggle kernels status`
  reports `KernelWorkerStatus.RUNNING` and `KernelWorkerStatus.COMPLETE`; the music kernel found its job file under
  `/kaggle/input`. The output of a failed batch run, the session limit and phone verification are still unchecked.
- **9:** still true; the CLI was installed with `uv tool` in its own Python 3.13 environment (entry 21).
- **15:** the finding stands; the film's moving-shot approach is now Wan through Diffusers (`EXPERIMENTS.md` E6), with
  the fallback per shot, not the fallback for every shot.
