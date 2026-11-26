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

28. **Gemini TTS refused requests at the free tier's per-minute limit (English narration for "The Tree That Was
    Always There", 2026-11-26).** HTTP 429: "limit: 3 requests per minute on Free Tier". The stage's retries wait 2 s
    and 4 s, too short to outlast it, so it stopped at s08 after 7 lines; 7 refused attempts are logged in
    `runs.jsonl`. *Did:* resumed with `--only` in groups of at most three, a minute apart; cached lines were not
    regenerated. *Status:* open; the narration stage does not pace its calls.
29. **The radio cut was narration only (found while producing the cut for "The Tree That Was Always There",
    2026-11-26).** `assemble --radio` mixed the narration over black without the chosen score, burned-in subtitles or
    the end card. *Did:* the radio cut is now the film without footage: the same ducked mix with the chosen score,
    the burned-in subtitles and the 3 s "Made with AI" end card over black, plus `final/radio_cut.srt` (tests in
    `tests/test_assemble.py`). *Status:* fixed; it stays narration only until a score is chosen.
30. **Loudness normalisation went dynamic because of the peak limit (the same radio cut).** Pass 1 measured -16.89
    LUFS, a range of 6.40 LU and a true peak of -2.02 dBTP. The +2.9 dB needed for -14 LUFS would lift the peak above
    the -1 dBTP ceiling, so `loudnorm` switched to dynamic mode although the range was within its target (unlike entry
    26). Measured on the mix: the gain varied from +2.1 to +7.1 dB between 1 s blocks, lifting quiet passages more
    than speech. The output meets -14.0 LUFS; the true peak of the AAC-encoded cut is -0.9 dBTP. *Status:* open; the
    fix for entry 26 does not cover this. A true-peak limiter before the gain would let the gain stay constant
    (not tried).

31. **Klein did not produce believable children for the reference ("The Tree That Was Always There" keyframes,
    2026-11-26).** Two attempts at a reference image with FLUX.2 [klein] 4B and no reference (`EXPERIMENTS.md` E9 steps
    1 and 2) gave children who looked younger than the established ages (about 7 to 10, with no clear gap between the
    narrator and the cousin) and did not read as Ethiopian; the first also made the house look stereotypically rural.
    *Did:* the filmmaker generated the reference image with Nano, outside the pipeline, and supplied it
    (`assets/the_tree_children_reference_photo.jpeg`). *Status:* resolved for this film; Klein still makes every
    keyframe.
32. **One reference image conditioned every shot (same keyframes).** `character.reference_image` was sent for every
    shot, and the children took over s03, a shot meant to show only the tree
    (`outputs/experiments/tree-keyframes-v1/s03_with_reference.jpg`). The reference sentence also read "the three children is the person in image 0". *Did:* a
    per-shot `character_reference: false` (no image for that shot) and the wording "Image 0 is the visual reference
    for …: keep the same faces, ages, builds and clothing as in image 0", with tests in `tests/test_keyframes.py`.
    *Status:* fixed.
33. **The reference's faces were too small to keep (same keyframes).** Cloudflare accepts input images under 512 px,
    so the 1408×768 Nano image was shrunk to 511×279 and each face reached Klein at about 24 to 28 px. s06 kept the
    clothing and courtyard but drifted on the faces (older, longer faces, lighter complexions, changed hair), and its
    first try added extra children (`outputs/experiments/tree-keyframes-v2/`). The prompt was checked and did not
    contradict the reference. *Did:* a children-only crop (451×502), sent unshrunk, faces about 67 to 76 px
    (`assets/the_tree_children_reference_crop.jpg`). *Status:* resolved; the backgrounds now come only from text and
    vary between shots.
34. **Children appeared in the shots meant to show no one (same keyframes).** With the reference switched off, s01 to
    s04, s09 and s11 still received the children's description ("When children appear they are always the same three
    …"), and each showed 3 or 4 other children whose faces did not match ours (the filmmaker described them as
    Asian-looking; `outputs/experiments/tree-keyframes-v3-before-fix/`). *Did:* shots marked
    `character_reference: false` now get only the style, the new `setting` field and the shot text. *Status:* fixed; the
    six were regenerated and show no people.
35. **The cut-tree scenes showed an intact tree (same keyframes).** s08 and s10, conditioned on the reference, copied
    its intact tree and its relaxed, smiling poses; s09 showed a full leafy tree
    (`outputs/experiments/tree-keyframes-v3-before-fix/`). *Did:* s08 to s10 now describe a short, bare, knee-high stump with no trunk,
    branches or leaves above it, s08 from behind and s10 with sad, subdued faces; the no-children `setting` text leaves
    the tree out. *Status:* fixed for the stump. Still open: in s08 the cousin faces the camera.
36. **Remaining artifacts in the accepted keyframes and the fallback film (`EXPERIMENTS.md` E9, E11).** In s07 the
    narrator sits in mid-air; the reference's toy car and leaning pose recur in s05, s06, s08 and s10; the courtyard,
    house and stump vary between shots; s06 shows reaching, not climbing; in the 9:16 export the centre crop cuts the
    cousin out of s06, s08 and s10 (entry 25; the fix is not on `main`). *Status:* open, accepted for the preview.

37. **Late-clip drift across the Wan production film (`EXPERIMENTS.md` E12, 2026-11-26).** All 11 clips were
    technically valid and started from their own keyframes, but late in the clip s02, s06, s09 and s10 broke the
    story (a stranger, extra children, a jug on the stump, changed faces), s11 had an object sweep through and s05's
    faces drifted. The assembly holds each 5.04 s clip's last frame for the rest of its shot, so the defects stay on
    screen. The drift was already seen in E6 and E10; its cause was not diagnosed. *Did:* recorded the defects and
    chose the FFmpeg film (E11) as the final film; the Wan film is kept as an experiment (`EXPERIMENTS.md`, "Model
    selection for the final film"). A community viewer later described the Wan film as erratic (`DESIGN.md`, section
    5). The lesson: technically successful generation is not the same as a usable storytelling result; every clip
    passed the automated checks, and the problems only showed when the clips were watched. *Status:* open for Wan.
    Options: regenerate those shots, use only the part of each clip before the drift, or cut to the keyframe still;
    set `portrait_crop_x` for s05 and s07.

38. **The tree is not one consistent place across the film (keyframes, both films).** Each keyframe is generated from
    text, and only the children have a reference image, so the tree's size and position vary between shots and it is
    missing from the shots of the house before it appears in the middle of the patio (`EXPERIMENTS.md` E9 noted the
    varying courtyard, house and stump). One community viewer said the tree sometimes seemed absent from the house
    before appearing in the middle of the scene and asked for it to be more consistent; the other found it smaller
    than expected (`DESIGN.md`, section 5). *Status:* open, not yet changed. A candidate fix is a
    reference image for the setting, used like the children's crop; not tried.

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
- **15:** resolved in the pipeline: the shots kernel now runs E6's Diffusers setup (`EXPERIMENTS.md` E10). The
  finding stands for Wan's own `generate.py`, which is no longer used.
- **23 (later entry):** fixed: `--fallback-only` makes pan-and-zoom clips with no Kaggle call.
- **25 (later entry):** the per-shot `portrait_crop_x` is now on `main` (without entry 26's loudness change, which is
  still open); "The Tree That Was Always There" sets it for s06 (0.65), s08 and s10 (0.72) so the cousin stays in
  the 9:16 frame.
- **15 (earlier note):** the finding stands; the film's moving-shot approach is now Wan through Diffusers (`EXPERIMENTS.md` E6), with
  the fallback per shot, not the fallback for every shot.
