# Experiments

One entry per experiment: hypothesis, change, result. Aim for about ten. Back every result with evidence in
`runs.jsonl`, `outputs/` or your notes.

## Template

### E1: <short title>
- **Hypothesis:** if I change ___, then ___, because ___.
- **Change:** what exactly (storyboard field, model parameter, prompt).
- **Result:** what happened, with numbers or a pointer to the log lines.
- **Decision:** what you kept and why.

### E1: FLUX.2 [klein] 4B reference-image keyframe
- **Hypothesis:** if a keyframe is conditioned on an approved image of the character, the character stays
  recognisable in a new scene, because the model sees her face rather than only a text description.
  `flux-1-schnell` (the pipeline's keyframe model) takes no reference image.
- **Change:** one standalone call outside the pipeline to `@cf/black-forest-labs/flux-2-klein-4b` on Cloudflare
  Workers AI.
  - Request: multipart REST, with the reference as `input_image_0`, 1280×704, `seed` 42. One call, no retry.
  - Reference: an approved crop of `outputs/keyframes/s03.jpg` showing only the woman (535×795), uploaded resized
    to 344×511, because Cloudflare requires input images smaller than 512×512.
  - Prompt: "The woman from image 0, wearing the same white netela shawl and silver earrings, walking along a dirt
    path through green highland terraces at sunrise, carrying a woven basket; warm golden morning light, soft
    film grain, painterly realism."
- **Result (observed):** HTTP 200 in 9.46 s; 107.80 neurons, as reported in the `cf-ai-neurons` response header.
  - Output: a 1280×704 JPEG.
  - Identity: a strong match to the reference (face shape, wrinkles, grey hair under the netela, smile).
  - Scene: terraces, sunrise, dirt path, basket and netela are all present.
  - Quality: sharp and coherent, with natural hands.
  - Differences: the earrings are simplified to a small pearl drop; the rendering is photographic rather than
    painterly; she keeps the reference's pose and expression and stands rather than walks.
  - Evidence: `outputs/experiments/flux2-klein-ref/` (git-ignored), containing `call.json`, `experiment.json`, the
    crops, the output and `side_by_side.jpg`. Not logged in `runs.jsonl`.
- **Conclusion:** reference conditioning worked for this one image. Consistency across several shots, and with
  substantially different poses, is untested.
- **Decision:** adopted as the default keyframe model, with this crop as `character.reference_image` (see E3).

### E2: Gemini Flash-Lite for the script stage
- **Hypothesis:** if the script stage uses `gemini-3.5-flash-lite` (Google's current stable Flash-Lite text model),
  it drafts the storyboard even while `gemini-3.8-flash` is over capacity, because it is a separate, lighter model.
- **Change:** the text model only; same storyboard (fictional "Morning Coffee", `target_seconds` 18, 3 shots planned).
- **Result (observed):**
  - `gemini-3.8-flash`, first: three attempts, all errors (120.3 s timeout, HTTP 503 "experiencing high demand",
    120.3 s timeout); nothing written (`runs.jsonl`, 2026-11-25T19:55–19:57Z).
  - `gemini-3.5-flash-lite`: first attempt, 4.79 s, `cost_usd` 0.0; 3 shots × 5.0 s (18 s with the end card); every
    check passed (schema, shot count, total, caption).
  - Draft quality: narration 1.6–2.0 words/s, which fits 5 s shots (confirmed by listening after TTS); music cues drift
    from `music.prompt`, which the script prompt does not show the model.
- **Conclusion:** Flash-Lite drafted reliably where Flash failed on capacity; one storyboard is not a quality
  comparison between the two.
- **Decision:** `gemini-3.5-flash-lite` is the default text model; `"text_model": "gemini-3.8-flash"` selects Flash.
  No automatic fallback (issue #6).

### E3: FLUX.2 [klein] 4B keyframes for all three shots
- **Hypothesis:** with the E1 crop as reference for every shot, the character stays consistent across the film, and
  prompts that name each action make the scenes follow the storyboard better than `flux-1-schnell` did.
- **Change:** pipeline keyframes with `flux-2-klein-4b`, 1280×704, per-shot seed, `assets/character_reference.jpg`
  (the E1 upload crop) as `input_image_0`; no reference sheet. Shot prompts rewritten to name the actions; then s02
  and s03 rewritten again and regenerated alone (`--only s02 s03`; s01 stayed cached).
- **Result (observed):** every call succeeded on the first attempt at 107.8 neurons each (`runs.jsonl`); about 5–8 s
  per image.
  - Identity: the same woman in all three shots and the reference (face, grey hair, white netela, earrings).
  - s01 roasting: green beans in a metal pan over a fire, stirred, with smoke; outdoors over open flame rather than
    a charcoal brazier.
  - s02 first version: a pestle over a shallow bowl, no pounding. Refined: a tall, deep wooden mortar with roasted
    beans and a long pestle, but held at rest with one hand, not mid-strike.
  - s03 first version: a teapot-shaped pot, cups with handles, and neighbours who read as elderly Europeans.
    Refined: neighbours in white netelas and traditional dress; the pot is still a small jug without a jebena's long
    neck, and the cups still have handles.
  - Style: photographic rather than painterly in every image.
  - Evidence: `outputs/keyframes/`; earlier versions in `outputs/experiments/schnell-keyframes-before-klein/` and
    `outputs/experiments/klein-keyframes-v1/` (git-ignored).
- **Conclusion:** the reference keeps the character consistent across three shots; scene objects with specific
  cultural shapes (jebena, handleless sini cups) and in-motion actions are not reliably followed from text.
- **Decision:** keep klein with the reference for keyframes; the jebena and cup shapes remain open.

### E4: ACE-Step 1.5 score on a Kaggle T4
- **Hypothesis:** the music kernel generates three usable candidates on Kaggle's T4 in one short GPU session,
  because ACE-Step's turbo model fits a 16 GB card (14.6 GB usable; its README recommends it for 8–16 GB GPUs).
- **Change:** first real `python -m pipeline.music`, with `MPLBACKEND=Agg` for the runner (FAILURES 14) and
  `HERITAGE_REMOTE_TIMEOUT_S=3600`. Caption: `music.prompt` plus the 3-cue sheet (263 characters); 20 s; seeds 42–44.
- **Result (measured):** run `music-20261125T213435Z-e8eba3`, status COMPLETE, first attempt.
  - Environment (kernel log): Tesla T4, 14.6 GB; ACE-Step ran in a uv-built Python 3.12 environment; float16 instead
    of bfloat16 and eager attention (chosen by ACE-Step for pre-Ampere GPUs); models downloaded from Hugging Face in
    41 s; the runner's language model initialised with the vllm backend.
  - Generation: 3 audio tensors at batch 3; VRAM check 8.55 GB free against about 1.40 GB needed. ACE-Step reported
    `llm_initialized=False, use_lm=False`, so the language model was not used, and our caption reached the DiT verbatim.
  - Time: runner 99.1 s, covering model download, loading and generation but not pip, git clone or `uv sync`
    (`runs.jsonl` repeats this batch total on each candidate line, seeds 42, 43, 44); about 215 s from submission to
    ingest.
  - Quota: 0.33 h → 0.38 h (`kaggle quota`).
  - Files: three 20.000 s, 48 kHz stereo 16-bit WAVs, all different; peak -1.0 dB; mean -23.1, -18.8 and -19.6 dB.
    Silence (below -50 dB for 2 s or more) starts at 11.7 s, 17.2 s and 15.7 s; the film is 18 s.
- **Unverified:** how closely each candidate follows the cue sheet (not checked); that candidate order matches seed
  order (assumed from the kernel's zip of results to items); why the language model the runner loads was not used.
- **Decision:** the filmmaker listened and chose candidate 3 (seed 44); `score.wav` is a byte-identical copy. Its
  silence from 15.7 s covers most of the 3 s end card.

### E5: Wan 2.2 TI2V-5B image-to-video with Diffusers on one Kaggle T4
- **Hypothesis:** the pipeline's Wan kernel cannot run on a T4 (FAILURES 15), but Hugging Face Diffusers'
  `WanImageToVideoPipeline`, which uses PyTorch SDPA instead of FlashAttention, can turn the s01 keyframe into a
  usable moving shot on a 16 GB T4 when the video model runs in float16.
- **Change:** a standalone private Kaggle script (`heritage-short-wan-diffusers-t4` v1, pushed 2026-11-25T22:12:19Z,
  hard limit 5400 s), outside the pipeline.
  - Model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (Apache 2.0).
  - Video model float16 with block-level group offloading (1 block per group, no streams); UMT5 text encoder bfloat16
    on CPU, used once and freed; VAE float32 on the GPU, tiled.
  - Input: `outputs/keyframes/s01.jpg` centre-cropped from 1280×704 to 1220×704, then resized to 832×480.
  - Prompt: s01's `motion_prompt`; Wan's default negative prompt; seed 661535228 (s01's shot seed).
  - 832×480, 81 frames, 24 fps, 50 steps, guidance 5.0; scheduler from the checkpoint (UniPC, flow shift 5.0).
  - Software (kernel log): Python 3.13.16, torch 2.11.0+cu130, diffusers 0.43.0, transformers 5.20.2.
- **Result (measured):** status ok, first attempt; no NaN or Inf in the latents over 50 steps.
  - Time: 2415.2 s in the notebook. Environment 33.7 s; download 140.2 s (31.85 GiB); text encoding on CPU
    211.9 s; model load 91.1 s; denoising 1501.9 s (30.04 s per step, 27.9–36.1 s); VAE decode 410.8 s; VAE
    encode 5.0 s; export 4.2 s.
  - Memory: peak 7141 MiB on GPU0 (nvidia-smi; PyTorch 5.05 GiB allocated, 6.83 GiB reserved); GPU1 unused. Host RAM
    peak 25,635 MiB used of 31.3 GiB (process 24,942 MiB).
  - Quota: Kaggle GPU 0.38 h → 1.05 h (`kaggle quota`).
  - Clip: H.264, 832×480, 24 fps, 81 frames, 3.375 s; decodes cleanly. First frame against the cropped input: PSNR
    34.2 dB; last frame: 12.6 dB. Mean frame-to-frame difference 0.024 (maximum 0.071), no jumps. Mean brightness
    0.43 in the first frame and 0.36 in the last (measured locally from `frame_first.png` and `frame_last.png`).
- **Observed (from sampled frames; not yet judged by the filmmaker):** a slow push-in, rising smoke, flickering
  flames and a stirring spoon; the same face, netela and earring throughout. The background changes after the first
  frame (the sun and hills recede). A small detached flame appears in the smoke at the end, and the last frames are
  darker and more orange.
- **Not logged in `runs.jsonl`:** it ran outside the pipeline. Evidence: `outputs/experiments/wan-diffusers-t4/`
  (git-ignored: clip, `experiment.json`, Kaggle log, sampled frames, `summary.json`, quota readings, the script).
- **Conclusion:** Wan 2.2 TI2V-5B can make a coherent moving shot from a Klein keyframe on a free T4, but one 3.4 s
  clip at 480p took about 32 min of generation plus about 8 min of setup in the same GPU session.
- **Decision:** see "Production decision" below.

## Production decision (2026-11-26)

E4 and E5 show that real music and real moving shots are technically feasible on Kaggle's free T4s. Neither is used
in the assessment film's production workflow:

- **Moving shots: the FFmpeg pan-and-zoom fallback.**
  - The pipeline's Wan kernel cannot run on a T4 (FAILURES 15); using Wan would mean replacing it with the E5
    Diffusers setup.
  - The E5 cost, measured for one 3.4 s, 480p clip, is about 32 min of generation plus about 8 min of setup in the
    same GPU session (0.67 h of quota). Three such clips in one session would take about 1.7 GPU hours
    (3 × 32 + 8 min; estimate). The storyboard's shots are 5 s, which needs about 121 frames, so real shots would
    cost more at any resolution (estimate; not measured).
  - Every regeneration also needs a Kaggle account, the CLI, quota and a session of 40 min or more.
  - The fallback is local, needs no GPU and is what the brief offers when video generation is impractical. It shows
    the Klein keyframes with a slow pan-and-zoom; the clips are marked `kind: fallback`.
- **Music: the score already chosen.** Production uses `outputs/music/score.wav`, the filmmaker's choice of candidate
  3 from E4 (`music.choice` in `storyboard.json`; sha256 `64422849…a631a`).
  - The music stage treats it as current and makes no Kaggle call while the candidates stay unchanged.
  - Regenerating would need Kaggle again and would produce different candidates, which would need a new listening
    choice.
  - `outputs/` is git-ignored, so the score exists only in this working copy; a clean clone would have to run the
    music stage on Kaggle (or be given the file).
- **Known gap:** `--fallback` still submits the Wan kernel to Kaggle first and only falls back when Kaggle does not
  deliver (`pipeline/shots.py`). Until there is a fallback-only option, run the shots stage with `KAGGLE_USERNAME`
  set to empty for that command, so the stage stops before any Kaggle call and falls back
  (`KAGGLE_USERNAME= python -m pipeline.shots --fallback`). This was checked on a scratch copy: no Kaggle CLI call,
  three `kind: fallback` clips.

## Ideas worth testing

- Does repeating the character description in every keyframe prompt keep the character recognisable?
  (Answered for FLUX.2 [klein] 4B with a reference image in E3; FLUX.1 schnell, which takes no seed or reference
  image, drifted.)
- Wan on a Kaggle T4: answered. The documented command cannot run there (FAILURES 15); Diffusers can, slowly (E5).
- Three ACE-Step candidates: which one fits the narration, and why?
- Gemini TTS: Amharic versus English narration quality and pacing.
