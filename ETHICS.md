# Ethics

You fill in the consent record, labelling plan and licence check; the pipeline cannot do it for you.
**If a real person appears or is voiced without a consent record here, the film scores zero** (assessment rule).

## Consent record

| Person | What is generated (face, likeness, voice, words) | Consent given (how, when, by whom) | Words they agreed to |
|---|---|---|---|
| | | | |

## Labelling plan

- The film ends with a visible "Made with AI" end card (`end_card` in `storyboard.json`; it must contain "AI").
- Switch the platform's AI-content disclosure on when posting. Leave SynthID and C2PA metadata intact.

## Licences (checked 2026-11-25 unless marked)

| Model | Licence | Notes |
|---|---|---|
| Wan 2.2 TI2V-5B | Apache 2.0 | model card, verified |
| ACE-Step 1.5 | MIT (code and weights) | the model card states generated music may be used commercially; verified |
| FLUX.1 schnell (via Cloudflare) | **to verify** | not checked |
| FLUX.2 [klein] 4B (via Cloudflare) | Apache 2.0 (open weights) | Hugging Face model card and BFL's klein page, checked 2026-11-25. Cloudflare links BFL's Terms of Service, not a licence, and does not say which weights it serves. Those terms (rev. 2026-08-01): BFL claims no ownership of outputs but takes a licence to use inputs and outputs to improve its models (opt-out: legal@blackforestlabs.ai); outputs must not be presented as human-made or as photographs of real events, and AI labels or metadata must not be removed; consent is required for real, identifiable people. BFL's Usage Policy not read |
| Gemini API and TTS | **to verify** | Google's terms; free-tier data use below |

## Data

Free-tier Gemini prompts and outputs may be used by Google to improve its products, and human reviewers may read
them (Google's terms, read 2026-11-25). Keep private family details out of text sent to Gemini unless you use a paid key.
Keyframes and prompts uploaded to Kaggle go in private datasets; confirm they are private.
`runs.jsonl` is tracked (it is assessment evidence) and records your narration text and prompts: review
`git diff runs.jsonl` for sensitive content before every commit, and before pushing to a public repository.
`outputs/` holds your media and stays git-ignored, so it cannot reach a public repository by accident; decide
deliberately what to publish with `git add -f outputs/<path>`.
