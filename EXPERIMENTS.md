# Experiments

One entry per experiment: hypothesis, change, result. Aim for about ten. Back every result with evidence in
`runs.jsonl`, `outputs/` or your notes. Nothing has been run yet.

## Template

### E1: <short title>
- **Hypothesis:** if I change ___, then ___, because ___.
- **Change:** what exactly (storyboard field, model parameter, prompt).
- **Result:** what happened, with numbers or a pointer to the log lines.
- **Decision:** what you kept and why.

## Ideas worth testing

- Does repeating the character description in every keyframe prompt keep the character recognisable?
  (No seed or reference image is available; judge across the first full set of keyframes.)
- Wan on a Kaggle T4: does the documented command run, and how long does one shot take?
- Three ACE-Step candidates: which one fits the narration, and why?
- Gemini TTS: Amharic versus English narration quality and pacing.
