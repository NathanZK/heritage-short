---
name: Pipeline issue
about: A change, experiment or blocker in the heritage-short pipeline
title: ""
labels: []
assignees: []
---

## Problem

What is missing, broken or unmeasured today? Describe the observable symptom,
not a proposed solution.

## Goal

What should be true when this is done? Describe the outcome, not the implementation.

## Stage and contract

- Stage(s): <!-- script / keyframes / shots / music / narration / assemble / docs / other -->
- `storyboard.json` or schema fields affected, if any.

## Failure modes

Plausible ways this could fail, for example quota or GPU limits, a model that
does not follow the prompt, an inconsistent character, a stale cache entry or an
unsupported language. For each, say what causes it and what breaks. Write
"None identified" if there are none. Do not use this section for acceptance
criteria or mitigations.

## Hypothesis (experiments only)

If I change ___, then ___ will improve/stay the same, measured by ___.

## Acceptance criteria

- Observable, verifiable outcomes: what is checked and the expected result.
- Evidence to be recorded: `runs.jsonl` entries, and the entry to add to
  EXPERIMENTS.md, FAILURES.md, COSTS.md, DESIGN.md or ETHICS.md.

## Validation

Commands or checks that apply, for example schema validation of `storyboard.json`,
a stage run, or a re-run showing only the changed shot regenerates. A passing
run alone does not show every criterion is met.
