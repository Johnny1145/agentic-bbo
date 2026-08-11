---
name: isolate-variable-effect
description: Propose one controlled hypothesis-test candidate when a successful trial changed multiple variables, variables often co-vary, or several competing explanations exist and a one-variable or single-reversal experiment would reduce uncertainty. Use for interpretability-focused control tests, not for routine exploitation.
---

# Isolate Variable Effect

## Overview

Use this conditional BBO skill to design one controlled trial that clarifies which variable change likely mattered.

## Goal

Reduce uncertainty by changing only one variable or by undoing one part of a confounded multi-variable change.

## Use When

- A successful candidate changed multiple variables at once.
- It is unclear which change caused the improvement.
- Multiple variables frequently vary together in history.
- Several plausible explanations compete.
- A controlled variable experiment would materially improve future decisions.

## Do Not Use When

- The desired next step is pure exploitation of a well-understood local effect.
- There is not enough history to define a meaningful control.
- The controlled test would be nearly identical to an already evaluated candidate.
- The candidate would violate constraints or destroy the tested context.

## Evidence To Check

- `compare_trials` for exact differences between source and improved trials.
- `find_nearest_trials` to avoid duplicates and confirm local context.
- `estimate_local_effects` to verify that evidence is currently ambiguous.
- `validate_candidate` before final output.

## Procedure

1. Select the ambiguous improvement or decline to explain.
2. Choose one variable or one component of a multi-variable change to isolate.
3. Start from a reference trial and apply only the isolated change, or start from the improved trial and revert only one change.
4. Keep all other variables unchanged.
5. Validate the candidate.
6. Return one candidate with `search_intent: "hypothesis_test"`.

## Tool Usage

Use `compare_trials` for the source and target trials. Use `validate_candidate` and nearest-neighbor checks before final output.

## Output Expectations

Record the tested hypothesis, parent/reference trials, isolated variable, and what a high or low score would imply.

## Positive Example

Situation:
Trial B improved substantially over Trial A, but x1, x2, and category were all changed.

Evidence:
The history does not contain another comparable transition that isolates any of the three changes.

Why this skill applies:
The source of the improvement is ambiguous, and a controlled trial can reduce that ambiguity.

Action:
Start from Trial A and reproduce only the x1 change. Keep x2 and category unchanged.

What the result would mean:
A clear improvement would support x1 as a useful factor. A score close to Trial A would weaken that hypothesis.

## Guardrails

- Prefer interpretability over predicted best score for this one trial.
- Keep the tested context intact.
- Do not claim causality; describe the result as evidence for or against a hypothesis.
