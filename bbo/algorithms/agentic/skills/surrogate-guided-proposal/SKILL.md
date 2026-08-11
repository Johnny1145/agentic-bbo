---
name: surrogate-guided-proposal
description: Propose one surrogate-guided candidate only when variables can be reasonably encoded, enough history exists, fit_and_check_surrogate reports usable validation signal, simple local rules are insufficient, and evaluator cost justifies model-based selection. Do not use with too little history, poor validation, unsuitable representations, repeated surrogate failure, changed search spaces, or training-fit-only evidence.
---

# Surrogate Guided Proposal

## Overview

Use this conditional BBO skill when a validated model can help choose one virtual candidate for real evaluation.

## Goal

Fit and validate a simple surrogate, score virtual candidates only if validation passes, and submit one valid evaluator-facing candidate.

## Use When

- Variables can be encoded for a numerical model.
- Sufficient evaluated history exists.
- `fit_and_check_surrogate` shows useful validation or ranking signal.
- Simple local trends are insufficient for the next decision.
- Real evaluator calls are costly.

## Do Not Use When

- History is too short.
- Surrogate validation is poor.
- The task representation makes distance or prediction meaningless.
- Recent surrogate proposals repeatedly failed under real evaluation.
- The search space changed.
- The model only fits the training set and lacks validation evidence.

## Evidence To Check

- `fit_and_check_surrogate` before any surrogate-guided candidate is accepted.
- `score_virtual_candidates` only with a validated `model_id`.
- `validate_candidate` for the selected candidate.
- `get_recent_search_actions` to detect repeated surrogate failures.

## Procedure

1. Call `fit_and_check_surrogate`.
2. Stop using this skill if `usable_signal` is false.
3. Generate virtual candidates internally using valid sampling, local proposals, or task priors.
4. Score virtual candidates using `score_virtual_candidates`.
5. Select one candidate using predicted objective, distance, validation status, and objective direction.
6. Validate the selected candidate.
7. Return one candidate with `search_intent: "surrogate_proposal"`.

## Tool Usage

Do not hard-code Gaussian Process as the only acceptable surrogate. Use whatever simple model the tool validates. Virtual candidates do not consume real budget.

## Output Expectations

Record model ID, validation evidence, virtual candidate selection reason, and expected evidence from real evaluation.

## Positive Example

Situation:
There are 24 numeric/categorical trials. Local comparisons are mixed, and no single variable direction dominates.

Evidence:
`fit_and_check_surrogate` returns `usable_signal: true`, leave-one-out ranking accuracy above baseline, and a selected model ID. Several virtual candidates validate successfully.

Why this skill applies:
The surrogate has validation evidence and can rank virtual candidates better than simple local rules.

Action:
Use `score_virtual_candidates`, select the best valid non-duplicate proposal with adequate distance from history, and submit it.

What the result would mean:
A strong real score would support continued model-guided proposals. A weak score would count against this surrogate region and should be recorded for future routing.

## Counterexample

Situation:
The tool can fit a model to eight trials and reports low training error.

Evidence:
`fit_and_check_surrogate` returns `usable_signal: false` because leave-one-out error does not beat the baseline.

Why this skill should not apply:
Training fit is not validation evidence.

Better action:
Use a history-comparison, initialization, or exploration move chosen by the controller.

## Guardrails

- Never use a surrogate proposal after failed validation.
- Never count virtual candidates as evaluated trials.
- Never consume real evaluator budget inside the tool.
