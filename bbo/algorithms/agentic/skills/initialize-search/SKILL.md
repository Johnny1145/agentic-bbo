---
name: initialize-search
description: Propose one information-rich initial black-box optimization candidate when there are no evaluated trials, too few observations for trends, poor initial space coverage, or unvisited important categories, bounds, or variable ranges. Do not use once a promising region or reliable surrogate is supported by history.
---

# Initialize Search

## Overview

Use this conditional BBO skill to improve early coverage before the controller has enough evidence for exploitation, interaction tests, or surrogate-guided proposals.

## Goal

Propose exactly one valid candidate with high coverage or information value. Internal virtual candidates are allowed, but only one evaluator-facing candidate may be returned.

## Use When

- No candidate has been evaluated.
- The history is too short to infer meaningful variable trends.
- Existing initial candidates cover only a small part of the search space.
- Important categories, boundaries, or numeric ranges have not been visited.

## Do Not Use When

- A promising region is already supported by enough observations.
- A reliable surrogate can be fitted and validated.
- The current problem is local stagnation rather than insufficient initial coverage.
- The candidate is only a minor local refinement around the incumbent.

## Evidence To Check

- `get_history_overview` for evaluated count and budget.
- `measure_search_coverage` for visited ranges, categories, and recent distance.
- `get_search_space` for bounds, defaults, and categories.
- `validate_candidate` before the final answer.

## Procedure

1. Inspect history size and coverage.
2. Identify the most valuable missing region, category, boundary anchor, or broad-space point.
3. If useful, generate virtual Sobol-like, Latin-hypercube-like, maximin, boundary, or domain-prior anchors.
4. Select one candidate that improves coverage and is not a duplicate.
5. Validate the candidate.
6. Return one candidate with `search_intent: "initialization"` and action metadata explaining the coverage gap.

## Tool Usage

Use `measure_search_coverage` when any history exists. Use `sample_candidates` only to create virtual options; sampled candidates do not consume evaluator budget. Use `validate_candidate` on the selected final candidate.

## Output Expectations

Return only the final candidate payload required by the controller. Natural-language rationale is enough for analysis. Do not emit a separate JSON diagnosis.

## Positive Example

Situation:
Only two trials exist. Both use category `adam`, middle learning rates, and default regularization.

Evidence:
`measure_search_coverage` reports that `sgd` and `adamw` are unvisited and that the learning-rate lower boundary has not been sampled.

Why this skill applies:
The run lacks initial coverage and no reliable trend can be inferred from two observations.

Action:
Propose one valid candidate using `adamw` and a low learning-rate anchor while keeping other parameters near safe defaults.

What the result would mean:
A strong score would make the low-rate `adamw` region worth follow-up. A weak score would reduce the value of that initial anchor and support trying a different unvisited region.

## Guardrails

- Do not submit a batch to the real evaluator.
- Do not repeat a candidate already in history.
- Do not claim that coverage alone proves quality.
- Use initialization only while coverage is the primary issue.
