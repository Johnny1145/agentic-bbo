---
name: refine-incumbent
description: Propose one controlled local refinement around a clearly strong incumbent when nearby candidates are also strong, recent local changes still improve the objective, and there is no evidence of local stagnation. Do not use for isolated incumbents, repeated failed local searches, or when exploration is more valuable.
---

# Refine Incumbent

## Overview

Use this conditional BBO skill when history supports local exploitation near the current best candidate.

## Goal

Make a small, evidence-backed modification around the incumbent while preserving unrelated high-performing settings.

## Use When

- There is a clearly strong incumbent.
- Several nearby candidates are also strong.
- Recent local modifications still improved the primary objective.
- No clear local stagnation has been observed.

## Do Not Use When

- The incumbent is an isolated high score with no local support.
- The same neighborhood has been searched repeatedly without improvement.
- High-performing candidates occupy unrelated regions.
- The main need is broader exploration or stagnation recovery.

## Evidence To Check

- `get_history_overview` for incumbent and recent best progression.
- `find_nearest_trials` around the incumbent.
- `estimate_local_effects` for variables considered for small changes.
- `get_recent_search_actions` to avoid repeating the same local move.
- `validate_candidate` before final output.

## Procedure

1. Treat the incumbent as the parent candidate.
2. Check nearby trials and recent actions.
3. Select one or a few variables with local positive evidence.
4. Use a step size smaller than or comparable to historically successful local steps.
5. Keep unrelated incumbent settings unchanged.
6. Validate the candidate.
7. Return one candidate with `search_intent: "exploitation"` and parent-trial metadata.

## Tool Usage

Use exact comparison or nearest-neighbor tools when the local evidence is not obvious. Do not estimate long parameter dictionaries by sight.

## Output Expectations

The final candidate should include parent trial IDs, modified variables, a short hypothesis, and the expected evidence that would support or weaken the local-refinement hypothesis.

## Positive Example

Situation:
Trial 24 is the incumbent. Trials 18, 21, and 23 are nearby and also strong.

Evidence:
`estimate_local_effects` shows two local comparisons where slightly lowering `dropout` improved the score, and no recent action has tested a smaller decrease from Trial 24.

Why this skill applies:
The incumbent is supported by a local high-performing neighborhood and a small variable change has positive local evidence.

Action:
Start from Trial 24 and decrease only `dropout` by a historically successful step size.

What the result would mean:
A better score would support continued local refinement of `dropout`. A worse score would suggest that the local optimum is near Trial 24 for that variable.

## Counterexample

Situation:
Trial 24 is best, but every nearest trial is much worse.

Evidence:
`find_nearest_trials` shows no high-performing neighbor near Trial 24, and the last three local refinements around it failed.

Why this skill should not apply:
The incumbent looks isolated and local exploitation has already stalled.

Better action:
Use `escape-search-stagnation` if the trajectory has collapsed, or let the controller choose a broader exploration move.

## Guardrails

- Change only variables with local evidence unless a small exploratory tweak is explicitly justified.
- Do not increase step size beyond successful historical local steps.
- Do not overwrite the search intent with repair metadata if a secondary repair is needed; preserve the original exploitation intent.
