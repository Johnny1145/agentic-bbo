---
name: escape-search-stagnation
description: Propose one stagnation-recovery candidate after tools confirm that best objective has not improved for several meaningful rounds, recent candidates are overly similar, the same parent/variables/skill recur, or a repeatedly recommended region has failed. Do not use for short histories, ongoing improvement, invalid-candidate failures, or normal early local refinement.
---

# Escape Search Stagnation

## Overview

Use this conditional BBO skill only after verifying that the trajectory has collapsed or stopped improving.

## Goal

Propose one candidate that is deliberately different from the recent failed trajectory while staying valid and evidence-informed.

## Use When

- Best objective has not improved for a meaningful sequence of rounds.
- Recent candidates are increasingly similar.
- The same parent, variables, or skill are repeatedly used.
- The current local region has been tested many times.
- A surrogate-recommended region has been rejected by the real evaluator several times.

## Do Not Use When

- History is still short.
- Recent trials still show meaningful improvement.
- The primary failure mode is invalid candidates.
- The run is in normal early local refinement.
- There is insufficient evidence that the trajectory has collapsed.

## Evidence To Check

- `get_history_overview` for last best trial and recent objective sequence.
- `measure_search_coverage` for trajectory collapse and underexplored regions.
- `get_recent_search_actions` for repeated parents, variables, or skills.
- `find_nearest_trials` to ensure the new candidate is meaningfully different.
- `validate_candidate` before final output.

## Procedure

1. Confirm stagnation with tools before using this skill.
2. Identify the likely cause: repeated parent, tiny local moves, exhausted region, failed surrogate area, or unvisited categories.
3. Choose one recovery action: underexplored region, unused category, different elite restart, opposite hypothesis, larger radius, or abandoning the incumbent as parent.
4. Build one valid non-duplicate candidate.
5. Validate that it is different from recent history.
6. Return one candidate with `search_intent: "stagnation_recovery"`.

## Tool Usage

Tool verification is required. Do not rely on a vague impression that progress feels slow.

## Output Expectations

Record the stagnation evidence, recovery move, parent/reference trials if any, and expected evidence.

## Positive Example

Situation:
The best score has not improved in eight trials. The last five candidates all modify only `dropout` around Trial 31.

Evidence:
`get_history_overview` shows last best at Trial 31. `measure_search_coverage` reports low recent pairwise distance and unvisited optimizer categories. `get_recent_search_actions` shows repeated exploitation from Trial 31.

Why this skill applies:
The recent trajectory has collapsed around one parent and one variable without improvement.

Action:
Propose one valid candidate from an underexplored optimizer category while retaining only broadly supported settings.

What the result would mean:
A strong result would support restarting from the underexplored region. A weak result would still provide evidence that the region is less promising.

## Counterexample

Situation:
The last two trials did not improve, but Trial 19 improved substantially three rounds ago.

Evidence:
`get_history_overview` shows recent progress, and `measure_search_coverage` does not show trajectory collapse.

Why this skill should not apply:
Two non-improving trials are not enough to diagnose stagnation.

Better action:
Use normal controller judgment, `refine-incumbent`, or `follow-promising-direction` if their triggers are satisfied.

## Guardrails

- Do not design a batch allocation; submit one candidate only.
- Do not use this skill when validation failures are the real problem.
- Do not move randomly without explaining the recovery hypothesis.
