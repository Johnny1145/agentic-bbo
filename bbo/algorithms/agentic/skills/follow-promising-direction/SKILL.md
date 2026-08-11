---
name: follow-promising-direction
description: Propose one cautious directional extrapolation when multiple comparable history changes show that increasing or decreasing a variable repeatedly improves the objective and the current best is near the observed trend edge with legal room remaining. Do not use for one-off trends, confounded multi-variable changes, context-dependent effects, balanced contradictory evidence, or boundary-saturated variables.
---

# Follow Promising Direction

## Overview

Use this conditional BBO skill to take one careful step along a historically supported direction.

## Goal

Extrapolate one variable or a small coherent variable group in a direction supported by repeated comparable evidence.

## Use When

- Multiple comparable history changes show a consistent direction.
- Increasing or decreasing a variable repeatedly improves the primary objective.
- The current best candidate sits at the edge of the observed trend.
- The variable still has legal room before its hard boundary.

## Do Not Use When

- Directional evidence comes from only one observation.
- Each apparent improvement changed many variables at once.
- The variable effect depends strongly on context.
- Supporting and contradicting evidence are similar in count or quality.
- The variable is already close to its hard boundary.

## Evidence To Check

- `compare_trials` for exact variable differences and objective deltas.
- `estimate_local_effects` for supporting and contradicting local comparisons.
- `find_nearest_trials` around the current best.
- `validate_candidate` before final output.

## Procedure

1. Identify the variable direction and supporting trial comparisons.
2. List relevant counterexamples.
3. Use a step no larger than historically successful changes.
4. Keep other variables unchanged when possible.
5. Validate the candidate and ensure it is not a duplicate.
6. Return one candidate with `search_intent: "directional_extrapolation"`.

## Tool Usage

Use `compare_trials` rather than manually aligning configs. Use `estimate_local_effects` when comparable evidence is scattered across history.

## Output Expectations

The final action metadata should name the direction, supporting comparisons, counterexamples, parent trial, and expected evidence.

## Positive Example

Situation:
The best trials have progressively higher `num_layers`, and Trial 30 is the current best at the highest tried value.

Evidence:
Three comparable transitions increased only `num_layers` or changed it with otherwise minor differences, and all improved the objective. One counterexample used a much larger simultaneous learning-rate change.

Why this skill applies:
The direction is repeatedly supported and the boundary still allows one cautious step.

Action:
Start from Trial 30 and increase `num_layers` by the same small increment used in successful comparisons.

What the result would mean:
A better score would support the extrapolation direction. A weaker score would suggest the trend is saturating or context-limited.

## Counterexample

Situation:
Trial 12 improved after `x1`, `x2`, and `category` all changed.

Evidence:
No comparable transition isolates `x1`, and another trial with higher `x1` performed worse.

Why this skill should not apply:
The direction is confounded and contradicted.

Better action:
Use `isolate-variable-effect` to test one factor or let the controller choose a direct exploration move.

## Guardrails

- Do not take a larger extrapolation step than the successful historical step scale.
- Do not ignore contradictory evidence.
- Do not extrapolate through a hard constraint or boundary.
