---
name: probe-variable-interaction
description: Propose one targeted interaction-test candidate when history shows that variables have context-dependent effects, high-performing candidates share a variable combination, task semantics suggest coupling, and there is evidence that a small joint modification may help. Do not use when basic single-variable effects are still unknown or the interaction is only a domain guess.
---

# Probe Variable Interaction

## Overview

Use this conditional BBO skill when the next useful evidence comes from testing a small variable combination rather than a single variable.

## Goal

Test one plausible interaction hypothesis with a targeted joint modification.

## Use When

- A variable behaves differently in different contexts.
- Single-variable patterns contradict each other.
- High-performing candidates share a combination rather than one value.
- Task semantics and history both support a plausible coupling.
- A joint modification has evidence and is likely legal.

## Do Not Use When

- There is too little history to understand basic single-variable effects.
- The interaction hypothesis comes from only one observation.
- The joint change is likely to violate constraints.
- The interaction is only an LLM domain guess without historical support.

## Evidence To Check

- `compare_trials` for candidate pairs that differ by the variable group.
- `estimate_local_effects` for context-dependent signs.
- `find_nearest_trials` for local support and duplicate avoidance.
- `validate_candidate` before final output.

## Procedure

1. Define a small variable group and the interaction hypothesis.
2. Identify supporting and contradicting historical comparisons.
3. Choose one parent trial whose context matches the hypothesis.
4. Apply the targeted joint change and keep unrelated variables unchanged.
5. Validate the candidate.
6. Return one candidate with `search_intent: "interaction_test"`.

## Tool Usage

Use tools to verify the interaction evidence. Do not invent an interaction from task semantics alone.

## Output Expectations

Record the variable group, parent/reference trials, hypothesis, and result interpretation.

## Positive Example

Situation:
Higher `batch_size` helps only when `learning_rate` is low. At high learning rates it repeatedly hurts.

Evidence:
`estimate_local_effects` shows opposite local effects for `batch_size` across two learning-rate regimes, and two strong trials share low learning rate plus high batch size.

Why this skill applies:
The variable effect appears context-dependent and the joint setting is historically supported.

Action:
Start from a strong low-learning-rate parent and increase `batch_size` one supported step while leaving unrelated variables unchanged.

What the result would mean:
A strong result would support the interaction hypothesis. A weak result would suggest that the apparent interaction is not robust in that local context.

## Guardrails

- Keep the variable group small.
- Do not use this skill to perform broad crossover.
- Do not submit a joint change that cannot be validated.
