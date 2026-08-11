---
name: recombine-complementary-elites
description: Propose one recombination candidate when multiple elite trials are strong for different, historically separable variable groups and combining those supported components should remain valid. Do not use for blind field-wise crossover, a single elite, strong variable dependencies, or when parent advantages cannot be localized.
---

# Recombine Complementary Elites

## Overview

Use this conditional BBO skill to combine supported components from multiple strong trials.

## Goal

Construct one candidate that merges complementary elite components without blind crossover.

## Use When

- Multiple high-performing candidates exist.
- Different elites appear strong in different variable groups.
- Those variable groups have some independent historical evidence.
- The combination is expected to satisfy constraints.

## Do Not Use When

- Only one meaningful elite exists.
- Variables have strong conditional dependency.
- The source of each parent's advantage cannot be localized.
- Direct combination would break structural or semantic validity.

## Evidence To Check

- `get_trial_history(mode="best")` for elites.
- `compare_trials` to locate variable-group differences.
- `estimate_local_effects` for independent support.
- `validate_candidate` to check the recombined candidate.

## Procedure

1. Select two or more elite parent trials.
2. Identify the supported component each parent contributes.
3. Verify that the components are likely compatible.
4. Build one recombined candidate using only supported components.
5. Validate the candidate and check for duplicates.
6. Return one candidate with `search_intent: "recombination"`.

## Tool Usage

Use exact comparison tools. Do not align long configs manually.

## Output Expectations

Record parent trials, inherited variable groups, compatibility rationale, and expected evidence.

## Positive Example

Situation:
Trial 17 is strong because its optimizer category and momentum settings work well. Trial 22 is strong because its regularization settings work well.

Evidence:
Comparisons show that optimizer/momentum changed independently from regularization in several trials, and neither component requires the other's parent context.

Why this skill applies:
There are complementary elite components with separable evidence.

Action:
Use Trial 17's optimizer and momentum with Trial 22's regularization, keeping other values from the stronger parent.

What the result would mean:
A strong score would support compatibility of the elite components. A weak score would suggest dependency between the inherited groups.

## Guardrails

- Do not perform blind per-field crossover.
- Do not recombine components that lack independent support.
- Do not combine values that break task semantics or validation.
