---
name: repair-invalid-candidate
description: Secondary corrective skill for minimally repairing an already generated candidate after validation reports a clear local schema, bounds, type, category, conditional, or duplicate-adjacent issue and the repair preserves the original search intent. Do not use when the candidate is semantically meaningless, repair would change the search action, constraints are unclear, the result duplicates history, or large rewrites are needed.
---

# Repair Invalid Candidate

## Overview

Use this as a secondary corrective skill after a primary controller or search skill has produced a candidate that fails validation.

## Goal

Make the smallest legal change that preserves the original search intent.

## Use When

- A candidate already exists.
- `validate_candidate` reports a clear and locally repairable violation.
- The repaired candidate still tests the original hypothesis or search action.
- The repair is small, explicit, and can be revalidated.

## Do Not Use When

- The candidate is semantically meaningless.
- Repair would fundamentally change the search action.
- Constraint definitions are unclear.
- The repaired candidate would exactly duplicate history.
- The repair requires rewriting most of the candidate.

## Evidence To Check

- `validate_candidate` violation fields and repairability.
- `find_nearest_trials` if duplicate or near-duplicate risk matters.
- Original candidate rationale and search intent.

## Procedure

1. State the original search intent.
2. Identify the exact validation violation.
3. Apply the minimal legal modification.
4. Preserve unrelated fields.
5. Re-run validation.
6. Return one candidate with the original `search_intent` plus repair metadata.

## Tool Usage

Use `validate_candidate` before and after repair. Do not let the validation tool mutate the candidate; the controller performs the explicit repair.

## Output Expectations

The final action metadata should include the original search intent, repaired field, before/after values, and why the repair preserves the intent.

## Positive Example

Original intent:
Increase model width while preserving the current training configuration.

Violation:
The width must be divisible by 8.

Repair:
Change the width from 130 to 128.

Why this preserves the intent:
The repaired value remains close to the proposed width and does not modify any unrelated variable.

What the result would mean:
The real evaluation still tests whether a wider model helps in the same training context.

## Counterexample

Situation:
A candidate violates three constraints and uses a category incompatible with the parent trial's tested hypothesis.

Evidence:
`validate_candidate` reports non-local repairs and duplicate risk after clipping numeric values.

Why this skill should not apply:
The repair would replace the search action rather than minimally fix it.

Better action:
Discard the candidate and have the controller generate a new valid proposal.

## Guardrails

- Do not use this as a primary search strategy.
- Do not hide the repair; record before/after changes.
- Do not change the original search intent unless the primary candidate must be abandoned.
