---
name: distill-search-memory
description: Maintain compact BBO search memory when raw history approaches context limits, a search phase ends, many trials are repetitive, old hypotheses conflict with new evidence, or the controller can no longer recover key facts from the raw trajectory. This skill summarizes memory and does not directly propose evaluator candidates.
---

# Distill Search Memory

## Overview

Use this maintenance skill to compress search knowledge without losing trial references.

## Goal

Create or update compact memory that preserves important observations, elites, counterexamples, failed directions, open questions, and trial IDs.

## Use When

- Raw history is near the context limit.
- A clear search phase has ended.
- Many history entries are repetitive.
- Old hypotheses conflict with new evidence.
- The controller is struggling to recover key trajectory facts.

## Do Not Use When

- The immediate task is to propose the next candidate.
- History is short and readable.
- The summary would omit trial IDs needed for later verification.
- The update would delete unresolved evidence.

## Evidence To Check

- `get_history_overview` for progression and recent state.
- `get_recent_search_actions` for repeated or stale actions.
- `compare_trials` for important claims.
- `memory_read` and `memory_write` if memory is enabled.

## Procedure

1. Identify the phase or repeated pattern to compact.
2. Preserve elites, key observations, counterexamples, failed directions, and open questions.
3. Keep supporting trial IDs for every claim.
4. Separate raw observations, tentative hypotheses, repeatedly supported findings, contradicted hypotheses, and open questions.
5. Remove only conclusions that are contradicted or unsupported; do not delete raw references.
6. Write concise Markdown or the repository's existing memory format.

## Tool Usage

Use memory tools if available. Use comparison tools for any non-obvious claim.

## Output Expectations

This skill does not return a real evaluator candidate. It should produce a compact memory note that future rounds can inspect.

## Positive Example

Situation:
Thirty trials contain many repeated local refinements around Trial 18, and the context is becoming crowded.

Evidence:
The incumbent, three failed directions, two supported local findings, and several contradicted hypotheses can be tied to specific trial IDs.

Why this skill applies:
The raw trajectory is repetitive and a phase has effectively ended.

Action:
Write a compact memory note with sections for observations, supported findings, contradicted hypotheses, and open questions, preserving all key trial IDs.

What the result would mean:
Future controller calls can use the memory note to avoid repeating stale moves while still checking raw trials when needed.

## Guardrails

- Do not invent numeric confidence scores.
- Do not delete key trial references.
- Do not propose a candidate from this skill.
