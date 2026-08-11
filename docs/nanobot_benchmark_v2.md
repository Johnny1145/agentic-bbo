# Nanobot Benchmark v2

This fork keeps the default Nanobot benchmark runner separate from
workflow-specific experiment policies.

See `docs/baseline_execution_logic.md` for the full baseline-by-baseline
execution map.

## Default outer runner

The default runner exposes the registered task exactly as defined by the core
benchmark: task name, task description, and prior knowledge are not masked.

```bash
uv run --extra nanobot python -m bbo.benchmark.nanobot list-tasks

uv run --extra nanobot python -m bbo.benchmark.nanobot run \
  --task branin_demo \
  --seed 1 \
  --skill-mode no-skill \
  --initial-random 20 \
  --optimizer-budget 100

uv run --extra nanobot python -m bbo.benchmark.nanobot matrix \
  --tasks all \
  --seeds 1,2,3,4 \
  --skill-modes both \
  --initial-random 20 \
  --optimizer-budget 100
```

Convenience scripts:

- `scripts/sglang/synthetic/nanobot.sh`
- `scripts/api/synthetic/nanobot.sh`
- `scripts/sglang/exp1/nanobot.sh`
- `scripts/api/exp1/nanobot.sh`

Nanobot defaults to `workspace_json` in this runner. The BBO workspace tool
limit remains configurable via `--agent-max-tool-calls`; Nanobot native
`read_file`, `exec`, and `write_file` calls are counted in summaries but are
not tightly limited by default.

Skill mode copies the built-in BBO skill library into `agent_workspace/skills/`
and writes `skills/index.json` with each skill's search intent and required
evidence tools. If a final candidate declares a built-in `search_action.skill`,
the same agent attempt must read that skill's `SKILL.md` and call the required
BBO workspace tools, including final candidate validation where required.

Run summaries include `tool_usage_summary` fields for BBO workspace tool counts,
Nanobot native tool counts, skill reads, accepted skill/search-intent counts,
non-validation numeric evidence calls, and skill evidence validation failures.

## Workflow policy layer

Experiments that restrict task identity, priors, metadata, or filesystem
visibility should live under `workflow/`.

The first policy workflow is:

```bash
bash scripts/sglang/exp1/nanobot.sh \
  --tasks branin_demo \
  --seeds 1,2 \
  --variant both \
  --initial-random 20 \
  --optimizer-budget 100
```

That workflow replaces agent-facing task ids with `restricted_task_###`,
removes the `prior_knowledge` section, and reduces agent-visible metadata.
It is not intended to be a full OS-level secrecy boundary.
Default run artifacts stay under `workflow/exp1/outputs/`.
