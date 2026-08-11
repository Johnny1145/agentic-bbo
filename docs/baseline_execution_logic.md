# Baseline Execution Logic

This document describes the runnable v2 benchmark paths after cleanup. The
default Nanobot path is full-prior: task name, task description, and prior
knowledge are exposed as registered by `bbo.tasks`.

## Default Nanobot Baseline

Entry points:

```bash
bash scripts/sglang/synthetic/nanobot.sh --dry-run --task branin_demo --seed 1
bash scripts/api/synthetic/nanobot.sh --dry-run --task branin_demo --seed 1
```

Direct CLI:

```bash
uv run --extra nanobot python -m bbo.benchmark.nanobot run \
  --task branin_demo \
  --seed 1 \
  --skill-mode no-skill \
  --initial-random 20 \
  --optimizer-budget 100

uv run --extra nanobot python -m bbo.benchmark.nanobot matrix \
  --tasks family:synthetic \
  --seeds 1,2 \
  --skill-modes both \
  --initial-random 20 \
  --optimizer-budget 100
```

Execution chain:

`bbo.benchmark.nanobot` expands task/seed/skill-mode cases, computes
`max_evaluations = initial_random + optimizer_budget`, then calls
`bbo.run.run_single_experiment`. The core creates the registered task,
constructs `NanobotBBOAlgorithm`, and `Experimenter` runs:

`algorithm.ask()` -> `task.evaluate()` -> `algorithm.tell()` -> JSONL logger.

Nanobot runs through `NanobotEngine`, which launches
`python -m bbo.algorithms.agentic.nanobot_runner ...` inside each run's
`agent_workspace`. The default tool mode is `workspace_json`.

Skill variants:

- `--skill-mode no-skill`: no packaged BBO skills are copied.
- `--skill-mode skill`: packaged BBO skills are copied; if the agent declares a
  skill, it must read that skill's `SKILL.md` in the same attempt.

Outputs:

`runs/nanobot_benchmark/full-prior/<skill-mode>/<task>/nanobot/seed_<seed>/`
unless `--results-root` is supplied. Each run writes `trials.jsonl`,
`summary.json`, agent logs, workspace files, and tool-use counts in the summary.

## Restricted-Prior Nanobot Workflow

Entry point:

```bash
bash scripts/sglang/exp1/nanobot.sh \
  --tasks branin_demo \
  --seeds 1,2 \
  --variant both \
  --initial-random 20 \
  --optimizer-budget 100
```

Execution chain:

The workflow creates the true registered task, wraps it in `RestrictedPriorTask`,
and passes that task object to the shared Nanobot runner helper. The wrapper
changes only agent-facing context:

- task id becomes `restricted_task_###`;
- `prior_knowledge` is removed from the description;
- identifying metadata such as display name and known optima are withheld.

Evaluation still delegates to the true underlying task. This workflow is a
policy experiment, not the default baseline and not a full OS-level secrecy
boundary.

## Core `bbo.run` Baselines

Use these for single-family, direct baseline runs:

```bash
uv run --extra optuna python -m bbo.run \
  --task ackley_2d_demo --algorithm optuna_tpe --seed 1 --max-evaluations 60 --no-plots

uv run python -m bbo.run \
  --task ackley_2d_demo --algorithm random_search --seed 1 --max-evaluations 60 --no-plots

uv run python -m bbo.run \
  --task ackley_2d_demo --algorithm pycma --seed 1 --max-evaluations 60 --no-plots
```

Supported baseline groups:

- `random_search` / `random`: uniform sampling from the task search space.
- `optuna_tpe`: Optuna TPE over compatible spaces.
- `pycma` / `cma_es`: CMA-ES, numeric spaces only; categorical spaces use the
  existing continuous conversion path when supported.
- `llambo`: LLM-based LLAMBO-style candidate generation, with heuristic or
  OpenAI-compatible backend.
- `opro`: OPRO-style prompt optimizer, with heuristic or OpenAI-compatible
  backend.
- `pablo` / `palbo`: planner/explorer/worker agentic optimizer.
- `agentic_openai_compatible`: function-calling general-agent baseline.
- `graph_ga`, `gpbo`, `graph_gpbo`: molecular SMILES baselines; pass
  `--molecular-initial-smiles-path` when an explicit initial pool is needed.

All use the same core loop: `create_task` -> `create_algorithm` ->
`Experimenter` -> `trials.jsonl` and `summary.json`.

## SGLang Matrix Runner

Family entry points:

```bash
python scripts/sglang/matrix.py --family synthetic --dry-run
python scripts/sglang/matrix.py --family dbtune --dry-run
python scripts/sglang/matrix.py --family molecule --dry-run
python scripts/sglang/matrix.py --family bboplace --dry-run
```

Per-baseline wrappers:

```bash
bash scripts/sglang/synthetic/nanobot.sh --dry-run
bash scripts/sglang/dbtune/random_search.sh --dry-run
bash scripts/sglang/molecule/gpbo.sh --dry-run
bash scripts/sglang/bboplace/optuna_tpe.sh --dry-run
```

The matrix runner reads `scripts/sglang/configs/*.toml`.

Dispatch model:

- If an algorithm has a `[commands.<algorithm>]` block, the runner executes the
  referenced workflow script with rendered args.
- If the algorithm is listed under `[generic_bbo_run].algorithms`, the runner
  executes `python -m bbo.run` directly for every selected task and seed.
- SGLang-backed algorithms receive provider/model/base-url args from the
  profile in `SGLANG_ARG_PROFILES`.

Selectors:

```bash
python scripts/sglang/matrix.py --family synthetic \
  --algorithm nanobot random_search optuna_tpe \
  --task ackley_2d_demo \
  --seed 1 2 \
  --dry-run
```

Selectors accept both space-separated and comma-separated values.

Family-specific execution:

- `synthetic`: historical fixed-initial OPRO/Nanobot scripts for `opro` and
  `nanobot`; generic `bbo.run` for random, Optuna, CMA-ES, LLAMBO, Pablo, and
  OpenAI-compatible agent baselines.
- `dbtune`: generic `scripts/sglang/dbtune/run_problem.sh` wrapper for all selected
  algorithms; surrogate tasks require the external surrogate evaluator service,
  and MariaDB tasks require the local MariaDB evaluator.
- `molecule`: shared GuacaMol OPRO/Nanobot workflow for LLM baselines; generic
  `bbo.run` for `graph_ga` and `gpbo`.
- `bboplace`: BBOPlace OPRO/Nanobot workflows plus random, Optuna, and CMA-ES
  BBOPlace workflow scripts. Requires BBOPlace evaluator URLs.
- `exp1`: restricted-prior Nanobot workflow. Its default output roots are
  `workflow/exp1/outputs/sglang` and `workflow/exp1/outputs/api`.

Default output root:

`workflow/script_runs/<family>_<timestamp>/...`

Set `RUN_ROOT` or a family-specific `*_RUN_ROOT` to override.
