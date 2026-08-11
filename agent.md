# Agentic BBO Project Map

This file is the code-oriented onboarding guide for agents and contributors. Read `AGENTS.md` first for repository rules, then use this map to locate the smallest correct change.

## What this project is

Agentic BBO is a benchmark framework, not a single optimizer. Its central contract is:

```text
TaskSpec + Algorithm.ask/tell + Experimenter + append-only trial log
```

A task owns the search space, objective directions, evaluator, budget, description, and any benchmark protocol. An algorithm proposes configurations and learns only through observations returned by the experiment loop. The core must not depend on any specific task family, optimizer library, LLM provider, or native agent harness.

## Start here

| Need | Primary location |
| --- | --- |
| Run one experiment from the CLI | `bbo/run.py` |
| Run a named benchmark from Python | `bbo/benchmark/runner.py` |
| Add or inspect an algorithm | `bbo/algorithms/registry.py` and the relevant family folder |
| Add or inspect a task | `bbo/tasks/registry.py` and the relevant family folder |
| Change ask/tell orchestration | `bbo/core/experimenter.py` |
| Change search-space types or validation | `bbo/core/space.py`, `bbo/core/conversion.py` |
| Change trial/result schemas | `bbo/core/trial.py`, `bbo/core/task.py` |
| Change JSONL/replay behavior | `bbo/core/logger.py`, `bbo/core/experimenter.py` |
| Change agent prompts or tools | `bbo/algorithms/agentic/prompts.py`, `tools/`, `workspace_*` |
| Change native Nanobot/Codex/Claude harness behavior | `bbo/algorithms/agentic/general_agent*.py` and compatibility adapters |
| Change task context shown to agents | `bbo/task_descriptions/<task_name>/` |
| Change plots | `bbo/core/plotting.py` |
| Understand protocol alignment | `bbo/algorithms/benchmark_protocol.py`, `bbo/algorithms/baseline_factory.py` |

## Repository map

### `bbo/core/`

Benchmark-agnostic contracts and orchestration:

- `algo.py`: `Algorithm`, ask/tell interfaces, and shared algorithm types.
- `space.py`: typed parameters and `SearchSpace` validation/sampling.
- `task.py`: `Task`, `TaskSpec`, objectives, and evaluation results.
- `trial.py`: suggestions, observations, statuses, and serialized trial records.
- `experimenter.py`: the authoritative ask → validate → evaluate → tell loop, budget handling, resume, and callbacks.
- `logger.py`: append-only JSONL metrics and replay inputs.
- `description.py` and `manifest.py`: deterministic task-card loading and policy metadata.
- `conversion.py`: numeric encodings for optimizers that cannot consume mixed spaces directly.
- `plotting.py`: run and comparison figures.
- `prompting.py`: benchmark prompt assembly shared outside individual agent implementations.

Do not put task-specific evaluation code, provider SDK calls, native-harness logic, or experiment-matrix policy in `bbo/core/`.

### `bbo/algorithms/`

All public algorithms are declared in `registry.py` through `AlgorithmSpec`.

- `traditional/`: random search, local perturbation, Sobol, and pycma.
- `model_based/`: Optuna TPE, GP-EI, TuRBO, GIT-BO, and PFN variants.
- `molecular/`: Graph GA and fingerprint/GP-based molecular optimization.
- `llm_based/`: LLAMBO, OPRO, and SkyDiscover strategy evolution.
- `agentic/`: unified agent algorithms, native harness engines, benchmark tools, adapters, prompts, packaged skills, state, memory, and serialization.
- `baseline_factory.py`: creates the same registered implementations used by standalone baselines and agent optimizer tools.
- `benchmark_protocol.py`: resolves shared initialization and candidate budgets from task metadata.

Canonical algorithm names should be stable. Add compatibility aliases only in the registry, and ensure aliases resolve to identical behavior.

### `bbo/tasks/`

All CLI-visible tasks flow through `tasks/registry.py`.

- `synthetic/`: the compact official COCO/BBOB suite, 24 functions at 10 dimensions.
- `scientific/`: tabular scientific tasks, QED/SELFIES tasks, molecular similarity, and direct-SMILES GuacaMol objectives.
- `dbtune/`: active HTTP surrogate tasks plus non-registered legacy/in-process helpers.
- `hpo/`: 25 LLAMBO/Bayesmark dataset-model tasks and aligned initializations.
- `bboplace/`: external HTTP BBOPlace adapter.
- `http_json.py`: shared HTTP evaluator plumbing.

A module being importable does not mean it is CLI-visible. The registries are authoritative.

### `bbo/task_descriptions/`

Runtime task context. Every registered agent benchmark should provide at least:

```text
background.md
goal.md
constraints.md
prior_knowledge.md
```

`evaluation.md`, `submission.md`, `environment.md`, and `manifest.json` are recommended when relevant. Localized files such as `background.zh.md` are documentation companions and must not change the deterministic default context loaded at runtime.

### Supporting directories

- `bbo/benchmark/`: reusable programmatic runner and Nanobot-facing benchmark entry point.
- `examples/`: small editable examples; keep them runnable against registered names.
- `scripts/`: batch wrappers and validation utilities; common logic belongs in shared scripts, not copied across every family.
- `docs/`: design notes, runbooks, protocol explanations, and task-family guides.
- `tests/`: behavior-focused unit and integration coverage.
- `workflow/`: local orchestration and experiment outputs. It is intentionally ignored and must not be required by package code or public documentation.

## Execution flow

1. `bbo.run` parses a registered task and algorithm name.
2. `bbo.tasks.create_task` constructs a `Task` and exposes its `TaskSpec`.
3. Task metadata supplies any shared initialization or candidate-budget protocol.
4. `bbo.algorithms.create_algorithm` constructs the registered optimizer.
5. `Experimenter` replays prior JSONL records when resuming.
6. The loop requests suggestions, validates them against the typed search space, evaluates the task, records the result, and calls `tell`.
7. Summaries, plots, agent workspaces, or generated strategies are written below the run directory.

Never bypass `Experimenter` for a public benchmark path unless the alternate runner preserves the same trial schema, logging, budget, and replay semantics.

## Important invariants

- JSONL histories are append-only. Resume is replay-based; do not rewrite completed observations.
- Evaluators return normalized `EvaluationResult` objects and never choose the next configuration.
- Algorithms do not call hidden evaluators or inspect ground-truth optima unavailable through task metadata.
- Search-space validation happens before evaluation.
- Objective direction is explicit; do not assume every task minimizes.
- Shared initial points belong to the task protocol and must be identical across comparable algorithms for a fixed seed.
- Registered optimizer tools used by agents must resolve to the same implementations and protocol as standalone baselines.
- Credentials and endpoint selection stay at runner/provider boundaries, never in prompts, fixtures, task cards, or committed config.
- Native harness configuration isolation supports reproducibility but is not a security boundary.
- `workflow/`, `runs/`, `artifacts/`, `.env`, `.apikey`, downloaded models, caches, and virtual environments are not source artifacts.

## Common changes

### Add a task

1. Choose or create the family under `bbo/tasks/`.
2. Define the typed search space, objective specification, budget, evaluator, metadata, and deterministic seed behavior.
3. Add the required task-description files.
4. Add the task to its family registry and then `bbo/tasks/registry.py`.
5. If fairness requires shared initialization or candidate budgets, expose them through task metadata and resolve them with `benchmark_protocol.py`.
6. Add focused evaluator, registry, description-loader, and CLI tests.

### Add an algorithm

1. Implement the `Algorithm` contract under the appropriate family.
2. Make randomness explicit and seedable.
3. Register a canonical name in `bbo/algorithms/registry.py`.
4. Set `numeric_only` or `categorical_to_continuous` accurately.
5. Reuse task protocol initialization rather than creating a private prefix.
6. Add CLI wiring only for algorithm-specific knobs that cannot use constructor defaults.
7. Test deterministic ask/tell behavior, categorical compatibility, budgets, logging, and replay.

### Add or change an agent tool

1. Put the implementation under `bbo/algorithms/agentic/tools/`.
2. Register it through the shared tool registry; do not create a harness-only duplicate.
3. Keep tool inputs/outputs JSON-serializable and auditable.
4. Update workspace wrappers if native shell/file agents need equivalent access.
5. Verify both function-calling and workspace modes when the tool is intended for both.
6. Ensure `no_tool` variants still mean no benchmark-injected tools or skills, while native harness tools may remain available.

### Change a cross-cutting schema

Trace all producers and consumers before editing:

```text
core schema
├── Experimenter and logger/replay
├── CLI and benchmark runner
├── algorithm adapters
├── agent tool serialization/workspace API
├── summaries and plots
└── tests and task descriptions
```

Prefer backward-compatible readers for existing JSONL artifacts. Add explicit schema/version metadata when compatibility cannot be inferred.

## Validation guide

Minimum checks for documentation or isolated Python changes:

```bash
uv run python -m compileall -q bbo examples tests
uv run pytest
```

Run focused tests first while iterating:

- core/search space: `tests/test_space.py`, `tests/test_experimenter.py`, `tests/test_unit_cube_converter.py`
- task descriptions: `tests/test_description.py`
- BBOB/protocols: `tests/test_bbob_tasks.py`, `tests/test_sobol_turbo.py`, `tests/test_aligned_numerical_workflow.py`
- HPO: `tests/test_hpo_bayesmark.py`
- general agents/tools: `tests/test_general_agent.py`, `tests/test_bbo_tools.py`, `tests/test_agentic_method_contract.py`
- native harnesses: `tests/test_native_harnesses.py`, `tests/test_native_harness_workflow.py`
- CLI: `tests/test_run_cli_smoke.py`

External-service and real-LLM tests should be clearly marked and must not be required for the default offline unit suite.

## Before committing

- Confirm every README command uses a currently registered task and algorithm.
- Inspect `git status --ignored` and the staged file list.
- Check for secrets, machine-specific absolute paths, generated outputs, caches, and large binary artifacts.
- Confirm `workflow/` is absent from the commit.
- Preserve unrelated user changes.
- Record the exact validation commands and results in the handoff.
