# Agentic BBO

Agentic BBO is a reproducible benchmark framework for black-box optimization agents and classical optimizers. It exposes one typed ask/tell loop across heterogeneous tasks, records append-only trial histories, and keeps algorithms, evaluators, prompts, and benchmark protocols separate.

- 73 registered tasks across COCO/BBOB, scientific and molecular optimization, DB tuning, BBOPlace, and Bayesmark HPO
- traditional, model-based, molecular, LLM-based, and native coding-agent optimizers
- shared initialization and candidate-budget protocols for comparable experiments
- JSONL logging, replay/resume, plots, and per-run agent workspaces

Languages: **English** (this file) · [中文](README.zh.md)

Start with [agent.md](agent.md) for a code-oriented project map. Repository-wide coding and validation rules live in [AGENTS.md](AGENTS.md).

## Quick start

Requirements: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Johnny1145/agentic-bbo.git
cd agentic-bbo
uv sync --extra dev

uv run python -m bbo.run \
  --task bbob_f01_d10 \
  --algorithm random_search \
  --max-evaluations 24 \
  --no-plots
```

The command evaluates the shared 20-point BBOB initial design followed by four optimizer suggestions. Results are written below `runs/demo/<task>/<algorithm>/seed_<seed>/`.

Use the CLI help to see every registered task, algorithm, and backend-specific option:

```bash
uv run python -m bbo.run --help
```

## Benchmark surface

### Task families

| Family | Count | Examples and notes |
| --- | ---: | --- |
| COCO/BBOB | 24 | `bbob_f01_d10` … `bbob_f24_d10`; 10D, instances 1–3, shared Sobol initialization |
| Scientific and molecular | 17 | tabular BO, QED/SELFIES, molecular similarity, and direct-SMILES GuacaMol tasks |
| DBTune | 6 | active HTTP sklearn-surrogate tasks named `knob_http_surrogate_*` |
| BBOPlace | 1 | `bboplace_bench`, backed by an external HTTP evaluator |
| Bayesmark HPO | 25 | five datasets × five sklearn models under `hpo_bayesmark_*` |

Task construction and discovery are centralized in `bbo/tasks/registry.py`. Each agent-facing task card lives in `bbo/task_descriptions/<task_name>/`.

### Algorithm families

| Family | Canonical entry points |
| --- | --- |
| Traditional | `random_search`, `local_perturbation`, `sobol_search`, `pycma` |
| Model-based | `optuna_tpe`, `gp_ei`, `botorch_turbo`, `git_bo`, `pfns4bo`, `pfns4bo_tabpfn_v2`, `pfns4bo_custom` |
| Molecular | `graph_ga`, `gpbo` |
| LLM-based | `llambo`, `opro`, `skydiscover_interleaved` |
| Agentic | `pablo`, `agentic_bo`, `agentic_nanobot`, `agentic_codex`, `agentic_claude_code`, `agentic_openai_compatible` |

Aliases such as `random`, `sobol`, `turbo`, `codex`, and `nanobot` are also registered. The authoritative list and compatibility metadata live in `bbo/algorithms/registry.py`.

## Architecture

```text
CLI / benchmark runner
        │
        ├── task registry ──> TaskSpec + evaluator + benchmark protocol
        │
        └── algorithm registry ──> ask/tell optimizer
                                      │
Experimenter: ask -> validate -> evaluate -> tell -> append JSONL
                                      │
                       summary, plots, replay state, agent artifacts
```

```text
.
├── agent.md                       # code-oriented project map
├── AGENTS.md                      # contribution rules for coding agents
├── bbo/
│   ├── core/                      # typed contracts, orchestration, logging, replay, plots
│   ├── algorithms/
│   │   ├── traditional/
│   │   ├── model_based/
│   │   ├── molecular/
│   │   ├── llm_based/
│   │   └── agentic/               # agent runtimes, tools, adapters, prompts, skills
│   ├── tasks/                     # evaluator implementations and task registries
│   ├── task_descriptions/         # deterministic task context shown to agents
│   ├── benchmark/                 # reusable named-benchmark runner
│   └── run.py                     # main CLI
├── examples/
├── scripts/
├── docs/
├── tests/
└── pyproject.toml
```

The local `workflow/` tree is intentionally excluded from version control. It contains machine-specific experiment orchestration and outputs, not reusable package code.

## Installation profiles

The base package includes NumPy/SciPy, plotting, pycma, and the pinned COCO runtime. Add only the extras required by an experiment:

| Extra | Purpose |
| --- | --- |
| `dev` | pytest and development checks |
| `task-host` | common scientific, molecular, Optuna, and HTTP-client dependencies |
| `hpo` | BoTorch/GPyTorch/sklearn stack for Bayesmark HPO |
| `optuna` | Optuna TPE only |
| `molecular` | graph/molecular BO dependencies |
| `pfns4bo` / `tabpfn` | PFN-based surrogate variants |
| `pablo` | OpenAI client for Pablo |
| `nanobot` / `general-agent` | native agent harness dependencies |
| `skydiscover` | online SkyDiscover meta-evolution |
| `interop` | ConfigSpace interoperability |

Examples:

```bash
uv sync --extra dev --extra task-host
uv sync --extra dev --extra hpo
uv sync --extra dev --extra general-agent
```

Some evaluators remain external by design. BBOPlace and active DBTune services require their documented HTTP services; LLM-backed algorithms require provider credentials; native Codex and Claude Code runs require their corresponding harnesses.

## Running experiments

Run a classical baseline:

```bash
uv run python -m bbo.run \
  --task bbob_f03_d10 \
  --algorithm pycma \
  --max-evaluations 120 \
  --seed 0
```

Run a Bayesmark HPO task with aligned initialization:

```bash
uv sync --extra dev --extra hpo
uv run python -m bbo.run \
  --task hpo_bayesmark_breast_svm \
  --algorithm botorch_turbo \
  --max-evaluations 30 \
  --seed 0
```

Run an offline LLM-style smoke test without credentials:

```bash
uv run python -m bbo.run \
  --task bbob_f01_d10 \
  --algorithm llambo \
  --llambo-backend heuristic \
  --max-evaluations 24 \
  --no-plots
```

For a minimal editable Python runner, see `examples/run_one_benchmark.py`. Batch entry points and SGLang/API wrappers are documented in `scripts/README.md`.

### Agentic optimizers

The general-agent runtime can expose benchmark tools through function calling or a generated workspace API. Native harness entry points are:

- `agentic_nanobot` / `nanobot`
- `agentic_codex` / `codex`
- `agentic_claude_code` / `claude_code`
- `agentic_openai_compatible`

Each run can record the agent workspace, state, memory, tool calls, LLM logs, reasoning metadata, and optimization trace. Benchmark-injected tools cover task context, search-space inspection, trial history, incumbents, candidate validation, sampling, registered optimizer suggestions, code execution, and optional web research.

Important: harness configuration isolation improves reproducibility but is not a security boundary. Run native coding agents in a dedicated container, VM, or restricted OS account.

Architecture and operational notes:

- `docs/agentic_bbo_unification.md`
- `docs/baseline_execution_logic.md`
- `docs/nanobot_benchmark_v2.md`
- `docs/sandboxfusion_bbo.md`

## Outputs and reproducibility

A single run normally contains:

```text
runs/demo/<task>/<algorithm>/seed_<seed>/
├── trials.jsonl          # append-only evaluations
├── summary.json          # final metrics and artifact paths
└── plots/                # optional trace, timing, regret, and comparison figures
```

Agentic and SkyDiscover algorithms add their own workspace or generated-strategy artifacts under the same run directory. `--resume` replays the JSONL history before continuing. Existing run directories are never silently overwritten; without `--resume`, the runner allocates a numbered sibling.

Task-owned protocol metadata defines comparable initialization and candidate budgets. In particular:

- COCO/BBOB uses 24 official 10D functions, instances selected deterministically from seeds, a shared 20-point scrambled-Sobol initialization, and 100 optimizer evaluations.
- Bayesmark HPO uses five shared LLAMBO initialization points followed by 25 optimizer evaluations.

See `docs/hpo_bayesmark.md` and `docs/baseline_execution_logic.md` for protocol details.

## Extending the benchmark

To add a task:

1. Implement or extend a family under `bbo/tasks/`.
2. Define a typed `SearchSpace`, `TaskSpec`, and normalized evaluator result.
3. Add `bbo/task_descriptions/<task_name>/` with at least `background.md`, `goal.md`, `constraints.md`, and `prior_knowledge.md`.
4. Register the task in the family registry and `bbo/tasks/registry.py`.
5. Add behavior-focused tests.

To add an algorithm:

1. Implement the `Algorithm` ask/tell contract outside `bbo/core/`.
2. Register one canonical name and any aliases in `bbo/algorithms/registry.py`.
3. Declare numeric/categorical compatibility in `AlgorithmSpec`.
4. Wire specialized CLI kwargs in `bbo/run.py` only when needed.
5. Test deterministic suggestions, replay behavior, and budget handling.

Read `agent.md` before changing cross-cutting contracts.

## Validation

```bash
uv run python -m compileall -q bbo examples tests
uv run pytest
```

Useful focused checks:

```bash
uv run pytest tests/test_run_cli_smoke.py
uv run pytest tests/test_bbob_tasks.py tests/test_hpo_bayesmark.py
uv run pytest tests/test_general_agent.py tests/test_native_harnesses.py
```

Keep API keys in environment variables. `.env`, `.apikey`, run outputs, downloaded surrogate models, virtual environments, and the local `workflow/` tree are ignored by Git.
