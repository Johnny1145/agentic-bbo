# Experiment Scripts

`scripts/` only keeps two experiment entrypoint families:

- `scripts/sglang/`: local SGLang/OpenAI-compatible endpoint runs.
- `scripts/api/`: remote API/OpenAI-compatible endpoint runs.

Both trees are organized as:

```text
scripts/<backend>/<task_family>/<baseline>.sh
```

Examples:

```bash
bash scripts/sglang/synthetic/nanobot.sh --dry-run --task branin_demo --seed 1
bash scripts/sglang/exp1/nanobot.sh --tasks branin_demo --seeds 1 --dry-run
bash scripts/sglang/exp1/nanobot_no_skill.sh --tasks branin_demo --seeds 1 --max-evaluations 20

API_MODEL=<model> OPENAI_API_KEY=<key> \
  bash scripts/api/synthetic/nanobot.sh --dry-run --task branin_demo --seed 1
API_MODEL=<model> OPENAI_API_KEY=<key> \
  bash scripts/api/exp1/nanobot_no_skill.sh --tasks branin_demo --seeds 1 --max-evaluations 20
```

## Backends

SGLang defaults:

```bash
export SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:18300/v1}"
export SGLANG_MODEL="${SGLANG_MODEL:-qwen3.5-9b}"
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-EMPTY}"
```

API defaults:

```bash
export API_BASE_URL="${API_BASE_URL:-https://api.openai.com/v1}"
export API_MODEL="<model>"
export API_KEY_ENV="${API_KEY_ENV:-OPENAI_API_KEY}"
```

The API runner maps those values onto the same OpenAI-compatible arguments used
by the SGLang matrix. It skips the local `/models` health check.

## Task Families

- `synthetic/`: classical synthetic functions.
- `dbtune/`: DBTune HTTP surrogate tasks.
- `molecule/`: GuacaMol molecule tasks.
- `bboplace/`: BBOPlace tasks.
- `exp1/`: restricted-prior Nanobot workflow where the agent sees
  `restricted_task_###` and no `prior_knowledge` section.

Shared matrix/config implementation lives in:

- `scripts/sglang/matrix.py`
- `scripts/api/matrix.py`
- `scripts/sglang/configs/*.toml`

DBTune support scripts are scoped under `scripts/sglang/dbtune/` because those
surrogate tasks need local evaluator/service handling.

`exp1` outputs default to `workflow/exp1/outputs/sglang` or
`workflow/exp1/outputs/api`, including summaries, JSONL logs, agent workspaces,
and LLM logs.

## SGLang Nanobot Validation

Dedicated five-evaluation Nanobot validation entrypoints live next to each task
family. They run real agent calls, set `--no-agent-allow-fallback`, and split
the two skill modes into separate bash scripts:

```bash
export RUN_ROOT="$PWD/workflow/script_runs/sglang_nanobot_validation_$(date +%Y%m%d_%H%M%S)"
export JOBS=4

bash scripts/sglang/synthetic/nanobot_no_skill.sh
bash scripts/sglang/synthetic/nanobot_skill.sh
bash scripts/sglang/molecule/nanobot_no_skill.sh
bash scripts/sglang/molecule/nanobot_skill.sh
bash scripts/sglang/bboplace/nanobot_no_skill.sh
bash scripts/sglang/bboplace/nanobot_skill.sh

python scripts/sglang/verify_nanobot_validation.py "$RUN_ROOT"
```

Defaults are `NANOBOT_VALIDATION_EVALUATIONS=5`, `SEEDS=1`, `JOBS=4`, and
`BBO_NANOBOT_MAX_TOKENS=2048`.
The scripts cover every task listed in `scripts/sglang/configs/{synthetic,molecule,bboplace}.toml`.
For a smoke subset, pass through the underlying workflow selector, for example
`--tasks branin_demo`, `--task guacamol_median1_smiles_demo`, or
`--benchmark adaptec1`, then validate with `--allow-subset`.

## Anonymous Exp1 Nanobot No-Skill

Use the dedicated no-skill wrapper when the experiment should expose anonymous
synthetic tasks without copying the BBO skill library:

```bash
export SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:18300/v1}"
export SGLANG_MODEL="${SGLANG_MODEL:-qwen3.5-9b}"
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-EMPTY}"

RUN_ROOT="$PWD/workflow/exp1/outputs/no_skill_$(date +%Y%m%d_%H%M%S)"
bash scripts/sglang/exp1/nanobot_no_skill.sh \
  --tasks branin_demo \
  --seeds 1 \
  --initial-random 0 \
  --max-evaluations 20 \
  --results-root "$RUN_ROOT" \
  --agent-max-retries 8 \
  --agent-timeout-seconds 300 \
  --no-plots

python scripts/sglang/exp1/verify_no_skill.py "$RUN_ROOT" --expected-evaluations 20
```

For a remote OpenAI-compatible API endpoint, use the matching API wrapper:

```bash
export API_BASE_URL="${API_BASE_URL:-https://api.openai.com/v1}"
export API_MODEL="<model>"
export OPENAI_API_KEY="<key>"

RUN_ROOT="$PWD/workflow/exp1/outputs/api_no_skill_$(date +%Y%m%d_%H%M%S)"
bash scripts/api/exp1/nanobot_no_skill.sh \
  --tasks branin_demo \
  --seeds 1 \
  --initial-random 0 \
  --max-evaluations 20 \
  --results-root "$RUN_ROOT" \
  --agent-max-retries 8 \
  --agent-timeout-seconds 300 \
  --no-plots
```

The wrapper pins `--variant no-skill`, disables fallback trials, disables web
search/code interpreter, and writes restricted-prior outputs under
`restricted-prior/no-skill/restricted_task_###/nanobot/seed_*`.

Molecule validation wrappers default to the packaged
`scripts/sglang/data/guacamol_init_smiles_smoke.txt` file so a source checkout
or zip can run smoke tests without machine-local GuacaMol CSV paths. Set
`SMILES_INIT_SOURCE` or `SMILES_INIT_CSV` to use a full external initialization
pool.
