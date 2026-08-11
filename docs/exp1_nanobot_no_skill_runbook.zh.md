# Exp1 Nanobot No-Skill 运行说明

本文档说明如何用 `scripts/` 跑 benchmark v2 的 exp1 Nanobot no-skill 版本，并如何做三类任务的 smoke 验证。

## 前置条件

需要在仓库根目录运行命令：

```bash
cd /path/to/agentic-bbo-nanobot-benchmark-v2
```

需要安装 `uv`，并有可用的本地 SGLang/OpenAI-compatible endpoint：

```bash
export SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:18300/v1}"
export SGLANG_MODEL="${SGLANG_MODEL:-qwen3.5-9b}"
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-EMPTY}"
```

检查模型服务：

```bash
curl "$SGLANG_BASE_URL/models"
```

BBOPlace 还需要 evaluator 服务。默认脚本会探测这些 `/health` endpoint：

```text
http://127.0.0.1:8270
http://127.0.0.1:8280
http://127.0.0.1:8281
http://127.0.0.1:8282
http://127.0.0.1:8283
http://127.0.0.1:8284
http://127.0.0.1:8285
http://127.0.0.1:8286
http://127.0.0.1:8287
```

如果服务在别的地址，设置：

```bash
export BBOPLACE_BASE_URLS="http://host1:port,http://host2:port"
```

## 跑 exp1 匿名 synthetic no-skill

使用 SGLang 专用 wrapper：

```bash
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
```

这个 wrapper 固定使用：

- `--variant no-skill`
- `--agent-tool-mode workspace_json`
- `--no-agent-enable-code-interpreter`
- `--agent-web-search-provider disabled`
- `--no-agent-allow-fallback`

输出结构：

```text
$RUN_ROOT/
  benchmark_summary.json
  task_alias_map.json
  restricted-prior/no-skill/restricted_task_001/nanobot/seed_1/
    trials.jsonl
    summary.json
    agent_calls.jsonl
    agent_tool_calls.jsonl
    agent_workspace/
```

检查 exp1 no-skill 结果：

```bash
python scripts/sglang/exp1/verify_no_skill.py "$RUN_ROOT" --expected-evaluations 20
```

如果使用远程 OpenAI-compatible API，用 API wrapper：

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

verifier 会检查：

- `benchmark_summary.json` 没有 failures
- 每个结果是 `skill_mode=no-skill`
- 每个结果是 `exposure_policy=restricted-prior`
- trial 数、成功数等于 expected evaluations
- agent workspace 中没有 `skills/*/SKILL.md` 文件
- agent-visible 文件中没有常见匿名泄漏字符串

## 三类任务 smoke 验证

下面的命令各跑一个任务、一个 seed、一轮 evaluation，用于确认脚本链路能启动并完成。

```bash
RUN_ROOT="$PWD/workflow/script_runs/sglang_nanobot_no_skill_smoke_$(date +%Y%m%d_%H%M%S)"
export RUN_ROOT
export NANOBOT_VALIDATION_EVALUATIONS=1
export SEEDS=1
export JOBS=1
export AGENT_MAX_RETRIES=8
export AGENT_TIMEOUT_SECONDS=300
export AGENT_MAX_TOOL_CALLS=64
```

匿名 synthetic：

```bash
bash scripts/sglang/exp1/nanobot_no_skill.sh \
  --tasks branin_demo \
  --seeds 1 \
  --initial-random 0 \
  --max-evaluations 1 \
  --results-root "$RUN_ROOT/exp1" \
  --no-plots

python scripts/sglang/exp1/verify_no_skill.py "$RUN_ROOT/exp1" --expected-evaluations 1
```

普通 synthetic no-skill：

```bash
bash scripts/sglang/synthetic/nanobot_no_skill.sh --tasks branin_demo
python scripts/sglang/verify_nanobot_validation.py "$RUN_ROOT" \
  --family synthetic \
  --mode no-skill \
  --expected-evaluations 1 \
  --allow-subset
```

molecule no-skill：

```bash
bash scripts/sglang/molecule/nanobot_no_skill.sh --task guacamol_median1_smiles_demo
python scripts/sglang/verify_nanobot_validation.py "$RUN_ROOT" \
  --family molecule \
  --mode no-skill \
  --expected-evaluations 1 \
  --allow-subset
```

`scripts/sglang/molecule/nanobot_no_skill.sh` 默认使用 repo 内置的：

```text
scripts/sglang/data/guacamol_init_smiles_smoke.txt
```

如果要使用完整 GuacaMol 初始池：

```bash
export SMILES_INIT_SOURCE=/path/to/guacamol_init_data.csv
```

BBOPlace no-skill：

```bash
bash scripts/sglang/bboplace/nanobot_no_skill.sh --benchmark adaptec1
python scripts/sglang/verify_nanobot_validation.py "$RUN_ROOT" \
  --family bboplace \
  --mode no-skill \
  --expected-evaluations 1 \
  --allow-subset
```

## 打包 zip

建议排除虚拟环境、缓存和历史输出：

```bash
cd /home/trx/cm
zip -r agentic-bbo-nanobot-benchmark-v2.zip agentic-bbo-nanobot-benchmark-v2 \
  -x 'agentic-bbo-nanobot-benchmark-v2/.venv/*' \
  -x 'agentic-bbo-nanobot-benchmark-v2/.pytest_cache/*' \
  -x 'agentic-bbo-nanobot-benchmark-v2/**/__pycache__/*' \
  -x 'agentic-bbo-nanobot-benchmark-v2/workflow/exp1/outputs/*' \
  -x 'agentic-bbo-nanobot-benchmark-v2/workflow/script_runs/*'
```
