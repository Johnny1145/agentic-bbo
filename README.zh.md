# Agentic BBO

Agentic BBO 是一个面向黑盒优化 Agent 与经典优化器的可复现 Benchmark 框架。项目用同一套强类型 ask/tell 循环连接异构任务，采用 append-only trial 历史，并把算法、评估器、提示词和 Benchmark 协议彼此分离。

- 73 个已注册任务，覆盖 COCO/BBOB、科学与分子优化、数据库调优、BBOPlace 和 Bayesmark HPO
- 传统、模型驱动、分子、LLM 驱动和原生 coding-agent 五类优化器
- 共享初始化与候选预算协议，便于进行公平对比
- JSONL 日志、replay/resume、可视化和逐次运行的 Agent workspace

语言版本：[English](README.md) · **中文**（本文件）

如果要阅读代码或让 Agent 修改项目，请先看 [agent.md](agent.md)；仓库级编码与验证规则位于 [AGENTS.md](AGENTS.md)。

## 快速开始

要求：Python 3.11+ 与 [uv](https://docs.astral.sh/uv/)。

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

该命令先评估 BBOB 共享的 20 点初始设计，再执行 4 次优化器建议。结果写入 `runs/demo/<task>/<algorithm>/seed_<seed>/`。

查看全部已注册任务、算法和 backend 参数：

```bash
uv run python -m bbo.run --help
```

## Benchmark 范围

### 任务族

| 任务族 | 数量 | 示例与说明 |
| --- | ---: | --- |
| COCO/BBOB | 24 | `bbob_f01_d10` … `bbob_f24_d10`；10D、实例 1–3、共享 Sobol 初始化 |
| 科学与分子优化 | 17 | 表格 BO、QED/SELFIES、分子相似度和 direct-SMILES GuacaMol |
| DBTune | 6 | 当前有效的 `knob_http_surrogate_*` HTTP sklearn-surrogate 任务 |
| BBOPlace | 1 | `bboplace_bench`，依赖外部 HTTP evaluator |
| Bayesmark HPO | 25 | 5 个数据集 × 5 个 sklearn 模型，名称为 `hpo_bayesmark_*` |

任务发现与构造集中在 `bbo/tasks/registry.py`。Agent 可见的任务卡位于 `bbo/task_descriptions/<task_name>/`。

### 算法族

| 算法族 | 主要入口 |
| --- | --- |
| 传统 | `random_search`、`local_perturbation`、`sobol_search`、`pycma` |
| 模型驱动 | `optuna_tpe`、`gp_ei`、`botorch_turbo`、`git_bo`、`pfns4bo` 及其变体 |
| 分子优化 | `graph_ga`、`gpbo` |
| LLM 驱动 | `llambo`、`opro`、`skydiscover_interleaved` |
| Agentic | `pablo`、`agentic_bo`、`agentic_nanobot`、`agentic_codex`、`agentic_claude_code`、`agentic_openai_compatible` |

`random`、`sobol`、`turbo`、`codex`、`nanobot` 等别名也已注册。权威清单位于 `bbo/algorithms/registry.py`。

## 架构

```text
CLI / benchmark runner
        │
        ├── task registry ──> TaskSpec + evaluator + benchmark protocol
        │
        └── algorithm registry ──> ask/tell optimizer
                                      │
Experimenter: ask -> validate -> evaluate -> tell -> append JSONL
                                      │
                       summary、plots、replay state、agent artifacts
```

```text
.
├── agent.md                       # 面向代码阅读的项目地图
├── AGENTS.md                      # coding agent 的仓库规则
├── bbo/
│   ├── core/                      # 类型契约、编排、日志、replay、绘图
│   ├── algorithms/
│   │   ├── traditional/
│   │   ├── model_based/
│   │   ├── molecular/
│   │   ├── llm_based/
│   │   └── agentic/               # Agent runtime、工具、适配器、提示词、skills
│   ├── tasks/                     # evaluator 实现与任务注册表
│   ├── task_descriptions/         # 确定性的 Agent 任务上下文
│   ├── benchmark/                 # 可复用 named-benchmark runner
│   └── run.py                     # 主 CLI
├── examples/
├── scripts/
├── docs/
├── tests/
└── pyproject.toml
```

本机的 `workflow/` 目录不会进入版本控制；其中保存机器相关的实验编排与输出，不属于可复用包代码。

## 安装组合

基础包包含 NumPy/SciPy、绘图、pycma 和固定版本的 COCO runtime。根据实验按需添加 extra：

| Extra | 用途 |
| --- | --- |
| `dev` | pytest 与开发检查 |
| `task-host` | 常用科学/分子任务、Optuna 与 HTTP client 依赖 |
| `hpo` | Bayesmark HPO 的 BoTorch/GPyTorch/sklearn 栈 |
| `optuna` | 仅安装 Optuna TPE |
| `molecular` | 图与分子 BO 依赖 |
| `pfns4bo` / `tabpfn` | PFN surrogate 变体 |
| `pablo` | Pablo 所需 OpenAI client |
| `nanobot` / `general-agent` | 原生 Agent harness 依赖 |
| `skydiscover` | 在线 SkyDiscover 元进化 |
| `interop` | ConfigSpace 互操作 |

```bash
uv sync --extra dev --extra task-host
uv sync --extra dev --extra hpo
uv sync --extra dev --extra general-agent
```

BBOPlace 和当前 DBTune 服务需要相应 HTTP evaluator；LLM 算法需要 provider 凭据；原生 Codex / Claude Code 运行需要对应 harness。

## 运行实验

经典 baseline：

```bash
uv run python -m bbo.run \
  --task bbob_f03_d10 \
  --algorithm pycma \
  --max-evaluations 120 \
  --seed 0
```

具有对齐初始化的 Bayesmark HPO：

```bash
uv sync --extra dev --extra hpo
uv run python -m bbo.run \
  --task hpo_bayesmark_breast_svm \
  --algorithm botorch_turbo \
  --max-evaluations 30 \
  --seed 0
```

无需凭据的 LLM 风格离线 smoke：

```bash
uv run python -m bbo.run \
  --task bbob_f01_d10 \
  --algorithm llambo \
  --llambo-backend heuristic \
  --max-evaluations 24 \
  --no-plots
```

可编辑的最小 Python 入口见 `examples/run_one_benchmark.py`；批量入口与 SGLang/API wrapper 见 `scripts/README.md`。

### Agentic 优化器

General-agent runtime 可通过 function calling 或生成的 workspace API 暴露 Benchmark 工具。原生 harness 入口包括：

- `agentic_nanobot` / `nanobot`
- `agentic_codex` / `codex`
- `agentic_claude_code` / `claude_code`
- `agentic_openai_compatible`

每个 run 可以记录 Agent workspace、state、memory、tool calls、LLM 日志、reasoning metadata 与优化轨迹。Benchmark 工具覆盖任务上下文、搜索空间、trial 历史、incumbent、候选验证、采样、注册优化器建议、代码执行和可选 Web 研究。

注意：harness 配置隔离用于提升可复现性，并不是安全边界。原生 coding agent 应在专用容器、虚拟机或受限系统账号下运行。

相关文档：

- `docs/agentic_bbo_unification.md`
- `docs/baseline_execution_logic.md`
- `docs/nanobot_benchmark_v2.md`
- `docs/sandboxfusion_bbo.md`

## 输出与可复现性

```text
runs/demo/<task>/<algorithm>/seed_<seed>/
├── trials.jsonl          # append-only evaluations
├── summary.json          # 最终指标与 artifact 路径
└── plots/                # 可选的轨迹、耗时、regret 与对比图
```

`--resume` 会先 replay JSONL 历史再继续运行。未指定 `--resume` 时，已有目录不会被静默覆盖，runner 会创建带编号的相邻目录。

任务自身的 protocol metadata 决定公平对比所用的初始化与候选预算：

- COCO/BBOB 使用 24 个官方 10D 函数、由 seed 确定的实例、共享 20 点 scrambled-Sobol 初始化和 100 次优化器评估。
- Bayesmark HPO 使用 5 个共享 LLAMBO 初始化点，再执行 25 次优化器评估。

协议细节见 `docs/hpo_bayesmark.md` 与 `docs/baseline_execution_logic.md`。

## 扩展 Benchmark

新增任务：

1. 在 `bbo/tasks/` 下新增或扩展任务族。
2. 定义强类型 `SearchSpace`、`TaskSpec` 和规范化 evaluator 结果。
3. 新增 `bbo/task_descriptions/<task_name>/`，至少包含 `background.md`、`goal.md`、`constraints.md`、`prior_knowledge.md`。
4. 在任务族注册表和 `bbo/tasks/registry.py` 中注册。
5. 添加面向行为的测试。

新增算法：

1. 在 `bbo/core/` 之外实现 `Algorithm` ask/tell 契约。
2. 在 `bbo/algorithms/registry.py` 注册规范名称和别名。
3. 在 `AlgorithmSpec` 声明数值/类别参数兼容性。
4. 只有确有需要时才在 `bbo/run.py` 接入专属 CLI 参数。
5. 测试确定性建议、replay 行为和预算处理。

修改跨模块契约前请先阅读 `agent.md`。

## 验证

```bash
uv run python -m compileall -q bbo examples tests
uv run pytest
```

常用的聚焦检查：

```bash
uv run pytest tests/test_run_cli_smoke.py
uv run pytest tests/test_bbob_tasks.py tests/test_hpo_bayesmark.py
uv run pytest tests/test_general_agent.py tests/test_native_harnesses.py
```

API key 必须保存在环境变量中。`.env`、`.apikey`、运行输出、下载的 surrogate 模型、虚拟环境和本机 `workflow/` 均由 Git 忽略。
