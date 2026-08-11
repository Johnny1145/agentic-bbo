# Agentic / Nanobot 运行流程小白版

这份文档解释当前仓库里 `agentic` 算法是怎么跑起来的，尤其是 `run.sh` 里接入 Nanobot 的那条路径。

先澄清一个名字：代码里实际叫 `nanobot`，没有看到 `nanobox` 这个实现。下面都按当前代码中的 `nanobot` 来说明。

## 一句话版本

这个系统把黑盒优化拆成两个角色：

- `Experimenter`：像裁判一样控制实验循环，负责“问算法要一个配置 -> 调 task 评估 -> 把结果告诉算法 -> 写日志”。
- `Nanobot agent`：像参谋一样看任务说明、历史结果和工具输出，然后提出下一批候选配置。

agent 自己不会直接消耗 benchmark 的评估预算。真正消耗预算的是 `Experimenter` 拿到候选配置后调用 `task.evaluate()` 的那一步。

## 当前 run.sh 跑的是什么

当前 `run.sh` 的核心命令是：

```bash
uv run --extra nanobot python -m bbo.run \
    --task branin_demo \
    --algorithm nanobot \
    --max-evaluations 20 \
    --agent-initial-random 2 \
    --agent-provider openai \
    --agent-model deepseek-reasoner \
    --agent-api-base "http://35.220.164.252:3888/v1/" \
    --agent-api-key-env NANOBOT_API_KEY \
    --agent-tool-mode workspace_json \
    --agent-code-backend sandboxfusion \
    --sandbox-fusion-base-url "$SANDBOX_FUSION_BASE_URL" \
    --agent-require-visible-cot \
    --no-agent-allow-fallback \
    --no-plots
```

对应含义：

- 任务是 `branin_demo`：一个二维连续优化 demo，参数是 `x1` 和 `x2`，目标是最小化 `loss`。
- 算法是 `nanobot`：注册表里它是 `agentic_nanobot` 的别名。
- 总预算是 20 次评估。
- 前 2 次用随机点，不问 LLM。
- 第 3 次开始调用 Nanobot，让它看历史后提出候选点。
- Nanobot 通过 OpenAI-compatible endpoint 连到 `deepseek-reasoner`。
- 工具模式是 `workspace_json`：不给模型原生 function calling，而是在运行目录里生成一套文件和 Python API，让 Nanobot 通过读写文件、运行脚本来用工具。
- `code_interpreter` 后端是 SandboxFusion，会请求 `${SANDBOX_FUSION_BASE_URL}/run_code` 跑分析代码。
- 要求捕获 visible reasoning；如果没有 reasoning trace，本次 agent 调用会失败。
- 禁用 fallback；如果 agent 没返回合法候选点，不会自动随机兜底。

## 关键代码地图

| 代码位置 | 作用 |
|---|---|
| `run.sh` | 当前 Nanobot 跑法的 shell 入口，设置 API key、SandboxFusion、web search 等参数。 |
| `bbo/run.py` | CLI 总入口；解析 `--algorithm nanobot`、`--agent-*` 参数，创建 task 和 algorithm。 |
| `bbo/algorithms/registry.py` | 算法注册表；把 `nanobot` 映射到 `NanobotBBOAlgorithm`。 |
| `bbo/core/experimenter.py` | 通用实验循环：`setup -> ask -> evaluate -> tell -> log`。 |
| `bbo/algorithms/agentic/general_agent.py` | agentic 算法主体；维护历史、工作区、prompt、候选队列、日志。 |
| `bbo/algorithms/agentic/general_agent_engines.py` | 不同 agent 后端的适配器；Nanobot、Claude Code、OpenAI-compatible 都在这里。 |
| `bbo/algorithms/agentic/nanobot_runner.py` | Nanobot CLI wrapper；给 Nanobot 打补丁并记录 LLM 日志/reasoning。 |
| `bbo/algorithms/agentic/workspace_python_api.py` | 复制到 agent 工作区里的 `bbo_tools.py`，是 Nanobot 推荐使用的 Python API。 |
| `bbo/algorithms/agentic/workspace_tool_cli.py` | 复制到 agent 工作区里的 `bbo_tool.py`，真正执行 workspace 工具。 |
| `bbo/algorithms/agentic/tools/*.py` | function-calling 模式用的工具实现；和 workspace 工具语义基本一致。 |
| `bbo/tasks/...` | 具体 benchmark task 的实现；例如 `branin_demo` 的目标函数在 `bbo/tasks/synthetic/branin.py`。 |

## 整体流程图

```text
run.sh
  |
  v
python -m bbo.run
  |
  |-- create_task("branin_demo")
  |-- create_algorithm("nanobot")
  v
Experimenter.run()
  |
  |-- algorithm.setup(task_spec, task_description)
  |      |
  |      |-- 创建 run_dir/agent_workspace
  |      |-- 写 task.md / space.json / history.jsonl / bbo_tools.py ...
  |      |-- 写 Nanobot config.json
  |
  |-- 循环直到 max_evaluations
         |
         |-- algorithm.ask()
         |     |
         |     |-- 前 2 次：随机采样
         |     |-- 后续：调用 Nanobot
         |            |
         |            |-- Nanobot 读工作区文件
         |            |-- Nanobot 可运行 Python 脚本，调用 BBO() 工具
         |            |-- Nanobot 最终只输出 {"candidates": [...]}
         |            |-- general_agent.py 校验 JSON 和搜索空间
         |
         |-- task.evaluate(candidate)
         |-- algorithm.tell(observation)
         |     |
         |     |-- 更新 history / incumbent / 工作区文件
         |
         |-- logger.log(observation)
```

## 一次 agent 调用里发生了什么

### 1. CLI 把参数传进算法

`bbo/run.py` 会识别 `agentic_nanobot`、`nanobot`、`agentic_claude_code`、`agentic_openai_compatible` 这些算法名，然后把所有 `--agent-*` 参数打包成 `algorithm_kwargs`。

这些参数包括模型、provider、API base、API key 环境变量、初始随机点数量、工具模式、代码执行后端、web search provider、fallback 策略、是否要求 visible CoT 等。

### 2. registry 找到 Nanobot 算法类

`bbo/algorithms/registry.py` 里：

- `agentic_nanobot` 的 factory 是 `NanobotBBOAlgorithm`。
- `nanobot` 是它的别名。

`NanobotBBOAlgorithm` 本身很薄，只是把通用 `GeneralAgentBBOAlgorithm` 的 `framework` 固定成 `nanobot`。

### 3. Experimenter 启动通用 ask/tell 循环

`Experimenter.run()` 的主逻辑是：

1. 检查 task 是否健康。
2. 读取 task 的 markdown 描述。
3. 调 `algorithm.setup(...)`。
4. 循环：
   - `algorithm.ask()`：要一个候选配置。
   - 校验候选配置是否符合搜索空间。
   - `task.evaluate(...)`：真正跑 benchmark。
   - `algorithm.tell(observation)`：把结果告诉算法。
   - 写 `trials.jsonl`。

所以无论算法是随机搜索、CMA-ES、OPRO 还是 Nanobot，外层实验循环都是同一个。

### 4. setup 创建 agent 工作区

`GeneralAgentBBOAlgorithm.setup()` 会在本次 run 目录下创建这些核心目录和文件：

```text
run_dir/
  agent_workspace/
    task.md
    manifest.json
    space.json
    objective.json
    incumbent.json
    history.jsonl
    tool_specs.json
    instructions.md
    python_environment.md
    bbo_tools.py
    bbo_tool.py
    bbo_tool_config.json
    gp_expected_improvement.py
    examples/gp_expected_improvement.py
  agent_state/
    config.json
  agent_memory/
    memory.jsonl
    memory_summary.json
  agent_calls.jsonl
  agent_prompts.jsonl
  agent_tool_calls.jsonl
  agent_web_sources.jsonl
  agent_optimization_trace.jsonl
  llm_logs/
  reasoning_traces/
  agent_reasoning_metadata.jsonl
```

对 Nanobot 来说，最重要的是 `agent_workspace/`：

- `task.md`：任务背景和要求。
- `space.json`：参数名、类型、上下界、类别选择。
- `history.jsonl`：最近已经评估过的 trial。
- `incumbent.json`：当前最好配置。
- `instructions.md`：告诉 agent 最终必须输出什么 JSON。
- `bbo_tools.py`：推荐的工具 API。
- `bbo_tool.py`：底层工具桥。
- `bbo_tool_config.json`：工具桥需要的路径、SandboxFusion URL、web search key 等配置。

### 5. 前几次可以不用 LLM，直接随机

`--agent-initial-random 2` 表示前 2 个 trial 直接从搜索空间随机采样。

这样做的原因很简单：黑盒优化一开始没有历史数据，先拿几个点让 agent 后面能看到“哪些配置好、哪些配置差”。

### 6. ask() 需要 LLM 时，会构造 prompt

随机阶段结束后，如果候选队列为空，`ask()` 会进入 `_fill_queue_from_agent()`：

1. 先刷新工作区文件，保证 `history.jsonl`、`incumbent.json` 是最新的。
2. 生成 `call_id`，例如 `agent_call_00000`。
3. 生成 prompt，要求 Nanobot 读取工作区文件、必要时运行脚本、最后只输出严格 JSON。
4. 把 prompt 记到 `agent_prompts.jsonl`。
5. 调 `_run_engine()` 启动 Nanobot。

最终 agent 必须输出类似：

```json
{
  "candidates": [
    {
      "config": {
        "x1": 3.14,
        "x2": 2.27
      },
      "rationale": "near a known promising Branin basin"
    }
  ]
}
```

注意：最终 stdout 只能是 JSON。解释、markdown、代码块都会影响解析。

### 7. Nanobot 是作为子进程启动的

`NanobotEngine.run_agent()` 会启动一个子进程：

```bash
python -m bbo.algorithms.agentic.nanobot_runner agent -m "<prompt>" --no-markdown -w <agent_workspace> -c <agent_state/config.json>
```

它还会把环境变量传进去：

- 模型 API key，比如 `OPENAI_API_KEY`。
- API base，比如 `OPENAI_BASE_URL`。
- web search key，比如 `SERPAPI_API_KEY`。
- `BBO_AGENT_CALL_ID`。
- `BBO_NANOBOT_REASONING_DIR`。
- `BBO_NANOBOT_REASONING_METADATA_PATH`。

`nanobot_runner.py` 再导入 Nanobot CLI 的 `app()` 来真正运行 agent。

### 8. nanobot_runner 做了两件额外的事

`nanobot_runner.py` 是一个 wrapper，不是优化算法本体。它主要做：

1. 如果设置了 `BBO_NANOBOT_NO_MAX_TOKENS=1`，就从 OpenAI-compatible 请求参数里去掉 `max_tokens`，兼容不接受这个参数的 endpoint。
2. 如果设置了 `BBO_NANOBOT_LOG_DIR`，就记录 Nanobot 的最终消息、usage、reasoning trace。

这就是为什么当前 `run.sh` 可以配 `deepseek-reasoner` 并要求 visible CoT。

### 9. Nanobot 怎么“用工具”

当前 `run.sh` 用的是：

```bash
--agent-tool-mode workspace_json
```

这表示 Nanobot 不走原生 function calling。它会在工作区里写 Python 脚本，然后通过：

```python
from bbo_tools import BBO

bbo = BBO()
space = bbo.search_space()
history = bbo.history()
incumbent = bbo.incumbent()
samples = bbo.sample(n=4)
check = bbo.validate([{"config": samples["candidates"][0]["config"]}])
```

这些调用最终会进入 `bbo_tool.py`，也就是 `workspace_tool_cli.py` 复制出来的文件。

可用工具包括：

| 工具/API | 做什么 |
|---|---|
| `task_context()` | 读取任务描述、manifest、约束。 |
| `manifest()` | 读取 agent-facing manifest。 |
| `search_space()` | 读取参数 schema。 |
| `objective()` | 读取目标名和优化方向。 |
| `history()` | 读取已经评估过的 trial。 |
| `incumbent()` | 读取当前最好配置。 |
| `sample()` | 随机采样，或围绕当前最好配置采样；不消耗评估预算。 |
| `validate()` | 校验候选点是否合法、是否重复；不消耗评估预算。 |
| `analyze_history()` | 做简单历史统计，比如 best trial、均值、相关性。 |
| `memory_read()` / `memory_write()` | 读写 agent 的长期记忆文件。 |
| `code_interpreter()` | 通过 SandboxFusion 跑分析代码；不消耗评估预算。 |
| `web_search()` | 通过 SerpAPI 搜索公开资料，并写 source 日志。 |
| `fetch_url()` | 拉取 manifest 允许的 URL，并写 source 日志。 |

### 10. SandboxFusion 在哪里用

SandboxFusion 只服务于 `code_interpreter()`。

当前路径里，`bbo_tool.py` 看到：

- `code_backend == "sandboxfusion"`
- `sandbox_fusion_base_url` 存在

就会 POST 到：

```text
<SANDBOX_FUSION_BASE_URL>/run_code
```

请求体大致是：

```json
{
  "code": "...",
  "language": "python"
}
```

它的用途是让 agent 跑离线分析代码，比如拟合一个简单 surrogate、分析历史、生成候选点。它不应该调用真实 task evaluator，也不会直接增加 trial 数。

### 11. 候选 JSON 会被严格校验

Nanobot 返回 stdout 后，`general_agent.py` 会：

1. 解析 JSON。
2. 要求顶层只有一个 key：`candidates`。
3. 要求 `candidates` 是非空 list。
4. 对每个 candidate，用 task 的 search space 做 `coerce_config`：
   - 参数必须完整。
   - 浮点/整数必须在范围内。
   - categorical 必须是合法选项。
5. 去掉本次响应内重复的 candidate。
6. 再去掉历史里已经见过的配置。
7. 合法 candidate 会进入内部队列 `_queue`。

如果 agent 输出非法：

- 默认 `allow_fallback=True` 时，会尝试随机兜底。
- 当前 `run.sh` 用了 `--no-agent-allow-fallback`，所以会直接失败。
- 当前还用了 `--agent-require-visible-cot`，如果没有捕获 reasoning，也会失败。

### 12. 真正评估发生在 task.evaluate()

当 `ask()` 返回一个合法 `TrialSuggestion` 后，`Experimenter` 才会调用：

```python
task.evaluate(suggestion)
```

以 `branin_demo` 为例：

- `x1` 范围是 `[-5.0, 10.0]`。
- `x2` 范围是 `[0.0, 15.0]`。
- 目标函数是 Branin-Hoo。
- 返回的 primary objective 叫 `loss`，方向是 minimize。

评估结果会被包装成 `TrialObservation`，里面有：

- config
- status
- objectives
- metrics
- elapsed_seconds
- metadata

然后：

1. `algorithm.tell(observation)` 更新 agent 内部历史和当前最优。
2. 工作区里的 `history.jsonl`、`incumbent.json` 被刷新。
3. `trials.jsonl` 记录这次 trial。
4. 下一轮 `ask()` 时，Nanobot 就能看到新的历史。

## 两种 agent 工具模式的区别

仓库里现在支持两种通用 agent 运行模式：

| 模式 | 适合谁 | 工具怎么调用 |
|---|---|---|
| `workspace_json` | Nanobot、Claude Code 这类能读写文件/跑命令的 agent | 通过工作区文件和 `from bbo_tools import BBO` 调工具。 |
| `function_calling` | OpenAI-compatible chat completions | 把工具 schema 传给模型，模型原生发 tool call，Python 端执行工具。 |

当前 `run.sh` 是 `workspace_json`。

`function_calling` 相关工具实现主要在 `bbo/algorithms/agentic/tools/`；`workspace_json` 相关工具实现主要在 `workspace_python_api.py` 和 `workspace_tool_cli.py`。

## 运行产物怎么看

一次 run 的目录大致是：

```text
runs/nanobot_deepseek_sandboxfusion_full/
  branin_demo/
    nanobot/
      seed_7/
        run_000/
          trials.jsonl
          summary.json
          agent_workspace/
          agent_state/
          agent_memory/
          agent_prompts.jsonl
          agent_calls.jsonl
          agent_tool_calls.jsonl
          agent_web_sources.jsonl
          agent_optimization_trace.jsonl
          llm_logs/
          reasoning_traces/
          agent_reasoning_metadata.jsonl
```

最常看的文件：

- `trials.jsonl`：每次真实 benchmark 评估的结果。
- `summary.json`：本次 run 总结、best objective、artifact 路径。
- `agent_prompts.jsonl`：每次发给 agent 的 prompt。
- `agent_calls.jsonl`：agent 返回了什么、是否解析成功、接收了几个候选点。
- `agent_tool_calls.jsonl`：agent 调过哪些 BBO 工具。
- `agent_workspace/history.jsonl`：agent 当前能看到的历史窗口。
- `reasoning_traces/` 和 `agent_reasoning_metadata.jsonl`：visible reasoning 记录。
- `llm_logs/`：Nanobot wrapper 记录的 LLM 消息快照。

## 小白理解版例子

把它想成“找最低分配方”的流程：

1. 系统告诉 agent：“你要调两个旋钮 `x1`、`x2`，目标是让 `loss` 越小越好。”
2. 前两次系统自己随机试两个配方。
3. 系统把这两次结果写进 `history.jsonl`。
4. Nanobot 读历史，可能再用 `BBO().analyze_history()` 或 SandboxFusion 跑个小脚本。
5. Nanobot 输出几个新配方。
6. Python 主程序检查这些配方有没有超范围。
7. 主程序拿第一个合法配方去跑真实目标函数。
8. 结果写回历史。
9. 重复，直到 20 次评估用完。

## 当前 Nanobot 路径最重要的调用链

```text
run.sh
-> bbo/run.py:main()
-> run_single_experiment()
-> create_task("branin_demo")
-> create_algorithm("nanobot")
-> NanobotBBOAlgorithm(...)
-> Experimenter.run()
-> GeneralAgentBBOAlgorithm.setup()
-> GeneralAgentBBOAlgorithm.ask()
-> GeneralAgentBBOAlgorithm._fill_queue_from_agent()
-> GeneralAgentBBOAlgorithm._run_engine()
-> NanobotEngine.run_agent()
-> python -m bbo.algorithms.agentic.nanobot_runner agent ...
-> Nanobot CLI
-> stdout: {"candidates": [...]}
-> parse_agent_candidate_payload()
-> Experimenter._normalize_suggestion()
-> task.evaluate()
-> GeneralAgentBBOAlgorithm.tell()
-> logger.log()
```

## 读代码时容易混淆的点

1. `nanobot` 不是一个单独的优化器逻辑文件；它复用 `GeneralAgentBBOAlgorithm`，只是后端 engine 是 `NanobotEngine`。
2. `workspace_json` 不是“直接 JSON 文件作为最终答案”这么简单；它会生成完整工作区，让 agent 通过 `bbo_tools.py` 调工具，最后 stdout 才必须是 strict JSON。
3. `code_interpreter()` 不是 benchmark evaluator；它只是分析沙箱，不消耗 trial budget。
4. `sample()` 和 `validate()` 也不消耗 trial budget；只有 `Experimenter` 调 `task.evaluate()` 才算一次真实评估。
5. `agent_tool_calls.jsonl` 记录工具调用，不等于真实 trial；真实 trial 看 `trials.jsonl`。
6. `--agent-require-visible-cot` 是额外约束：即使候选 JSON 合法，只要 wrapper 没捕获到 visible reasoning，也会判失败。
7. `--no-agent-allow-fallback` 会让失败更硬：agent 输出不合法时不会随机补一个点。

## 最短源码阅读顺序

如果只想快速读懂当前流程，建议按这个顺序看：

1. `run.sh`
2. `bbo/run.py` 里 agent 参数和 `run_single_experiment()`
3. `bbo/algorithms/registry.py` 里 `nanobot` 的注册
4. `bbo/core/experimenter.py` 的 `run()`
5. `bbo/algorithms/agentic/general_agent.py` 的 `setup()`、`ask()`、`tell()`、`_fill_queue_from_agent()`
6. `bbo/algorithms/agentic/general_agent_engines.py` 的 `NanobotEngine`
7. `bbo/algorithms/agentic/nanobot_runner.py`
8. `bbo/algorithms/agentic/workspace_python_api.py`
9. `bbo/algorithms/agentic/workspace_tool_cli.py`
10. 当前 task 的实现，比如 `bbo/tasks/synthetic/branin.py`
