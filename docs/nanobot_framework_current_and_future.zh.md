# Nanobot 框架当前实现与未来改进方向

这份文档说明当前仓库里的 `nanobot` agentic BBO 框架是怎么实现的，并整理几个后续可以做的改进方向。重点会放在：

- Nanobot 在整个黑盒优化流程里扮演什么角色。
- 代码从 CLI 到 Nanobot 子进程是怎么串起来的。
- workspace、prompt、工具、候选 JSON、日志分别怎么工作。
- 目前已有的 GP/LCB 分析 demo 是什么。
- 未来如何给 agent 更强的数据分析 demo，比如 GP、TPE、随机森林、局部敏感性分析等。

## 一句话总结

当前 `nanobot` 不是一个单独写死的优化算法，而是：

```text
GeneralAgentBBOAlgorithm 通用 ask/tell 优化框架
  +
NanobotEngine 后端适配器
  +
nanobot_runner.py 兼容补丁和日志包装
  +
agent_workspace 里的文件、Python API、辅助脚本
```

它的核心思想是：让 Nanobot 像一个 coding agent 一样，在 workspace 里读任务、读历史、运行分析脚本，然后输出下一批候选配置。真正的 benchmark 评估仍然由外层 `Experimenter` 执行，Nanobot 不直接调用目标函数，也不直接消耗评估预算。

## 当前实现的主要代码位置

| 文件 | 作用 |
|---|---|
| `bbo/algorithms/registry.py` | 把 `nanobot` 和 `agentic_nanobot` 注册到算法表。 |
| `bbo/run.py` | CLI 和实验入口，把 `--agent-*` 参数打包传给算法。 |
| `bbo/algorithms/agentic/general_agent.py` | 通用 agentic BBO 主体，负责 `setup()`、`ask()`、`tell()`、workspace、prompt、候选解析和状态持久化。 |
| `bbo/algorithms/agentic/general_agent_engines.py` | agent engine 适配层，里面有 `NanobotEngine`。 |
| `bbo/algorithms/agentic/nanobot_runner.py` | Nanobot CLI wrapper，给 Nanobot 做兼容 patch、记录 LLM log 和 reasoning metadata。 |
| `bbo/algorithms/agentic/workspace_python_api.py` | 会复制成 workspace 里的 `bbo_tools.py`，给 agent 使用。 |
| `bbo/algorithms/agentic/workspace_tool_cli.py` | 会复制成 workspace 里的 `bbo_tool.py`，执行 workspace 工具调用。 |
| `bbo/algorithms/agentic/gp_expected_improvement_example.py` | 会复制成 workspace 里的 GP/LCB 候选生成 demo。 |

## 算法注册：`nanobot` 怎么接入 BBO

在 `bbo/algorithms/registry.py` 中：

- `agentic_nanobot` 的 factory 是 `NanobotBBOAlgorithm`。
- `nanobot` 是 `agentic_nanobot` 的别名。

`NanobotBBOAlgorithm` 本身非常薄：

```python
class NanobotBBOAlgorithm(GeneralAgentBBOAlgorithm):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            framework="nanobot",
            algorithm_name="agentic_nanobot",
            **kwargs,
        )
```

也就是说，Nanobot 没有自己重新实现一套黑盒优化循环。它复用 `GeneralAgentBBOAlgorithm`，只是把后端框架固定为 `nanobot`。

## 外层流程：Experimenter 控制真实评估

Nanobot 只负责提候选配置，真实 objective 评估由 `Experimenter` 控制。

整体循环是：

```text
Experimenter.run()
  -> algorithm.setup(task_spec)
  -> algorithm.ask()
      -> 返回一个 TrialSuggestion
  -> task.evaluate(suggestion)
      -> 真正消耗一次 benchmark budget
  -> algorithm.tell(observation)
      -> 把结果写回算法历史
  -> 重复直到 max_evaluations
```

这点很重要：agent 可以分析历史、采样、验证候选、运行离线脚本，但不能直接“偷跑” benchmark evaluator。只有 `task.evaluate()` 那一步才是真实评估。

## setup 阶段：创建 Nanobot 工作区

`GeneralAgentBBOAlgorithm.setup()` 会创建运行目录和 workspace：

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
    bbo_workspace_audit.py
  agent_state/
    config.json
  agent_memory/
    memory.jsonl
    memory_summary.json
  agent_prompts.jsonl
  agent_calls.jsonl
  agent_tool_calls.jsonl
  agent_web_sources.jsonl
  agent_optimization_trace.jsonl
  llm_logs/
  reasoning_traces/
  agent_reasoning_metadata.jsonl
```

对 Nanobot 来说，最核心的是 `agent_workspace/`。它通过读这些文件来理解：

- 当前任务是什么。
- 搜索空间有哪些参数。
- 主目标是 minimize 还是 maximize。
- 最近历史 trial 表现如何。
- 当前最好配置是什么。
- 最终输出 JSON 应该长什么样。
- 可以用哪些 BBO workspace 工具。

## Nanobot config 是怎么写的

当 `framework == "nanobot"` 时，`_build_framework_config()` 会写：

```text
agent_state/config.json
```

里面会包含：

```json
{
  "agents": {
    "defaults": {
      "workspace": "<agent_workspace>",
      "provider": "auto 或 provider_key",
      "model": "可选 model"
    }
  },
  "providers": {
    "openai/custom/anthropic/...": {
      "api_key": "可选",
      "api_base": "可选"
    }
  },
  "channels": {
    "send_progress": false,
    "send_tool_hints": false
  },
  "tools": {
    "restrict_to_workspace": true
  }
}
```

provider 名会被 `_nanobot_provider_key()` 转换：

| 输入 provider | Nanobot provider key |
|---|---|
| `openai` | `openai` |
| `anthropic` | `anthropic` |
| `google` | `gemini` |
| `ollama` | `ollama` |
| `azure` | `azure_openai` |
| 其他 | `custom` |

`restrict_to_workspace: true` 的意思是限制 Nanobot 在工作区内操作，降低它乱读乱写外部文件的风险。

## ask 阶段：什么时候调用 Nanobot

`ask()` 的逻辑是：

```text
如果历史数量 < initial_random
  -> 从 search_space 随机采样
否则如果候选队列为空
  -> 调 _fill_queue_from_agent()
否则
  -> 从候选队列弹出一个配置
```

`_fill_queue_from_agent()` 会：

1. 刷新 workspace 文件，保证历史和 incumbent 是最新的。
2. 生成 `call_id`，例如 `agent_call_00000`。
3. 构造 prompt。
4. 把 prompt 写入 `agent_prompts.jsonl`。
5. 调 `_run_engine()`，最终进入 `NanobotEngine.run_agent()`。
6. 解析 Nanobot 返回的 JSON。
7. 校验候选配置。
8. 去重。
9. 把合法候选放入 `_queue`。

如果 Nanobot 返回不合法 JSON，或者候选不符合搜索空间，代码会重试。重试仍失败时，如果 `allow_fallback=True`，会随机采样兜底；如果 `allow_fallback=False`，会直接报错。

## prompt：Nanobot 收到什么指令

默认 prompt 风格是 `workspace`。它不会把所有信息都塞进 prompt，而是让 Nanobot 去读 workspace 文件。

普通调用的 prompt 大致是：

```text
You are an optimization agent for task `<task_name>`.
Workspace path: <agent_workspace>
Call id: agent_call_00000
Attempt: 0

Read `instructions.md`, `task.md`, `manifest.json`, `space.json`,
`objective.json`, `history.jsonl`, `incumbent.json`, `tool_specs.json`,
`python_environment.md`, `bbo_workspace_audit.py`, `gp_expected_improvement.py`,
and `examples/gp_expected_improvement.py` in the workspace.

If native function-calling tools are unavailable, use the workspace Python
API: `from bbo_tools import BBO`. At minimum, inspect search space,
history, incumbent, and validate your final candidate list.

Current best primary objective: <best_score>
Objective direction: minimize/maximize

Produce candidate configurations now. Your entire stdout must be the
strict JSON object described in `instructions.md`.
```

第一次 agent 调用时，还会要求 Nanobot 先运行：

```bash
python3 bbo_workspace_audit.py
```

这个 audit 脚本不是优化算法，只是工具链体检。它会检查 `BBO()` API、历史读取、采样、验证、memory、code interpreter、web search、fetch URL 等是否能正常调用。

## NanobotEngine：怎么启动 Nanobot 子进程

`NanobotEngine.run_agent()` 会启动一个 Python 子进程：

```bash
python -m bbo.algorithms.agentic.nanobot_runner \
  agent \
  -m "<prompt>" \
  --no-markdown \
  -s "<call_id>" \
  -w "<agent_workspace>" \
  -c "<agent_state/config.json>"
```

其中：

- `-m` 是传给 Nanobot 的 prompt。
- `--no-markdown` 要求最终不要用 markdown 包裹。
- `-s` 用 `call_id` 当 Nanobot session id。
- `-w` 指定 workspace。
- `-c` 指定 Nanobot 配置文件。

环境变量会带上：

- API key / API base。
- `BBO_AGENT_CALL_ID`。
- `BBO_AGENT_MODEL_REQUESTED`。
- `BBO_AGENT_PROVIDER`。
- `BBO_AGENT_REQUIRE_VISIBLE_COT`。
- `BBO_NANOBOT_REASONING_DIR`。
- `BBO_NANOBOT_REASONING_METADATA_PATH`。
- `BBO_NANOBOT_NO_MAX_TOKENS=1`。
- `BBO_NANOBOT_PARSE_TEXT_TOOL_CALLS=1`。
- `BBO_NANOBOT_LOG_DIR`。

## nanobot_runner.py 做了什么

`nanobot_runner.py` 是一个 wrapper。它最后仍然会导入并执行 Nanobot 的 CLI：

```python
from nanobot.cli.commands import app
app()
```

但在执行前，它会做几类兼容 patch。

### 1. 去掉不兼容的 `max_tokens`

一些 OpenAI-compatible endpoint 不接受 `max_tokens` 参数。设置 `BBO_NANOBOT_NO_MAX_TOKENS=1` 后，runner 会 patch Nanobot provider，把请求里的 `max_tokens` 移除。

### 2. 把 XML 样式工具调用转成 Nanobot 结构化工具调用

有些本地模型会输出这种文本：

```xml
<tool_call>
<function=exec>
<parameter=command>
python3 bbo_workspace_audit.py
</parameter>
</function>
</tool_call>
```

但 Nanobot runner 只执行结构化 tool call。`_patch_parse_text_tool_calls()` 会把这种文本解析成 Nanobot 的 `ToolCallRequest`，让 Nanobot 能继续执行工具。

### 3. 记录 LLM 日志和 reasoning trace

如果设置了 `BBO_NANOBOT_LOG_DIR`，runner 会记录：

- Nanobot 最终消息列表。
- duration。
- usage。
- assistant reasoning 内容。
- reasoning token 数。
- 每次 agent call 对应的 trace 文件。

这些会写到：

```text
llm_logs/
reasoning_traces/
agent_reasoning_metadata.jsonl
```

如果开启 `require_visible_cot`，`general_agent.py` 会检查 metadata 里是否捕获到可见推理。没有捕获到时，这次 agent 调用会被判失败。

## Nanobot 怎么使用 BBO 工具

当前 Nanobot 这条路径主要依赖 workspace 工具，而不是注入式 function calling。

推荐方式是在 workspace 里写 Python 脚本：

```python
from bbo_tools import BBO

bbo = BBO()
space = bbo.search_space()
history = bbo.history(mode="recent", limit=40)
incumbent = bbo.incumbent()
analysis = bbo.analyze_history(limit=100)
sample = bbo.sample(n=8, strategy="around_incumbent")
checked = bbo.validate([item["config"] for item in sample["candidates"]])
```

这些方法底层会通过 `bbo_tool.py` 执行工具，并写日志到：

```text
agent_tool_calls.jsonl
```

当前 workspace API 包括：

| API | 作用 |
|---|---|
| `task_context()` | 读取任务背景和 manifest。 |
| `manifest()` | 读取 manifest。 |
| `search_space()` | 读取参数 schema、默认值、维度。 |
| `objective()` | 读取主目标和优化方向。 |
| `tool_specs()` | 读取工具规格。 |
| `history()` | 读取历史 trial。 |
| `incumbent()` | 读取当前最好配置。 |
| `validate()` | 校验候选是否合法、是否重复。 |
| `sample()` | 采样候选，不消耗真实评估预算。 |
| `analyze_history()` | 做轻量历史统计。 |
| `memory_read()` / `memory_write()` | 读写 agent memory。 |
| `code_interpreter()` | 通过 SandboxFusion 执行离线分析代码。 |
| `web_search()` | 搜索网页。 |
| `fetch_url()` | 抓取 URL。 |

## 一个重要现状：Nanobot 不支持注入式 BBO function-calling tools

`NanobotEngine.run_agent()` 里有一个明确判断：

```python
if tools:
    return AgentResult(
        status="failed",
        answer="",
        error="NanobotEngine does not support injected BBO function-calling tools in this runtime.",
        returncode=-2,
    )
```

这意味着：Nanobot 当前不支持像 `OpenAICompatibleToolEngine` 那样，把 BBO tools schema 直接注入给模型做原生 function calling。

对 Nanobot 更合适的运行方式是：

```bash
--agent-prompt-style workspace
--agent-tool-mode workspace_json
```

这里的 `workspace_json` 含义是：不注入原生 function-calling tools，但 workspace 里仍然有 `bbo_tools.py`、`bbo_tool.py`、`tool_specs.json`，Nanobot 可以通过文件和脚本使用这些工具。

当前全局默认值是 `tool_mode=function_calling`、`prompt_style=workspace`。这对 OpenAI-compatible 后端合理，但对真实 Nanobot 后端会触发上面的拒绝逻辑。因此这是一个值得修的框架级默认/兼容问题。

## 当前已有的分析 demo：GP/LCB 候选生成脚本

当前已经有一个可编辑的分析 demo：

```text
agent_workspace/gp_expected_improvement.py
agent_workspace/examples/gp_expected_improvement.py
```

源码来自：

```text
bbo/algorithms/agentic/gp_expected_improvement_example.py
```

它做的事情是：

1. `from bbo_tools import BBO`。
2. 读取搜索空间、目标方向、全部历史。
3. 找出成功 trial。
4. 如果成功 trial 太少，就 fallback 到 `BBO().sample(...)`。
5. 如果历史足够，就用 `scikit-learn` 的 `GaussianProcessRegressor` 拟合 GP。
6. 对 maximize 任务，把目标取负，统一成 minimize。
7. 把 config 编码成数值向量：
   - float / int 归一化到 `[0, 1]`。
   - categorical 用 choices index 简单编码。
8. 从搜索空间采样候选池。
9. 用 GP 预测候选池的 mean 和 std。
10. 用 LCB acquisition：

```text
score = mean - 1.5 * std
```

11. 选择 score 最小的候选。
12. 用 `BBO().validate(...)` 校验。
13. 最终打印 strict JSON：

```json
{
  "candidates": [
    {
      "config": {
        "param": "value"
      },
      "rationale": "GP/LCB workspace analysis or validated fallback sample"
    }
  ]
}
```

这个 demo 已经能帮助 agent 做基础 BO 风格建模，但它目前仍然偏“示例脚本”，不是一个完整的分析框架。

## 当前实现的优点

### 1. 复用通用 ask/tell 框架

Nanobot 不需要自己实现实验循环、历史更新、候选校验、状态持久化，这些都由 `GeneralAgentBBOAlgorithm` 管。

### 2. workspace 设计适合 coding agent

Nanobot 可以读文件、写脚本、运行命令。把任务上下文拆成文件，比把所有内容塞进 prompt 更适合这类 agent。

### 3. 候选输出有强校验

agent 输出后，Python 侧仍会严格检查 JSON 和搜索空间，不会直接相信 agent。

### 4. 工具调用和 LLM 调用都有日志

`agent_prompts.jsonl`、`agent_calls.jsonl`、`agent_tool_calls.jsonl`、`llm_logs/`、`reasoning_traces/` 都能帮助 debug。

### 5. 已经有可编辑的 GP demo

agent 不只是“拍脑袋”出候选，它可以运行 `gp_expected_improvement.py`，基于历史拟合一个简单 surrogate。

## 当前限制和容易踩坑的地方

### 1. Nanobot 默认 tool mode 不够友好

全局默认是 `function_calling`，但真实 `NanobotEngine` 不支持注入式 BBO function-calling tools。实际跑 Nanobot 时应该显式传：

```bash
--agent-tool-mode workspace_json
```

否则 Nanobot engine 会因为收到 `tools` 参数而返回 failed。默认允许 fallback 时，流程可能退化成随机候选；禁用 fallback 时会直接失败。

### 2. workspace tools 和 function-calling tools 是两套入口

它们能力相似，但实现不是同一套代码路径：

- function-calling tools 在 `tools/*.py`。
- workspace tools 在 `workspace_tool_cli.py` 和 `workspace_python_api.py`。

这会带来维护成本：新增一个工具时，可能要同时改多个地方。

### 3. GP demo 的 acquisition 名字和实现不完全一致

脚本名叫 `gp_expected_improvement.py`，但当前核心排序是 LCB：

```text
mean - 1.5 * std
```

它不是标准 EI。这个命名可能会误导后续使用者。

### 4. categorical 编码比较简单

当前 GP demo 对 categorical 参数只是用 choices index 编码。这个做法简单，但可能让模型误以为类别之间有线性顺序。

### 5. 高维任务里 GP 可能不稳

GP 对高维、稀疏历史、混合类型参数不一定稳。脚本有 fallback，但还缺少更系统的模型选择和诊断。

### 6. 分析脚本输出缺少结构化诊断

当前 demo 最终只输出候选 JSON。它不会额外保存一份结构化分析报告，比如：

- 训练样本数。
- 使用了哪个模型。
- acquisition 分数。
- 是否 fallback。
- 哪些参数最敏感。
- 候选为什么被选中。

这些信息如果能写入 workspace，会更利于复盘和改进 agent。

## 未来改进方向

### 方向 1：给 Nanobot 设置框架级默认 `workspace_json`

当前全局默认是：

```text
prompt_style = workspace
tool_mode = function_calling
```

但 Nanobot 更适合：

```text
prompt_style = workspace
tool_mode = workspace_json
```

可以做一个框架级默认策略：

```text
如果 framework == nanobot 且用户没有显式指定 tool_mode
  -> 默认使用 workspace_json
否则
  -> 保持用户配置
```

收益：

- 避免真实 Nanobot 跑起来直接拒绝 injected tools。
- 降低新手使用成本。
- 和现有 Nanobot workflow 脚本保持一致。

### 方向 2：把 GP demo 扩展成“分析 recipe”目录

现在只有一个 `gp_expected_improvement.py`。可以扩展成：

```text
agent_workspace/
  analysis_recipes/
    gp_lcb.py
    gp_ei.py
    random_forest_ei.py
    tpe_like.py
    local_sensitivity.py
    history_report.py
    ensemble_candidates.py
  run_analysis_recipe.py
```

每个 recipe 都遵守统一接口：

```text
读 BBO() 上下文
  -> 分析历史
  -> 生成候选
  -> validate
  -> 输出 candidates JSON
  -> 额外写 analysis_report.json
```

收益：

- agent 不需要从零写分析代码。
- 不同任务可以选择更合适的 recipe。
- 分析过程更可复现。

### 方向 3：增加标准 EI / PI / UCB / LCB acquisition demo

当前脚本实际是 LCB。可以明确拆成几种 acquisition：

| acquisition | 用途 |
|---|---|
| EI | 常见 BO acquisition，平衡探索和利用。 |
| PI | 更偏向寻找超过当前 best 的区域。 |
| UCB/LCB | 通过均值和不确定性控制探索强度。 |
| Thompson sampling | 从 posterior 采样，增加多样性。 |

对 minimize 任务，可以统一把 acquisition 写成“越小越优”或“越大越优”，避免方向混乱。

### 方向 4：增加非 GP 的代理模型 demo

GP 不适合所有任务。可以增加：

| 方法 | 适合场景 |
|---|---|
| Random Forest surrogate | 混合类型参数、高维、非平滑目标。 |
| ExtraTrees surrogate | 小样本、高噪声、需要多样性。 |
| Gradient Boosting surrogate | 中等维度、非线性但样本稍多。 |
| TPE-like density ratio | 类似 Optuna TPE，对 categorical 和条件空间更自然。 |
| kNN / local interpolation | 很小样本时作为简单 baseline。 |

agent 可以根据 `history_size`、维度、参数类型选择模型：

```text
history < dim + 1
  -> fallback sampling
mixed categorical-heavy
  -> RF / TPE
low-dimensional continuous
  -> GP
many observations
  -> ensemble
```

### 方向 5：给 agent 一个 `history_report.py`

这个 demo 不直接生成候选，而是帮 agent 做数据理解：

```text
读 history
  -> 统计成功/失败 trial
  -> 找 best / worst
  -> 计算数值参数相关性
  -> 按 categorical 分组看均值
  -> 检查重复配置
  -> 检查失败原因
  -> 输出 history_report.md/json
```

收益：

- agent 在建模前先理解数据质量。
- prompt 可以要求 Nanobot 先读 `history_report.json` 再提候选。
- 对小白也更容易 debug 为什么 agent 选了某些参数。

### 方向 6：候选生成脚本保存结构化分析报告

建议每次分析脚本除了 stdout 的 candidates JSON，再写：

```text
analysis_report.json
```

内容包括：

```json
{
  "recipe": "gp_lcb",
  "history_size": 80,
  "success_count": 75,
  "model": "GaussianProcessRegressor",
  "fallback": false,
  "candidate_pool_size": 256,
  "acquisition": "LCB",
  "top_candidates": [
    {
      "config": {},
      "predicted_mean": 0.12,
      "predicted_std": 0.03,
      "acquisition_score": 0.075
    }
  ],
  "warnings": []
}
```

收益：

- 人可以看懂 agent 的分析依据。
- 后续可以用日志评估哪个 recipe 更有效。
- 失败时更容易定位是模型问题、数据问题还是候选校验问题。

### 方向 7：把 workspace tools 和 function-calling tools 合并语义来源

现在两套工具入口能力相近，但代码路径分开。未来可以让 workspace CLI/API 自动复用 `BaseBBOTool` 定义，或者至少从同一份 tool spec 生成：

- function-calling schema。
- CLI handler。
- `BBO()` Python API 文档。

收益：

- 减少工具行为不一致。
- 新增工具时不用改三四个地方。
- `tool_specs.json` 更可信。

### 方向 8：让 Nanobot 支持注入式 BBO tool calling

现在 `NanobotEngine` 收到 `tools` 就直接失败。未来可以考虑：

1. 把 BBO tools 转成 Nanobot 原生工具格式。
2. 在 `nanobot_runner.py` 里桥接 tool execution。
3. 或者让 Nanobot 的工具系统直接调用 `BBOToolRegistry.execute_tool()`。

收益：

- Nanobot 可以同时使用原生工具调用和 workspace 文件。
- 和 OpenAI-compatible 后端行为更一致。
- tool call 的参数 schema 和执行路径更标准。

但这需要认真设计权限、日志和错误处理，不能只是简单把函数暴露出去。

### 方向 9：增加 recipe selector，让 agent 不用猜该跑哪个 demo

可以提供一个脚本：

```bash
python3 choose_analysis_recipe.py
```

它根据任务情况输出建议：

```json
{
  "recommended": "random_forest_ei.py",
  "reason": "mixed categorical parameters and history_size=60",
  "alternatives": ["tpe_like.py", "gp_lcb.py"]
}
```

规则可以先简单写死：

- 历史太少：采样或围绕 incumbent jitter。
- 低维连续：GP。
- categorical 多：RF/TPE。
- 高维：RF/ExtraTrees。
- 失败 trial 多：先跑 history report。

### 方向 10：加入“候选组合器”

很多时候单一模型不稳。可以让脚本组合多种候选来源：

```text
30% GP acquisition
30% RF acquisition
20% around incumbent
20% random exploration
```

然后统一：

```text
去重 -> validate -> 排序 -> 输出 top K
```

收益：

- 减少单一代理模型失效的风险。
- 在历史较少时仍有探索。
- 对混合搜索空间更稳。

### 方向 11：增加任务类型专用 demo

比如 BBOPlace 这类宏布局任务，有大量 `x_i/y_i` 坐标参数。通用 GP 对这种结构不一定合适。

可以提供专用 demo：

```text
bboplace_coordinate_patterns.py
```

做一些结构化候选：

- 围绕 incumbent 小幅扰动坐标。
- 对拥挤区域做局部扩散。
- 按宏的初始顺序生成网格/簇状布局。
- 保持 x/y 成对结构，避免全默认坐标。

收益：

- 比纯随机或通用 GP 更符合任务结构。
- 更容易让 agent 学会坐标类任务的合理变换。

### 方向 12：把 prompt 明确引导到分析 demo

当前 prompt 已经说“history 足够时运行 `gp_expected_improvement.py` 或改造示例”。可以进一步具体化：

```text
If history_size < dim + 1:
  use sample_candidates around incumbent.
If history_size >= dim + 1 and mostly continuous:
  run analysis_recipes/gp_lcb.py.
If many categorical parameters:
  run analysis_recipes/random_forest_ei.py.
Always inspect analysis_report.json before final JSON.
```

收益：

- agent 更少盲目决策。
- 工具使用更稳定。
- 输出候选更可解释。

## 推荐的近期落地顺序

如果按投入产出比排序，我建议先做：

1. **修 Nanobot 默认 tool mode**
   - Nanobot 默认或自动降级到 `workspace_json`。
   - 这是稳定性优先级最高的问题。

2. **重命名或拆分 GP demo**
   - 当前脚本叫 expected improvement，但实现是 LCB。
   - 可以改成 `gp_lcb.py`，另写真正 `gp_ei.py`。

3. **增加 `history_report.py`**
   - 先不复杂建模，只做数据质量和历史统计。
   - 对 agent 和人类 debug 都有用。

4. **增加 `random_forest_surrogate.py`**
   - 比 GP 更适合混合类型和高维任务。
   - 可以用候选池打分，选择 top K。

5. **让分析脚本写 `analysis_report.json`**
   - 让每次候选生成有证据可看。
   - 后续可以自动比较不同 recipe 的效果。

6. **做 `analysis_recipes/` 统一入口**
   - 把 GP、RF、TPE、history report 都放进去。
   - prompt 只需要告诉 agent “根据情况运行 recipe”。

## 小结

当前 Nanobot 框架已经完成了一个可用闭环：

```text
CLI 选择 nanobot
  -> 通用 agentic BBO 创建 workspace
  -> Nanobot 子进程读取 workspace 和 prompt
  -> Nanobot 可运行 BBO Python API 和分析脚本
  -> Nanobot 输出 strict candidates JSON
  -> Python 主程序校验候选
  -> Experimenter 真实评估
  -> observation 写回历史
```

现在最值得加强的不是“让 agent 能不能跑起来”，而是“让 agent 更稳定、更系统地做数据分析”。已有的 `gp_expected_improvement.py` 是一个好的起点，但未来应该发展成一组标准分析 recipe：GP、EI/LCB/UCB、随机森林、TPE、history report、任务专用候选生成器，并配套结构化报告和更清晰的 prompt 引导。
