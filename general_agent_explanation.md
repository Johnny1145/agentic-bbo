# `general_agent.py` 代码说明

源文件：`/home/trx/cm/agentic-bbo/bbo/algorithms/agentic/general_agent.py`

这份文件实现的是一个“让通用 AI agent 来做黑盒优化”的算法包装器。你可以把它理解成：

- 外部有一个黑盒目标函数，比如“给一组参数，跑一次实验，得到一个分数”。
- 这个文件里的算法负责不断提出下一组参数。
- 参数可以由 Nanobot、Claude Code、OpenAI-compatible agent 这类外部 agent 来生成。
- 算法通过 `ask()` 给出候选参数，通过 `tell()` 接收实验结果，然后继续下一轮。

如果用最简单的话概括主流程，就是：

```text
setup() 初始化任务
  -> ask() 要一个候选配置
  -> 外部系统拿这个配置去跑实验
  -> tell() 把实验结果还给算法
  -> 算法记录历史、更新当前最优、让 agent 参考历史继续提建议
```

文件里很多函数名前面有 `_`。这通常表示“内部函数”，也就是类自己用的辅助方法，外部代码一般不应该直接调用。

## 一、几个核心概念

### 黑盒优化

黑盒优化的意思是：我们不知道目标函数内部怎么算，只能试一组参数，看结果好不好。比如：

```text
输入：learning_rate=0.01, batch_size=32
输出：accuracy=0.91
```

算法的目标就是少试几次，尽快找到更好的参数。

### `ask()` / `tell()` 模式

这个文件继承的 `Algorithm` 使用典型的 `ask()` / `tell()` 接口：

- `ask()`：算法说“下一次请试这组参数”。
- 外部评估器：实际运行实验。
- `tell()`：外部评估器把实验结果告诉算法。

这个模式把“提建议”和“跑实验”分开，方便算法不关心实验怎么执行。

### agent workspace

这个算法会给外部 agent 准备一个工作目录，里面写入：

- 任务描述 `task.md`
- 参数空间 `space.json`
- 历史实验记录 `history.jsonl`
- 当前最优结果 `incumbent.json`
- 工具说明 `tool_specs.json`
- 辅助脚本 `bbo_tools.py`、`bbo_tool.py`
- 给 agent 看的操作说明 `instructions.md`

外部 agent 可以读这些文件，再输出 JSON 格式的候选参数。

## 二、类总览

这个文件里有 8 个类：

| 类名 | 作用 |
|---|---|
| `GeneralAgentValidationError` | agent 返回的候选结果格式不对时抛出的异常 |
| `ParsedAgentCandidate` | 表示“刚从 agent 原始回答里解析出来”的一个候选配置 |
| `AgentCandidateEntry` | 表示“已经排队，准备被 `ask()` 返回”的候选配置 |
| `GeneralAgentConfig` | 存放整个 agent 优化器的配置项 |
| `GeneralAgentBBOAlgorithm` | 核心算法类，负责 ask/tell、调用 agent、解析候选、记录状态 |
| `NanobotBBOAlgorithm` | 使用 Nanobot 作为后端 agent 的快捷子类 |
| `ClaudeCodeBBOAlgorithm` | 使用 Claude Code 作为后端 agent 的快捷子类 |
| `OpenAICompatibleBBOAlgorithm` | 使用 OpenAI-compatible function calling agent 的快捷子类 |

## 三、`GeneralAgentValidationError`

位置：`general_agent.py:59`

```python
class GeneralAgentValidationError(ValueError):
```

这是一个自定义异常类。

它的作用很简单：当 agent 返回的内容不符合要求时，代码会抛出这个异常。比如 agent 没有返回合法 JSON、没有 `candidates` 字段、参数超出搜索空间范围等。

它继承自 `ValueError`，说明这类错误本质上是“值不合法”。

类内没有自己定义函数。

## 四、`ParsedAgentCandidate`

位置：`general_agent.py:63`

```python
@dataclass(frozen=True)
class ParsedAgentCandidate:
```

这是一个数据容器，用来保存“从 agent 原始输出里解析出来的候选配置”。

字段：

| 字段 | 含义 |
|---|---|
| `config` | 实际参数配置，例如 `{"x": 1.0, "y": 2.0}` |
| `candidate_index` | 这是 agent 返回的第几个候选，从 0 开始 |
| `metadata` | 附加信息，比如 agent 给出的理由 `rationale` |

`frozen=True` 表示这个对象创建以后不建议再修改。你可以把它理解成“只读记录”。

类内没有手写函数，但 `@dataclass` 会自动生成 `__init__`、比较等基础方法。

## 五、`AgentCandidateEntry`

位置：`general_agent.py:72`

```python
@dataclass
class AgentCandidateEntry:
```

这也是一个数据容器，用来保存“已经通过验证、放进队列、等待 `ask()` 返回”的候选配置。

为什么需要队列？因为一次 agent 调用可能返回多个候选配置，但 `ask()` 一次只返回一个，所以多出来的候选会先放在 `_queue` 里。

字段：

| 字段 | 含义 |
|---|---|
| `config` | 参数配置 |
| `call_id` | 这是哪一次 agent 调用产生的 |
| `candidate_index` | 这是该次 agent 返回的第几个候选 |
| `metadata` | 附加信息，例如 fallback 原因、rationale 等 |

类内没有手写函数，但 `@dataclass` 会自动生成 `__init__` 等基础方法。

## 六、`GeneralAgentConfig`

位置：`general_agent.py:82`

```python
@dataclass(frozen=True)
class GeneralAgentConfig:
```

这是核心配置类。它不负责执行逻辑，只负责保存配置。

常用字段解释：

| 字段 | 通俗解释 |
|---|---|
| `framework` | 使用哪种 agent 后端，例如 `nanobot`、`claude_code` |
| `algorithm_name` | 算法显示名称 |
| `timeout_seconds` | 每次调用 agent 最多等多久 |
| `max_retries` | agent 输出不合法时最多重试几次 |
| `history_limit` | 给 agent 看的最近历史记录数量 |
| `candidates_per_call` | 希望 agent 每次返回几个候选配置 |
| `model` | 使用的模型名 |
| `provider` | 模型服务商，比如 OpenAI、Anthropic |
| `api_base` | 自定义 API 地址 |
| `api_key_env` | 从哪个环境变量读取 API key |
| `initial_random` | 前几次不用 agent，先随机采样 |
| `run_dir` | 运行输出目录 |
| `resume` | 是否从之前保存的状态恢复 |
| `tool_mode` | agent 调工具的方式，支持 `function_calling` 和 `workspace_json` |
| `prompt_style` | prompt 风格，支持 `workspace` 和 `direct_json` |
| `max_tool_calls` | 单次 agent 调用最多允许多少次工具调用 |
| `enable_memory` | 是否启用 agent 记忆 |
| `enable_code_interpreter` | 是否启用代码解释器工具 |
| `code_backend` | 代码执行后端，例如 `sandboxfusion`、`mock`、`disabled` |
| `web_search_provider` | 网页搜索后端 |
| `allow_fallback` | agent 失败时是否允许随机兜底 |
| `require_visible_cot` | 是否要求捕获可见推理记录 |

`frozen=True` 表示配置对象创建后不应该被改动。这样可以避免运行中配置突然变化导致行为混乱。

类内没有手写函数。

## 七、`GeneralAgentBBOAlgorithm`

位置：`general_agent.py:112`

这是本文件最核心的类。它继承自 `Algorithm`，实现黑盒优化算法接口。

它主要负责 5 件事：

1. 初始化任务、搜索空间、工作目录和日志文件。
2. 在 `ask()` 时返回下一组候选参数。
3. 在 `tell()` 时接收实验结果，并更新历史和当前最优。
4. 调用外部 agent 生成候选配置。
5. 解析、校验、去重、排队 agent 返回的候选配置。

### `__init__`

位置：`general_agent.py:115`

初始化算法对象。

它做的事情很多，但可以分成三类：

1. 检查用户传入的参数是否合法。
   - `timeout_seconds` 必须大于 0。
   - `max_retries` 不能小于 0。
   - `history_limit` 不能小于 0。
   - `candidates_per_call` 必须大于 0。
   - `initial_random` 不能小于 0。
   - `tool_mode` 只能是 `function_calling` 或 `workspace_json`。
   - `prompt_style` 只能是 `workspace` 或 `direct_json`。

2. 创建 `GeneralAgentConfig`。
   - 把所有配置集中放进 `self.config`。
   - 框架名会先通过 `normalize_agent_framework()` 标准化。

3. 初始化运行时状态。
   - `_task_spec`：当前任务。
   - `_search_space`：参数搜索空间。
   - `_history`：已经评估过的历史结果。
   - `_queue`：agent 已经生成但还没有被 `ask()` 返回的候选。
   - `_seen_config_ids`：已经见过的配置，用于去重。
   - `_best`：当前最优结果。
   - `_call_index`：第几次调用 agent。
   - `_workspace_dir`、`_state_dir`、`_memory_dir`：工作目录、状态目录、记忆目录。

### `name`

位置：`general_agent.py:215`

这是一个属性方法。

返回算法名字，也就是 `self.config.algorithm_name`。

比如 `NanobotBBOAlgorithm` 默认会返回 `agentic_nanobot`。

### `artifact_paths`

位置：`general_agent.py:219`

也是一个属性方法。

返回算法运行过程中产生的重要文件路径，比如：

- agent 工作目录
- prompt 日志
- agent 调用日志
- 状态文件
- 历史记录
- 工具调用日志

它返回的是 `dict(self._artifacts)`，也就是返回一份拷贝，避免外部代码直接改内部状态。

### `setup`

位置：`general_agent.py:222`

初始化一次具体优化任务。

这是使用算法前必须调用的方法。它会：

1. 保存任务信息。
   - `TaskSpec`
   - 搜索空间
   - 主目标名称
   - 优化方向：最小化或最大化

2. 初始化随机数。
   - 用传入的 `seed` 创建 `random.Random`。

3. 准备运行目录。
   - `agent_workspace`
   - `agent_state`
   - `agent_memory`
   - `reasoning_traces`
   - `llm_logs`

4. 加载任务 manifest。
   - `load_BBO_manifest(task_spec)`

5. 创建 memory store 和 tool registry。
   - memory 用于 agent 记忆。
   - tool registry 用于管理 BBO 工具。

6. 创建 agent 工作副本 `AgentWorkCopy`。
   - 这里把配置路径、项目根目录、日志目录、推理记录路径等都交给 agent engine。

7. 注册 artifact 路径。
   - 方便外部查看算法产生的各种文件。

8. 重置历史、队列、最优结果等运行状态。

9. 尝试加载 resume 快照。

10. 写出 workspace 上下文文件，并持久化状态。

简单说，`setup()` 就是“把 agent 做优化前需要的所有环境准备好”。

### `ask`

位置：`general_agent.py:317`

返回下一组要评估的参数配置。

逻辑是：

1. 先检查 `setup()` 是否已经调用。
2. 如果当前历史数量还少于 `initial_random`，就随机采样。
3. 如果候选队列 `_queue` 是空的，就调用 `_fill_queue_from_agent()` 让 agent 生成新候选。
4. 如果队列仍然为空，说明 agent 没有成功产生配置，抛出错误。
5. 从队列头部取出一个候选。
6. 给候选加上元数据，比如 agent 框架、engine、call id、模型、provider。
7. 保存状态。
8. 返回 `TrialSuggestion`。

重点：`ask()` 一次只返回一个候选，即使 agent 一次生成了多个。

### `tell`

位置：`general_agent.py:338`

接收一次实验结果。

它会：

1. 调用 `_ingest_observation()` 把结果写进历史。
2. 更新当前最优结果。
3. 重新写 workspace 文件，让 agent 下一轮能看到最新历史。
4. 保存状态。

简单说，`tell()` 是“告诉算法刚才那组参数跑出来怎么样”。

### `replay`

位置：`general_agent.py:343`

用一批已有历史记录重建算法状态。

常见场景是恢复运行或复现实验。

它会：

1. 清空当前历史、队列、已见配置、当前最优。
2. 逐条吃入传入的历史 observation。
3. 因为是 replay，所以不会重复写优化 trace。
4. 如果开启 `resume`，尝试从状态快照恢复未消费的候选队列。
5. 重写 workspace 上下文。
6. 保存状态。

### `incumbents`

位置：`general_agent.py:355`

返回当前最优结果列表。

如果已经有最优结果，返回 `[self._best]`。

如果还没有任何成功结果，返回空列表 `[]`。

这里用列表是因为有些算法可能支持多个 incumbent，但这个类只维护一个当前最优。

### `_initial_random_suggestion`

位置：`general_agent.py:358`

在早期阶段随机生成候选配置。

用途是：有些优化任务一开始没有历史数据，agent 可能很难判断，所以可以先随机试几次。

逻辑：

1. 从搜索空间随机采样。
2. 用 `stable_config_identity()` 给配置生成稳定 ID。
3. 如果这个配置以前没见过，就返回。
4. 最多尝试 100 次避免重复。
5. 如果 100 次都重复，最后还是采一个返回。

### `_fill_queue_from_agent`

位置：`general_agent.py:378`

这是“调用 agent 生成候选”的核心函数。

它做的是一个重试循环：

1. 写 workspace 上下文，确保 agent 看到最新信息。
2. 生成 `call_id`，例如 `agent_call_00000`。
3. 构造 prompt。
4. 把 prompt 写入 `agent_prompts.jsonl` 日志。
5. 调用 `_run_engine()` 运行外部 agent。
6. 记录 agent 返回状态、回答、日志等。
7. 如果 agent 调用失败，就记录错误并重试。
8. 如果要求可见推理，但没有捕获到推理记录，也算失败。
9. 用 `parse_agent_candidate_payload()` 解析 agent 的 JSON 输出。
10. 用 `_enqueue_candidates()` 把合法、不重复的候选放入队列。
11. 如果有候选入队，就成功返回。
12. 如果所有重试都失败：
    - 若 `allow_fallback=False`，直接报错。
    - 若允许 fallback，就用随机采样生成一个候选兜底。

这可以理解为：它负责“催 agent 交作业，并检查作业是否合格”。

### `_run_engine`

位置：`general_agent.py:463`

真正调用外部 agent engine。

如果 `tool_mode` 是 `function_calling`，它会：

1. 获取工具注册表。
2. 创建工具上下文。
3. 把工具规格传给 agent。
4. 定义 `_execute_tool()`，让 agent 可以调用工具。

然后调用：

```python
self._engine.run_agent(...)
```

因为 `run_agent()` 是异步协程，所以最后用 `_run_coro_sync()` 把它同步执行。

### `_agent_call_env`

位置：`general_agent.py:496`

构造一次 agent 调用需要的环境变量。

包括：

- `BBO_AGENT_CALL_ID`
- `BBO_AGENT_MODEL_REQUESTED`
- `BBO_AGENT_PROVIDER`
- `BBO_AGENT_REQUIRE_VISIBLE_COT`

如果当前框架是 `nanobot`，还会额外设置推理记录目录和元数据路径。

### `_reasoning_metadata_for_call`

位置：`general_agent.py:508`

从 `agent_reasoning_metadata.jsonl` 里读取某次 agent 调用的推理元数据。

它会逐行读 JSONL：

- 跳过空行。
- 跳过无法解析的坏行。
- 找到 `call_id` 匹配的记录。
- 如果同一个 `call_id` 有多条，返回最后一条。

### `_call_has_visible_reasoning`

位置：`general_agent.py:524`

检查某次 agent 调用是否记录了“可见推理”。

它调用 `_reasoning_metadata_for_call()`，然后检查返回记录里的 `reasoning_visible` 是否为真。

### `_build_tool_registry`

位置：`general_agent.py:528`

创建 agent 可以调用的 BBO 工具集合。

默认包含：

- 核心 BBO 工具
- 可选 memory 工具
- 可选 code interpreter 工具
- web search 工具
- fetch URL 工具

同时给工具注册一个 `BBOToolCallLogger`，把工具调用写进日志。

### `_build_tool_context`

位置：`general_agent.py:535`

创建工具执行时需要的上下文。

上下文里包含：

- 当前任务
- 任务描述
- manifest
- workspace 目录
- state 目录
- 历史记录
- 当前最优结果
- memory store
- code backend
- web search provider
- source logger
- 随机种子

你可以把它理解成“工具运行时需要知道的全部背景资料”。

### `_build_code_backend`

位置：`general_agent.py:555`

根据配置创建代码执行后端。

支持：

- `mock`：假的后端，通常用于测试。
- `disabled` / `local_disabled` / `none`：禁用代码执行。
- `sandboxfusion`：使用 SandboxFusion 执行代码。

如果配置了 `sandboxfusion` 但没有提供 base URL，就会自动退回 disabled。

如果传入未知后端名，会抛出 `ValueError`。

### `_build_web_search_provider`

位置：`general_agent.py:568`

根据配置创建网页搜索 provider。

实际创建逻辑交给 `create_BBO_web_search_provider()`。

### `_require_tool_registry`

位置：`general_agent.py:574`

确保工具注册表已经初始化。

如果 `_tool_registry` 还是 `None`，说明 `setup()` 还没正确完成，会抛出 `RuntimeError`。

### `_enqueue_candidates`

位置：`general_agent.py:579`

把解析后的候选配置放进 `_queue`。

它会去重：

1. 给每个配置生成稳定 ID。
2. 如果这个 ID 已经在 `_seen_config_ids` 里，就跳过。
3. 否则加入 `_seen_config_ids`，再放入队列。

返回值是成功入队的候选数量。

### `_fallback_candidate`

位置：`general_agent.py:597`

当 agent 多次失败时，用随机采样生成一个兜底候选。

它最多尝试 500 次避免重复。

如果成功，会返回一个 `AgentCandidateEntry`，metadata 中会标明：

- `agent_source = "fallback_random"`
- `agent_fallback_reason = reason`

如果 500 次都找不到新配置，返回 `None`。

### `_ingest_observation`

位置：`general_agent.py:616`

吸收一次实验结果，也就是把 `tell()` 传入的 observation 写进算法内部状态。

它会：

1. 把 observation 加入 `_history`。
2. 把该配置加入 `_seen_config_ids`，避免以后重复建议。
3. 如果实验成功，并且包含主目标分数，就构造一个 `Incumbent`。
4. 根据优化方向更新 `_best`：
   - 最小化任务：分数更小就更好。
   - 最大化任务：分数更大就更好。
5. 如果不是 replay，还会把当前 step、trial、best 等写进优化 trace。

### `_write_workspace_context`

位置：`general_agent.py:655`

把当前任务状态写入 agent workspace。

它会生成或更新很多文件：

- `task.md`
- `space.json`
- `manifest.json`
- `tool_specs.json`
- `objective.json`
- `incumbent.json`
- `history.jsonl`
- `bbo_tool.py`
- `bbo_tools.py`
- `gp_expected_improvement.py`
- `bbo_workspace_audit.py`
- `python_environment.md`
- `instructions.md`

这一步很关键，因为外部 agent 不是直接读 Python 对象，而是主要通过这些文件理解任务和历史。

### `_write_history_jsonl`

位置：`general_agent.py:693`

把最近的历史 observation 写入 `history.jsonl`。

JSONL 的意思是“一行一个 JSON”。这种格式适合不断追加或逐行读取历史记录。

它写入的是 `_observation_summary()` 生成的简化版本。

### `_write_workspace_tool_bridge`

位置：`general_agent.py:700`

把工具桥接脚本写入 workspace。

主要做三件事：

1. 把 `workspace_tool_cli.py` 的内容复制成 `bbo_tool.py`。
2. 给 `bbo_tool.py` 设置可执行权限。
3. 写入 `bbo_tool_config.json`，里面包括 workspace 路径、日志路径、memory 路径、code backend、web search 配置等。

它还会尽量把配置文件权限设为 `0o600`，减少 API key 等敏感信息泄露风险。

### `_write_workspace_python_api`

位置：`general_agent.py:737`

把 `workspace_python_api.py` 复制到 workspace，文件名是 `bbo_tools.py`。

agent 可以通过：

```python
from bbo_tools import BBO
```

来访问任务上下文、历史、采样、验证、memory、web search 等功能。

### `_write_workspace_gp_example`

位置：`general_agent.py:743`

写入一个高斯过程 / expected improvement 示例脚本。

它会：

1. 创建 `examples/` 目录。
2. 把 `gp_expected_improvement_example.py` 复制为 `examples/gp_expected_improvement.py`。
3. 在 workspace 根目录创建一个入口脚本 `gp_expected_improvement.py`。

这样 agent 可以直接运行：

```bash
python3 gp_expected_improvement.py
```

来参考或改造一个基于历史数据生成候选的示例。

### `_write_workspace_audit_script`

位置：`general_agent.py:777`

生成 `bbo_workspace_audit.py`。

这个脚本的作用是检查 workspace 里的 BBO API 是否能正常用。

它会尝试调用：

- `task_context`
- `manifest`
- `search_space`
- `objective`
- `tool_specs`
- `history`
- `incumbent`
- `sample`
- `analyze_history`
- `memory_write`
- `memory_read`
- `code_interpreter`
- `web_search`
- `fetch_url`
- `validate`

每个调用都会用 `safe_call()` 包起来，所以即使某个工具失败，也会记录错误，而不是整个脚本崩掉。

第一次 agent 调用时，prompt 还会要求 agent 先运行这个 audit 脚本，用来确认工具链可用。

### `_render_python_environment`

位置：`general_agent.py:869`

返回一段 Markdown 文本，说明 workspace Python 环境怎么用。

内容包括：

- 推荐用 `from bbo_tools import BBO`
- 推荐写小 Python 脚本分析历史和验证候选
- 推荐库包括标准库、`numpy`、`scikit-learn`
- SandboxFusion 镜像里建议预装哪些库

### `_render_task_markdown`

位置：`general_agent.py:897`

生成 `task.md` 内容。

如果传入的任务描述 `TaskDescriptionBundle` 已经有 `rendered_context`，就直接返回。

否则生成一个简单兜底内容：

```markdown
# task_name

No structured task description was available.
```

### `_render_instructions`

位置：`general_agent.py:903`

生成给 agent 看的 `instructions.md`。

这份说明告诉 agent：

- 它正在做黑盒优化。
- 不要自己评估 objective。
- 应该读哪些 workspace 文件。
- 输出必须是严格 JSON。
- 每个候选必须包含完整参数。
- 数值不能超出 bounds。
- 分类值必须来自合法 choices。
- 最好使用 BBO 工具检查历史、采样和验证。

这是约束 agent 输出格式的关键文档。

### `_build_agent_prompt`

位置：`general_agent.py:956`

构造传给 agent 的 prompt。

如果 `prompt_style == "direct_json"`，直接转交给 `_build_direct_json_prompt()`。

否则构造 workspace 风格 prompt，告诉 agent：

- 当前任务名
- workspace 路径
- call id
- attempt index
- 要读取哪些文件
- 当前最好分数
- 优化方向
- 最终只能输出严格 JSON

第一次调用时还会加入一个特殊要求：先运行 `python3 bbo_workspace_audit.py`，用于检查工具是否可用。

### `_build_direct_json_prompt`

位置：`general_agent.py:1022`

构造“直接 JSON 上下文”风格的 prompt。

这种模式不让 agent 读文件，也不让它调用工具，而是把任务上下文、搜索空间、当前最优、最近历史直接塞进 prompt 里。

它还专门处理一种特殊搜索空间：成对的坐标参数，比如：

```text
x_0, y_0, x_1, y_1, ...
```

如果检测到这种结构，就允许 agent 用更短的格式返回：

```json
{"candidates": [{"x": [...], "y": [...], "rationale": "short reason"}]}
```

这样可以避免 prompt 和输出被大量 `x_i` / `y_i` 字段撑得太长。

### `_build_framework_config`

位置：`general_agent.py:1112`

根据不同 agent 框架写配置文件。

如果框架是 `nanobot`：

- 创建 `agent_state/config.json`
- 设置 workspace
- 设置 provider
- 设置 model
- 设置 API key 和 API base
- 限制工具只能访问 workspace

如果框架是 `claude_code`：

- 确保 `agent_state/settings.json` 存在。

其他框架暂时不需要写特殊配置，返回 `None`。

### `_agent_env`

位置：`general_agent.py:1154`

构造 agent 进程需要的环境变量。

它会根据 provider 设置不同的 key：

- OpenAI：`OPENAI_API_KEY`
- Anthropic：`ANTHROPIC_API_KEY`
- Google：`GOOGLE_API_KEY`
- 自定义 provider：使用 `api_key_env`

它也会处理：

- OpenAI / Anthropic API base URL
- web search API key
- `SERPAPI_ENDPOINT`

### `_openai_compatible_config`

位置：`general_agent.py:1181`

返回 OpenAI-compatible engine 需要的配置字典：

```python
{
    "api_key": ...,
    "api_base": ...,
    "model": ...,
}
```

### `_claude_config`

位置：`general_agent.py:1188`

返回 Claude Code engine 需要的配置。

它根据 provider 不同设置不同环境变量：

- `claude`：使用 `CLAUDE_CODE_OAUTH_TOKEN`
- `anthropic` 或未设置 provider：使用 `ANTHROPIC_API_KEY`
- 其他 provider：使用 `ANTHROPIC_AUTH_TOKEN`

返回内容包括：

- `env`
- `model`

### `_api_key`

位置：`general_agent.py:1210`

从环境变量读取 API key。

如果没有配置 `api_key_env`，返回 `None`。

否则读取：

```python
os.environ.get(self.config.api_key_env)
```

### `_restore_queue_from_snapshot`

位置：`general_agent.py:1215`

从之前保存的状态快照里恢复候选队列。

只有当：

- `resume=True`
- 已经加载到 `_loaded_resume_snapshot`

才会执行。

恢复时会：

1. 读取快照里的 `queue`。
2. 跳过格式不对的项。
3. 用当前搜索空间重新校验 config。
4. 跳过重复配置。
5. 恢复成 `AgentCandidateEntry`。
6. 恢复 `_call_index`，避免 call id 重复。

### `_load_resume_snapshot`

位置：`general_agent.py:1242`

读取上次保存的 `agent_state.json`。

如果没有开启 `resume`，或者文件不存在，返回空字典。

如果 JSON 读取失败，也返回空字典。

### `_persist_state`

位置：`general_agent.py:1251`

把当前算法状态写入 `agent_state.json`。

保存的内容包括：

- 算法名
- 框架名
- engine 名
- call index
- history size
- queue
- seen config ids
- 当前最好配置和分数
- 模型、provider
- tool 配置
- fallback 配置
- visible CoT 配置

这让实验可以之后恢复或排查。

### `_agent_calls_path`

位置：`general_agent.py:1279`

属性方法，返回 agent 调用日志路径：

```text
run_dir / "agent_calls.jsonl"
```

### `_agent_prompts_path`

位置：`general_agent.py:1284`

属性方法，返回 prompt 日志路径：

```text
run_dir / "agent_prompts.jsonl"
```

### `_agent_state_path`

位置：`general_agent.py:1289`

属性方法，返回状态文件路径：

```text
run_dir / "agent_state.json"
```

### `_agent_optimization_trace_path`

位置：`general_agent.py:1294`

属性方法，返回优化过程 trace 文件路径：

```text
run_dir / "agent_optimization_trace.jsonl"
```

这个文件记录每一步 trial 和当时最优结果。

### `_agent_tool_calls_path`

位置：`general_agent.py:1299`

属性方法，返回工具调用日志路径：

```text
run_dir / "agent_tool_calls.jsonl"
```

### `_agent_tool_specs_path`

位置：`general_agent.py:1304`

属性方法，返回工具规格 JSON 文件路径：

```text
run_dir / "agent_tool_specs.json"
```

### `_agent_sources_path`

位置：`general_agent.py:1309`

属性方法，返回网页来源记录路径：

```text
run_dir / "agent_web_sources.jsonl"
```

当 agent 做 web search 或 fetch URL 时，可以把来源记录在这里。

### `_agent_memory_path`

位置：`general_agent.py:1314`

属性方法，返回 memory 记录路径：

```text
memory_dir / "memory.jsonl"
```

### `_agent_memory_summary_path`

位置：`general_agent.py:1319`

属性方法，返回 memory 摘要路径：

```text
memory_dir / "memory_summary.json"
```

### `_agent_reasoning_traces_dir`

位置：`general_agent.py:1324`

属性方法，返回推理记录目录：

```text
run_dir / "reasoning_traces"
```

### `_agent_reasoning_metadata_path`

位置：`general_agent.py:1329`

属性方法，返回推理元数据日志路径：

```text
run_dir / "agent_reasoning_metadata.jsonl"
```

### `_require_ready`

位置：`general_agent.py:1333`

检查算法是否已经完成 `setup()`。

如果 `_task_spec` 或 `_search_space` 还是 `None`，说明还没初始化，抛出：

```text
GeneralAgentBBOAlgorithm.setup() must be called before use.
```

### `_require_task_spec`

位置：`general_agent.py:1337`

先调用 `_require_ready()`，然后返回当前 `TaskSpec`。

这样写的好处是：调用方不用到处判断 `self._task_spec` 是否为 `None`。

### `_require_search_space`

位置：`general_agent.py:1342`

先调用 `_require_ready()`，然后返回当前 `SearchSpace`。

和 `_require_task_spec()` 类似，这是一个内部保护函数。

## 八、三个快捷子类

这三个类都很薄，只是帮你少写 `framework=...`。

### `NanobotBBOAlgorithm`

位置：`general_agent.py:1348`

```python
class NanobotBBOAlgorithm(GeneralAgentBBOAlgorithm):
```

这是使用 Nanobot 后端的算法类。

#### `__init__`

位置：`general_agent.py:1351`

调用父类初始化，并固定：

```python
framework="nanobot"
algorithm_name="agentic_nanobot"
```

也就是说，你创建 `NanobotBBOAlgorithm(...)` 时，不需要自己传这两个参数。

### `ClaudeCodeBBOAlgorithm`

位置：`general_agent.py:1355`

```python
class ClaudeCodeBBOAlgorithm(GeneralAgentBBOAlgorithm):
```

这是使用 Claude Code 后端的算法类。

#### `__init__`

位置：`general_agent.py:1358`

调用父类初始化，并固定：

```python
framework="claude_code"
algorithm_name="agentic_claude_code"
```

### `OpenAICompatibleBBOAlgorithm`

位置：`general_agent.py:1362`

```python
class OpenAICompatibleBBOAlgorithm(GeneralAgentBBOAlgorithm):
```

这是使用 OpenAI-compatible function calling 后端的算法类。

#### `__init__`

位置：`general_agent.py:1365`

调用父类初始化，并固定：

```python
framework="openai_compatible"
algorithm_name="agentic_openai_compatible"
```

## 九、类外辅助函数

虽然用户主要关心类和类内函数，但这个文件底部的类外函数也很重要，因为它们负责解析 agent 输出、处理特殊坐标格式、总结 observation 和运行异步任务。

### `parse_agent_candidate_payload`

位置：`general_agent.py:1369`

把 agent 返回的原始文本解析成 `ParsedAgentCandidate` 列表。

agent 理想情况下应该返回：

```json
{
  "candidates": [
    {
      "config": {"param_name": "value"},
      "rationale": "short reason"
    }
  ]
}
```

这个函数会：

1. 尝试用 `parse_json_object()` 解析完整 JSON。
2. 如果失败，尝试从文本中抽取包含 `candidates` 的 JSON 对象。
3. 检查顶层只能有一个 key：`candidates`。
4. 检查 `candidates` 必须是非空列表。
5. 逐个候选检查格式。
6. 支持两种候选写法：
   - 标准写法：`{"config": {...}, "rationale": "..."}`
   - 特殊坐标简写：`{"x": [...], "y": [...], "rationale": "..."}`
7. 用搜索空间 `coerce_config()` 校验并规范化配置。
8. 去掉重复配置。
9. 如果没有任何合法候选，抛出 `GeneralAgentValidationError`。

这是保护系统不被 agent 错误输出污染的关键函数。

### `_extract_candidates_json_object`

位置：`general_agent.py:1422`

当 agent 输出不是纯 JSON 时，尝试从一大段文本中找到包含 `candidates` 的 JSON 对象。

它会：

1. 去掉首尾空白。
2. 如果文本为空或以 markdown 代码块开头，直接放弃。
3. 从每个 `{` 开始尝试用 JSON decoder 解码。
4. 如果还不行，就用 `_balanced_json_object_texts()` 找出所有大括号平衡的对象片段。
5. 对这些片段用更宽松的 JSON 加载方式解析。

### `_balanced_json_object_texts`

位置：`general_agent.py:1443`

扫描文本，找出所有“大括号成对匹配”的 JSON 对象片段。

难点是：字符串里的 `{` 和 `}` 不应该算结构括号。

所以这个函数会维护几个状态：

- `depth`：当前大括号嵌套层数。
- `in_string`：当前是否在字符串里。
- `escaped`：当前字符是否被反斜杠转义。

当 `depth` 从 1 回到 0，就说明找到一个完整对象。

### `_loads_lenient_json_object`

位置：`general_agent.py:1478`

用“宽松一点”的方式加载 JSON 对象。

它会依次尝试：

1. 直接 `json.loads()`。
2. 先把字符串里的控制字符转义，再 `json.loads()`。
3. 如果安装了 `json_repair`，用它修复并加载。

如果最后成功得到字典，就返回字典；否则返回 `None`。

### `_escape_control_chars_in_strings`

位置：`general_agent.py:1498`

把 JSON 字符串内部的控制字符转义。

例如字符串里直接出现换行，会被替换成：

```text
\n
```

它同样会维护：

- 是否在字符串中
- 当前字符是否被转义

这样只处理字符串内部的换行、回车、制表符，不会误改 JSON 结构外的内容。

### `search_space_schema`

位置：`general_agent.py:1528`

把 `SearchSpace` 转成 JSON-friendly schema。

它只是简单调用：

```python
search_space_to_schema(search_space)
```

这个函数提供了一个本文件对外暴露的统一入口。

### `_paired_xy_parameter_count`

位置：`general_agent.py:1532`

检测搜索空间里是否存在成对坐标参数：

```text
x_0, y_0, x_1, y_1, ...
```

它从 0 开始数，只要 `x_i` 和 `y_i` 同时存在，就继续。

如果没有任何成对坐标，返回 0。

如果存在连续完整的坐标对，返回坐标对数量。

### `_compact_xy_candidate_to_config`

位置：`general_agent.py:1543`

把 agent 返回的紧凑坐标格式转换成完整 config。

例如 agent 返回：

```json
{"x": [1, 2], "y": [3, 4]}
```

如果搜索空间需要：

```text
x_0, y_0, x_1, y_1
```

它会转换成：

```json
{
  "x_0": 1,
  "x_1": 2,
  "y_0": 3,
  "y_1": 4
}
```

它还会：

- 截断过长数组。
- 如果数组略短，用参数默认值补齐。
- 对数值做边界裁剪。

### `_clip_numeric_to_bounds`

位置：`general_agent.py:1570`

把数值限制在搜索空间允许范围内。

例如参数范围是 `[0, 10]`：

- 输入 `-3`，输出 `0`
- 输入 `15`，输出 `10`
- 输入 `6`，输出 `6`

如果参数没有 `low/high`，或者值不是数字，就原样返回。

### `_default_xy_values`

位置：`general_agent.py:1583`

为缺失的 `x_i` 或 `y_i` 坐标补默认值。

它会调用对应参数的：

```python
effective_default()
```

比如要补 `x_3` 到 `x_5`，就依次读取这些参数的默认值。

### `_compact_xy_arrays`

位置：`general_agent.py:1590`

把完整 config 压缩回：

```python
{
    "x": [...],
    "y": [...]
}
```

它主要用于生成 prompt 上下文，让坐标类任务的历史记录更短、更好读。

### `_compact_xy_incumbent`

位置：`general_agent.py:1597`

把当前最优结果压缩成坐标数组形式。

返回内容包括：

- `x`
- `y`
- `score`
- `objectives`
- `trial_id`

这个函数只用于成对坐标任务的 direct JSON prompt。

### `_compact_xy_observation_summary`

位置：`general_agent.py:1612`

把一次 observation 压缩成坐标数组形式。

包含：

- `trial_id`
- `status`
- `x`
- `y`
- 主目标值 `objective`
- 可选错误类型和错误消息

这可以让 agent 在 prompt 里看到更简洁的历史。

### `_observation_summary`

位置：`general_agent.py:1630`

把普通 observation 转成字典。

包含：

- trial id
- config
- budget
- status
- objectives
- metrics
- elapsed seconds
- error type
- error message
- timestamp

这个函数用于写 `history.jsonl` 和优化 trace。

### `_run_coro_sync`

位置：`general_agent.py:1645`

同步运行一个异步协程。

如果当前线程没有运行中的 event loop，直接：

```python
asyncio.run(coro)
```

如果已经有 event loop，比如在某些 notebook 或异步环境里，就开一个新线程，在新线程里运行 `asyncio.run(coro)`，再把结果或异常带回主线程。

这里面还定义了一个嵌套小函数 `_runner()`。它只在新线程里使用，负责真正运行协程，并把结果放进 `result_box`，或者把异常放进 `error_box`。

这样可以避免常见错误：

```text
asyncio.run() cannot be called from a running event loop
```

### `_nanobot_provider_key`

位置：`general_agent.py:1668`

把通用 provider 名字转换成 Nanobot 配置里使用的 provider key。

映射关系：

| 输入 | 输出 |
|---|---|
| `openai` | `openai` |
| `anthropic` | `anthropic` |
| `google` | `gemini` |
| `ollama` | `ollama` |
| `azure` | `azure_openai` |
| 其他 | `custom` |

## 十、整体运行链路

把所有东西串起来看，一轮优化大概是这样：

```text
外部代码调用 setup()
  -> 创建 workspace、日志、工具、memory、配置文件
  -> 写 task.md / space.json / instructions.md 等

外部代码调用 ask()
  -> 如果需要初始随机，先随机采样
  -> 否则如果队列为空，调用 agent
      -> 构造 prompt
      -> 运行 agent engine
      -> 解析 agent JSON 输出
      -> 校验搜索空间
      -> 去重
      -> 放入队列
  -> 从队列取一个 TrialSuggestion 返回

外部评估器运行这个 TrialSuggestion
  -> 得到 TrialObservation

外部代码调用 tell(observation)
  -> 记录历史
  -> 更新当前最优
  -> 写 trace
  -> 重写 workspace，让 agent 下次看到新历史
```

## 十一、最重要的设计点

### 1. agent 不直接控制实验

agent 只负责提出候选参数，不负责真正评估 objective。

这能避免 agent 私自改结果文件，也让评估逻辑保持可控。

### 2. 所有 agent 输出都要校验

`parse_agent_candidate_payload()` 和 `search_space.coerce_config()` 会检查：

- JSON 格式是否正确
- 是否有 `candidates`
- 候选是否完整
- 参数是否符合搜索空间
- 是否重复

所以即使 agent 输出不稳定，也不会直接污染优化流程。

### 3. agent 可以一次返回多个候选

这些候选会进入 `_queue`。

之后每次 `ask()` 只弹出一个。

这能减少频繁调用 LLM 的成本。

### 4. 有 fallback 机制

如果 agent 连续失败，并且 `allow_fallback=True`，算法会随机采样一个候选继续跑。

这样优化流程不会因为一次 agent 输出坏 JSON 就完全中断。

### 5. workspace 是 agent 和算法之间的桥

算法把当前世界写成文件，agent 通过读文件理解任务，再输出 JSON。

这种设计让不同 agent 后端更容易接入，因为它们不一定都能直接访问 Python 对象。
