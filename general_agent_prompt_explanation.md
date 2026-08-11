# `general_agent.py` 里的 agent prompt 是什么样的

源文件：`/home/trx/cm/agentic-bbo/bbo/algorithms/agentic/general_agent.py`

这份说明只讲一件事：这个算法具体给 agent 发了什么 prompt，这个 prompt 是怎么一步步构造出来的，以及为什么这么设计。

## 一句话总览

`general_agent.py` 不是一直把所有任务信息都塞进一条超长 prompt。

默认情况下，它采用的是：

```text
把详细上下文写进 agent_workspace 里的多个文件
  +
给 agent 一条较短 prompt，要求它去读这些文件
  +
要求最终只输出严格 JSON 候选配置
```

所以真正发给 agent 的 prompt，默认更像一个“任务入口说明”：

```text
你是某个任务的优化 agent。
workspace 在哪里。
这是第几次调用。
请读哪些文件。
请检查搜索空间、历史、当前最优，并验证候选。
最后只能输出 instructions.md 里规定的 JSON。
```

## Prompt 是在哪里构造的

关键函数有 3 个：

| 函数 | 作用 |
|---|---|
| `_build_agent_prompt()` | 构造默认 workspace 风格 prompt |
| `_build_direct_json_prompt()` | 构造 direct JSON 风格 prompt |
| `_render_instructions()` | 生成 workspace 里的 `instructions.md`，默认 prompt 会要求 agent 阅读它 |

调用链路是：

```text
ask()
  -> 如果候选队列为空
  -> _fill_queue_from_agent()
      -> _write_workspace_context()
      -> call_id = "agent_call_00000" 这种形式
      -> prompt = _build_agent_prompt(call_id, attempt_index)
      -> 把 prompt 写入 agent_prompts.jsonl
      -> _run_engine(prompt, call_id)
          -> self._engine.run_agent("", prompt, ...)
```

注意最后一行：

```python
self._engine.run_agent("", prompt, ...)
```

这里传给 engine 的第一个参数 `session_id` 是空字符串，第二个参数 `prompt` 才是真正的消息内容。

不同后端拿到这条消息的方式略有不同：

- Nanobot：作为 CLI 参数 `-m message` 传进去。
- Claude Code：作为 `query(prompt=message, options=opts)` 的 prompt。
- OpenAI-compatible：作为一条 `{"role": "user", "content": message}` 消息。

也就是说，这里没有单独的 system prompt；主要指令都在这个 user prompt 和 workspace 文件里。

## 默认 prompt 风格：`workspace`

默认配置是：

```python
prompt_style="workspace"
```

这种模式下，`_build_agent_prompt()` 会返回下面这个模板。

### 普通调用时的 prompt 模板

```text
You are an optimization agent for task `{task_spec.name}`.
Workspace path: {self._workspace_dir}
Call id: {call_id}
Attempt: {attempt_index}

Read `instructions.md`, `task.md`, `manifest.json`, `space.json`,
`objective.json`, `history.jsonl`, `incumbent.json`, `tool_specs.json`,
`python_environment.md`, `bbo_workspace_audit.py`, `gp_expected_improvement.py`, and
`examples/gp_expected_improvement.py`
in the workspace.

If native function-calling tools are unavailable, use the workspace Python
API: `from bbo_tools import BBO`. At minimum, inspect search space,
history, incumbent, and validate your final candidate list. When enough
history exists, run `python3 gp_expected_improvement.py` from the workspace
or adapt the GP example script before proposing candidates. Use relative
paths only, and recover to validated sampling if any command fails.

Current best primary objective: {best_score}
Objective direction: {task_spec.primary_objective.direction.value}
{audit_instruction}

Produce candidate configurations now. Your entire stdout must be the
strict JSON object described in `instructions.md`.
```

其中花括号里的内容会在运行时替换：

| 占位符 | 实际来源 | 含义 |
|---|---|---|
| `{task_spec.name}` | 当前 `TaskSpec` | 任务名 |
| `{self._workspace_dir}` | `setup()` 创建的 workspace 目录 | agent 要读文件的位置 |
| `{call_id}` | `_fill_queue_from_agent()` 生成 | 例如 `agent_call_00000` |
| `{attempt_index}` | retry 循环下标 | 第几次尝试，第一次是 0 |
| `{best_score}` | `self._best.score` | 当前最好主目标分数，没有则是 `None` |
| `{direction}` | `task_spec.primary_objective.direction.value` | `minimize` 或 `maximize` |
| `{audit_instruction}` | 首次调用时额外加入 | 要求先跑 workspace audit |

### 第一次调用会多一段 audit prompt

当满足：

```python
call_id == "agent_call_00000" and attempt_index == 0
```

prompt 中的 `{audit_instruction}` 会变成下面这段：

```text
Tool audit requirement for this first agent call:
Your first tool action must execute `python3 bbo_workspace_audit.py`
from this workspace. This requirement overrides file-reading order:
run the audit before any manual `read_file` calls and before final
candidate JSON. If your local runtime emits XML-style tool calls,
emit exactly this `exec` call first:

<tool_call>
<function=exec>
<parameter=command>
python3 bbo_workspace_audit.py
</parameter>
</function>
</tool_call>

The audit imports `BBO` from `bbo_tools` and calls every available
workspace BBO tool/API at least once:
`task_context`, `manifest`, `search_space`, `objective`,
`tool_specs`, `history`, `incumbent`, `sample`, `analyze_history`,
`memory_write`, `memory_read`, `code_interpreter`, `web_search`,
`fetch_url`, and `validate`. After the audit, inspect the workspace
files as needed and validate the final candidates. This audit is for
logging/observability; still return only the strict final candidates
JSON on stdout.
```

通俗理解：第一次调用时，代码强制 agent 先跑一次体检脚本，确认 workspace 里的工具能不能用。这个 audit 的目的不是优化本身，而是让日志里能看到工具是否可用。

## `instructions.md` 也是 prompt 的一部分

默认 workspace prompt 很短，因为很多具体规则被写进了 workspace 里的 `instructions.md`。

这个文件由 `_render_instructions()` 生成，主要内容是：

```text
# Agentic BBO Candidate Protocol

You are proposing configurations for a black-box optimization benchmark.
Do not evaluate the objective yourself and do not modify benchmark result files.

Files in this workspace:
- task.md: task background, goal, constraints, and prior knowledge.
- manifest.json: agent benchmark construction, tool policy, and provenance.
- space.json: exact parameter schema. Every candidate must include every parameter exactly once.
- objective.json: primary objective name and optimization direction.
- history.jsonl: recent evaluated trials.
- incumbent.json: current best known configuration, if any.
- tool_specs.json: available BBO function-calling tools when the backend supports tools.
- bbo_tools.py: preferred Python API for BBO tools when using shell/file tools.
- bbo_workspace_audit.py: first-call observability script that exercises all workspace BBO APIs.
- gp_expected_improvement.py: root-level relative entrypoint for the GP/LCB example.
- examples/gp_expected_improvement.py: editable GP/LCB candidate-generation example.
- python_environment.md: Python and sandbox library guidance.
- bbo_tool.py: lower-level CLI bridge for BBO tools; use only as a fallback.

Task: {task_spec.name}
Primary objective: {task_spec.primary_objective.name}
Direction: {task_spec.primary_objective.direction.value}

Preferred Python workflow:
- Write and run small Python scripts in this workspace.
- Use relative paths from the workspace. Do not execute absolute paths.
- Import the BBO API with `from bbo_tools import BBO`.
- Use `BBO().search_space()`, `BBO().history(...)`, `BBO().incumbent()`,
  and `BBO().validate(...)` before emitting final candidates.
- When enough history exists, run `python3 gp_expected_improvement.py` or adapt
  `examples/gp_expected_improvement.py`
  to fit a GP-style surrogate and propose candidates.
- Tool/API calls are append-only logged to agent_tool_calls.jsonl.

Print only raw JSON to stdout, with this exact shape:
{"candidates": [{"config": {"param_name": "value"}, "rationale": "short reason"}]}

Requirements:
- Return {self.config.candidates_per_call} candidate configurations when possible.
- Use BBO tools for task context, history, memory, validation, code analysis, or web research when available.
- Validate proposed candidates with `BBO().validate(...)` before final output.
- If any script or command fails, recover with `BBO().sample(...)` and `BBO().validate(...)`; never return an error message as the final answer.
- Do not include markdown fences, comments, prose, or partial configurations.
- Float and integer values must stay within their declared bounds.
- Categorical values must be one of the declared choices.
```

这份文件非常关键，因为最后一行 prompt 说：

```text
Your entire stdout must be the strict JSON object described in `instructions.md`.
```

所以默认模式下，真正约束输出格式的详细规则主要在 `instructions.md` 里。

## workspace 里还会写哪些上下文

在构造 prompt 之前，`_fill_queue_from_agent()` 每次都会先调用：

```python
self._write_workspace_context()
```

这个函数会把当前最新状态写进 workspace。主要文件包括：

| 文件 | 内容 |
|---|---|
| `task.md` | 任务背景、目标、约束、先验知识 |
| `space.json` | 参数搜索空间，每个参数的类型、范围、choices |
| `manifest.json` | benchmark 构造和 provenance |
| `objective.json` | 主目标名、优化方向、全部目标 |
| `incumbent.json` | 当前最优配置和分数 |
| `history.jsonl` | 最近若干条实验历史 |
| `tool_specs.json` | 可用工具说明 |
| `bbo_tools.py` | workspace Python API |
| `bbo_tool.py` | CLI 工具桥 |
| `gp_expected_improvement.py` | GP/LCB 示例入口 |
| `examples/gp_expected_improvement.py` | 可修改的候选生成示例 |
| `bbo_workspace_audit.py` | 工具体检脚本 |
| `python_environment.md` | Python 环境说明 |
| `instructions.md` | 最终输出协议和要求 |

也就是说，默认 prompt 本身不包含完整搜索空间和完整历史；它让 agent 去 workspace 里读。

## direct JSON prompt 风格：`direct_json`

如果配置成：

```python
prompt_style="direct_json"
```

那么 `_build_agent_prompt()` 不再返回 workspace prompt，而是直接调用：

```python
self._build_direct_json_prompt(call_id=call_id, attempt_index=attempt_index)
```

这种模式下，prompt 会把上下文直接塞进一个 JSON 块里，并明确禁止 agent 读文件或调用工具。

### direct JSON prompt 模板

```text
You are proposing configurations for a black-box optimization benchmark.

Use only the JSON context embedded below. Do not call tools. Do not emit
XML, HTML, markdown, code fences, `<tool_call>` tags, or prose. Do not try
to read files. The local runtime may expose tools, but this task requires
a direct final answer.

Optimize the primary objective according to its direction. Use the
search-space bounds exactly. Every candidate config must include every
parameter exactly once, avoid duplicate configurations when possible, and
use numeric values for numeric parameters.

Return only one raw JSON object with this exact shape:
{response_shape}
{compact_note}

Return exactly {self.config.candidates_per_call} candidate configurations when possible.

BBO_CONTEXT_JSON:
{context_json}
```

这里的 `{context_json}` 是代码现场拼出来的，结构大概是：

```json
{
  "task": {
    "name": "任务名",
    "primary_objective": "主目标名",
    "direction": "minimize 或 maximize",
    "candidates_per_call": 4,
    "call_id": "agent_call_00000",
    "attempt_index": 0
  },
  "task_context": "任务描述文本",
  "search_space": {
    "parameters": [
      {
        "name": "参数名",
        "type": "float/int/categorical",
        "low": 0,
        "high": 1
      }
    ]
  },
  "incumbent": null,
  "recent_history": [],
  "seen_config_count": 0
}
```

真实运行时，`parameters`、`incumbent`、`recent_history` 会根据当前任务和历史填充。

### direct JSON 的输出格式

普通搜索空间下，`response_shape` 是：

```json
{"candidates": [{"config": {"param_name": "value"}, "rationale": "short reason"}]}
```

如果检测到搜索空间是成对坐标形式：

```text
x_0, y_0, x_1, y_1, ...
```

则允许更紧凑的输出：

```json
{"candidates": [{"x": [... exactly N numbers ...], "y": [... exactly N numbers ...], "rationale": "short reason"}]}
```

对应的额外说明是：

```text
This task has paired macro-placement coordinates. Prefer the compact
response shape above: each candidate may contain `x` and `y` arrays
instead of spelling out every `x_i`/`y_i` key. Each array must contain
exactly N numbers; `x[k]` maps to `x_k`, and `y[k]`
maps to `y_k`.
Do not return more than N x values or more than
N y values. Do not return an all-default coordinate
vector such as every value being 112.0. Return a new pattern that is
not identical to the incumbent or recent history.
```

这里的 `N` 是 `_paired_xy_parameter_count(search_space)` 算出来的坐标对数量。

## direct JSON 的上下文是怎么构造的

`_build_direct_json_prompt()` 会收集这些信息：

1. `task_spec`
   - 任务名
   - 主目标名
   - 优化方向
   - 搜索空间

2. `history`
   - 最近 `history_limit` 条 observation。
   - 如果 `history_limit=0`，则不给历史。

3. `task_context`
   - 来自 `_render_task_markdown()`。
   - 普通任务最多保留 12000 字符。
   - 成对坐标任务最多保留 6000 字符。

4. `incumbent`
   - 当前最好结果。
   - 普通任务保留完整 config。
   - 成对坐标任务压缩成 `x` / `y` 数组。

5. `recent_history`
   - 普通任务使用 `_observation_summary()`。
   - 成对坐标任务使用 `_compact_xy_observation_summary()`。

6. `seen_config_count`
   - 已经见过多少个配置。

然后用：

```python
json.dumps(to_jsonable(context), ensure_ascii=False, sort_keys=True, indent=2)
```

把上下文格式化为漂亮的 JSON 放进 prompt。

## 两种 prompt 风格的区别

| 对比项 | `workspace` 默认风格 | `direct_json` 风格 |
|---|---|---|
| 信息来源 | workspace 文件 | prompt 内嵌 JSON |
| 是否鼓励读文件 | 是 | 否，明确禁止 |
| 是否鼓励用工具 | 是，有工具就用 | 否，明确禁止 |
| prompt 长度 | 相对短 | 可能较长 |
| 适合场景 | agent 能操作文件、运行脚本、调用工具 | 想要简单直接、只让模型输出 JSON |
| 历史和搜索空间 | 写在文件里 | 直接塞进 prompt |
| 可调试性 | 高，文件和工具日志都在 workspace | 高，完整上下文在 `agent_prompts.jsonl` |

## `tool_mode` 和 `prompt_style` 不是一回事

这里容易混淆。

### `prompt_style`

控制 prompt 长什么样：

- `workspace`：让 agent 去读 workspace 文件。
- `direct_json`：把 JSON 上下文直接放进 prompt。

### `tool_mode`

控制是否给支持 function calling 的 engine 注入工具：

- `function_calling`：构造工具列表和工具执行器，传给 engine。
- `workspace_json`：不注入 function-calling 工具，但 workspace 里的 `bbo_tools.py` / `bbo_tool.py` 仍然存在。

所以可能出现这些组合：

| 组合 | 含义 |
|---|---|
| `prompt_style=workspace`, `tool_mode=function_calling` | prompt 让 agent 读文件，同时支持原生 function calling 工具 |
| `prompt_style=workspace`, `tool_mode=workspace_json` | prompt 让 agent 读文件，但工具主要通过 workspace API/CLI 使用 |
| `prompt_style=direct_json`, `tool_mode=function_calling` | prompt 禁止用工具，但 engine 代码层可能仍传工具；prompt 指令要求不要调用 |
| `prompt_style=direct_json`, `tool_mode=workspace_json` | 最纯粹的“只看 JSON，上来就回答”模式 |

从源码意图看，`direct_json` 更适合不想让 agent 自己探索 workspace 的情况。

## 为什么默认要这么设计

### 1. 避免 prompt 过长

搜索空间、历史、manifest、工具说明都可能很长。

如果每次都塞进一条 prompt，token 成本会很高，也更容易截断。

默认模式把这些内容拆成文件，只在 prompt 里告诉 agent 去读哪些文件。这样更像真实 coding agent 的工作方式。

### 2. 让 agent 看到最新状态

每次调用 agent 前都会执行：

```python
self._write_workspace_context()
```

所以 `history.jsonl`、`incumbent.json`、`objective.json` 等文件都是最新的。

这很重要，因为 `tell()` 之后当前最优和历史会变化，下一轮 agent 应该基于最新结果提候选。

### 3. 把任务协议写清楚

`instructions.md` 明确告诉 agent：

- 不要自己评估 objective。
- 不要修改 benchmark 结果文件。
- 每个候选必须包含完整参数。
- 最终只输出 raw JSON。
- 不能输出 markdown 代码块。
- 数值必须在范围内。
- 分类值必须合法。

这能降低 agent 输出“解释文字”“半截 JSON”“参数缺失”等问题的概率。

### 4. 给 agent 工具，但要求先验证

prompt 和 `instructions.md` 都强调：

```text
validate your final candidate list
```

也就是让 agent 在最终输出前先用 `BBO().validate(...)` 检查候选是否合法。

但最终代码不会完全相信 agent。agent 输出后，Python 侧还会用：

```python
parse_agent_candidate_payload(result.answer, search_space)
```

再校验一次。

这是双保险：

```text
agent 自己先检查
  +
算法代码再检查
```

### 5. 第一次 audit 是为了可观测性

第一次 prompt 强制运行 `bbo_workspace_audit.py`，主要是为了确认：

- workspace Python API 是否可导入。
- search space 能不能读。
- history / incumbent 能不能读。
- sample / validate 能不能用。
- memory、code interpreter、web search、fetch URL 是否可用。

这些结果会写进工具调用日志，方便后面排查“agent 为什么没用工具”或“工具是不是坏了”。

### 6. direct JSON 是给简单后端准备的

有些后端不适合像 coding agent 一样读文件、跑脚本、调用工具。

`direct_json` 模式就把上下文直接放进 prompt，告诉模型：

```text
不要读文件，不要调用工具，只看下面 JSON，直接返回 JSON。
```

这种模式更封闭、更可控，但也牺牲了 workspace 探索能力。

### 7. 成对坐标压缩是为了节省 token

有些任务有很多参数：

```text
x_0, y_0, x_1, y_1, ... x_99, y_99
```

如果每个候选都完整写所有键，会非常长。

所以 direct JSON 模式下允许 agent 返回：

```json
{"x": [...], "y": [...]}
```

后续代码再用 `_compact_xy_candidate_to_config()` 转回完整配置。

这能显著减少 prompt 和输出长度。

## 最终 agent 必须返回什么

无论哪种 prompt 风格，最终都希望 agent 返回一个顶层只有 `candidates` 的 JSON 对象。

普通格式：

```json
{
  "candidates": [
    {
      "config": {
        "param_a": 1,
        "param_b": "choice"
      },
      "rationale": "short reason"
    }
  ]
}
```

成对坐标任务在 direct JSON 模式下还可以返回：

```json
{
  "candidates": [
    {
      "x": [1.0, 2.0, 3.0],
      "y": [4.0, 5.0, 6.0],
      "rationale": "short reason"
    }
  ]
}
```

然后代码会执行：

```text
parse_agent_candidate_payload()
  -> 解析 JSON
  -> 检查只有 candidates 这个顶层 key
  -> 检查 candidates 是非空列表
  -> 转换紧凑 x/y 格式
  -> search_space.coerce_config()
  -> 去重
  -> 放进 ask() 的候选队列
```

## 怎么查看真实运行时的 prompt

每次调用 agent 前，代码都会把 prompt 写入：

```text
{run_dir}/agent_prompts.jsonl
```

每行是一个 JSON，形如：

```json
{
  "call_id": "agent_call_00000",
  "attempt_index": 0,
  "prompt": "真正发给 agent 的 prompt 字符串",
  "timestamp": 1234567890.0
}
```

所以如果你想看某次实验里 agent 真的收到了什么，不需要猜源码模板，直接看对应 run 目录下的 `agent_prompts.jsonl` 就行。

## 最后用很白话的话总结

默认模式下，代码不是对 agent 说一大堆完整背景，而是先给 agent 准备一个工作文件夹，然后发一条类似这样的消息：

```text
你现在是优化 agent。
任务叫这个，workspace 在这里。
请读 instructions.md、task.md、space.json、history.jsonl、incumbent.json 等文件。
如果能用工具，就用工具检查搜索空间、历史和候选是否合法。
如果历史够多，可以参考 GP 示例脚本。
最终不要废话，只输出 instructions.md 要求的 JSON。
```

这样设计的核心原因是：

```text
让 agent 像 coding agent 一样工作，
把长上下文放文件里，
把短指令放 prompt 里，
把最终输出限制成机器可解析的 JSON，
再由 Python 代码做二次校验。
```
