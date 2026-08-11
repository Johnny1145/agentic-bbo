# `general_agent.py` 里的 workspace tools 说明

源文件相关位置：

- `/home/trx/cm/agentic-bbo/bbo/algorithms/agentic/general_agent.py`
- `/home/trx/cm/agentic-bbo/bbo/algorithms/agentic/tools/`
- `/home/trx/cm/agentic-bbo/bbo/algorithms/agentic/workspace_python_api.py`
- `/home/trx/cm/agentic-bbo/bbo/algorithms/agentic/workspace_tool_cli.py`

这份说明回答两个问题：

1. 现在 workspace 里给 agent 的 tools 有哪些？
2. agent 支不支持自己构建新的 tool？

## 一句话结论

当前系统有两层工具：

```text
原生 function-calling tools
  -> 由 Python 代码里的 BBOToolRegistry 注册，给支持 function calling 的模型用

workspace tools
  -> 复制到 agent_workspace 里的 bbo_tools.py / bbo_tool.py，给能读写文件、运行脚本的 agent 用
```

agent 可以自己写 Python 脚本、分析脚本、候选生成脚本，这一点是支持的。

但 agent 不能在运行中把自己写的脚本注册成新的原生 BBO tool。真正新增一个受框架管理、出现在 `tool_specs.json` 里的 tool，需要改源码。

## 工具是在哪里创建的

在 `GeneralAgentBBOAlgorithm._build_tool_registry()` 里：

```python
def _build_tool_registry(self) -> BBOToolRegistry:
    tools = create_core_BBO_tools(enable_memory=self.config.enable_memory)
    if self.config.enable_code_interpreter:
        tools.append(CodeInterpreterTool())
    tools.extend([WebSearchTool(), FetchURLTool()])
    return BBOToolRegistry(tools, logger=BBOToolCallLogger(self._agent_tool_calls_path))
```

通俗解释：

1. 先创建一批核心 BBO 工具。
2. 如果开启 memory，就加 memory 工具。
3. 如果开启 code interpreter，就加代码执行工具。
4. 总是加入 web search 和 fetch URL 工具。
5. 用 `BBOToolRegistry` 注册这些工具。
6. 工具调用会写进 `agent_tool_calls.jsonl` 日志。

## 原生 function-calling tools 有哪些

这些工具来自 `bbo/algorithms/agentic/tools/`。

如果当前 engine 支持原生 function calling，比如 `OpenAICompatibleToolEngine`，这些工具会作为模型的 tools 传进去。

### 核心工具

| 工具名 | 作用 |
|---|---|
| `get_task_context` | 读取任务文档、manifest、目标信息、约束 |
| `get_search_space` | 读取搜索空间 schema、默认值、参数顺序 |
| `get_trial_history` | 读取已经评估过的 trial 历史 |
| `get_incumbent` | 读取当前最好配置和分数 |
| `get_history_overview` | 汇总 incumbent、best-so-far、recent objective 和 search action |
| `summarize_objective_metrics` | 汇总 objective 进展、recent delta 和 agent 可见 numeric metrics |
| `compare_trials` | 精确比较 trial 的目标值、变量差异和 action metadata |
| `find_nearest_trials` | 按搜索空间归一化距离找最近历史 trial |
| `estimate_local_effects` | 用局部可比历史估计变量效果 |
| `measure_search_coverage` | 汇总搜索覆盖、未覆盖区域和近期距离 |
| `fit_and_check_surrogate` | 拟合并验证简单 surrogate，不消耗 evaluator budget |
| `score_virtual_candidates` | 用已验证 surrogate 给虚拟候选打分 |
| `validate_candidate` | 校验单个候选并报告 bounds、重复、repair 信息 |
| `validate_candidates` | 检查候选配置是否合法、是否重复 |
| `get_recent_search_actions` | 读取近期 search_action metadata |
| `sample_candidates` | 从搜索空间采样合法候选，不消耗 objective budget |
| `analyze_history` | 对历史结果做轻量统计分析 |

### memory 工具

只有 `enable_memory=True` 时才会注册。

| 工具名 | 作用 |
|---|---|
| `memory_read` | 读取 agent 之前写过的记忆 |
| `memory_write` | 写入假设、经验、失败记录、策略笔记 |

### code interpreter 工具

只有 `enable_code_interpreter=True` 时才会注册。

| 工具名 | 作用 |
|---|---|
| `code_interpreter` | 在配置好的 sandbox 后端里运行分析代码，不消耗 trial budget |

这个工具默认后端配置是 `sandboxfusion`，但如果没有设置 `SANDBOX_FUSION_BASE_URL` 或 `sandbox_fusion_base_url`，实际会退化成 disabled backend，也就是能返回“代码执行未启用”的明确结果。

### web 工具

| 工具名 | 作用 |
|---|---|
| `web_search` | 搜索公开网页资料，并记录 source metadata |
| `fetch_url` | 抓取 manifest 允许的 URL，并记录 source metadata |

注意：`web_search` 是否真的能搜，取决于 `web_search_provider`。默认是 `disabled`，也就是工具名存在，但搜索能力可能不可用。

## workspace 里实际会出现哪些工具文件

每次 agent 调用前，`_write_workspace_context()` 会重写 workspace 上下文，其中包括工具相关文件。

主要有：

| 文件 | 作用 |
|---|---|
| `tool_specs.json` | 当前可用工具的规格说明 |
| `bbo_tools.py` | 给 agent 用的 Python API |
| `bbo_tool.py` | 给 agent 用的 CLI 工具桥 |
| `bbo_tool_config.json` | workspace 工具配置，比如路径、日志、memory、web search、code backend |
| `bbo_workspace_audit.py` | 第一次调用时用于检查所有 workspace 工具是否可用 |

## `bbo_tools.py` 暴露的方法

`workspace_python_api.py` 会被复制到 workspace，文件名叫 `bbo_tools.py`。

agent 可以这样用：

```python
from bbo_tools import BBO

bbo = BBO()
space = bbo.search_space()
history = bbo.history(limit=20)
```

`BBO` 类提供这些方法：

| 方法 | 底层 tool 名 | 作用 |
|---|---|---|
| `task_context()` | `get_task_context` | 读任务上下文 |
| `manifest()` | `get_manifest` | 读 manifest |
| `search_space()` | `get_search_space` | 读搜索空间 |
| `objective()` | `get_objective` | 读目标定义 |
| `tool_specs()` | `get_tool_specs` | 读工具规格 |
| `history()` | `get_trial_history` | 读 trial 历史 |
| `incumbent()` | `get_incumbent` | 读当前最好结果 |
| `history_overview()` | `get_history_overview` | 读紧凑历史概览 |
| `compare_trials(...)` | `compare_trials` | 精确比较 trial |
| `find_nearest_trials(...)` | `find_nearest_trials` | 找最近历史 trial |
| `estimate_local_effects(...)` | `estimate_local_effects` | 估计局部变量效果 |
| `measure_search_coverage()` | `measure_search_coverage` | 看搜索覆盖 |
| `summarize_objective_metrics()` | `summarize_objective_metrics` | 汇总 objective/metric 进展 |
| `fit_and_check_surrogate()` | `fit_and_check_surrogate` | 验证 surrogate 信号 |
| `score_virtual_candidates(...)` | `score_virtual_candidates` | 给虚拟候选打分 |
| `validate_candidate(...)` | `validate_candidate` | 校验单个候选 |
| `validate(candidates)` | `validate_candidates` | 校验候选配置 |
| `recent_search_actions()` | `get_recent_search_actions` | 读近期 search action |
| `sample(...)` | `sample_candidates` | 随机采样或围绕 incumbent 采样 |
| `analyze_history()` | `analyze_history` | 分析历史统计 |
| `memory_read()` | `memory_read` | 读 memory |
| `memory_write()` | `memory_write` | 写 memory |
| `code_interpreter()` | `code_interpreter` | 调代码解释器 |
| `web_search()` | `web_search` | 搜索网页 |
| `fetch_url()` | `fetch_url` | 抓取 URL |

这就是 workspace 里最推荐 agent 使用的工具入口。

## `bbo_tool.py` CLI 支持哪些 tool 名

`workspace_tool_cli.py` 会被复制为 workspace 里的 `bbo_tool.py`。

agent 也可以直接用命令行调用：

```bash
python3 bbo_tool.py get_search_space '{}'
python3 bbo_tool.py get_trial_history '{"mode": "recent", "limit": 20}'
python3 bbo_tool.py validate_candidates '{"candidates": [{"config": {...}}]}'
```

CLI 支持这些 tool 名：

| tool 名 | 作用 |
|---|---|
| `get_task_context` | 读任务上下文 |
| `get_manifest` | 读 manifest |
| `get_search_space` | 读搜索空间 |
| `get_objective` | 读目标 |
| `get_tool_specs` | 读工具规格 |
| `get_trial_history` | 读 trial 历史 |
| `get_incumbent` | 读当前最好结果 |
| `get_history_overview` | 读紧凑历史概览 |
| `compare_trials` | 精确比较 trial |
| `find_nearest_trials` | 找最近历史 trial |
| `estimate_local_effects` | 估计局部变量效果 |
| `measure_search_coverage` | 看搜索覆盖 |
| `summarize_objective_metrics` | 汇总 objective/metric 进展 |
| `fit_and_check_surrogate` | 验证 surrogate 信号 |
| `score_virtual_candidates` | 给虚拟候选打分 |
| `validate_candidate` | 校验单个候选 |
| `validate_candidates` | 校验候选 |
| `get_recent_search_actions` | 读近期 search action |
| `sample_candidates` | 采样候选 |
| `analyze_history` | 分析历史 |
| `memory_read` | 读 memory |
| `memory_write` | 写 memory |
| `code_interpreter` | 调代码解释器 |
| `web_search` | 搜索网页 |
| `fetch_url` | 抓 URL |

CLI 还支持几个别名：

| 别名 | 等价于 |
|---|---|
| `get_history` | `get_trial_history` |
| `get_space` | `get_search_space` |
| `get_objective` | `get_objective` |
| `get_tool_specs` | `get_tool_specs` |
| `get_manifest` | `get_manifest` |

## 原生 tools 和 workspace tools 的区别

| 对比项 | 原生 function-calling tools | workspace tools |
|---|---|---|
| 使用入口 | 模型直接 function call | agent 运行 `bbo_tools.py` 或 `bbo_tool.py` |
| 注册位置 | `BBOToolRegistry` | `workspace_tool_cli.py` 的 handler 表 |
| 规格来源 | `BaseBBOTool.function_spec()` | `tool_specs.json` + CLI/API 文档 |
| 日志 | `agent_tool_calls.jsonl` | 同样写入 `agent_tool_calls.jsonl` |
| 适合对象 | OpenAI-compatible 这类支持 tools 的后端 | Nanobot、Claude Code、shell/file agent |
| 是否动态扩展 | 不支持运行中动态扩展 | 可以写脚本，但不是注册成 BBO tool |

## agent 支不支持自己构建工具

答案要分两层。

### 支持：agent 可以自己写辅助脚本

这是支持的，而且 prompt 里明确鼓励：

```text
Write and run small Python scripts in this workspace.
```

agent 可以在 workspace 里写自己的文件，比如：

```text
my_analysis.py
my_surrogate_model.py
candidate_generator.py
```

这些脚本可以：

- import `BBO`
- 读取搜索空间
- 读取历史
- 做统计分析
- 拟合简单 surrogate model
- 调用 `bbo.validate(...)` 校验候选
- 最后帮助 agent 生成 JSON 候选

例如：

```python
from bbo_tools import BBO

bbo = BBO()
space = bbo.search_space()
history = bbo.history(mode="best", limit=20)
sample = bbo.sample(n=16, strategy="around_incumbent")
valid = bbo.validate([item["config"] for item in sample["candidates"]])

print(valid)
```

这种“自建工具”本质上是 agent 自己写的分析程序，不是框架级注册工具。

### 不支持：agent 不能运行中注册新的原生 BBO tool

如果你说的“构建工具”是指：

```text
新增一个 tool 名
让它出现在 tool_specs.json
让模型可以原生 function call 它
让 BBOToolRegistry 管理它
让它自动进入工具日志和上下文
```

那当前代码不支持 agent 在运行中自己完成。

原因是：

1. 原生工具列表在 `setup()` 时由 `_build_tool_registry()` 创建。
2. 注册表 `BBOToolRegistry` 只接收 Python 侧传入的 `BaseBBOTool` 实例。
3. 每次给模型的 tools 来自 `registry.get_tool_specs()`。
4. workspace 里的 `tool_specs.json` 是由 Python 主程序写出来的，不是 agent 自由声明的。
5. `bbo_tool.py` 的 CLI handler 表也是源码固定的。

所以 agent 自己写一个 `my_tool.py`，不会自动变成：

```text
tool_specs.json 里的新 tool
OpenAI function-calling 的新 function
BBOToolRegistry 里的新 handler
```

## agent 能不能修改 `bbo_tool.py` 来“伪造”工具

技术上，如果 agent 有文件写权限，它可能能在当前 workspace 里改 `bbo_tool.py` 或写另一个脚本。

但这不算受支持的扩展方式，原因是：

1. `_write_workspace_context()` 每次 agent 调用前都会重新写 `bbo_tool.py`、`bbo_tools.py`、`tool_specs.json` 等文件。
2. 改出来的新工具不会进入 Python 主程序里的 `BBOToolRegistry`。
3. 原生 function-calling 模型不会知道这个新工具。
4. 日志、权限、manifest policy、参数 schema 都不会自动接入。
5. 这容易让实验不可复现。

推荐做法是：agent 可以写自己的 `my_analysis.py`，但不要修改系统生成的 `bbo_tool.py` 和 `bbo_tools.py`。

## 如果真的想新增框架级工具，该怎么做

需要改源码，而不是让 agent 运行中自己注册。

大致步骤是：

1. 在 `bbo/algorithms/agentic/tools/` 下新增一个继承 `BaseBBOTool` 的类。

```python
class MyTool(BaseBBOTool):
    name = "my_tool"
    description = "..."
    parameters_schema = {
        "type": "object",
        "properties": {
            "arg": {"type": "string"}
        },
        "required": ["arg"],
    }

    async def execute(self, context: BBOToolContext, arg: str, **kwargs):
        return {"answer": arg}
```

2. 在 `general_agent.py` 的 `_build_tool_registry()` 里把它 append 进去。

```python
tools.append(MyTool())
```

3. 如果也希望 workspace CLI 支持它，需要改 `workspace_tool_cli.py`：

```python
handlers = {
    ...
    "my_tool": _my_tool,
}
```

4. 如果也希望 `BBO()` Python API 有快捷方法，需要改 `workspace_python_api.py`：

```python
def my_tool(self, arg: str):
    return self.call("my_tool", {"arg": arg})
```

5. 补测试，确认：

- `tool_specs.json` 里出现新工具。
- 原生 function calling 可以调用。
- workspace CLI 可以调用。
- `bbo_tools.py` API 可以调用。
- 调用日志正常写入。

## 推荐 agent 实际怎么用工具

### 第一步：读上下文

```python
from bbo_tools import BBO

bbo = BBO()
task = bbo.task_context()
space = bbo.search_space()
objective = bbo.objective()
history = bbo.history(mode="recent", limit=40)
incumbent = bbo.incumbent()
```

### 第二步：分析历史

```python
analysis = bbo.analyze_history(limit=100)
metrics = bbo.summarize_objective_metrics(recent_limit=10)
```

如果历史足够，也可以运行 workspace 自带的：

```bash
python3 gp_expected_improvement.py
```

### 第三步：生成候选

可以随机采样：

```python
sample = bbo.sample(n=8, strategy="random")
```

也可以围绕当前最好配置采样：

```python
sample = bbo.sample(n=8, strategy="around_incumbent", jitter_fraction=0.1)
```

### 第四步：验证候选

```python
candidates = [item["config"] for item in sample["candidates"]]
checked = bbo.validate(candidates)
```

### 第五步：最终只输出 JSON

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

## 总结

当前 workspace 里的工具能力已经覆盖了 BBO agent 最需要的事情：

- 读任务
- 读搜索空间
- 读目标
- 读历史
- 读当前最优
- 校验候选
- 采样候选
- 分析历史
- 读写 memory
- 运行分析代码
- 搜索网页
- 抓取网页

agent 支持自己写脚本来做更复杂的分析，这可以理解成“自建临时工具”。

但 agent 不支持把这些临时脚本动态注册成框架级 tool。框架级 tool 的增删改，需要改 Python 源码里的工具类、注册表、CLI 桥和 Python API。
