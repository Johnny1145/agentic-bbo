# Agentic BBO 组件架构

## 唯一外层协议

Agentic 方法与所有其他优化器一样，只实现 benchmark 的 `Algorithm`
(`setup/ask/tell/replay/incumbents`)。不再经过
`Algorithm -> AgenticPolicy -> Algorithm` 的双重转换。

`create_agentic_method()` 从 `AgenticMethodSpec` 创建原始算法，并用
`EventedAlgorithm` 添加透明事件记录。该装饰器把 `replay()`、`seed()` 和
`incumbents()` 委托给原始算法，因此不会丢失 PABLO 或 GeneralAgent 的
方法专属恢复逻辑。

`AgenticPolicyAlgorithm`、`AlgorithmPolicyAdapter` 和旧 protocol 目前只为
import 兼容保留，不是正式 registry 的执行路径。

## 可组合组件

- `RoleSpec`：角色名、模型 route、tool profile、prompt builder 和 response parser。
- `ToolProfile`：工具 allowlist、optimizer backend allowlist 和决策约束。
- `SingleAgentMethod`：将一个 role、tool profile 和 runtime 组合成通用
  `GeneralAgentBBOAlgorithm`。
- `AgenticMethodSpec`：方法 factory、aliases、roles、capabilities 和 resume 能力。
- `EventedAlgorithm`：不改变优化语义的 append-only deliberation event 装饰器。

Agentic BO 当前由一个 `SingleAgentMethod` 声明构造，而不是复制新的 agent
loop。PABLO 的 planner、explorer、worker 作为三个 `RoleSpec` 声明，并继续由
专用 hierarchical controller 维护其方法状态。

## 新增方法

普通单角色方法只需要声明组件并在 agentic method registry 增加一个 spec：

```python
MY_METHOD = SingleAgentMethod(
    name="my_method",
    role=RoleSpec(name="proposer", tool_profile="my_tools"),
    tools=ToolProfile(
        name="my_tools",
        tools=("get_trial_history", "validate_candidate"),
    ),
)
```

factory 只需调用 `MY_METHOD.build(**kwargs)`。不需要增加新的 Algorithm 子类、
policy adapter、runtime loop、candidate validator 或 event logger。层级方法只需
实现其 controller 状态机并复用相同 role/runtime/tool 组件。

## 执行边界

```text
Method/controller -> Agent runtime -> Tool service -> candidate
                                             |
                                             +-- never calls evaluator
candidate -> Candidate validation -> Experimenter.evaluate -> observation
```

Objective evaluator 仍只能由 `Experimenter` 调用。optimizer tools 必须返回
`evaluator_called=false` 和 `budget_consumed=false`。

## 兼容性

- 保留 `agentic_bo`、`pablo`、`palbo`、`llambo` 和现有 runtime 算法名。
- 保留原 CLI 和 JSONL 顶层格式。
## 工具传输统一

函数调用 runtime 与 workspace JSON CLI 现在都调用
`tools/tool_service.py -> BBOToolRegistry -> core_tools.py`。CLI 不再维护一份
历史分析、候选验证、采样、surrogate 和诊断算法；它只负责从 workspace
快照重建 `BBOToolContext`，以及保留 memory/code/web/optimizer 的宿主适配。
这消除了两种 agent 协议之间结果逐渐分叉的主要来源。

## PABLO 反馈状态机

PABLO 每个 Explorer 或 Worker search state 一次只提出一个候选，然后必须等待
真实 `tell(observation)`：

- 改进：失败计数清零；Worker 将成功候选设为新的局部 seed。
- 未改进或失败：当前 state 的连续失败数加一。
- 达到 `max_fails`：结束当前 state，转到 agenda 中下一个 role/task/seed。
- agenda 耗尽：Planner 生成下一轮任务并重建 Explorer/Worker agenda。

`agenda` 和 `active_search` 都写入 resume state，因此中断恢复不会重新开始
已经进行中的局部搜索。
## General Agent 与 workspace 分层

`general_agent.py` 现在只保留配置、ask/tell 协调、Agent runtime 调用、
workspace materialization、prompt 构造和恢复。两个纯逻辑子系统已独立：

- `agent_candidate.py`：候选 JSON 提取、容错解析、search-space coercion 和
  BBOPLACE 紧凑坐标转换。
- `agent_skill_audit.py`：skill 声明、读取记录、证据工具、搜索动作模式与
  合规性判断。

workspace 执行也拆成两层：

- `workspace_tool_cli.py`：复制进 workspace 的稳定薄入口，并保留
  `bbo_tools.py` 需要的兼容 forwarding API。
- `workspace_tool_runtime.py`：包内 optimizer host、memory、code、web 和日志适配。
- `tools/tool_service.py`：两种 transport 共用的核心 BBO tool 业务实现。
## Prompt 管理

Prompt 分成四层，避免 harness、tools 和 workflow 互相复制文本：

1. task description：只描述优化问题与领域知识。
2. benchmark protocol：统一候选格式、workspace 文件和 evaluator 边界。
3. `ToolProfile`：只声明可用能力和 optimizer backend。
4. `PromptProfile`：声明方法特有策略；controller/workflow 只负责调用顺序。

内置 profile 包括 `general_bbo`、`native_harness`、`native_tools` 和
`agentic_bo`。普通 Codex tool run 可以选择 `native_tools`；Agentic BO
组件自己声明 `agentic_bo`，调用者不需要手工同步 prompt 与 tool allowlist。

## 最小 Python 入口

`bbo.benchmark.run_benchmark` 接受已经创建的 Task 与 Algorithm；
`run_named_benchmark` 接受 registry 名称和各自 kwargs。执行设置单独放在
`BenchmarkRunConfig`。可编辑示例见 `examples/run_one_benchmark.py`。
## 单 Agent 与多 Agent Prompt 契约

`PromptProfile` 表示一个 role 的 prompt additions；`WorkflowPromptProfile`
表示完整的 `role -> PromptProfile` 映射。单 Agent 方法同样被规范化成只有一个
role 的 workflow，因此 controller 使用同一个解析接口：

```python
workflow.for_role(role).compose_bundle(prompt)
```

`AgenticMethodSpec` 在构造时校验声明的 `RoleSpec` 集合与 workflow prompt
roles 完全相等。多角色 workflow 不允许传入一个原子 PromptProfile 后静默复用。
PABLO 内置 planner、explorer、worker 三个独立 profile，并在三个真实 role
调用点显式解析对应 profile。
