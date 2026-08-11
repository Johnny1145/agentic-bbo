# Agentic BBO 框架严格架构审查

> 审查对象：`/home/trx/agentic-bbo-multiharness-benchmark-v5`
> 审查日期：2026-08-11
> 审查重点：代码简洁度、模块边界、新 single-agent / multi-agent 方法的接入成本，以及 EoH 一类 evolve-code 方法的兼容性。

## 1. 结论先行

当前仓库的 **BBO 外层内核是清楚而且可复用的**：`Task`、`Algorithm`、`TrialSuggestion`、`TrialObservation`、`Experimenter` 和 JSONL replay 形成了一个容易理解的串行 ask/tell 闭环。传统优化器或已经自行实现完整控制流的方法，只要实现 `Algorithm` 就能接入。

但当前的 **agentic 扩展层还没有达到“声明几个 role，就能轻量接入任意 agentic 方法”的程度**：

- single-agent 有可复用的 runtime，但声明层较薄，仍与 `GeneralAgentBBOAlgorithm` 的大量具体约定绑定；
- multi-agent 只有 PABLO 这一条专用 controller 路径，没有通用编排协议；
- `RoleSpec`、`AgenticMethodSpec` 中多个字段只描述能力，没有驱动执行；
- agentic method registry、全局 algorithm registry 和 CLI 分支存在重复登记；
- evolve-code 可以直接把程序源码放进 `StringParam`，但仓库还没有可复用的 program task 辅助类、通用 controller 和安全执行组件；
- 当前 SkyDiscover runner-enabled 路径存在正式 benchmark 不应接受的 oracle、执行隔离和 replay 风险。

严格评级如下：

| 维度 | 评级 | 结论 |
| --- | --- | --- |
| BBO 核心边界 | B+ | 小而清楚，适合配置型黑盒优化；错误处理和恢复契约仍可加固。 |
| 新传统/模型优化器接入 | B | 实现 `Algorithm` 并注册即可，但 CLI 仍有集中式分支。 |
| 新 single-agent 方法接入 | B- | 可复用 general-agent loop，但方法声明没有完全控制 runtime。 |
| 新 multi-agent 方法接入 | C- | 必须自行实现 controller、状态机、role 调用和恢复，PABLO 不是通用模板。 |
| 新 evolve-code 方法接入 | C+ | 核心协议已经能表达“程序 → fitness”，但 EoH 的 controller、代码执行和恢复仍要自行实现。 |
| 模块简洁度 | C | 多个 600–3000 行模块承担过多职责，扩展时容易继续堆条件分支。 |
| 协议可审计性 | C+ | trial、tool、agent call 日志较丰富，但内部动作、预算和跨 runtime 语义未统一。 |

一句话判断：**现在是“所有东西最终都能被包成 Algorithm”，还不是“所有 agentic 方法都能用一套小而稳定的组件协议自然表达”。**

## 2. 审查标准

本审查用以下问题判断架构是否真正可扩展：

1. 新方法是否只需要实现自己的决策逻辑，而不复制通用 runtime、候选解析、日志和恢复？
2. 方法声明中的 role、model route、prompt、parser、tool profile 是否真的驱动执行？
3. single-agent 与 multi-agent 是否共享同一个 role invocation 接口？
4. 方法是否能拥有内部状态、并行工作和多种内部对象，而不破坏 benchmark 的 ask/tell 边界？
5. benchmark evaluation 是否仍由 `Experimenter`/`Task` 独占，agent/tool 调用是否能被观测和限制？
6. 新增方法是否只需新增一个模块和一条注册信息？
7. replay 是否恢复方法真实状态，而不是仅仅尝试重新运行当前代码？
8. 生成代码是否在受限、可超时、可审计的环境中执行？

## 3. 当前实现的真实分层

### 3.1 Benchmark kernel

核心协议位于：

- `bbo/core/algo.py`：`Algorithm.setup/ask/tell/replay/incumbents`；
- `bbo/core/task.py`：配置型 `Task` 与 `TaskSpec`；
- `bbo/core/trial.py`：suggestion、evaluation、observation 和持久化 record；
- `bbo/core/experimenter.py`：串行 ask → evaluate → tell → log；
- `bbo/core/logger.py`：append-only JSONL 与 replay。

这层最大的优点是边界非常明确：算法只提交一个配置，只有 `Experimenter` 调用真实 evaluator。这个边界应当保留。

`TrialSuggestion.config` 必须是 `SearchSpace` 中的参数字典，但这并不把它限制为数值配置。仓库已有 `StringParam`，分子任务也已经使用字符串作为候选。因此程序搜索可以自然表达为 `config={"program": source}`：源码是 benchmark 输入，`Task.evaluate()` 返回 fitness。父代、operator、generation 等属于方法状态或 trial metadata，不需要升级成新的公共候选类型。

### 3.2 Agentic 声明与注册

当前有三组相关抽象：

- `components.py` 中的 `RoleSpec`、`ToolProfile`、`SingleAgentMethod`；
- `method_spec.py` 中的 `AgenticMethodSpec` 和 `AGENTIC_METHOD_REGISTRY`；
- `protocol.py` / `policy_algorithm.py` 中的 `AgenticPolicy` 适配协议。

正式 registry 路径并不使用 `AgenticPolicyAlgorithm`，而是由 method factory 创建一个完整 `Algorithm`，再套 `EventedAlgorithm`。仓库自己的 `docs/agentic_bbo_unification.md` 也说明旧 policy 层仅为 import 兼容保留。

因此当前公开 API 同时暴露了两种心智模型：

```text
正式路径：MethodSpec -> factory -> Algorithm -> EventedAlgorithm
兼容路径：AgenticPolicy -> AgenticPolicyAlgorithm -> Algorithm
```

这会让新贡献者无法立即判断应该扩展哪一个接口。

### 3.3 General-agent runtime

`GeneralAgentBBOAlgorithm` 已经复用了不少重要能力：

- workspace 和上下文物化；
- prompt 构造；
- native harness / OpenAI-compatible engine 调用；
- function-calling 与 workspace JSON tools；
- 候选解析、校验、去重、重试和 fallback；
- history、memory、agent call、tool call 和 reasoning 日志；
- 方法本地状态持久化。

但 `general_agent.py` 约 3000 行，仍同时负责配置校验、workspace 生命周期、框架配置、prompt、候选队列、resume、skill 安装与审计、tool policy、fallback 和多种 harness 差异。相关 engine、tool 和 compatibility 模块也非常大：

| 模块 | 约行数 | 主要问题 |
| --- | ---: | --- |
| `agentic/general_agent.py` | 3056 | orchestration、workspace、policy、state、prompt 和 harness 配置集中在同一类。 |
| `agentic/tools/core_tools.py` | 2505 | 多种分析工具、schema、统计逻辑和服务适配集中。 |
| `agentic/tools/optimizer_tools.py` | 1359 | backend 生命周期、工具动作与 workspace 恢复耦合。 |
| `agentic/general_agent_engines.py` | 988 | 四种 runtime 的进程、SDK、协议与日志处理放在同一模块。 |
| `agentic/pablo.py` | 693 | 方法控制器与 role 调用、状态和候选逻辑仍为专用实现。 |

这不是单纯的文件长度问题。真正的问题是新增一个能力时通常需要同时理解多个不相干的职责，导致“继续加开关”比“组合小组件”更容易。

### 3.4 PABLO multi-agent 路径

PABLO 实现了 planner、explorer、worker、agenda、active search、failure counter 和 resume state，证明核心 ask/tell 可以承载 multi-agent 方法。

但这只能证明“可以专门实现一个 multi-agent Algorithm”，不能证明框架已有通用 multi-agent 能力：

- `RoleSpec` 没有负责实际 role invocation；
- 没有通用 controller/graph 执行器；
- 没有通用 shared state 或 message envelope；
- role 的顺序、分支、重试、终止和反馈都写在 PABLO 内；
- 新 multi-agent 方法无法仅靠声明 roles 和 transitions 接入。

### 3.5 SkyDiscover evolve-code 路径

`SkydiscoverInterleavedAlgorithm` 直接实现 `Algorithm`，定期进化并加载一个定义 `suggest_next_config()` 的 Python 文件。这是一个有价值的可行性原型，但它是专项集成：

- 位于 `llm_based/`，不经过 agentic method registry；
- 自己管理 generated 目录、round、环境变量和 strategy 文件；
- 自己定义代码 contract 和 meta evaluator；
- 自己处理 runner 兼容、fallback 和 replay；
- 事件、预算和状态没有复用统一的 agentic controller 协议。

它证明“外层 Algorithm 可以托住 evolve-code”，但没有提供可复用的 evolve-code 基础设施。

## 4. 主要架构问题

### A1. 声明字段没有成为执行契约 — P0

`RoleSpec` 声明了 `model_route`、`tool_profile`、`prompt_profile`、`prompt_builder` 和 `response_parser`。当前 `SingleAgentMethod.build()` 实际只把 role name 和一个方法级 prompt profile 传给 `GeneralAgentBBOAlgorithm`；`prompt_builder` 与 `response_parser` 没有通用消费点，`model_route` 也没有由通用 single-agent runtime 解析。

`AgenticMethodSpec.controller`、`capabilities`、`supports_multiple_roles`、`supports_resume` 同样主要用于描述和测试，没有约束 factory 产物或驱动 controller。

风险：registry 看起来比实际能力更声明式，新增方法后很容易出现“spec 说支持、runtime 并未实现”的漂移。

建议：声明字段必须满足二选一——由统一 runner 消费并验证，或者删除。不要保留装饰性协议。

### A2. Agentic 扩展存在双轨协议 — P1

`AgenticPolicy`、`AgenticPolicyAlgorithm` 和 `AlgorithmPolicyAdapter` 仍是公开导出，但正式方法不走这条路径。事实上前两者已经接近所需的通用 controller 与 adapter；问题不是缺少新抽象，而是已有抽象没有成为正式入口。

建议：直接把 `AgenticPolicy` + `AgenticPolicyAlgorithm` 做成唯一正式 controller 路径，或者一次性重命名后保留兼容 alias；`AlgorithmPolicyAdapter` 等重复方向再移入 `compat`。不要新增第三套 controller API。

### A3. Registry 与构造逻辑重复 — P0

新增一个正式 agentic 方法目前可能需要修改：

1. `AGENTIC_METHOD_REGISTRY`；
2. `ALGORITHM_REGISTRY`；
3. 顶层 registry 的 factory wrapper；
4. `bbo/run.py` 的算法分支或参数传递；
5. agentic `__init__` 导出；
6. method contract 测试；
7. 方法专属 event mapper。

aliases 也会在不同层重复出现。`llambo` 被 agentic method registry 包装，但顶层仍将它分类为 `llm_based`，说明 registry 已经同时承担“执行入口”和“论文方法分类”两种职责。

建议：只保留一个权威 `MethodSpec`，由它生成算法别名、family、factory、capability、CLI/schema 和事件映射。

### A4. Single-agent 组件只能配置现有 loop，不能替换方法语义 — P1

`SingleAgentMethod` 的本质是把若干 kwargs 填入 `GeneralAgentBBOAlgorithm`。如果新方法仍然是“读取相同 workspace → 调一个 agent → 解析 candidates JSON → 提交一个配置”，它很省代码。

一旦方法需要以下任一能力，就要进入大类或另写 Algorithm：

- 一个 round 内多次不同目的的 role call；
- structured intermediate decisions；
- 非 candidate 的内部对象；
- 方法自己的 branching 或 termination；
- 内部批量/并行工作；
- 多阶段程序生成、筛选或角色协作；
- 与候选提交不同的状态更新节奏。

因此它是“preset builder”，还不是通用 single-agent controller abstraction。

### A5. Multi-agent 没有通用编排层 — P0

当前 `GeneralAgentEngine.run_agent()` 能执行一次 agent 调用，但它没有被抽象成可由任意 controller 调度的 role runtime。PABLO 自己处理 role 和 provider；native harness 方法又走 general-agent loop。

新增 planner/reviewer/proposer 方法时，贡献者仍需决定并重新实现：

- role 的 model routing；
- prompt 与 response parser；
- role 间消息和共享状态；
- 谁可以调用哪些工具；
- role failure 如何影响本轮；
- 哪个结果构成 commit；
- snapshot/replay 如何恢复中间阶段。

建议：把“执行一次 role”与“决定下一步执行哪个 role”彻底分开。前者属于 runtime，后者属于 controller。

### A6. State 与 replay 不是统一能力 — P0

外层 `Experimenter` 只 replay trial history，不读取 `AgenticPolicy.snapshot()`。General agent、PABLO、SkyDiscover 分别维护自己的 state 文件和恢复逻辑。

这会产生两个状态源：

```text
trials.jsonl：已经提交并观察的 benchmark trial
method state：候选队列、agent memory、agenda、population、当前生成程序等
```

二者没有统一 commit point。进程在“内部状态已写、trial 未写”或相反位置退出时，各方法必须自行解决一致性。

建议：先规定最小 controller snapshot envelope，至少记录 last committed trial id 和无法由 trial history 推导的私有状态。对于程序优化，已提交源码本身就在 trial config 中；只有源码过大时才有必要额外做内容寻址存储。

### A7. Event schema 对复杂 controller 表达不足 — P1

当前通用事件适合单候选 agentic BO：context、probe、propose、reconfigure、role_call、candidate、commit、observation、fallback、stop、error。

对 multi-agent 和 evolve-code，还缺少：

- message/role transition；
- parent selection；
- program generated/validated/rejected；
- population updated；
- checkpoint saved/restored；
- sandbox timeout/resource violation。

如果每个新方法继续在 metadata 中发明字段，后续无法统一分析 agent 行为。

### A8. CLI 是集中式依赖注入容器 — P1

`bbo/run.py` 通过一个很长的参数表和算法名 `if/elif` 组装 kwargs。新增方法即使 registry 已注册，只要需要新参数，仍需修改中心 CLI。

建议：每个 spec 提供 typed config/schema 或 `add_arguments/config_from_namespace`；通用 runner 只解析公共参数并委托方法 spec。

## 5. EoH / evolve-code 兼容性问题

### E1. 现有 ask/tell 已经能直接表达 EoH

对这里讨论的 EoH 任务，一次 trial 就是：

```text
EoH.ask()
  -> TrialSuggestion(config={"program": source})
  -> ProgramOptimizationTask.evaluate(suggestion)
  -> EvaluationResult(fitness)
  -> EoH.tell(observation)
  -> 更新方法私有 population
```

`program` 使用现有 `StringParam`，和分子任务使用 SMILES 字符串在协议层没有本质区别。fitness 完全由外部 benchmark `Task` 计算并消耗一次 trial；EoH 内部不另设 evaluator。

父代、operator、generation、thought 和 selection score 可以放在 `TrialSuggestion.metadata` 与 EoH controller 的私有状态中。它们有审计价值，但不应成为所有算法都必须理解的公共数据类。

### E2. 不要把当前 SkyDiscover 桥接误当成 EoH 抽象

当前 SkyDiscover 进化的是 `suggest_next_config()`，再由该程序产生普通 BBO 配置；这是“进化 optimizer/proposer”的双层问题。这里的 EoH 则直接把待评估程序作为 Task 输入，是单层 benchmark：程序本身就是 candidate。

因此 SkyDiscover 的 contract-only fitness、known-optimum distance 和 interleaved strategy refresh 都不应该复用到 EoH 接口。尤其 known-optimum selection 使用了测试 oracle，不能作为正式 benchmark 的公平基线；但这是 SkyDiscover 当前专项设计的问题，不是 EoH 接入必须增加公共 meta-evaluator 的理由。

### E3. 程序评估属于 Task，实现仍需隔离 — P0

“外部 Task eval”并不意味着可以在 benchmark 主进程里直接 `exec`。`ProgramOptimizationTask` 应把源码交给受限 evaluator：至少有独立进程、hard timeout、process-tree kill、资源上限、只读输入和结构化输出。这里的 sandbox 是 program task 的实现组件，不是 agentic 框架的新协议层。

当前 `generated_solver.load_suggest_next_config()` 和 SkyDiscover meta evaluator 使用 importlib 直接加载生成模块，仍不适合作为不可信代码 benchmark 的实现。

### E4. Replay 可以依靠 trial 中的完整源码

只要每个已提交 trial 的 `config["program"]` 保存完整源码，外层 replay 就能恢复所有已观察的 `(program, fitness)`；EoH controller 再按顺序重建 population 即可，不必先引入公共 artifact store。

需要额外 checkpoint 的只有两类状态：无法由已观察 trial 推导的 pending proposal，以及生成器自身的随机数/模型会话状态。若源码使 JSONL 体积不可接受，再把源码移到 content-addressed blob 并在 config 中保存引用；这应是存储优化，而不是 benchmark 类型系统的前提。

### E5. Population 应是 EoH 私有状态

不同 evolve-code 方法的 population 语义差异很大：有的存完整代际，有的只保留 archive，有的用岛模型或 MAP-Elites。强制抽象统一 `PopulationState` 很可能只得到一个过度宽泛的数据袋。

框架只需允许 controller 拥有、序列化和恢复私有状态；EoH 自己定义 individual、population、selection 和 operator。公共层负责 ask/tell、role invocation、日志和 snapshot 生命周期。

### E6. Fallback 会掩盖方法失败 — P1

SkyDiscover 刷新失败会回到初始模板，strategy 调用失败会随机采样。作为 smoke-test 这很实用；作为正式方法比较，如果 summary 仍只显示 trial 成功，就会把“evolve 方法失败”伪装成“该方法提出了随机候选”。

正式运行应默认 fail closed，或者把 fallback 作为单独算法条件并在主指标中报告。EoH 的无效程序也应由 Task 返回明确失败/惩罚结果，不能静默替换成另一程序。

## 6. 三类新方法的实际接入成本

### 6.1 与现有 loop 完全同构的 single-agent

如果方法只改变 prompt 和工具组合，可以复用 `SingleAgentMethod`，代码量不大。但仍需跨 registry、export、测试和可能的 CLI 改动。

判断：**可以接，但注册面不够单一，声明与执行也没有完全闭环。**

### 6.2 新 multi-agent controller

角色声明可以复用，实际 controller、role invocation、消息、状态、恢复和 event mapping 仍需专门编写。

判断：**核心能承载，但框架没有提供足够组件；接入成本接近重新实现一个方法 runtime。**

### 6.3 EoH 类代码进化方法

最小版本只需：一个以 `StringParam("program")` 为输入的 Task，以及一个 EoH `AgenticPolicy`，再由现有 `AgenticPolicyAlgorithm` 适配到 ask/tell。generation、operator、selection 和 population 都留在方法内部；Task 负责程序执行与 fitness。

但为了让接入代码真正简洁，框架还应把现有 controller-to-Algorithm adapter 做成正式入口，补齐统一 role runtime、snapshot hook，以及可选的 task-side 安全 program runner helper。现有 SkyDiscover 代码只能作为隔离与 replay 风险样例，不能作为 EoH 基类。

判断：**核心协议兼容，公共组件不足；不需要 EoH 专属层。**

## 7. 值得保留的设计

严格审查并不意味着应推倒重写。以下部分应当保留：

- `Algorithm` 作为 benchmark 外边界；
- `Experimenter` 独占真实 evaluator 调用权；
- typed search space 与 candidate coercion；
- append-only trial ledger；
- workspace 是 coding agent 的可理解交互面；
- function-calling 与 workspace tool 共用 tool service 的方向；
- 每个 run 独立 state/config 目录；
- fixed initialization 与 task-owned protocol；
- method-specific controller 可以存在，但应实现统一 controller 接口。

## 8. 修复优先级

### P0：先建立真实扩展协议

1. 将现有 `AgenticPolicy` 与 `AgenticPolicyAlgorithm` 正式化为唯一、最小的 controller 生命周期和 adapter；若重命名，只保留兼容 alias。
2. 让 role spec、method spec 真正驱动执行和验证。
3. 统一 controller snapshot 与 trial replay 的恢复规则。
4. 为 program-as-input task 提供可选的隔离执行 helper；不要引入公共 artifact/population 层。
5. 禁止不可信生成代码在宿主 benchmark 进程直接执行。
6. 移除 SkyDiscover 正式路径中的 known-optimum meta selection。

### P1：降低新增方法改动面

1. 合并 method/algorithm registry。
2. 把 CLI 参数定义下放到 method spec。
3. 拆分 general-agent 的 workspace、state、prompt、tool policy 和 orchestration。
4. 扩展通用事件 schema。
5. 把 PABLO 迁移为统一 controller 的验证样例；另加一个最小 EoH adapter 验证 program-string 路径。

### P2：清理与长期维护

1. 将旧 policy adapter 移入 compat 或删除公开推荐。
2. 统一命名、family 分类和 aliases。
3. 为第三方方法提供 entry-point/plugin discovery。
4. 增加架构 contract tests，而不是只测 registry 元数据。

## 9. 最终判断

如果目标只是“研究人员愿意写一个完整 `Algorithm` 类，就都能放进 benchmark”，当前框架已经满足。

如果目标是“研究人员只写方法本身的 50–150 行控制逻辑，就能接入 single-agent、multi-agent 或 evolve-code，并自动获得 runtime、工具、日志和恢复”，当前框架还没有满足。下一步不应新增 EoH 专属架构层，而应把 controller、role runtime、state/replay 和 registry 四个公共边界做实；代码候选继续使用现有 `StringParam` 和 `Task.evaluate()`。
