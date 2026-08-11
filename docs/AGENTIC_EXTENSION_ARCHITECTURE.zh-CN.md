# Agentic BBO 统一扩展架构建议（精简版）

> 适用仓库：/home/trx/agentic-bbo-multiharness-benchmark-v5
> 修订日期：2026-08-11
> 目标：让 single-agent、multi-agent 和 EoH 类方法用少量代码接入，同时不为 EoH 增加专属公共层。

## 1. 核心结论

这次修订撤回此前的三层 artifact-evolution 设计。对当前讨论的 EoH 任务，正确模型非常直接：

~~~text
程序源码
  -> TrialSuggestion(config={"program": source})
  -> benchmark Task.evaluate(...)
  -> fitness
  -> TrialObservation
  -> EoH 私有 population 更新
~~~

公开框架不需要新增 CandidateArtifact、ArtifactEvaluation、PopulationState 或 ArtifactBudget。

原因如下：

1. 现有 SearchSpace 已经支持 StringParam；代码和 SMILES 一样，都可以是字符串候选。
2. 一段程序就是一次 benchmark trial，fitness 由外部 Task 计算。
3. EoH 的 population、operator、parent、thought 和 generation 是方法内部语义。
4. 不同 evolve-code 方法的私有状态差异很大，强行统一 population 类型只会增加空泛抽象。
5. 框架真正缺少的是通用 controller 生命周期、统一 role runtime、状态恢复 hook 和简化后的 registry。
6. 程序执行隔离仍然必要，但它属于 program task 的实现，不是 agentic 方法协议的新层。

因此推荐结构只有两条主边界：

~~~text
Benchmark kernel:
    Experimenter -> Algorithm -> TrialSuggestion -> Task -> TrialObservation

Method runtime:
    Algorithm adapter -> method controller -> optional RoleRuntime / ToolService
~~~

EoH、single-agent 和 multi-agent 的区别只存在于 controller 内部。

## 2. Program 作为 StringParam 是否足够

足够，而且这是当前仓库最自然的表达。

现有 StringParam 已经具备：

- 字符串类型转换和验证；
- min_length、max_length 和 pattern；
- JSONL 可序列化；
- 与 TrialSuggestion、TrialRecord 和 replay 的直接兼容。

一个 program task 的搜索空间可以定义为：

~~~python
SearchSpace(
    [
        StringParam(
            "program",
            default=TEMPLATE_PROGRAM,
            min_length=1,
            max_length=100_000,
        )
    ]
)
~~~

这里要注意两个现有行为：

- StringParam.sample() 会抛出 TypeError，所以 RandomSearchAlgorithm 不能用于开放程序空间；这不是协议缺陷，而是算法必须有代码生成能力。
- Task.sanity_check() 会调用 search_space.defaults()，因此 program StringParam 必须提供一个合法的默认模板。

程序相关的附加信息不需要塞进 config：

| 信息 | 推荐位置 |
| --- | --- |
| 完整源码 | suggestion.config["program"] |
| parent ids | suggestion.metadata |
| operator | suggestion.metadata |
| generation / proposal index | suggestion.metadata |
| thought / rationale | suggestion.metadata，必要时截断 |
| fitness / task metrics | TrialObservation.objectives / metrics |
| population | EoH controller 私有状态 |
| checkpoint | controller snapshot |
| sandbox 日志 | EvaluationResult.metrics / metadata |

只有当源码非常大、导致 trials.jsonl 明显不可控时，才值得把源码移到 content-addressed blob。那是存储优化，不应改变 Task 和 Algorithm 的公共类型。

## 3. 现有代码中已经接近可用的 Controller

仓库已有：

- bbo/algorithms/agentic/protocol.py 中的 AgenticPolicy；
- OptimizationContext；
- CommitCandidate；
- bbo/algorithms/agentic/policy_algorithm.py 中的 AgenticPolicyAlgorithm。

它们已经非常接近所需的通用 controller：

~~~python
class AgenticPolicy(Protocol):
    @property
    def name(self) -> str: ...

    def setup(self, context: OptimizationContext) -> None: ...
    def deliberate(self, context: OptimizationContext) -> PolicyDecision: ...
    def observe(self, observation: TrialObservation) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...
    def restore(self, state: Mapping[str, Any]) -> None: ...
~~~

最简洁的做法不是再新增 EoHControllerBase 或 ArtifactController，而是把这组现有协议正式化：

- 语义上把 AgenticPolicy 视为通用 MethodController；
- AgenticPolicyAlgorithm 就是 controller-to-Algorithm adapter；
- EoH 的 population 直接存在 policy/controller 实例里；
- single-agent 和 multi-agent 也实现同一个协议；
- registry 的正式路径统一创建 controller，再套同一个 adapter。

名称是否改成 Controller 是次要问题。如果要改，应该用兼容 alias 完成，不能同时长期维护 Policy 和 Controller 两套接口。

### 3.1 当前实现还差什么

现有 AgenticPolicy 路径有四个实质缺口：

1. 正式 AgenticMethodSpec registry 基本不使用它，而是让 factory 直接创建完整 Algorithm。
2. AgenticPolicyAlgorithm 没有调用 snapshot()/restore()，这两个方法目前是死契约。
3. StopOptimization 最终变成 ask() 异常，Experimenter 没有正常的提前停止协议。
4. OptimizationContext.incumbent 没有在 adapter 中填充。

这些问题应修在通用 adapter，而不是为 EoH 另开一层。

## 4. 最小公共架构

推荐的公共模块保持很小：

~~~text
bbo/core/
    algo.py                 # 保持现有 Algorithm
    task.py                 # 保持现有 Task
    trial.py                # 保持现有 trial 数据类

bbo/algorithms/agentic/
    protocol.py             # 唯一 MethodController 协议
    policy_algorithm.py     # 唯一 controller -> Algorithm adapter
    role_runtime.py         # 执行一次 role，统一模型/工具/日志
    method_spec.py          # 唯一 agentic 方法声明

bbo/tasks/program/
    task.py                 # 可选的 program-as-string Task helper
    runner.py               # task-owned sandbox runner
~~~

不新增 evolve/、artifact/、population/ 等框架目录。某个方法自己的代码可以放在：

~~~text
bbo/algorithms/agentic/methods/eoh/
    controller.py
    operators.py
    prompts.py
    state.py
~~~

这里的 state.py 是 EoH 私有实现，不是公共协议。

### 4.1 Controller 的职责

Controller 负责：

- 根据历史和私有状态生成下一个候选；
- 选择要调用的 role、operator 或工具；
- 接收 TrialObservation；
- 更新方法私有状态；
- 返回 incumbent；
- 可选地 snapshot/restore。

Controller 不负责：

- 直接调用 Task.evaluate()；
- 分配 trial_id；
- 统计 benchmark evaluation 次数；
- 写通用 trial ledger；
- 定义另一套 candidate/evaluation 数据模型。

### 4.2 RoleRuntime 的职责

RoleRuntime 只做“一次 role invocation”：

~~~python
class RoleRuntime(Protocol):
    def invoke(
        self,
        role: RoleSpec,
        *,
        prompt: str,
        context: Mapping[str, Any],
    ) -> Any: ...
~~~

它统一处理 model route、tool profile、prompt builder、response parser、超时、重试和调用日志。Controller 决定什么时候调用哪个 role。

不建议把 InvokeRole、RunTool、EvaluateArtifact 等内部动作和当前简单的 CommitCandidate 合并成大型 action union。保留 CommitCandidate 作为唯一提交结果；当前是串行 outer loop，controller 直接同步调用注入的 runtime 更简单。未来真的需要暂停/恢复内部 action 时，再以真实需求扩展。

## 5. 当前 API 下的 program Task 骨架

下面代码刻意贴近当前仓库 API。ProgramRunner 是 Task 的内部依赖；它不是算法可调用的第二个 evaluator。

~~~python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from bbo.core import (
    EvaluationResult,
    ObjectiveDirection,
    ObjectiveSpec,
    SearchSpace,
    StringParam,
    Task,
    TaskDescriptionRef,
    TaskSpec,
    TrialStatus,
    TrialSuggestion,
)


class InvalidProgramError(ValueError):
    pass


class ProgramExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProgramScore:
    fitness: float
    metrics: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ProgramRunner(Protocol):
    """Task-owned implementation that runs and scores one source string."""

    def run(self, source: str) -> ProgramScore: ...


class ProgramOptimizationTask(Task):
    def __init__(
        self,
        *,
        name: str,
        runner: ProgramRunner,
        template_program: str,
        description_dir: Path,
        max_evaluations: int,
        direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE,
        max_program_length: int = 100_000,
    ) -> None:
        self._runner = runner
        self._spec = TaskSpec(
            name=name,
            search_space=SearchSpace(
                [
                    StringParam(
                        "program",
                        default=template_program,
                        min_length=1,
                        max_length=max_program_length,
                    )
                ]
            ),
            objectives=(ObjectiveSpec("fitness", direction),),
            max_evaluations=max_evaluations,
            description_ref=TaskDescriptionRef.from_directory(
                name,
                description_dir,
            ),
            metadata={
                "representation": "python_source",
                "program_parameter": "program",
            },
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def evaluate(self, suggestion: TrialSuggestion) -> EvaluationResult:
        source = suggestion.config["program"]
        started = time.perf_counter()

        try:
            score = self._runner.run(source)
        except InvalidProgramError as exc:
            return EvaluationResult(
                status=TrialStatus.INVALID,
                elapsed_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        except ProgramExecutionError as exc:
            return EvaluationResult(
                status=TrialStatus.FAILED,
                elapsed_seconds=time.perf_counter() - started,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        return EvaluationResult(
            status=TrialStatus.SUCCESS,
            objectives={"fitness": float(score.fitness)},
            metrics=dict(score.metrics),
            metadata=dict(score.metadata),
            elapsed_seconds=time.perf_counter() - started,
        )
~~~

实际 ProgramRunner 应在独立进程或容器中运行不可信程序，并施加 hard timeout、process-tree kill、CPU/内存限制、只读输入和结构化输出。这个 runner 可以被多个 program tasks 复用，但仍属于 task-side utility。

如果 benchmark 评估的是其他语言，config 仍然可以是 program 字符串，只需替换 runner。公共 Algorithm 协议无需变化。

## 6. 当前 API 下的 EoH 方法骨架

### 6.1 EoH 私有类型

这些类型都放在 EoH 方法包内：

~~~python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

from bbo.core import (
    Incumbent,
    ObjectiveDirection,
    StringParam,
    TrialObservation,
)
from bbo.algorithms.agentic.protocol import (
    CommitCandidate,
    OptimizationContext,
)


@dataclass(frozen=True)
class ProgramProposal:
    source: str
    operator: str
    parent_ids: tuple[str, ...] = ()
    thought: str = ""


@dataclass(frozen=True)
class Individual:
    proposal_id: str
    source: str
    fitness: float | None
    status: str
    operator: str
    parent_ids: tuple[str, ...]


class EoHEngine(Protocol):
    """LLM/operator implementation owned by EoH, not by the framework."""

    def propose(
        self,
        *,
        population: Sequence[Individual],
        seed: int,
        task_context: str,
    ) -> ProgramProposal: ...
~~~

### 6.2 直接实现现有 AgenticPolicy

~~~python
class EoHPolicy:
    def __init__(
        self,
        engine: EoHEngine,
        *,
        population_size: int = 20,
    ) -> None:
        self.engine = engine
        self.population_size = population_size
        self.population: list[Individual] = []
        self._seed = 0
        self._proposal_index = 0
        self._objective_name = ""
        self._direction = ObjectiveDirection.MAXIMIZE
        self._task_context = ""
        self._pending_id: str | None = None

    @property
    def name(self) -> str:
        return "eoh"

    def setup(self, context: OptimizationContext) -> None:
        params = tuple(context.task_spec.search_space)
        if len(params) != 1:
            raise ValueError("EoH expects exactly one program parameter.")
        param = params[0]
        if not isinstance(param, StringParam) or param.name != "program":
            raise TypeError(
                "EoH expects SearchSpace([StringParam('program', ...)])."
            )

        self.population = []
        self._seed = int(context.seed)
        self._proposal_index = 0
        self._objective_name = context.task_spec.primary_objective.name
        self._direction = context.task_spec.primary_objective.direction
        self._task_context = (
            context.description.rendered_context
            if context.description is not None
            else ""
        )
        self._pending_id = None

    def deliberate(self, context: OptimizationContext) -> CommitCandidate:
        if self._pending_id is not None:
            raise RuntimeError("Previous EoH proposal has not been observed.")

        index = self._proposal_index
        proposal_id = f"eoh-{index:06d}"
        proposal = self.engine.propose(
            population=tuple(self.population),
            seed=self._seed + index,
            task_context=self._task_context,
        )

        self._proposal_index += 1
        self._pending_id = proposal_id
        return CommitCandidate(
            config={"program": proposal.source},
            metadata={
                "eoh.proposal_id": proposal_id,
                "eoh.proposal_index": index,
                "eoh.operator": proposal.operator,
                "eoh.parent_ids": list(proposal.parent_ids),
                "eoh.thought": proposal.thought,
            },
        )

    def observe(self, observation: TrialObservation) -> None:
        metadata = observation.suggestion.metadata
        proposal_id = str(
            metadata.get(
                "eoh.proposal_id",
                f"replay-{observation.suggestion.trial_id}",
            )
        )
        proposal_index = metadata.get("eoh.proposal_index")
        if isinstance(proposal_index, int):
            self._proposal_index = max(
                self._proposal_index,
                proposal_index + 1,
            )

        fitness = None
        if observation.success:
            value = observation.objectives.get(self._objective_name)
            if value is not None:
                fitness = float(value)

        self.population.append(
            Individual(
                proposal_id=proposal_id,
                source=str(observation.suggestion.config["program"]),
                fitness=fitness,
                status=observation.status.value,
                operator=str(metadata.get("eoh.operator", "unknown")),
                parent_ids=tuple(metadata.get("eoh.parent_ids", ())),
            )
        )
        self.population = self._select(self.population)
        self._pending_id = None

    def _select(
        self,
        population: Sequence[Individual],
    ) -> list[Individual]:
        valid = [item for item in population if item.fitness is not None]
        reverse = self._direction == ObjectiveDirection.MAXIMIZE
        valid.sort(
            key=lambda item: float(item.fitness),
            reverse=reverse,
        )
        return valid[: self.population_size]

    def incumbents(self) -> list[Incumbent]:
        if not self.population:
            return []
        best = self.population[0]
        return [
            Incumbent(
                config={"program": best.source},
                score=best.fitness,
                objectives={self._objective_name: float(best.fitness)},
                metadata={"eoh.proposal_id": best.proposal_id},
            )
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "proposal_index": self._proposal_index,
            "population": [asdict(item) for item in self.population],
            "pending_id": self._pending_id,
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        self._proposal_index = int(state.get("proposal_index", 0))
        self.population = [
            Individual(**raw)
            for raw in state.get("population", [])
        ]
        pending = state.get("pending_id")
        self._pending_id = str(pending) if pending is not None else None
~~~

这段实现的关键点是：

- EoH 不调用 task；
- deliberate() 只提交程序；
- observe() 只消费外部返回的 fitness；
- population 完全私有；
- trial metadata 足以重建 lineage；
- replay 时，adapter 依次调用 observe() 就能重建已提交 population；
- generation/operator/selection 可以任意替换，不影响 benchmark kernel。

这个骨架是稳妥的 sequential steady-state 版本。若要实现整代并行，需要先扩展 benchmark kernel 的 batch ask/tell 或异步 trial 协议，不应偷偷在 EoH 内直接调用 Task。

### 6.3 用现有 adapter 包装

当前仓库已经能这样创建 Algorithm：

~~~python
from bbo.algorithms.agentic.policy_algorithm import AgenticPolicyAlgorithm

algorithm = AgenticPolicyAlgorithm(
    EoHPolicy(
        engine=my_eoh_engine,
        population_size=20,
    )
)
~~~

这比新增 EoHAlgorithm 基类更简洁。若团队暂时不想把 AgenticPolicy 作为正式 API，也可以写一个很薄的 EoHAlgorithm，但它实际上会重复 AgenticPolicyAlgorithm 的 setup/ask/tell/replay 逻辑，不推荐长期保留。

## 7. Single-agent 与 Multi-agent 如何使用同一结构

### 7.1 Single-agent

single-agent controller 的 deliberate() 只需要：

1. 构造 prompt；
2. 调用 RoleRuntime.invoke()；
3. parser 得到 config；
4. 返回 CommitCandidate；
5. observe() 更新 memory。

~~~python
class SingleAgentController:
    def deliberate(self, context):
        reply = self.runtime.invoke(
            self.role,
            prompt=self.prompt_builder(context),
            context={"evaluation_index": context.evaluation_index},
        )
        config = self.response_parser(reply)
        return CommitCandidate(config=config)
~~~

### 7.2 Multi-agent

multi-agent controller 仍然返回一个 CommitCandidate，只是内部调用多个 role：

~~~python
class ProposeReviewController:
    def deliberate(self, context):
        draft = self.runtime.invoke(
            self.proposer,
            prompt=self.build_proposer_prompt(context),
            context={},
        )
        review = self.runtime.invoke(
            self.reviewer,
            prompt=self.build_review_prompt(context, draft),
            context={"draft": draft},
        )
        config = self.final_parser(draft, review)
        return CommitCandidate(config=config)
~~~

roles、messages、agenda 或 graph 都是 controller 私有逻辑。只有当多个方法真的共享同一种 graph 语义时，才提取可选 graph helper；不要先把 graph 变成所有 agentic 方法的必经层。

### 7.3 EoH

EoH controller 内部的 EoHEngine 可以使用同一个 RoleRuntime 生成、交叉或改写代码。外层依然只看到 CommitCandidate(config={"program": source})。

因此三种方法共享的是 controller 生命周期和 role service，不是相同的私有状态结构。

## 8. Replay 与状态恢复

建议明确两级状态源：

| 状态 | 权威来源 | 恢复方式 |
| --- | --- | --- |
| 已提交 program | trials.jsonl 的 config | replay |
| 对应 fitness/status | trials.jsonl 的 observation | replay |
| lineage | suggestion metadata | replay |
| 已观察 population | EoH observe() | 由 replay 重建 |
| pending proposal | controller snapshot | 可选恢复或丢弃后重提 |
| 模型会话、随机数内部状态 | controller snapshot | restore |
| 通用 trial id / eval time | MetricLogger resume state | 现有逻辑 |

对最小 EoH 实现，只用 replay 就足够恢复已观察 population。snapshot 不是接入前置条件。

但当前 AgenticPolicyAlgorithm 声明 snapshot/restore 后从不调用，建议补充以下能力：

~~~python
class ControllerStateStore(Protocol):
    def load(self) -> Mapping[str, Any] | None: ...
    def save(
        self,
        *,
        last_committed_trial_id: int | None,
        controller_state: Mapping[str, Any],
    ) -> None: ...
~~~

保存时机应在 tell() 完成后。若进程死在 ask() 与 Task.evaluate() 之间，可以丢弃 pending proposal并重新生成；若要求严格复用 pending，则 ask() 后也要保存 pending state。

不建议 replay 时重新执行历史程序或重新调用 LLM。历史 program 和 fitness 已在 trial ledger 中，恢复只需 observe()。

## 9. Program Task 的严格要求

程序是字符串输入，不代表它可以被当作普通可信数据。正式 benchmark 至少应固定：

- 源码最大长度；
- 入口函数或命令 contract；
- 输入输出 schema；
- Python/编译器/runtime 版本；
- hard timeout；
- CPU、内存、进程数和临时磁盘限制；
- 网络策略；
- 只读 benchmark 输入；
- stdout/stderr 截断；
- INVALID、FAILED、SUCCESS 的判定；
- 无效程序是否占 trial。建议占用，因为已经提交并进入 Task.evaluate()；
- 随机种子和重复性规则；
- sandbox/runtime 版本写入 result metadata。

Program Task 必须只从公开 task context 和正式 evaluator 获得评分信息。不能读取 known optimum 或测试答案给 controller 额外 selection signal。

## 10. 注册方式

### 10.1 当前仓库的最小改动

一个 EoH 方法目前大致需要：

1. 新增 EoHPolicy 与 EoHEngine 实现；
2. factory 返回 AgenticPolicyAlgorithm(EoHPolicy(...))；
3. 在 AGENTIC_METHOD_REGISTRY 注册；
4. 在顶层 ALGORITHM_REGISTRY 再加一次入口；
5. 若有专属 CLI 参数，修改 bbo/run.py；
6. 添加 task registry 和 program task description。

示意 factory：

~~~python
def _eoh_factory(**kwargs):
    run_dir = kwargs.pop("run_dir", None)
    engine = build_eoh_engine(**kwargs)
    policy = EoHPolicy(engine=engine)
    return AgenticPolicyAlgorithm(policy)
~~~

这里 run_dir 应最终由统一 state/event service 消费，而不是每个 factory 自己丢弃或解释。

### 10.2 推荐收敛

权威 MethodSpec 应同时提供：

- name / aliases / family；
- controller factory；
- typed method config；
- roles 与 tool profiles；
- runtime requirements；
- event mapper；
- capability flags。

顶层 create_algorithm() 应能从同一 spec 自动发现 agentic 方法。这样新增 EoH 只写方法模块和一条 spec，不再同步两个 registry 和 CLI 分支。

## 11. 建议实施顺序

### Phase 1：最小 EoH，可立即做

1. 实现 ProgramOptimizationTask 与隔离 ProgramRunner。
2. 实现 EoHPolicy，直接使用现有 AgenticPolicyAlgorithm。
3. 加入 EoH factory 和当前两处 registry。
4. 加 sequential smoke、invalid program、replay 和 incumbent 测试。
5. 明确一段 program 等于一个 benchmark trial。

### Phase 2：把 controller 路径做成正式入口

1. 让 AgenticMethodSpec.factory 创建 controller，而不是任意完整 Algorithm。
2. registry 统一套 AgenticPolicyAlgorithm。
3. 让 RoleSpec 的 model_route、prompt_builder、response_parser 和 tool_profile 真正被 RoleRuntime 消费。
4. 用同一 adapter 迁移一个 single-agent 方法和 PABLO。
5. 将旧名称保留为兼容 alias，不新增第二套 protocol。

### Phase 3：恢复和注册收敛

1. 接通 snapshot()/restore()。
2. 定义 tell 后的 state commit point。
3. 修复正常 StopOptimization 支持，避免把停止当异常。
4. 合并 agentic 与顶层 algorithm registry。
5. 把方法 CLI 参数下放到 typed config。

不建议的 Phase：

- 不新增通用 artifact 层；
- 不新增通用 population 层；
- 不新增 EoH 专属 benchmark loop；
- 不让算法内部调用 Task；
- 不把 SkyDiscover 的双层 meta-evolution 当成 program task 模板。

## 12. 验收测试

### Program Task

- 多行源码可以通过 StringParam 和 JSONL round-trip；
- 缺少 program、超长源码和错误类型被正确判定；
- 默认模板能通过 sanity_check；
- 每次成功 ask 只触发一次 Task.evaluate；
- 无效程序占用一个 trial 且不会静默替换；
- sandbox timeout 能杀死整个进程树；
- result 只暴露正式 fitness 和允许的 metrics。

### EoH Controller

- setup 拒绝非单一 program StringParam 的任务；
- deliberate 不调用 Task 或持有 Task 实例；
- 一个 pending proposal 未 observe 前不能再次 propose；
- observe 对成功、失败和无效程序都能更新一致状态；
- maximize/minimize selection 都正确；
- replay 后 population、proposal_index 和 incumbent 一致；
- snapshot 是 JSON serializable；
- lineage metadata 随 TrialRecord 完整 round-trip。

### 通用扩展

- 一个 single-agent controller 和一个 multi-agent controller 使用同一 adapter；
- RoleSpec 字段由 runtime 实际消费，而不是只做 registry metadata；
- 新方法只增加一个方法模块和一个 MethodSpec；
- controller 不需要了解 Experimenter、MetricLogger 或 task registry。

## 13. 最终建议

你的判断是对的：为 EoH 单开 artifact evolution 层会让当前框架变得冗余。

最小且长期合理的设计是：

~~~text
program 是 StringParam
fitness 来自外部 Task.evaluate
population 是 EoH 私有状态
AgenticPolicy/Controller 是统一方法扩展点
AgenticPolicyAlgorithm 是唯一 adapter
RoleRuntime 是可复用服务
~~~

当前核心已经能跑这种方法。真正应补的是让现有 policy/controller 路径成为正式、完整且可恢复的扩展入口，而不是再发明一套 EoH 协议。
