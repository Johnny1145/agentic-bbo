# Agent 指南

更新时间：2026-04-19 CST

## 0. 任务定位

本 workflow 用来驱动后续执行 agent，在远程 `Johnny1145/agentic-bbo` 仓库中接入 `optuna_tpe` 算法，并完成最小验证。

本任务是算法接入与 smoke 验证，不是完整 benchmark 大规模实验。

固定范围：

- 目标仓库：`https://github.com/Johnny1145/agentic-bbo`
- 目标算法：`optuna_tpe`
- 目标 scientific tasks：
  - `her_demo`
  - `oer_demo`
  - `molecule_qed_demo`
- 目标 synthetic task：
  - `branin_demo`
- 依赖策略：
  - `optuna` 与 `bo-tutorial` 保持 optional
  - 不污染基础安装
- 远程 host / 路径 / 分支：创建 workflow 时未知，执行阶段必须补全

## 1. 执行前必读顺序

开始修改远程仓库前，必须按顺序阅读：

1. `Agent.md`
2. `request.md`
3. `exp_task.md`
4. `algorithm_manifest.md`
5. `dependency_manifest.md`
6. `exp_result.md`
7. `remote_codex_prompt.md`
8. 目标仓库 `README.md`
9. 目标仓库 `bbo/algorithms/registry.py`
10. 目标仓库 `bbo/run.py`
11. 目标仓库 `bbo/core/algo.py`
12. 目标仓库 `bbo/core/adapters.py`
13. 目标仓库 `bbo/core/experimenter.py`
14. 目标仓库 `bbo/core/space.py`

如果文件之间存在冲突，优先级如下：

1. `exp_task.md`
2. `algorithm_manifest.md`
3. `dependency_manifest.md`
4. `request.md`
5. `Agent.md`
6. 远程仓库现有源码

## 2. 执行规则

- 不要在本地这个 workflow 工作台里实现仓库代码；只在远程目标仓库中修改。
- 不允许把 `optuna` 依赖加入基础 `[project].dependencies`。
- 不允许把 `optuna_tpe` 做成多 sampler 总线；只允许一个明确算法入口。
- 不允许 silent fallback 到 random search。
- 不允许把 mixed/categorical task 隐式降级为 numeric-only。
- 不允许忽略 replay / resume 语义。
- 不允许伪造 smoke 结果、summary 或 artifact 路径。
- 除非现有接口明确无法表达，否则不要修改 `bbo/core/`。
- 所有远程路径、分支、commit、安装命令、验证命令、失败原因都必须写入 `exp_result.md`。

## 3. 远程实现建议结构

推荐新增：

```text
bbo/algorithms/model_based/
  __init__.py
  optuna_tpe.py
  optuna_utils.py
```

推荐修改：

```text
bbo/algorithms/__init__.py
bbo/algorithms/registry.py
bbo/run.py
pyproject.toml
README.md
README.zh.md
examples/run_optuna_tpe_demo.py
tests/
```

## 4. 完成定义

只有同时满足以下条件，本 workflow 才算完成：

1. 远程 repo 路径、分支、commit 已记录。
2. `optuna_tpe` 已注册为公开算法名。
3. `optuna` 已作为 optional extra 接入。
4. `optuna_tpe` 可通过 `python -m bbo.run --algorithm optuna_tpe` 调用。
5. `optuna_tpe` 能覆盖 `branin_demo`、`her_demo`、`oer_demo`、`molecule_qed_demo`。
6. replay / resume 行为已通过测试或 smoke 间接验证。
7. `compileall`、`pytest` 和 smoke 命令状态已写入 `exp_result.md`。
8. 每个 smoke run 的 `trials.jsonl` 和 `summary.json` 路径已记录。
9. `summary.json` 包含 `best_primary_objective`。
10. `exp_result.md` 被更新为 `completed`，或记录了明确 blocker。
