# 用户请求

更新时间：2026-04-19 CST

## 1. 原始需求

创建一个新的 workflow，用来指导后续 agent 在远程 `Johnny1145/agentic-bbo` 仓库中接入 `optuna_tpe`，并完成 smoke 级别验证。

后续执行 agent 需要完成：

- 在 `agentic-bbo` 中新增 `optuna_tpe` 算法实现。
- 将 `optuna` 作为 optional extra 接入，而不是基础依赖。
- 让 `optuna_tpe` 兼容 mixed/categorical search space。
- 让 `python -m bbo.run --algorithm optuna_tpe --task <task>` 能运行。
- 对 `branin_demo`、`her_demo`、`oer_demo`、`molecule_qed_demo` 跑最小 smoke。
- 记录所有命令、结果、artifact 路径和 blocker。

## 2. 固定决策

- Workflow 目录：
  - `D:\Users\Johnny\Documents\New project\workflow\agentic_bbo_optuna_tpe_remote_20260419_v1`
- 目标仓库：
  - `https://github.com/Johnny1145/agentic-bbo`
- 算法名：
  - `optuna_tpe`
- Optuna 范围：
  - 只做 v1 单算法入口
  - 不做多 sampler 总线
- 验证矩阵：
  - `branin_demo`
  - `her_demo`
  - `oer_demo`
  - `molecule_qed_demo`
- 依赖：
  - `optuna`
  - scientific task smoke 需要的 `bo-tutorial`

## 3. 不做事项

- 不在这轮 workflow 中接入多个 Optuna sampler。
- 不把 `optuna_tpe` 合并进 `suite`。
- 不做完整实验矩阵。
- 不修改无关 logger、plotting 或 task 代码。
- 不把本地 workflow 目录误当成目标仓库实现代码。

## 4. 执行阶段必须补全的远程信息

执行 agent 必须在 `exp_result.md` 中补全：

- 远程 host
- 远程仓库路径
- 远程分支
- 远程 Python 版本
- 远程 uv 版本
- `agentic-bbo` base commit/ref
- artifact root
- smoke 输出根目录

## 5. 交付物

本 workflow 的交付物是一个可直接交给远程 agent 执行的 `optuna_tpe` 接入包，而不是本地代码实现。

如果远程执行被阻塞，必须在 `exp_result.md` 中记录可恢复的 blocker，便于下一个 agent 接着做。
