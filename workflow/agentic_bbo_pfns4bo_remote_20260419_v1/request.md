# 用户请求

更新时间：2026-04-19 CST

## 1. 原始需求

创建一个新的 workflow，用来指导后续 agent 在远程 `Johnny1145/agentic-bbo` 仓库中接入 `pfns4bo`，并完成 smoke 级别验证。

后续执行 agent 需要完成：

- 在 `agentic-bbo` 中新增 `pfns4bo` 算法实现。
- 将 `pfns4bo` 相关依赖接入为 optional extra。
- numeric tasks 使用 PFNs4BO 官方连续接口。
- `oer_demo` 与 `molecule_qed_demo` 采用固定的 pool-based 离散接口。
- 对指定任务运行最小 smoke。
- 记录所有命令、结果、artifact 路径和 blocker。

## 2. 固定决策

- Workflow 目录：
  - `D:\Users\Johnny\Documents\New project\workflow\agentic_bbo_pfns4bo_remote_20260419_v1`
- 目标仓库：
  - `https://github.com/Johnny1145/agentic-bbo`
- 算法名：
  - `pfns4bo`
- PFNs 范围：
  - 覆盖全部既定 scientific tasks
  - mixed/categorical 部分明确采用 pool-based 语义
- 默认模型：
  - `hebo_plus`
- 默认 pool size：
  - `256`
- 验证矩阵：
  - `branin_demo`
  - `her_demo`
  - `hea_demo`
  - `bh_demo`
  - `oer_demo`
  - `molecule_qed_demo`

## 3. 不做事项

- 不在这轮 workflow 中训练新的 PFN 模型。
- 不在这轮 workflow 中修改 scientific task 本体的数据语义。
- 不强求 mixed/categorical task 保持逐点连续优化语义。
- 不把失败伪装成自动降级成功。
- 不把本地 workflow 目录误当成目标仓库实现代码。

## 4. 执行阶段必须补全的远程信息

执行 agent 必须在 `exp_result.md` 中补全：

- 远程 host
- 远程仓库路径
- 远程分支
- 远程 Python 版本
- 远程 uv 版本
- `agentic-bbo` base commit/ref
- PFNs 模型缓存路径
- PFNs 模型下载状态
- artifact root
- smoke 输出根目录

## 5. 交付物

本 workflow 的交付物是一个可直接交给远程 agent 执行的 `pfns4bo` 接入包，而不是本地代码实现。

如果远程执行被阻塞，必须在 `exp_result.md` 中记录可恢复的 blocker，便于下一个 agent 接着做。
