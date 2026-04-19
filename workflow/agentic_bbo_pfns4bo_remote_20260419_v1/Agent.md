# Agent 指南

更新时间：2026-04-19 CST

## 0. 任务定位

本 workflow 用来驱动后续执行 agent，在远程 `Johnny1145/agentic-bbo` 仓库中接入 `pfns4bo` 算法，并完成 smoke 级别验证。

本任务是算法接入与 smoke 验证，不是完整 benchmark 复现实验。

固定范围：

- 目标仓库：`https://github.com/Johnny1145/agentic-bbo`
- 目标算法：`pfns4bo`
- 验证矩阵：
  - `branin_demo`
  - `her_demo`
  - `hea_demo`
  - `bh_demo`
  - `oer_demo`
  - `molecule_qed_demo`
- 后端策略：
  - numeric task 走连续接口
  - OER / molecule 走 pool-based 离散接口
- PFNs 模型默认：
  - `hebo_plus`
- 默认 pool size：
  - `256`
- 远程 host / 路径 / 分支：创建 workflow 时未知，执行阶段必须补全

## 1. 执行前必读顺序

开始修改远程仓库前，必须按顺序阅读：

1. `Agent.md`
2. `request.md`
3. `exp_task.md`
4. `algorithm_manifest.md`
5. `encoding_manifest.md`
6. `dependency_manifest.md`
7. `exp_result.md`
8. `remote_codex_prompt.md`
9. 目标仓库 `README.md`
10. 目标仓库 `bbo/algorithms/registry.py`
11. 目标仓库 `bbo/run.py`
12. 目标仓库 `bbo/core/algo.py`
13. 目标仓库 `bbo/core/adapters.py`
14. 目标仓库 `bbo/core/experimenter.py`
15. 目标仓库 scientific tasks 的搜索空间实现

如果文件之间存在冲突，优先级如下：

1. `exp_task.md`
2. `encoding_manifest.md`
3. `algorithm_manifest.md`
4. `dependency_manifest.md`
5. `request.md`
6. `Agent.md`
7. 远程仓库现有源码

## 2. 执行规则

- 不要在本地这个 workflow 工作台里实现仓库代码；只在远程目标仓库中修改。
- 不允许把 `pfns4bo` 依赖加入基础安装。
- 不允许把 mixed/categorical task 静默 fallback 到 random search。
- 不允许用假模型、假 surrogate 或假 descriptor 代替正式实现。
- `oer_demo` 与 `molecule_qed_demo` 必须按 `encoding_manifest.md` 的固定编码规则执行。
- 候选池抽样、编码与 tie-break 必须受 seed 控制。
- 如模型下载、依赖、设备或环境阻塞，必须显式记录 blocker。
- 除非现有接口明确无法表达，否则不要修改 `bbo/core/`。

## 3. 远程实现建议结构

推荐新增：

```text
bbo/algorithms/model_based/
  __init__.py
  pfns4bo.py
  pfns4bo_utils.py
  pfns4bo_encoding.py
```

推荐修改：

```text
bbo/algorithms/__init__.py
bbo/algorithms/registry.py
bbo/run.py
pyproject.toml
README.md
README.zh.md
examples/run_pfns4bo_demo.py
tests/
```

## 4. 完成定义

只有同时满足以下条件，本 workflow 才算完成：

1. 远程 repo 路径、分支、commit 已记录。
2. `pfns4bo` 已注册为公开算法名。
3. `pfns4bo` 已作为 optional extra 接入。
4. 数值任务可走连续接口。
5. `oer_demo` 与 `molecule_qed_demo` 可走 pool-based 接口。
6. PFNs 模型路径、缓存路径与下载状态已记录。
7. `compileall`、`pytest` 和 smoke 命令状态已写入 `exp_result.md`。
8. 每个 smoke run 的 `trials.jsonl` 与 `summary.json` 路径已记录。
9. `summary.json` 包含 `best_primary_objective`。
10. `exp_result.md` 被更新为 `completed`，或记录了明确 blocker。
