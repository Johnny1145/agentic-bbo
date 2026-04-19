# 算法清单

更新时间：2026-04-19 CST

本清单固定 `pfns4bo` 的实现语义，执行 agent 不应在远程临时改变。

## 1. 公开算法名

- 算法名：`pfns4bo`
- family：`model_based`
- 目标文件：`bbo/algorithms/model_based/pfns4bo.py`

## 2. 核心行为

- 统一对外暴露一个算法名：`pfns4bo`
- 内部根据 task/search space 类型路由到不同后端
- 默认模型：`hebo_plus`
- 默认 device：由 CLI 指定；未显式指定时由实现选择安全默认值
- 默认 pool size：`256`

## 3. 后端路由

必须固定为：

- fully numeric task：
  - 使用 PFNs4BO 连续接口
- `oer_demo`：
  - 使用 pool-based 离散接口
- `molecule_qed_demo`：
  - 使用 pool-based 离散接口

推荐将：

- `branin_demo`
- `her_demo`
- `hea_demo`
- `bh_demo`

视为 numeric 路线验证对象。

## 4. replay / resume 语义

必须支持：

- 从 logger 历史重建内部状态
- replay 后继续 ask/tell
- 候选池生成与 tie-break 保持可复现

## 5. 模型管理

必须记录：

- 模型名称
- 模型文件实际路径
- 是否自动下载
- 若自动下载失败，失败原因

不允许：

- 使用未记录来源的模型文件
- 自动换成别的模型却不记录

## 6. smoke 目标

必须至少覆盖：

- `branin_demo`
- `her_demo`
- `hea_demo`
- `bh_demo`
- `oer_demo`
- `molecule_qed_demo`

## 7. 失败时的记录要求

如果远程执行失败，必须在 `exp_result.md` 中明确记录：

- 失败命令
- 失败 task
- 报错摘要
- 是否为依赖问题、模型下载问题、设备问题、编码问题或 replay 问题
- 推荐下一步
