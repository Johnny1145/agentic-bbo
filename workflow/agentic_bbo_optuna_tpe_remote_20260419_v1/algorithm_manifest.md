# 算法清单

更新时间：2026-04-19 CST

本清单固定 `optuna_tpe` 的实现语义，执行 agent 不应在远程临时改变。

## 1. 公开算法名

- 算法名：`optuna_tpe`
- family：`model_based`
- 目标文件：`bbo/algorithms/model_based/optuna_tpe.py`

## 2. 核心行为

- 单目标优化
- 使用 Optuna 官方 ask/tell 风格
- 使用内存内 `Study`
- 默认 sampler 固定为：

```python
optuna.samplers.TPESampler(seed=<run_seed>, n_startup_trials=5)
```

## 3. 搜索空间映射

必须显式覆盖：

- `FloatParam`
  - 保持上下界
  - 若原参数是 `log=True`，映射时保留 log 语义
- `IntParam`
  - 保持上下界
  - 若原参数是 `log=True`，映射时保留 log 语义
- `CategoricalParam`
  - choices 顺序必须稳定

不允许：

- 丢失默认值语义
- 把 categorical 转成伪 numeric 再交给 TPE
- 改写 task 的原始搜索空间定义

## 4. Replay / Resume 语义

必须支持：

- 从 logger 历史中重建内部状态
- replay 后继续 ask/tell
- incumbent 结果与历史一致

推荐做法：

- 用历史 observation 逐条重建 completed / failed / invalid trial
- 明确处理 primary objective 的 minimize 方向

## 5. 限制

- 仅支持单目标
- 不需要在本轮支持多个 sampler
- 不需要在本轮支持多目标或 pruner
- 不需要在本轮提供额外 CLI sampler 参数

## 6. smoke 目标

必须至少覆盖：

- `branin_demo`
- `her_demo`
- `oer_demo`
- `molecule_qed_demo`

这四个 task 共同验证：

- synthetic numeric
- scientific numeric
- scientific mixed
- scientific large categorical

## 7. 失败时的记录要求

如果远程执行失败，必须在 `exp_result.md` 中明确记录：

- 失败命令
- 失败 task
- 报错摘要
- 是否为依赖问题、搜索空间映射问题、resume 问题或 task 语义问题
- 推荐下一步
