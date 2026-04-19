# 实验任务单

更新时间：2026-04-19 CST

## 0. 任务定位

本任务用于驱动后续执行 agent，在远程 `Johnny1145/agentic-bbo` 仓库中接入 `optuna_tpe` 算法，并完成最小 smoke 验证。

本 workflow 是算法接入任务，不是大规模 benchmark 复现实验。

## 1. 固定来源

### 1.1 目标仓库

- Target repo：
  - `https://github.com/Johnny1145/agentic-bbo`

### 1.2 关键现状

- scientific tasks 已经存在于远程仓库的 `main` 基线上。
- `bo-tutorial` extra 已经存在于远程仓库的 `main` 基线上。
- 本轮新增重点是：
  - 算法实现
  - CLI 接入
  - optional extra `optuna`
  - README / example / tests

### 1.3 开始实现前必须记录

开始修改前必须在 `exp_result.md` 中补全：

- remote host
- remote repo path
- remote branch
- Python version
- `uv` version
- `agentic-bbo` base commit/ref
- artifact root
- smoke output root

## 2. 范围

### 2.1 必须完成

- 在 `bbo/algorithms/` 下新增 model-based 算法目录或等价结构。
- 新增 `optuna_tpe` 算法实现。
- 注册公开算法名 `optuna_tpe`。
- 在 `pyproject.toml` 中新增 optional extra：`optuna`。
- 更新 `bbo/run.py` 使 CLI 能运行 `optuna_tpe`。
- 保持现有 `random_search` / `pycma` / `suite` 行为兼容。
- 为 `optuna_tpe` 新增示例脚本。
- 更新 `README.md` 与 `README.zh.md`。
- 新增测试覆盖：
  - 搜索空间映射
  - mixed/categorical 支持
  - replay / resume 语义
  - incumbents / summary 产出
- 跑最小 smoke：
  - `branin_demo`
  - `her_demo`
  - `oer_demo`
  - `molecule_qed_demo`
- 将所有命令、输出、路径、commit 和 blocker 写入 `exp_result.md`。

### 2.2 明确不做

- 不做多个 Optuna sampler 的统一入口。
- 不增加新的 CLI sampler 参数。
- 不把 `optuna_tpe` 放入默认 `suite`。
- 不改 scientific task 的数据语义。
- 不做完整 benchmark 对比。
- 不伪造 resume / replay。

## 3. 实现要求

### 3.1 推荐新增文件

新增：

```text
bbo/algorithms/model_based/
  __init__.py
  optuna_tpe.py
  optuna_utils.py
```

### 3.2 Registry 与 CLI

修改 `bbo/algorithms/registry.py`：

- 注册 `optuna_tpe`
- family 固定为 `model_based`
- mixed/categorical task 不得标记为 `numeric_only`

修改 `bbo/algorithms/__init__.py`：

- 导出 `OptunaTpeAlgorithm`

修改 `bbo/run.py`：

- `--algorithm` choices 来自完整算法注册表
- `suite` 仍只保留传统算法路径
- `run_single_experiment()` 不改主流程，只增加 `optuna_tpe` 可选分支

### 3.3 Optuna 语义

必须固定为：

- 单目标优化
- 内存内 `Study`
- `TPESampler(seed=<run_seed>, n_startup_trials=5)`
- 使用 ask/tell 风格

搜索空间映射必须满足：

- `FloatParam` -> Optuna float distribution
- `IntParam` -> Optuna int distribution
- `CategoricalParam` -> Optuna categorical distribution

如任务不是单目标，必须报清晰错误，不允许隐式降级。

### 3.4 Replay / Resume

必须支持从现有 `TrialObservation` 历史重建内部状态。

最小要求：

- logger 历史可 replay
- replay 后继续 ask 不应破坏 trial 顺序
- incumbents 与 summary 保持一致

## 4. 示例与文档

必须新增：

```text
examples/run_optuna_tpe_demo.py
```

README 中至少补充：

- 安装命令
- `optuna_tpe` 的算法名
- mixed/categorical 支持说明
- smoke 运行示例
- `suite` 不包含 `optuna_tpe`

## 5. 必跑验证命令

实现完成后，在远程仓库上运行：

```bash
uv sync --extra dev --extra optuna --extra bo-tutorial
uv run python -m compileall -q bbo examples tests
uv run pytest
uv run python -m bbo.run --algorithm optuna_tpe --task branin_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm optuna_tpe --task her_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm optuna_tpe --task oer_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm optuna_tpe --task molecule_qed_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
```

## 6. 验收标准

- `optuna_tpe` 已被注册并可从 CLI 调用。
- `optuna` 已作为 optional extra 接入。
- mixed/categorical task 可运行，不报 numeric-only 错误。
- replay / resume 有测试覆盖。
- 四个 smoke run 均写出 `trials.jsonl`。
- 四个 smoke run 均写出 `summary.json`。
- 每个 `summary.json` 包含 `best_primary_objective`。
- artifact 路径均写入 `exp_result.md`。
- 失败的 smoke 不得隐藏；blocker 必须包含命令、错误摘要、原因判断和下一步。

## 7. 结果记录要求

至少将以下内容写入 `exp_result.md`：

- 远程 benchmark 信息
- git branch 与 commit
- `optuna` 安装命令与状态
- 修改文件列表
- 测试状态
- 每个 smoke 的命令、状态、JSONL、summary、plot 路径
- blocker
- next steps
