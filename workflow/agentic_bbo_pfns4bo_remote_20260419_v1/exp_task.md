# 实验任务单

更新时间：2026-04-19 CST

## 0. 任务定位

本任务用于驱动后续执行 agent，在远程 `Johnny1145/agentic-bbo` 仓库中接入 `pfns4bo` 算法，并完成最小 smoke 验证。

本 workflow 是算法接入任务，不是完整 benchmark 大规模实验。

## 1. 固定来源

### 1.1 目标仓库

- Target repo：
  - `https://github.com/Johnny1145/agentic-bbo`

### 1.2 PFNs 来源

- PFNs4BO 仓库：
  - `https://github.com/automl/PFNs4BO`
- 论文：
  - `https://arxiv.org/abs/2305.17535`
- 默认模型：
  - `hebo_plus`

### 1.3 开始实现前必须记录

开始修改前必须在 `exp_result.md` 中补全：

- remote host
- remote repo path
- remote branch
- Python version
- `uv` version
- `agentic-bbo` base commit/ref
- PFNs model cache path
- PFNs model download status
- artifact root
- smoke output root

## 2. 范围

### 2.1 必须完成

- 在 `bbo/algorithms/` 下新增 model-based 算法目录或等价结构。
- 新增 `pfns4bo` 算法实现。
- 注册公开算法名 `pfns4bo`。
- 在 `pyproject.toml` 中新增 optional extra：`pfns4bo`。
- 更新 `bbo/run.py` 使 CLI 能运行 `pfns4bo`。
- 为 `pfns4bo` 增加必要 CLI 参数：
  - `--pfns-device`
  - `--pfns-pool-size`
  - `--pfns-model`
- 保持现有 `random_search` / `pycma` / `suite` 行为兼容。
- 为 `pfns4bo` 新增示例脚本。
- 更新 `README.md` 与 `README.zh.md`。
- 新增测试覆盖：
  - 后端路由
  - OER 编码
  - molecule descriptor 与归一化
  - pool 生成可复现性
  - replay / resume
- 跑最小 smoke：
  - `branin_demo`
  - `her_demo`
  - `hea_demo`
  - `bh_demo`
  - `oer_demo`
  - `molecule_qed_demo`
- 将所有命令、输出、路径、commit 和 blocker 写入 `exp_result.md`。

### 2.2 明确不做

- 不训练新模型。
- 不修改 scientific task 原始目标定义。
- 不把 mixed/categorical task 强行改成连续 task。
- 不 silent fallback 到 random search。
- 不伪造 pool 或 fake descriptor。

## 3. 实现要求

### 3.1 推荐新增文件

新增：

```text
bbo/algorithms/model_based/
  __init__.py
  pfns4bo.py
  pfns4bo_utils.py
  pfns4bo_encoding.py
```

### 3.2 Registry 与 CLI

修改 `bbo/algorithms/registry.py`：

- 注册 `pfns4bo`
- family 固定为 `model_based`
- `numeric_only=False`

修改 `bbo/run.py`：

- `--algorithm` choices 来自完整算法注册表
- 新增 `--pfns-device`
- 新增 `--pfns-pool-size`
- 新增 `--pfns-model`
- `suite` 不包含 `pfns4bo`

### 3.3 后端路由

必须固定为：

- numeric task：
  - 走 PFNs4BO 连续接口
- `oer_demo`：
  - 走 pool-based 离散接口
- `molecule_qed_demo`：
  - 走 pool-based 离散接口

不得在远程临时改变这些规则。

### 3.4 pool-based 规则

固定要求：

- 默认 pool size：`256`
- 候选池生成必须受 seed 控制
- 打分与 tie-break 必须受 seed 控制
- replay / resume 后必须仍然稳定

## 4. 编码要求

编码细节以 `encoding_manifest.md` 为准。

最关键约束：

- OER：
  - `Metal_1/2/3` one-hot
  - 数值与整数特征 min-max 归一化
- Molecule：
  - 候选来自原始 SMILES 列表
  - 使用固定 RDKit 描述符集
  - 描述符做 dataset-level min-max 归一化

## 5. 必跑验证命令

实现完成后，在远程仓库上运行：

```bash
uv sync --extra dev --extra pfns4bo --extra bo-tutorial
uv run python -m compileall -q bbo examples tests
uv run pytest
uv run python -m bbo.run --algorithm pfns4bo --task branin_demo --max-evaluations 6 --results-root artifacts/pfns4bo_smoke
uv run python -m bbo.run --algorithm pfns4bo --task her_demo --max-evaluations 6 --results-root artifacts/pfns4bo_smoke
uv run python -m bbo.run --algorithm pfns4bo --task hea_demo --max-evaluations 6 --results-root artifacts/pfns4bo_smoke
uv run python -m bbo.run --algorithm pfns4bo --task bh_demo --max-evaluations 6 --results-root artifacts/pfns4bo_smoke
uv run python -m bbo.run --algorithm pfns4bo --task oer_demo --max-evaluations 6 --results-root artifacts/pfns4bo_smoke --pfns-pool-size 256
uv run python -m bbo.run --algorithm pfns4bo --task molecule_qed_demo --max-evaluations 6 --results-root artifacts/pfns4bo_smoke --pfns-pool-size 256
```

## 6. 验收标准

- `pfns4bo` 已被注册并可从 CLI 调用。
- `pfns4bo` 已作为 optional extra 接入。
- numeric tasks 可通过连续接口运行。
- `oer_demo` 与 `molecule_qed_demo` 可通过 pool-based 路线运行。
- PFNs 模型路径与下载状态已记录。
- 六个 smoke run 均写出 `trials.jsonl`。
- 六个 smoke run 均写出 `summary.json`。
- 每个 `summary.json` 包含 `best_primary_objective`。
- artifact 路径均写入 `exp_result.md`。
- 失败的 smoke 不得隐藏；blocker 必须包含命令、错误摘要、原因判断和下一步。

## 7. 结果记录要求

至少将以下内容写入 `exp_result.md`：

- 远程 benchmark 信息
- git branch 与 commit
- `pfns4bo` 安装命令与状态
- PFNs 模型缓存路径与下载状态
- 修改文件列表
- 测试状态
- 每个 smoke 的命令、状态、JSONL、summary、plot 路径
- blocker
- next steps
