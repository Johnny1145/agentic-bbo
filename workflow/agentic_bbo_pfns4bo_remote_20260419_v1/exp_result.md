# 实验结果记录

更新时间：2026-04-19 CST

## 0. 状态

- 总体状态：`not_started`
- 当前阶段：workflow 已准备，远程 `pfns4bo` 接入尚未开始
- 执行 agent：TBD
- 最后更新：本地 workflow 起草 agent

## 1. 远程 Benchmark 信息

开始实现前必须补全：

| 字段 | 值 |
|---|---|
| remote host | TBD |
| remote repo path | TBD |
| remote branch | TBD |
| remote Python version | TBD |
| remote uv version | TBD |
| agentic-bbo base commit/ref | TBD |
| PFNs model cache path | TBD |
| PFNs model download status | TBD |
| artifact root | TBD |
| smoke output root | TBD |

## 2. 依赖安装记录

| 步骤 | 命令 | 状态 | 备注 |
|---|---|---|---|
| 安装 PFNs + scientific smoke 依赖 | `uv sync --extra dev --extra pfns4bo --extra bo-tutorial` | not_started | smoke 前必须完成 |
| PFNs 导入检查 | `uv run python -c "import pfns4bo; print('pfns4bo import ok')"` | not_started | 必须完成 |
| 组合依赖导入检查 | `uv run python -c "import pfns4bo, pandas, sklearn, scipy, openpyxl, tqdm; from rdkit import Chem; from rdkit.Chem import QED; print('pfns4bo + bo-tutorial deps ok')"` | not_started | scientific smoke 前建议完成 |

## 3. 模型状态记录

| 项目 | 值 |
|---|---|
| model name | `hebo_plus` |
| model file path | TBD |
| auto download attempted | TBD |
| auto download result | TBD |
| manual staging path | TBD |
| device used | TBD |

## 4. 实现状态

| 组件 | 状态 | 文件/路径 | 备注 |
|---|---|---|---|
| model_based package | not_started | `bbo/algorithms/model_based/` | TBD |
| pfns adapter | not_started | `bbo/algorithms/model_based/pfns4bo.py` | TBD |
| pfns helper | not_started | `bbo/algorithms/model_based/pfns4bo_utils.py` | TBD |
| encoding helper | not_started | `bbo/algorithms/model_based/pfns4bo_encoding.py` | TBD |
| registry | not_started | `bbo/algorithms/registry.py` | TBD |
| exports | not_started | `bbo/algorithms/__init__.py` | TBD |
| CLI | not_started | `bbo/run.py` | TBD |
| optional extra | not_started | `pyproject.toml` | TBD |
| examples | not_started | `examples/run_pfns4bo_demo.py` | TBD |
| README | not_started | `README.md`, `README.zh.md` | TBD |
| tests | not_started | `tests/` | TBD |

## 5. 验证命令记录

| 命令 | 状态 | 输出/log path | 备注 |
|---|---|---|---|
| `uv sync --extra dev --extra pfns4bo --extra bo-tutorial` | not_started | TBD | TBD |
| `uv run python -m compileall -q bbo examples tests` | not_started | TBD | TBD |
| `uv run pytest` | not_started | TBD | TBD |
| `uv run python -m bbo.run --algorithm pfns4bo --task branin_demo --max-evaluations 6 --results-root artifacts/pfns4bo_smoke` | not_started | TBD | TBD |
| `uv run python -m bbo.run --algorithm pfns4bo --task her_demo --max-evaluations 6 --results-root artifacts/pfns4bo_smoke` | not_started | TBD | TBD |
| `uv run python -m bbo.run --algorithm pfns4bo --task hea_demo --max-evaluations 6 --results-root artifacts/pfns4bo_smoke` | not_started | TBD | TBD |
| `uv run python -m bbo.run --algorithm pfns4bo --task bh_demo --max-evaluations 6 --results-root artifacts/pfns4bo_smoke` | not_started | TBD | TBD |
| `uv run python -m bbo.run --algorithm pfns4bo --task oer_demo --max-evaluations 6 --results-root artifacts/pfns4bo_smoke --pfns-pool-size 256` | not_started | TBD | TBD |
| `uv run python -m bbo.run --algorithm pfns4bo --task molecule_qed_demo --max-evaluations 6 --results-root artifacts/pfns4bo_smoke --pfns-pool-size 256` | not_started | TBD | TBD |

## 6. Smoke 产物

| Task | JSONL path | Summary path | Plot paths | best_primary_objective | 状态 |
|---|---|---|---|---:|---|
| `branin_demo` | TBD | TBD | TBD | TBD | not_started |
| `her_demo` | TBD | TBD | TBD | TBD | not_started |
| `hea_demo` | TBD | TBD | TBD | TBD | not_started |
| `bh_demo` | TBD | TBD | TBD | TBD | not_started |
| `oer_demo` | TBD | TBD | TBD | TBD | not_started |
| `molecule_qed_demo` | TBD | TBD | TBD | TBD | not_started |

## 7. 修改文件列表

实现后记录：

```text
TBD
```

## 8. 当前 Blockers

- 远程 benchmark host / path / branch 尚未填写。
- PFNs 模型缓存路径与下载策略尚未在远程环境确认。
- 本 workflow 创建阶段尚未修改远程代码。

## 9. 下一步

1. 补全远程 benchmark 信息。
2. 在远程仓库中创建 feature branch。
3. 接入 `pfns4bo` optional extra。
4. 实现并注册 `pfns4bo`。
5. 落实连续与 pool-based 两套后端。
6. 更新 CLI、README、examples、tests。
7. 运行 compile、pytest 与六个 smoke。
8. 用真实执行结果更新本文件。
