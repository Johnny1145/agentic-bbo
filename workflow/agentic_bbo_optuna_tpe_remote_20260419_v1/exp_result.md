# 实验结果记录

更新时间：2026-04-20 Australia/Perth

## 0. 状态

- 总体状态：`completed`
- 当前阶段：`optuna_tpe` 已接入、验证完成，workflow 收尾
- 执行 agent：Codex GPT-5
- 最后更新：2026-04-20 Australia/Perth 远程执行

## 1. 远程 Benchmark 信息

开始实现前必须补全：

| 字段 | 值 |
|---|---|
| remote host | `amax3` |
| remote repo path | `/home/trx/cm/agentic-bbo` |
| remote branch | `feat/optuna-tpe` |
| remote Python version | `Python 3.11.5` |
| remote uv version | `uv 0.11.1 (a6042f67f 2026-03-24 x86_64-unknown-linux-gnu)` |
| agentic-bbo base commit/ref | `395f31bd85c19fb6f42118ccd12ef1ff9f852877` |
| artifact root | `/home/trx/cm/agentic-bbo/artifacts` |
| smoke output root | `/home/trx/cm/agentic-bbo/artifacts/optuna_tpe_smoke` |

## 2. 依赖安装记录

| 步骤 | 命令 | 状态 | 备注 |
|---|---|---|---|
| 安装 Optuna + scientific smoke 依赖 | `uv sync --extra dev --extra optuna --extra bo-tutorial` | completed | 见 `workflow/agentic_bbo_optuna_tpe_remote_20260419_v1/logs/uv_sync.log`；安装到 `optuna==4.8.0` |
| Optuna 导入检查 | `uv run python -c "import optuna; print(optuna.__version__)"` | completed | 输出 `4.8.0`；见 `workflow/agentic_bbo_optuna_tpe_remote_20260419_v1/logs/optuna_import_check.log` |
| 组合依赖导入检查 | `uv run python -c "import optuna, pandas, sklearn, scipy, openpyxl, tqdm; from rdkit import Chem; from rdkit.Chem import QED; print('optuna + bo-tutorial deps ok')"` | completed | 输出 `optuna + bo-tutorial deps ok`；见 `workflow/agentic_bbo_optuna_tpe_remote_20260419_v1/logs/deps_import_check.log` |

## 3. 实现状态

| 组件 | 状态 | 文件/路径 | 备注 |
|---|---|---|---|
| model_based package | completed | `bbo/algorithms/model_based/` | 新增 package 并导出 `OptunaTpeAlgorithm` |
| optuna adapter | completed | `bbo/algorithms/model_based/optuna_tpe.py` | ask/tell、单目标校验、TPE sampler、resume/replay 历史重建 |
| optuna helper | completed | `bbo/algorithms/model_based/optuna_utils.py` | search-space 映射与 lazy import |
| registry | completed | `bbo/algorithms/registry.py` | 注册公开算法名 `optuna_tpe`，family=`model_based`，非 numeric-only |
| exports | completed | `bbo/algorithms/__init__.py` | 导出 `OptunaTpeAlgorithm` |
| CLI | completed | `bbo/run.py` | `--algorithm` choices 覆盖完整 registry，`suite` 仍保留原传统算法逻辑 |
| optional extra | completed | `pyproject.toml`, `uv.lock` | 新增 optional extra `optuna` 并更新 lock |
| examples | completed | `examples/run_optuna_tpe_demo.py` | 新增 Branin demo 示例 |
| README | completed | `README.md`, `README.zh.md` | 增补安装、CLI、mixed/categorical、smoke 与 suite 说明 |
| tests | completed | `tests/test_optuna_tpe.py` | 覆盖空间映射、mixed/categorical、multi-objective 拒绝、replay/resume、summary 输出 |

## 4. 验证命令记录

| 命令 | 状态 | 输出/log path | 备注 |
|---|---|---|---|
| `uv sync --extra dev --extra optuna --extra bo-tutorial` | completed | `workflow/agentic_bbo_optuna_tpe_remote_20260419_v1/logs/uv_sync.log` | 安装完成，新增 `optuna==4.8.0` |
| `uv run python -m compileall -q bbo examples tests` | completed | `workflow/agentic_bbo_optuna_tpe_remote_20260419_v1/logs/compileall.log` | 通过 |
| `uv run pytest` | completed | `workflow/agentic_bbo_optuna_tpe_remote_20260419_v1/logs/pytest.log` | `23 passed in 35.90s` |
| `uv run python -m bbo.run --algorithm optuna_tpe --task branin_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke` | completed | `workflow/agentic_bbo_optuna_tpe_remote_20260419_v1/logs/branin_demo.log` | 6/6 成功，summary 含 `best_primary_objective` |
| `uv run python -m bbo.run --algorithm optuna_tpe --task her_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke` | completed | `workflow/agentic_bbo_optuna_tpe_remote_20260419_v1/logs/her_demo.log` | 6/6 成功，summary 含 `best_primary_objective` |
| `uv run python -m bbo.run --algorithm optuna_tpe --task oer_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke` | completed | `workflow/agentic_bbo_optuna_tpe_remote_20260419_v1/logs/oer_demo.log` | mixed search space 运行成功 |
| `uv run python -m bbo.run --algorithm optuna_tpe --task molecule_qed_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke` | completed | `workflow/agentic_bbo_optuna_tpe_remote_20260419_v1/logs/molecule_qed_demo.log` | categorical-only search space 运行成功 |

## 5. Smoke 产物

| Task | JSONL path | Summary path | Plot paths | best_primary_objective | 状态 |
|---|---|---|---|---:|---|
| `branin_demo` | `artifacts/optuna_tpe_smoke/branin_demo/optuna_tpe/seed_7/trials.jsonl` | `artifacts/optuna_tpe_smoke/branin_demo/optuna_tpe/seed_7/summary.json` | `artifacts/optuna_tpe_smoke/branin_demo/optuna_tpe/seed_7/plots/trace.png`<br>`artifacts/optuna_tpe_smoke/branin_demo/optuna_tpe/seed_7/plots/distribution.png`<br>`artifacts/optuna_tpe_smoke/branin_demo/optuna_tpe/seed_7/plots/landscape.png` | 5.211335324193094 | completed |
| `her_demo` | `artifacts/optuna_tpe_smoke/her_demo/optuna_tpe/seed_7/trials.jsonl` | `artifacts/optuna_tpe_smoke/her_demo/optuna_tpe/seed_7/summary.json` | `artifacts/optuna_tpe_smoke/her_demo/optuna_tpe/seed_7/plots/trace.png`<br>`artifacts/optuna_tpe_smoke/her_demo/optuna_tpe/seed_7/plots/distribution.png` | 22.53392410810001 | completed |
| `oer_demo` | `artifacts/optuna_tpe_smoke/oer_demo/optuna_tpe/seed_7/trials.jsonl` | `artifacts/optuna_tpe_smoke/oer_demo/optuna_tpe/seed_7/summary.json` | `artifacts/optuna_tpe_smoke/oer_demo/optuna_tpe/seed_7/plots/trace.png`<br>`artifacts/optuna_tpe_smoke/oer_demo/optuna_tpe/seed_7/plots/distribution.png` | 269.2332136346673 | completed |
| `molecule_qed_demo` | `artifacts/optuna_tpe_smoke/molecule_qed_demo/optuna_tpe/seed_7/trials.jsonl` | `artifacts/optuna_tpe_smoke/molecule_qed_demo/optuna_tpe/seed_7/summary.json` | `artifacts/optuna_tpe_smoke/molecule_qed_demo/optuna_tpe/seed_7/plots/trace.png`<br>`artifacts/optuna_tpe_smoke/molecule_qed_demo/optuna_tpe/seed_7/plots/distribution.png` | 0.06854463065121497 | completed |

## 6. 修改文件列表

```text
README.md
README.zh.md
bbo/algorithms/__init__.py
bbo/algorithms/model_based/__init__.py
bbo/algorithms/model_based/optuna_tpe.py
bbo/algorithms/model_based/optuna_utils.py
bbo/algorithms/registry.py
bbo/run.py
examples/run_optuna_tpe_demo.py
pyproject.toml
tests/test_optuna_tpe.py
uv.lock
workflow/agentic_bbo_optuna_tpe_remote_20260419_v1/exp_result.md
```

## 7. 当前 Blockers

- 无。四个 smoke 已真实执行完成。

## 8. 下一步

1. 如需提交远程结果，可在 `feat/optuna-tpe` 上整理 commit。
2. 如需扩展 benchmark，可继续增加更长预算或更多 seed 的 Optuna 对比实验。
