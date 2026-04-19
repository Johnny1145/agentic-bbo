# 远程 Codex 执行 Prompt

请在远程环境中执行以下 workflow：

```text
D:\Users\Johnny\Documents\New project\workflow\agentic_bbo_optuna_tpe_remote_20260419_v1
```

或同步到远程后的同名目录。

## 1. 必读文件

请按顺序阅读：

1. `Agent.md`
2. `request.md`
3. `exp_task.md`
4. `algorithm_manifest.md`
5. `dependency_manifest.md`
6. `exp_result.md`

然后继续阅读目标仓库：

7. `README.md`
8. `bbo/algorithms/registry.py`
9. `bbo/run.py`
10. `bbo/core/algo.py`
11. `bbo/core/adapters.py`
12. `bbo/core/experimenter.py`
13. `bbo/core/space.py`

## 2. 任务目标

在远程 `Johnny1145/agentic-bbo` 仓库中接入 `optuna_tpe` 算法，并让它能用于以下 smoke 验证：

```text
branin_demo
her_demo
oer_demo
molecule_qed_demo
```

目标仓库：

```text
https://github.com/Johnny1145/agentic-bbo
```

## 3. 开始修改前必须做

先在 `exp_result.md` 中补全：

- remote host
- remote repo path
- remote branch
- Python version
- `uv` version
- `agentic-bbo` base commit/ref
- artifact root
- smoke output root

## 4. 必须实现

- 新增 `optuna_tpe` 算法实现。
- 注册公开算法名 `optuna_tpe`。
- 新增 optional extra：`optuna`。
- 保持 `suite` 仅包含现有传统算法。
- 让 `python -m bbo.run --algorithm optuna_tpe --task <task>` 可运行。
- mixed/categorical task 不允许报 numeric-only 限制。
- 为 `optuna_tpe` 新增 example、README 更新和测试。
- 支持 replay / resume 的历史重建。

不允许把 `optuna_tpe` 做成多 sampler 总线。
不允许 silent fallback 到 random search。

## 5. 必跑验证

运行：

```bash
uv sync --extra dev --extra optuna --extra bo-tutorial
uv run python -m compileall -q bbo examples tests
uv run pytest
uv run python -m bbo.run --algorithm optuna_tpe --task branin_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm optuna_tpe --task her_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm optuna_tpe --task oer_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
uv run python -m bbo.run --algorithm optuna_tpe --task molecule_qed_demo --max-evaluations 6 --results-root artifacts/optuna_tpe_smoke
```

## 6. 完成后必须更新

更新 `exp_result.md`：

- 依赖安装状态
- 修改文件列表
- 测试状态
- smoke 命令状态
- JSONL 路径
- summary 路径
- plot 路径
- blocker 或 completed 状态

## 7. 最终要求

只有当以下条件同时满足时，才可将 workflow 标记为完成：

1. `optuna_tpe` 已可从 CLI 调用
2. `optuna` 已作为 optional extra 接入
3. 四个 smoke 已真实执行或记录可恢复 blocker
4. `exp_result.md` 已完整更新
