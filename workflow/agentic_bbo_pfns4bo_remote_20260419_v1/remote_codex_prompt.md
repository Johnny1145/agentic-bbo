# 远程 Codex 执行 Prompt

请在远程环境中执行以下 workflow：

```text
D:\Users\Johnny\Documents\New project\workflow\agentic_bbo_pfns4bo_remote_20260419_v1
```

或同步到远程后的同名目录。

## 1. 必读文件

请按顺序阅读：

1. `Agent.md`
2. `request.md`
3. `exp_task.md`
4. `algorithm_manifest.md`
5. `encoding_manifest.md`
6. `dependency_manifest.md`
7. `exp_result.md`

然后继续阅读目标仓库：

8. `README.md`
9. `bbo/algorithms/registry.py`
10. `bbo/run.py`
11. `bbo/core/algo.py`
12. `bbo/core/adapters.py`
13. `bbo/core/experimenter.py`
14. scientific tasks 的搜索空间实现

## 2. 任务目标

在远程 `Johnny1145/agentic-bbo` 仓库中接入 `pfns4bo` 算法，并让它能用于以下 smoke 验证：

```text
branin_demo
her_demo
hea_demo
bh_demo
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
- PFNs model cache path
- PFNs model download status
- artifact root
- smoke output root

## 4. 必须实现

- 新增 `pfns4bo` 算法实现。
- 注册公开算法名 `pfns4bo`。
- 新增 optional extra：`pfns4bo`。
- 保持 `suite` 仅包含现有传统算法。
- 让 `python -m bbo.run --algorithm pfns4bo --task <task>` 可运行。
- numeric tasks 走连续接口。
- `oer_demo` 与 `molecule_qed_demo` 走 pool-based 离散接口。
- 为 `pfns4bo` 新增 example、README 更新和测试。
- 支持 replay / resume 的历史重建。

不允许 silent fallback 到 random search。
不允许伪造模型下载成功。
不允许偏离 `encoding_manifest.md` 的固定编码规则。

## 5. 必跑验证

运行：

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

## 6. 完成后必须更新

更新 `exp_result.md`：

- 依赖安装状态
- 模型缓存路径与下载状态
- 修改文件列表
- 测试状态
- smoke 命令状态
- JSONL 路径
- summary 路径
- plot 路径
- blocker 或 completed 状态

## 7. 最终要求

只有当以下条件同时满足时，才可将 workflow 标记为完成：

1. `pfns4bo` 已可从 CLI 调用
2. `pfns4bo` 已作为 optional extra 接入
3. numeric 与 pool-based 两条路线都已真实验证或记录可恢复 blocker
4. `exp_result.md` 已完整更新
