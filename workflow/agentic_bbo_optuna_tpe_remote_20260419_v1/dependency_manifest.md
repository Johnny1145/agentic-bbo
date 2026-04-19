# 依赖清单

更新时间：2026-04-19 CST

本 workflow 需要把 Optuna 相关依赖接入 benchmark，但不能让基础安装变重。

## 1. 推荐 optional extras

建议在 `pyproject.toml` 中新增：

### `optuna`

```toml
optuna = [
  "optuna>=4.0",
]
```

用途：

- 提供 `TPESampler`
- 提供 ask/tell 风格的 `Study`

## 2. scientific smoke 所需额外依赖

由于本 workflow 的验证矩阵包含 scientific tasks，远程 smoke 还需要现有：

```text
bo-tutorial
```

执行时推荐使用：

```bash
uv sync --extra dev --extra optuna --extra bo-tutorial
```

## 3. 依赖导入验证

安装后必须验证：

```bash
uv run python -c "import optuna; print(optuna.__version__)"
```

如果要同时验证 scientific smoke 所需依赖，可执行：

```bash
uv run python -c "import optuna, pandas, sklearn, scipy, openpyxl, tqdm; from rdkit import Chem; from rdkit.Chem import QED; print('optuna + bo-tutorial deps ok')"
```

## 4. 规则

- 不要把 `optuna` 放进基础 `[project].dependencies`。
- 不要把 `optuna` 混进 `bo-tutorial` extra。
- 不要在 smoke 中隐式依赖未声明的 extras。
- 安装命令、导入检查和失败摘要必须写入 `exp_result.md`。
