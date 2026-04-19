# 依赖清单

更新时间：2026-04-19 CST

本 workflow 需要把 PFNs4BO 相关依赖接入 benchmark，但不能让基础安装变重。

## 1. 推荐 optional extras

建议在 `pyproject.toml` 中新增：

### `pfns4bo`

建议至少覆盖：

```toml
pfns4bo = [
  "pfns4bo==0.1.5",
]
```

如果远程环境实测发现 wheel / solver 对间接依赖不稳定，可以在远程实现时补充显式依赖，但必须写入 `exp_result.md`。

## 2. scientific smoke 所需额外依赖

由于本 workflow 的验证矩阵包含 scientific tasks，远程 smoke 还需要现有：

```text
bo-tutorial
```

执行时推荐使用：

```bash
uv sync --extra dev --extra pfns4bo --extra bo-tutorial
```

## 3. 依赖导入验证

安装后必须验证：

```bash
uv run python -c "import pfns4bo; print('pfns4bo import ok')"
```

如果需要确认模型与 scientific smoke 相关依赖，可执行：

```bash
uv run python -c "import pfns4bo, pandas, sklearn, scipy, openpyxl, tqdm; from rdkit import Chem; from rdkit.Chem import QED; print('pfns4bo + bo-tutorial deps ok')"
```

## 4. 模型下载与缓存

PFNs 模型默认允许自动下载，但必须记录：

- 下载是否发生
- 下载源
- 缓存路径
- 若失败，失败摘要

如果远程环境禁止联网，允许手动预置模型文件，但必须把实际路径写入 `exp_result.md`。

## 5. 规则

- 不要把 `pfns4bo` 放进基础 `[project].dependencies`。
- 不要把 `pfns4bo` 混进 `bo-tutorial` extra。
- 不要把模型下载失败伪装成算法运行成功。
- 安装命令、导入检查、模型下载状态和失败摘要必须写入 `exp_result.md`。
