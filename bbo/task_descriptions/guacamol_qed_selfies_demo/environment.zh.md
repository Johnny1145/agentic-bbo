# 环境设置

一种已验证的环境配置方式是：

```bash
uv sync --extra dev --extra bo-tutorial
```

最小 smoke 命令：

```bash
uv run --extra bo-tutorial python -m bbo.run --algorithm random_search --task guacamol_qed_selfies_demo --max-evaluations 3 --no-plots
```

任务需要 RDKit、`selfies`，以及仓库内置的 `bbo/tasks/scientific/data/examples/Molecule/zinc.txt.gz` 文件。
