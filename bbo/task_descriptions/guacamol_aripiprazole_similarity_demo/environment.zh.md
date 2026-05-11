# 环境设置

一种已验证的环境配置方式是：

```bash
uv sync --extra dev --extra bo-tutorial
```

最小 smoke 命令：

```bash
uv run --extra bo-tutorial python -m bbo.run --algorithm random_search --task guacamol_aripiprazole_similarity_demo --max-evaluations 3 --no-plots
```

任务需要 RDKit、`selfies`，以及仓库内置的分子数据文件。
