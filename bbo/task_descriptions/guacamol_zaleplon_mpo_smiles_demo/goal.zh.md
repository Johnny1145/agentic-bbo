# 目标

实现来源：`bbo/tasks/scientific/guacamol_smiles.py`，`_score_mol()`。

最大化 `zaleplon_mpo_score`。

优化目标：最小化 `zaleplon_mpo_loss = 1 - zaleplon_mpo_score`。

任务定义中记录的 fingerprint type：`ECFP4`。

实现中使用 `_geometric_mean()`，合并 PMO/PyTDC 0.3.6 的 Zaleplon ECFP4 相似度和分子式分量。
