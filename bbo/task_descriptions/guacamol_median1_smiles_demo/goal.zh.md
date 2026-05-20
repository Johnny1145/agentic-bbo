# 目标

实现来源：`bbo/tasks/scientific/guacamol_smiles.py`，`_score_mol()`。

最大化 `median1_score`。

优化目标：最小化 `median1_loss = 1 - median1_score`。

任务定义中记录的 fingerprint type：`ECFP4`。

实现中使用 `_geometric_mean()` 计算 score。
