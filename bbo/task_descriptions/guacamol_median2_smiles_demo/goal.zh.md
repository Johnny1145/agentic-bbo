# 目标

实现来源：`bbo/tasks/scientific/guacamol_smiles.py`，`_score_mol()`。

最大化 `median2_score`。

优化目标：最小化 `median2_loss = 1 - median2_score`。

任务定义中记录的 fingerprint type：`ECFP6`。

实现中使用 `_geometric_mean()`，合并到 tadalafil 和 sildenafil 的 ECFP6 相似度。
