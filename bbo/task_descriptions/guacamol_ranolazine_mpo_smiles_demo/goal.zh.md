# 目标

实现来源：`bbo/tasks/scientific/guacamol_smiles.py`，`_score_mol()`。

最大化 `ranolazine_mpo_score`。

优化目标：最小化 `ranolazine_mpo_loss = 1 - ranolazine_mpo_score`。

任务定义中记录的 fingerprint type：`AP`。

实现中使用 `_geometric_mean()`，合并 Ranolazine AP 相似度、logP、氟原子数量和 TPSA modifier。
