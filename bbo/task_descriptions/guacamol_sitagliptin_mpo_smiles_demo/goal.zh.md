# 目标

实现来源：`bbo/tasks/scientific/guacamol_smiles.py`，`_score_mol()`。

最大化 `sitagliptin_mpo_score`。

优化目标：最小化 `sitagliptin_mpo_loss = 1 - sitagliptin_mpo_score`。

任务定义中记录的 fingerprint type：`ECFP4`。

实现中使用 `_geometric_mean()`，合并 Sitagliptin ECFP4 去相似度、logP、TPSA 和分子式 modifier。
