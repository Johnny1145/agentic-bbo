# 目标

实现来源：`bbo/tasks/scientific/guacamol_smiles.py`，`_score_mol()`。

最大化 `osimertinib_mpo_score`。

优化目标：最小化 `osimertinib_mpo_loss = 1 - osimertinib_mpo_score`。

任务定义中记录的 fingerprint type：`FCFP4, ECFP6`。

实现中使用 `_geometric_mean()`，合并 Osimertinib FCFP4 相似度、ECFP6 去相似度、TPSA 和 logP modifier。
