# 目标

实现来源：`bbo/tasks/scientific/guacamol_smiles.py`，`_score_mol()`。

最大化 `perindopril_mpo_score`。

优化目标：最小化 `perindopril_mpo_loss = 1 - perindopril_mpo_score`。

任务定义中记录的 fingerprint type：`ECFP4`。

实现中使用 `_geometric_mean()`，合并 Perindopril ECFP4 相似度和两个芳香环数量 modifier。
