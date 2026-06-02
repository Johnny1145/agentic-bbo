# 目标

实现来源：`bbo/tasks/scientific/guacamol_smiles.py`，`_score_mol()`。

最大化 `valsartan_smarts_score`。

优化目标：最小化 `valsartan_smarts_loss = 1 - valsartan_smarts_score`。

任务定义中记录的 fingerprint type：`none`。

实现中使用 `_geometric_mean()`，合并 Valsartan SMARTS 匹配、logP、TPSA 和 Bertz complexity modifier。
