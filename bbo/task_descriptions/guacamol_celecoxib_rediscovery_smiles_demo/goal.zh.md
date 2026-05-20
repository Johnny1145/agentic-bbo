# 目标

实现来源：`bbo/tasks/scientific/guacamol_smiles.py`，`_score_mol()`。

最大化 `celecoxib_rediscovery_score`。

优化目标：最小化 `celecoxib_rediscovery_loss = 1 - celecoxib_rediscovery_score`。

任务定义中记录的 fingerprint type：`ECFP4`。
