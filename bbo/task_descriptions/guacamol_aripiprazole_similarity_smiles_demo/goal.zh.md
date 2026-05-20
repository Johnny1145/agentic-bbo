# 目标

实现来源：`bbo/tasks/scientific/guacamol_smiles.py`，`_score_mol()`。

最大化 `aripiprazole_similarity_score`。

优化目标：最小化 `aripiprazole_similarity_loss = 1 - aripiprazole_similarity_score`。

任务定义中记录的 fingerprint type：`FCFP4`。

源 benchmark 字符串中记录的 threshold：`0.75`。
