# 目标

最大化 GuacaMol Zaleplon MPO score。

该 benchmark 将以下 benchmark 定义的 scoring components 合并为一个标量分子 score：

- 使用 ECFP4 与 zaleplon 的相似度；
- C19H17N3O2 分子式匹配。

components 使用几何平均聚合。在本地 loss-minimization 接口中，优化 `zaleplon_mpo_loss = 1 - zaleplon_mpo_score`。
