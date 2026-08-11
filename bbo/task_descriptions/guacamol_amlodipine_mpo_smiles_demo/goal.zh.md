# 目标

最大化 GuacaMol Amlodipine MPO score。

该 benchmark 将以下 benchmark 定义的 scoring components 合并为一个标量分子 score：

- 使用 ECFP4 与 amlodipine 的相似度；
- 总环数量，并使用中心为 3 的 Gaussian modifier。

components 使用几何平均聚合。在本地 loss-minimization 接口中，优化 `amlodipine_mpo_loss = 1 - amlodipine_mpo_score`。
