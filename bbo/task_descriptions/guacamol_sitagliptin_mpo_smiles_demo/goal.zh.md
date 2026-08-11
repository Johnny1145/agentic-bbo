# 目标

最大化 GuacaMol Sitagliptin MPO score。

该 benchmark 将以下 benchmark 定义的 scoring components 合并为一个标量分子 score：

- 与 sitagliptin 的 ECFP4 相似度，并使用中心为 0 的 Gaussian modifier，用于鼓励与 sitagliptin 不相似；
- logP，并使用中心为 2.0165 的 Gaussian modifier；
- TPSA，并使用中心为 77.04 的 Gaussian modifier；
- C16H15F6N5O 分子式匹配。

components 使用几何平均聚合。在本地 loss-minimization 接口中，优化 `sitagliptin_mpo_loss = 1 - sitagliptin_mpo_score`。
