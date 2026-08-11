# 目标

最大化 GuacaMol Median Molecules 1 score。

该 benchmark 根据候选分子与两个目标分子的相似度打分：camphor 和 menthol。score 使用几何平均合并候选分子与 camphor 的 ECFP4 Tanimoto 相似度，以及候选分子与 menthol 的 ECFP4 Tanimoto 相似度。

在本地 loss-minimization 接口中，优化 `median1_loss = 1 - median1_score`。
