# 目标

最大化 GuacaMol Median Molecules 2 score。

该 benchmark 根据候选分子与两个目标分子的相似度打分：tadalafil 和 sildenafil。score 使用几何平均合并候选分子与 tadalafil 的 ECFP6 Tanimoto 相似度，以及候选分子与 sildenafil 的 ECFP6 Tanimoto 相似度。

在本地 loss-minimization 接口中，优化 `median2_loss = 1 - median2_score`。
