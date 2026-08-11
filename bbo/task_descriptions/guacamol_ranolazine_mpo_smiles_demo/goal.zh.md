# 目标

最大化 GuacaMol Ranolazine MPO score。

该 benchmark 将以下 benchmark 定义的 scoring components 合并为一个标量分子 score：

- 使用 atom-pair fingerprints (AP) 与 ranolazine 的相似度，并在 0.7 使用 thresholded modifier；
- logP，并使用中心为 7 的 MaxGaussian modifier；
- TPSA，并使用中心为 95 的 MaxGaussian modifier；
- 氟原子数量，并使用中心为 1 的 Gaussian modifier。

components 使用几何平均聚合。在本地 loss-minimization 接口中，优化 `ranolazine_mpo_loss = 1 - ranolazine_mpo_score`。
