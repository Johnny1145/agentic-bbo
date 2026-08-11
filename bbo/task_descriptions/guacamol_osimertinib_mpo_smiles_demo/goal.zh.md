# 目标

最大化 GuacaMol Osimertinib MPO score。

该 benchmark 将以下 benchmark 定义的 scoring components 合并为一个标量分子 score：

- 使用 FCFP4 与 osimertinib 的相似度，并在 0.8 使用 thresholded modifier；
- 使用 ECFP6 与 osimertinib 的相似度，并使用中心为 0.85 的 MinGaussian modifier；
- TPSA，并使用中心为 100 的 MaxGaussian modifier；
- logP，并使用中心为 1 的 MinGaussian modifier。

components 使用几何平均聚合。在本地 loss-minimization 接口中，优化 `osimertinib_mpo_loss = 1 - osimertinib_mpo_score`。
