# 目标

最大化 GuacaMol Fexofenadine MPO score。

该 benchmark 将以下 benchmark 定义的 scoring components 合并为一个标量分子 score：

- 使用 atom-pair fingerprints (AP) 与 fexofenadine 的相似度，并在 0.8 使用 thresholded modifier；
- TPSA，并使用中心为 90 的 MaxGaussian modifier；
- logP，并使用中心为 4 的 MinGaussian modifier。

components 使用几何平均聚合。在本地 loss-minimization 接口中，优化 `fexofenadine_mpo_loss = 1 - fexofenadine_mpo_score`。
