# 目标

最小化 `fexofenadine_mpo_loss`，其中：

- `fexofenadine_mpo_loss = 1.0 - fexofenadine_mpo_score`
- `fexofenadine_mpo_score` 是 GuacaMol 风格的 Fexofenadine 相似度、TPSA 分量和 logP 分量的几何平均

loss 越低，表示解码分子越好地满足组合 MPO 目标。
