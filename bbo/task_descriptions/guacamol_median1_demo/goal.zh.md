# 目标

最小化 `median1_loss`，其中：

- `median1_loss = 1.0 - median1_score`
- `median1_score` 是与 camphor 和 menthol 的 ECFP4 Tanimoto 相似度的几何平均

loss 越低，表示解码分子越能在两个目标结构之间取得平衡。
