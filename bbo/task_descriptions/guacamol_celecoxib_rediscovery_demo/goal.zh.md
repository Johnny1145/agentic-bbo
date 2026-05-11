# 目标

最小化 `celecoxib_rediscovery_loss`，其中：

- `celecoxib_rediscovery_loss = 1.0 - celecoxib_rediscovery_score`
- `celecoxib_rediscovery_score` 是与 Celecoxib 的阈值化 ECFP4 Tanimoto 相似度

在 GuacaMol 相似度目标下，得分 `1.0` 表示精确 rediscovery。
