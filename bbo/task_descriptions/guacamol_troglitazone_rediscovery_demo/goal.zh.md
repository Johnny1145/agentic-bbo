# 目标

最小化 `troglitazone_rediscovery_loss`，其中：

- `troglitazone_rediscovery_loss = 1.0 - troglitazone_rediscovery_score`
- `troglitazone_rediscovery_score` 是与 Troglitazone 的阈值化 ECFP4 Tanimoto 相似度

在 GuacaMol 相似度目标下，得分 `1.0` 表示精确 rediscovery。
