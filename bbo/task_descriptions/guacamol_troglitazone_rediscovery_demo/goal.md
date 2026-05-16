# Goal

Minimize `troglitazone_rediscovery_loss`, where:

- `troglitazone_rediscovery_loss = 1.0 - troglitazone_rediscovery_score`
- `troglitazone_rediscovery_score` is thresholded ECFP4 Tanimoto similarity to Troglitazone

A score of `1.0` corresponds to exact rediscovery under the GuacaMol similarity objective.
