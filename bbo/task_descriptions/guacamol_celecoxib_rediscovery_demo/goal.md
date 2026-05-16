# Goal

Minimize `celecoxib_rediscovery_loss`, where:

- `celecoxib_rediscovery_loss = 1.0 - celecoxib_rediscovery_score`
- `celecoxib_rediscovery_score` is thresholded ECFP4 Tanimoto similarity to Celecoxib

A score of `1.0` corresponds to exact rediscovery under the GuacaMol similarity objective.
