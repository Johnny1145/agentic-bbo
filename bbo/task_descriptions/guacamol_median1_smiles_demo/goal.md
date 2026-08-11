# Goal

Maximize the GuacaMol Median Molecules 1 score.

This benchmark scores a molecule by its similarity to two target molecules: camphor and menthol. The score combines ECFP4 Tanimoto similarity to camphor and ECFP4 Tanimoto similarity to menthol using a geometric mean.

In the local loss-minimization interface, optimize `median1_loss = 1 - median1_score`.
