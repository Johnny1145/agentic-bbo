# Goal

Maximize the GuacaMol Median Molecules 2 score.

This benchmark scores a molecule by its similarity to two target molecules: tadalafil and sildenafil. The score combines ECFP6 Tanimoto similarity to tadalafil and ECFP6 Tanimoto similarity to sildenafil using a geometric mean.

In the local loss-minimization interface, optimize `median2_loss = 1 - median2_score`.
