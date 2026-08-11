# Goal

Maximize the GuacaMol Amlodipine MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- similarity to amlodipine using ECFP4;
- total number of rings, with a Gaussian modifier centered at 3.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `amlodipine_mpo_loss = 1 - amlodipine_mpo_score`.
