# Goal

Maximize the GuacaMol Perindopril MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- similarity to perindopril using ECFP4;
- number of aromatic rings, with a Gaussian modifier centered at 2.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `perindopril_mpo_loss = 1 - perindopril_mpo_score`.
