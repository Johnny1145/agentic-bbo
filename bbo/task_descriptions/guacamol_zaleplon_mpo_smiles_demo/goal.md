# Goal

Maximize the GuacaMol Zaleplon MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- similarity to zaleplon using ECFP4;
- molecular formula matching for C19H17N3O2.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `zaleplon_mpo_loss = 1 - zaleplon_mpo_score`.
