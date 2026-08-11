# Goal

Maximize the GuacaMol Sitagliptin MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- ECFP4 similarity to sitagliptin, with a Gaussian modifier centered at 0, encouraging dissimilarity to sitagliptin;
- logP, with a Gaussian modifier centered at 2.0165;
- TPSA, with a Gaussian modifier centered at 77.04;
- molecular formula matching for C16H15F6N5O.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `sitagliptin_mpo_loss = 1 - sitagliptin_mpo_score`.
