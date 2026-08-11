# Goal

Maximize the GuacaMol Valsartan SMARTS score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- presence of a valsartan-related SMARTS pattern;
- logP, with a Gaussian modifier centered at 2.0165;
- TPSA, with a Gaussian modifier centered at 77.04;
- Bertz complexity, with a Gaussian modifier centered at 896.38.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `valsartan_smarts_loss = 1 - valsartan_smarts_score`.
