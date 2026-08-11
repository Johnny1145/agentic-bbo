# Goal

Maximize the GuacaMol Ranolazine MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- similarity to ranolazine using atom-pair fingerprints (AP), with a thresholded modifier at 0.7;
- logP, with a MaxGaussian modifier centered at 7;
- TPSA, with a MaxGaussian modifier centered at 95;
- number of fluorine atoms, with a Gaussian modifier centered at 1.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `ranolazine_mpo_loss = 1 - ranolazine_mpo_score`.
