# Goal

Maximize the GuacaMol Osimertinib MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- similarity to osimertinib using FCFP4, with a thresholded modifier at 0.8;
- similarity to osimertinib using ECFP6, with a MinGaussian modifier centered at 0.85;
- TPSA, with a MaxGaussian modifier centered at 100;
- logP, with a MinGaussian modifier centered at 1.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `osimertinib_mpo_loss = 1 - osimertinib_mpo_score`.
