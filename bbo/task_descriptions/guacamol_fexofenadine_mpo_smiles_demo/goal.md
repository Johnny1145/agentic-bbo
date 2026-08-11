# Goal

Maximize the GuacaMol Fexofenadine MPO score.

This benchmark combines the following benchmark-defined scoring components into one scalar molecule score:

- similarity to fexofenadine using atom-pair fingerprints (AP), with a thresholded modifier at 0.8;
- TPSA, with a MaxGaussian modifier centered at 90;
- logP, with a MinGaussian modifier centered at 4.

The components are aggregated by a geometric mean. In the local loss-minimization interface, optimize `fexofenadine_mpo_loss = 1 - fexofenadine_mpo_score`.
